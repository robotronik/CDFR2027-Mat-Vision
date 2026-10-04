"""Flotte de robots pilotée par le mat.

Le mat pilote le robot principal, un robot « chasseur » et un essaim de petits
robots à travers leur API REST embarquée (``/get_robot``, ``/get_strategies``,
``/set_strat``, ``/set_color``). Le robot principal est vu par la caméra grâce à
son tag ArUco ; le chasseur et les petits robots n'ont pas de tag : ils
déclarent leur position à chaque requête (``POST /fleet/report``) et reçoivent
en retour les éléments de jeu détectés par le mat.

Ce module ne dépend que de la bibliothèque standard (``urllib``) : aucune
dépendance supplémentaire n'est ajoutée au mat.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import RobotConfig, RobotsConfig
from .vision import VisionEngine

log = logging.getLogger(__name__)

__all__ = ["Fleet", "FleetError", "RobotTarget"]

#: Correspondance entre l'entier renvoyé par le robot et le nom de couleur.
COLOR_NAMES: dict[int, str] = {0: "", 1: "blue", 2: "yellow"}
#: Correspondance inverse (nom de couleur -> entier attendu par ``/set_color``).
COLOR_IDS: dict[str, int] = {"blue": 1, "yellow": 2}
#: Couleur opposée, utilisée pour désigner le robot adverse.
OPPOSITE_COLOR: dict[str, str] = {"blue": "yellow", "yellow": "blue", "": ""}

#: Durée de validité du cache d'état d'un robot (s) : évite d'interroger un robot
#: plusieurs fois pour la même image d'interface.
STATUS_CACHE_S = 1.0
#: Garde-fou sur le nombre de points conservés par trajectoire.
MAX_PATH_POINTS = 600


class FleetError(Exception):
    """Erreur de pilotage d'un robot, convertie en réponse HTTP par l'API."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class RobotTarget:
    """Un robot de la flotte : clé stable, rôle et configuration."""

    key: str                       # "main", "hunter" ou "swarm/<index>"
    role: str                      # "main" | "hunter" | "swarm"
    index: int | None
    config: RobotConfig

    @property
    def name(self) -> str:
        return self.config.name or self.key

    @property
    def configured(self) -> bool:
        """Un robot sans adresse n'est jamais contacté (aucun appel réseau)."""
        return bool(self.config.host) and self.config.enabled

    @property
    def base_url(self) -> str:
        return self.config.base_url


class Fleet:
    """Client REST de la flotte + suivi des trajectoires."""

    def __init__(self, config: RobotsConfig, engine: VisionEngine) -> None:
        self.config = config
        self.engine = engine
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._paths: dict[str, deque[tuple[float, float, float, float]]] = {}
        self._reports: dict[str, dict[str, float]] = {}
        self._last_error: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Cibles
    # ------------------------------------------------------------------ #
    def targets(self) -> list[RobotTarget]:
        result: list[RobotTarget] = []
        for key, robot in self.config.targets():
            role, _, index = key.partition("/")
            result.append(
                RobotTarget(
                    key=key,
                    role=role,
                    index=int(index) if index else None,
                    config=robot,
                )
            )
        return result

    def _target(self, key: str) -> RobotTarget | None:
        for target in self.targets():
            if target.key == key:
                return target
        return None

    def _resolve(self, key: str) -> list[RobotTarget]:
        """Résout une clé d'interface vers une ou plusieurs cibles.

        ``all`` s'applique à toute la flotte (couleur commune) ; ``swarm`` aux
        petits robots ; ``main`` et ``hunter`` désignent les robots uniques ;
        ``swarm/<i>`` un petit robot précis ; tout autre texte est comparé au nom
        du robot.
        """
        key = (key or "").strip()
        if key == "all":
            matches = self.targets()
        elif key == "swarm":
            matches = [t for t in self.targets() if t.role == "swarm"]
        elif key in ("main", "hunter"):
            target = self._target(key)
            matches = [target] if target else []
        elif key.startswith("swarm/"):
            target = self._target(key)
            matches = [target] if target else []
        else:
            matches = [t for t in self.targets() if t.name.lower() == key.lower()]
        if not matches:
            raise FleetError(f"robot inconnu : {key!r}", 404)
        return matches

    # ------------------------------------------------------------------ #
    # Appels REST vers les robots
    # ------------------------------------------------------------------ #
    def _request(
        self,
        base_url: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urlrequest.Request(
            base_url.rstrip("/") + path, data=data, headers=headers, method=method
        )
        try:
            with urlrequest.urlopen(request, timeout=self.config.request_timeout_s) as response:
                raw = response.read()
        except urlerror.HTTPError as exc:
            # Le robot renvoie un message exploitable (ex. « Cannot change the
            # strategy in the current state ») : on le relaie à l'interface.
            detail = ""
            try:
                body = exc.read().decode("utf-8", "replace")
                detail = (json.loads(body) or {}).get("message", "") if body else ""
            except (ValueError, AttributeError):
                detail = ""
            raise FleetError(detail or f"HTTP {exc.code}", exc.code) from exc
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8", "replace"))
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    def _fetch_state(self, target: RobotTarget) -> dict[str, Any]:
        """État d'un robot : en ligne, couleur, stratégie courante et liste."""
        if not target.configured:
            return {
                "key": target.key,
                "name": target.name,
                "role": target.role,
                "index": target.index,
                "host": target.config.host,
                "port": target.config.port,
                "configured": False,
                "online": False,
                "error": "adresse non renseignée",
            }

        now = time.monotonic()
        cached = self._cache.get(target.key)
        if cached is not None and now - cached[0] < STATUS_CACHE_S:
            return cached[1]

        state: dict[str, Any] = {
            "key": target.key,
            "name": target.name,
            "role": target.role,
            "index": target.index,
            "host": target.config.host,
            "port": target.config.port,
            "configured": True,
        }
        try:
            robot = self._request(target.base_url, "/get_robot")
        except (FleetError, urlerror.URLError, OSError, ValueError) as exc:
            state.update({"online": False, "error": getattr(exc, "message", str(exc))})
        else:
            color_id = int(robot.get("team") or 0)
            state.update(
                {
                    "online": True,
                    "color_id": color_id,
                    "color": COLOR_NAMES.get(color_id, ""),
                    "strategy": robot.get("strategy") or "",
                    "status": robot.get("status"),
                    "score": robot.get("score"),
                    "time": robot.get("time"),
                    "runid": robot.get("runid"),
                    "error": "",
                }
            )
            try:
                listed = self._request(target.base_url, "/get_strategies")
                state["strategies"] = [
                    name for name in (listed.get("strategies") or []) if name
                ]
            except (FleetError, urlerror.URLError, OSError, ValueError) as exc:
                state["strategies"] = []
                log.debug("Stratégies indisponibles pour %s : %s", target.key, exc)

        self._cache[target.key] = (now, state)
        return state

    def _invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def _cached_state(self, key: str) -> dict[str, Any]:
        """Dernier état connu d'un robot, sans aucun appel réseau."""
        with self._lock:
            cached = self._cache.get(key)
        return cached[1] if cached is not None else {}

    # ------------------------------------------------------------------ #
    # Couleur de l'équipe / robot adverse
    # ------------------------------------------------------------------ #
    def our_color(self, *, allow_fetch: bool = True) -> str:
        """Couleur de notre équipe : lue sur le robot principal, sinon configurée.

        La vue live appelle cette méthode avec ``allow_fetch=False`` : elle ne
        déclenche alors aucun appel réseau (elle se contente du dernier état
        connu) pour rester fluide même si un robot est injoignable.
        """
        main = self._target("main")
        if main is not None and main.configured:
            if allow_fetch:
                color = self._fetch_state(main).get("color") or ""
            else:
                color = self._cached_state(main.key).get("color") or ""
            if color:
                return color
        color = (self.config.our_color or "").strip().lower()
        return color if color in COLOR_IDS else ""

    def opponent_color(self, *, allow_fetch: bool = True) -> str:
        return OPPOSITE_COLOR.get(self.our_color(allow_fetch=allow_fetch), "")

    # ------------------------------------------------------------------ #
    # API de pilotage
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        robots = [self._fetch_state(target) for target in self.targets()]
        return {
            "robots": robots,
            "our_color": self.our_color(),
            "opponent_color": self.opponent_color(),
            "table": self._table_info(),
            "map_image": self.config.map_image or None,
        }

    def strategies(self) -> dict[str, Any]:
        return {
            "robots": [
                {
                    "key": state["key"],
                    "name": state["name"],
                    "role": state["role"],
                    "index": state["index"],
                    "online": state.get("online", False),
                    "strategy": state.get("strategy", ""),
                    "strategies": state.get("strategies", []),
                }
                for state in (self._fetch_state(t) for t in self.targets())
            ]
        }

    def set_strategy(self, key: str, strategy: str) -> dict[str, Any]:
        return self._apply(key, "/set_strat", {"strat": strategy})

    def set_color(self, key: str, color: int | str) -> dict[str, Any]:
        color_id = self._color_id(color)
        if color_id not in COLOR_IDS.values():
            raise FleetError("couleur invalide (1 = bleu, 2 = jaune)", 400)
        return self._apply(key, "/set_color", {"color": color_id})

    def _apply(self, key: str, path: str, payload: dict) -> dict[str, Any]:
        targets = self._resolve(key)
        results: list[dict[str, Any]] = []
        failed = False
        for target in targets:
            if not target.configured:
                results.append({"key": target.key, "ok": False, "message": "adresse non renseignée"})
                failed = True
                continue
            try:
                self._request(target.base_url, path, method="POST", payload=payload)
            except (FleetError, urlerror.URLError, OSError, ValueError) as exc:
                results.append(
                    {
                        "key": target.key,
                        "ok": False,
                        "status": getattr(exc, "status", 502),
                        "message": getattr(exc, "message", str(exc)),
                    }
                )
                failed = True
            else:
                self._invalidate(target.key)
                results.append({"key": target.key, "ok": True, "message": "ok"})
        if failed and all(not r["ok"] for r in results):
            raise FleetError(results[0]["message"], results[0].get("status", 502))
        return {"ok": not failed, "results": results}

    @staticmethod
    def _color_id(color: int | str) -> int:
        if isinstance(color, str):
            return COLOR_IDS.get(color.strip().lower(), 0)
        try:
            return int(color)
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------ #
    # Déclarations de position (chasseur et essaim)
    # ------------------------------------------------------------------ #
    def report(self, key: str, x: float, y: float, a: float = 0.0) -> dict[str, Any]:
        """Enregistre la position déclarée d'un robot sans tag.

        Le robot reçoit en retour les éléments de jeu vus par le mat.
        """
        targets = self._resolve(key)
        if len(targets) != 1:
            raise FleetError("précisez un robot (clé ou nom)", 400)
        target = targets[0]
        if target.role == "main":
            raise FleetError("le robot principal est vu par la caméra", 400)

        now = time.monotonic()
        position = {"x": float(x), "y": float(y), "a": float(a), "ts": time.time()}
        with self._lock:
            self._reports[target.key] = position
            self._push_path(target.key, float(x), float(y), float(a), now)

        live = self.live()
        return {
            "robot": target.key,
            "name": target.name,
            "position": position,
            "our_color": live["our_color"],
            "opponent_color": live["opponent_color"],
            "objects": live["objects"],
            "opponents": [
                {"id": o["id"], "x": o["x"], "y": o["y"], "a": o["a"]}
                for o in live["opponents"]
            ],
        }

    # ------------------------------------------------------------------ #
    # Vue temps réel (table live)
    # ------------------------------------------------------------------ #
    def live(self) -> dict[str, Any]:
        now = time.monotonic()
        payload = self.engine.objects()
        objects = [obj for obj in payload["objects"] if obj.get("label") != ""]
        our_color = self.our_color(allow_fetch=False)
        opponent_color = self.opponent_color(allow_fetch=False)

        main_position = self._camera_position(objects, our_color)
        opponents = [obj for obj in objects if opponent_color and obj["label"] == opponent_color]
        game_objects = [
            obj for obj in objects if obj["label"] not in COLOR_IDS
        ]

        with self._lock:
            if main_position is not None:
                self._push_path("main", main_position["x"], main_position["y"], main_position.get("a", 0.0), now)
            for opponent in opponents:
                self._push_path(
                    f"opponent/{opponent['id']}",
                    opponent["x"],
                    opponent["y"],
                    opponent.get("a", 0.0),
                    now,
                )
            robots = self._live_robots(our_color, main_position)
            opponents_live = [
                {**self._public_object(opponent), "path": self._path(f"opponent/{opponent['id']}")}
                for opponent in opponents
            ]

        return {
            "our_color": our_color,
            "opponent_color": opponent_color,
            "table": self._table_info(),
            "map_image": self.config.map_image or None,
            "robots": robots,
            "opponents": opponents_live,
            "objects": [self._public_object(obj) for obj in game_objects],
            "timestamp": time.time(),
        }

    def _live_robots(self, our_color: str, main_position: dict | None) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for target in self.targets():
            if target.role == "main":
                position = main_position
                source = "camera"
            else:
                position = self._reports.get(target.key)
                source = "report"
            color = our_color
            if target.configured:
                color = self._cached_state(target.key).get("color") or our_color
            entry: dict[str, Any] = {
                "key": target.key,
                "role": target.role,
                "index": target.index,
                "name": target.name,
                "color": color,
                "source": source,
                "online": bool(target.configured),
                "x": position.get("x") if position else None,
                "y": position.get("y") if position else None,
                "a": position.get("a", 0.0) if position else None,
                "path": self._path(target.key),
            }
            entries.append(entry)
        return entries

    @staticmethod
    def _public_object(obj: dict) -> dict[str, Any]:
        return {
            "id": obj.get("id"),
            "label": obj.get("label"),
            "x": obj.get("x"),
            "y": obj.get("y"),
            "a": obj.get("a", 0.0),
        }

    def _camera_position(self, objects: list[dict], our_color: str) -> dict | None:
        if not our_color:
            return None
        candidates = [obj for obj in objects if obj["label"] == our_color]
        main = self._target("main")
        if main is not None and main.config.tag is not None:
            tagged = [obj for obj in candidates if obj.get("id") == main.config.tag]
            candidates = tagged or candidates
        return candidates[0] if candidates else None

    def _table_info(self) -> dict[str, Any]:
        table = self.engine.table_info()
        return {
            "width_mm": table.get("width_mm"),
            "height_mm": table.get("height_mm"),
            "test_mode": bool(table.get("test_mode")),
        }

    # ------------------------------------------------------------------ #
    # Trajectoires
    # ------------------------------------------------------------------ #
    def _push_path(self, key: str, x: float, y: float, a: float, now: float) -> None:
        path = self._paths.setdefault(key, deque(maxlen=MAX_PATH_POINTS))
        path.append((now, x, y, a))
        self._prune_path(key, now)

    def _prune_path(self, key: str, now: float) -> None:
        path = self._paths.get(key)
        if not path:
            return
        window = max(1.0, self.config.path_window_s)
        while path and now - path[0][0] > window:
            path.popleft()

    def _path(self, key: str) -> list[list[float]]:
        path = self._paths.get(key)
        if not path:
            return []
        return [[round(x, 1), round(y, 1)] for _, x, y, _ in path]

    # ------------------------------------------------------------------ #
    # Déclaration côté serveur (arrêt propre)
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        with self._lock:
            self._paths.clear()
            self._reports.clear()
            self._cache.clear()
