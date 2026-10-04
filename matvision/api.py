"""API REST (Flask) exposant le mat de vision sur le réseau local.

L'interface web est servie sur ``/`` (négociation de contenu : un navigateur
reçoit la page HTML, ``curl`` reçoit le JSON de découverte) et sur ``/ui``.
Le reste répond en JSON, sauf ``/preview`` (JPEG) et ``/stream`` (MJPEG).

Avec ``create_app(..., web_ui=False)`` (mode ``--match`` du serveur), l'interface
web, les statiques et les sorties image ne sont pas enregistrées : seules les
routes JSON subsistent, ce qui évite tout encodage JPEG dans la boucle.

La liste des routes n'est écrite qu'à un seul endroit : la fonction
``_discovery`` ci-dessous.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from typing import Any, Callable

from flask import Flask, Response, g, jsonify, render_template, request

from . import __version__
from .fleet import Fleet, FleetError
from .vision import VisionEngine

log = logging.getLogger(__name__)

__all__ = ["create_app"]

#: Délai avant l'arrêt effectif, pour laisser la réponse HTTP partir.
SHUTDOWN_DELAY_S = 0.3

#: Au-delà de ce seuil (ms), une requête est signalée en WARNING.
SLOW_REQUEST_MS = 1000.0


def create_app(
    engine: VisionEngine,
    *,
    cors: bool = True,
    on_shutdown: Callable[[], None] | None = None,
    web_ui: bool = True,
) -> Flask:
    """Construit l'application Flask autour d'un :class:`VisionEngine`.

    :param on_shutdown: appelée par ``POST /shutdown`` pour arrêter le serveur.
        Sans ce rappel, la route répond ``501`` (utile en test).
    :param web_ui: quand ``False`` (mode match), l'interface web et les sorties
        image ne sont **pas** servies : ni page HTML, ni fichiers statiques, ni
        ``/preview``, ni ``/stream`` (donc aucun encodage JPEG). Seules les
        routes JSON subsistent, pour alléger la boucle de traitement.
    """
    app = Flask(__name__, static_folder="static" if web_ui else None)
    # Interface servie depuis un réseau local : on veut voir les mises à jour
    # immédiatement, donc pas de mise en cache des fichiers statiques.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    if cors:
        try:
            from flask_cors import CORS

            CORS(app, resources={r"/*": {"origins": "*"}})
        except ImportError:  # pragma: no cover - dépendance optionnelle
            log.warning("flask-cors absent : les requêtes cross-origin seront bloquées")

    config = engine.config

    log.info(
        "API Flask créée (interface web %s, CORS %s)",
        "activée" if web_ui else "désactivée",
        "activé" if cors else "désactivé",
    )

    # ------------------------------------------------------------------ #
    # Chronométrage des requêtes (console uniquement, aucun fichier de log)
    # ------------------------------------------------------------------ #
    @app.before_request
    def _start_request_timer() -> None:
        g.request_started = time.perf_counter()

    @app.after_request
    def _log_request(response: Response) -> Response:
        started = getattr(g, "request_started", None)
        if started is None or (response.mimetype or "").startswith("multipart"):
            return response
        duration_ms = (time.perf_counter() - started) * 1000.0
        level = logging.WARNING if duration_ms >= SLOW_REQUEST_MS else logging.DEBUG
        log.log(
            level,
            "%s %s -> %d en %.1f ms",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        return response

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _json(payload: Any, status: int = 200) -> Response:
        response = jsonify(payload)
        response.status_code = status
        response.headers["Cache-Control"] = "no-store"
        return response

    def _fleet_call(func: Callable[..., dict], *args: Any) -> Response:
        try:
            return _json(func(*args))
        except FleetError as exc:
            return _json({"message": exc.message}, exc.status)

    fleet = Fleet(config.robots, engine)

    # ------------------------------------------------------------------ #
    # Découverte & interface web
    # ------------------------------------------------------------------ #
    def _discovery() -> dict[str, Any]:
        endpoints: dict[str, str] = {
            "GET /health": "test de vie",
            "GET /status": "état complet du moteur",
            "GET|POST /start": "démarre la détection",
            "GET|POST /stop": "arrête la détection",
            "GET|POST /reset": "réinitialise le suivi",
            "GET /objects": "objets détectés (repère table, mm)",
            "GET /objects/<clé>": "objets d'un tag ou d'un libellé",
            "GET /position": "pose de la caméra dans le repère table",
            "GET /table": "géométrie de la table et tags de coin",
            "GET /config": "configuration effective",
            "POST /calibration/start": "calibration automatique (damier)",
            "GET /calibration/status": "progression de la calibration",
            "POST /calibration/stop": "annule la calibration",
            "GET /calibration/result": "intrinsèques courantes",
            "POST /snapshot": "enregistre l'image courante sur le disque",
            "POST /shutdown": "arrête le serveur et le processus",
            "GET /fleet": "état des robots (principal, chasseur, essaim)",
            "GET /fleet/strategies": "stratégies disponibles par robot",
            "POST /fleet/<cible>/strategy": "change la stratégie (cible = main|hunter|swarm|swarm/<i>)",
            "POST /fleet/<cible>/color": "change la couleur (mêmes cibles)",
            "GET /fleet/live": "positions et trajectoires pour la table live",
            "POST /fleet/report": "position déclarée d'un robot sans tag + objets de jeu",
        }
        if web_ui:
            endpoints.update(
                {
                    "GET /": "interface web (navigateur) ou ce JSON (curl)",
                    "GET /ui": "interface web de pilotage",
                    "GET /preview": "image annotée (JPEG)",
                    "GET /stream": "flux MJPEG temps réel",
                }
            )
        return {
            "name": "matvision",
            "version": __version__,
            "description": "Mat de vision ArUco (LattePanda Delta + Logitech 4K Stream)",
            "ui": "/ui" if web_ui else None,
            "match_mode": not web_ui,
            "endpoints": endpoints,
        }

    @app.get("/api")
    def api_index() -> Response:
        return _json(_discovery())

    if web_ui:

        def _preview_response(quality: int | None) -> Response:
            data = engine.preview_jpeg(quality)
            if data is None:
                return _json({"message": "aucune image disponible"}, 503)
            return Response(
                data,
                mimetype="image/jpeg",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        def _page() -> Response:
            response = Response(
                render_template("index.html", version=__version__), mimetype="text/html"
            )
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/")
        def index() -> Response:
            """Interface web pour un navigateur, découverte JSON pour le reste."""
            best = request.accept_mimetypes.best_match(["text/html", "application/json"])
            if best == "text/html" and (
                request.accept_mimetypes["text/html"]
                > request.accept_mimetypes["application/json"]
            ):
                return _page()
            return _json(_discovery())

        @app.get("/ui")
        def ui() -> Response:
            return _page()

    @app.get("/health")
    def health() -> Response:
        status = engine.status()
        return _json(
            {
                "ok": status["running"] and bool(status["camera"]["opened"]),
                "mode": status["mode"],
                "uptime_s": status["uptime_s"],
                "error": status["error"],
            }
        )

    # ------------------------------------------------------------------ #
    # État
    # ------------------------------------------------------------------ #
    @app.get("/status")
    def status() -> Response:
        return _json(engine.status())

    @app.get("/config")
    def get_config() -> Response:
        return _json(asdict(config))

    @app.get("/table")
    def table() -> Response:
        return _json(engine.table_info())

    @app.get("/position")
    def position() -> Response:
        return _json(engine.camera_position())

    # ------------------------------------------------------------------ #
    # Commandes
    # ------------------------------------------------------------------ #
    @app.route("/start", methods=["GET", "POST"])
    def start() -> Response:
        return _json(engine.start_detection())

    @app.route("/stop", methods=["GET", "POST"])
    def stop() -> Response:
        return _json(engine.stop())

    @app.route("/reset", methods=["GET", "POST"])
    @app.route("/reset_tracking", methods=["GET", "POST"])
    def reset() -> Response:
        return _json(engine.reset())

    # ------------------------------------------------------------------ #
    # Objets
    # ------------------------------------------------------------------ #
    @app.get("/objects")
    def objects() -> Response:
        return _json(engine.objects(label=request.args.get("label")))

    @app.get("/objects/<key>")
    def object_by_key(key: str) -> Response:
        """Objets d'un libellé (``blue``, ``element``…) ou d'un identifiant de tag (``13``)."""
        payload = engine.objects()
        matches = payload["by_label"].get(key) or [
            item for item in payload["objects"] if str(item["id"]) == key
        ]
        if not matches:
            return _json({"message": f"aucun objet pour {key!r}"}, 404)
        return _json({"count": len(matches), "objects": matches})

    # ------------------------------------------------------------------ #
    # Flotte de robots (principal, chasseur, essaim)
    # ------------------------------------------------------------------ #
    @app.get("/fleet")
    def fleet_status() -> Response:
        return _json(fleet.status())

    @app.get("/fleet/strategies")
    def fleet_strategies() -> Response:
        return _json(fleet.strategies())

    @app.post("/fleet/<path:target>/strategy")
    def fleet_set_strategy(target: str) -> Response:
        body = request.get_json(silent=True) or {}
        name = body.get("strat") or body.get("strategy") or request.args.get("strat")
        if not name:
            return _json({"message": "stratégie manquante"}, 400)
        return _fleet_call(fleet.set_strategy, target, str(name))

    @app.post("/fleet/<path:target>/color")
    def fleet_set_color(target: str) -> Response:
        body = request.get_json(silent=True) or {}
        color = body.get("color", request.args.get("color"))
        if color is None:
            return _json({"message": "couleur manquante"}, 400)
        return _fleet_call(fleet.set_color, target, color)

    @app.get("/fleet/live")
    def fleet_live() -> Response:
        return _json(fleet.live())

    @app.route("/fleet/report", methods=["GET", "POST"])
    def fleet_report() -> Response:
        """Position déclarée par un robot sans tag ; renvoie les objets de jeu."""
        body = request.get_json(silent=True) or request.args
        key = body.get("robot") or body.get("key")
        if not key:
            return _json({"message": "robot manquant"}, 400)
        try:
            x = float(body["x"])
            y = float(body["y"])
            a = float(body.get("a", 0.0))
        except (KeyError, TypeError, ValueError):
            return _json({"message": "position invalide (x, y requis)"}, 400)
        return _fleet_call(fleet.report, str(key), x, y, a)

    # ------------------------------------------------------------------ #
    # Aperçu (absent en mode match : aucun encodage JPEG)
    # ------------------------------------------------------------------ #
    if web_ui:

        @app.get("/preview")
        @app.get("/preview.jpg")
        def preview() -> Response:
            quality = request.args.get("quality", type=int)
            return _preview_response(quality)

        @app.get("/stream")
        def stream() -> Response:
            """Flux MJPEG temps réel, consommable directement par une balise ``<img>``.

            Paramètres : ``fps`` (défaut 10), ``quality`` (défaut config),
            ``frames`` (0 = illimité) et ``max_seconds`` (garde-fou, défaut 300 s).
            """
            quality = request.args.get("quality", type=int)
            fps = request.args.get("fps", type=float, default=10.0) or 10.0
            max_frames = request.args.get("frames", type=int, default=0) or 0
            max_seconds = request.args.get("max_seconds", type=float, default=300.0) or 0.0
            interval = 1.0 / max(1.0, min(float(fps), 60.0))

            def generate():
                boundary = b"--frame\r\n"
                started = time.monotonic()
                sent = 0
                while True:
                    if max_frames and sent >= max_frames:
                        break
                    if max_seconds and (time.monotonic() - started) > max_seconds:
                        break
                    data = engine.preview_jpeg(quality)
                    if data is None:
                        time.sleep(0.2)
                        continue
                    yield (
                        boundary
                        + b"Content-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode("ascii")
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                    sent += 1
                    time.sleep(interval)

            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )

    @app.post("/snapshot")
    def snapshot() -> Response:
        from .config import DATA_DIR

        target = request.args.get("path")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not target:
            import time as _time

            target = str(DATA_DIR / f"snapshot_{int(_time.time())}.jpg")
        ok = engine.save_frame(target)
        if not ok:
            return _json({"message": "aucune image à enregistrer"}, 503)
        return _json({"message": "image enregistrée", "path": target})

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    @app.route("/calibration/start", methods=["GET", "POST"])
    def calibration_start() -> Response:
        return _json(engine.start_calibration())

    @app.get("/calibration/status")
    def calibration_status() -> Response:
        return _json(engine.calibration_status())

    @app.route("/calibration/stop", methods=["GET", "POST"])
    @app.route("/calibration/cancel", methods=["GET", "POST"])
    def calibration_stop() -> Response:
        return _json(engine.cancel_calibration())

    @app.get("/calibration/result")
    def calibration_result() -> Response:
        intrinsics = engine.reload_intrinsics()
        if intrinsics is None:
            return _json({"message": "aucune calibration disponible", "file": str(config.intrinsics_path)}, 404)
        return _json(intrinsics.summary())

    # ------------------------------------------------------------------ #
    # Arrêt
    # ------------------------------------------------------------------ #
    @app.post("/shutdown")
    def shutdown() -> Response:
        """Arrête le moteur puis le processus (arrêt différé de ``SHUTDOWN_DELAY_S``)."""
        if on_shutdown is None:
            return _json({"message": "arrêt non câblé sur ce serveur"}, 501)

        timer = threading.Timer(SHUTDOWN_DELAY_S, on_shutdown)
        timer.daemon = True
        timer.start()
        log.info("Arrêt demandé via POST /shutdown")
        return _json({"message": "arrêt du serveur en cours"})

    # ------------------------------------------------------------------ #
    # Erreurs
    # ------------------------------------------------------------------ #
    @app.errorhandler(404)
    def not_found(_error: Any) -> Response:
        return _json({"message": "route inconnue"}, 404)

    @app.errorhandler(500)
    def server_error(error: Any) -> Response:  # pragma: no cover
        log.exception("Erreur non gérée : %s", error)
        return _json({"message": "erreur interne", "detail": str(error)}, 500)

    return app
