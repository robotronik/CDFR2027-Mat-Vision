"""API REST (Flask) exposant le mat de vision sur le réseau local.

L'interface web est servie sur ``/`` (négociation de contenu : un navigateur
reçoit la page HTML, ``curl`` reçoit le JSON de découverte) et sur ``/ui``.
Le reste répond en JSON, sauf ``/preview`` (JPEG) et ``/stream`` (MJPEG).

La liste des routes n'est écrite qu'à un seul endroit : la fonction
``_discovery`` ci-dessous.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from . import __version__
from .vision import VisionEngine

log = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(engine: VisionEngine, *, cors: bool = True) -> Flask:
    """Construit l'application Flask autour d'un :class:`VisionEngine`."""
    app = Flask(__name__)
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

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _json(payload: Any, status: int = 200) -> Response:
        response = jsonify(payload)
        response.status_code = status
        response.headers["Cache-Control"] = "no-store"
        return response

    def _preview_response(quality: int | None) -> Response:
        data = engine.preview_jpeg(quality)
        if data is None:
            return _json({"message": "aucune image disponible"}, 503)
        return Response(
            data,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    # ------------------------------------------------------------------ #
    # Découverte & interface web
    # ------------------------------------------------------------------ #
    def _discovery() -> dict[str, Any]:
        return {
            "name": "matvision",
            "version": __version__,
            "description": "Mat de vision ArUco (LattePanda Delta + Logitech 4K Stream)",
            "ui": "/ui",
            "endpoints": {
                "GET /": "interface web (navigateur) ou ce JSON (curl)",
                "GET /ui": "interface web de pilotage",
                "GET /health": "test de vie",
                "GET /status": "état complet du moteur",
                "GET|POST /start": "démarre la détection",
                "GET|POST /stop": "arrête la détection",
                "GET|POST /reset": "réinitialise le suivi",
                "GET /objects": "objets détectés (repère table, mm)",
                "GET /objects/<clé>": "objets d'un tag ou d'un libellé",
                "GET /position": "pose de la caméra dans le repère table",
                "GET /table": "géométrie de la table et tags de coin",
                "GET /preview": "image annotée (JPEG)",
                "GET /stream": "flux MJPEG temps réel",
                "GET /config": "configuration effective",
                "POST /calibration/start": "calibration automatique (damier)",
                "GET /calibration/status": "progression de la calibration",
                "POST /calibration/stop": "annule la calibration",
                "GET /calibration/result": "intrinsèques courantes",
                "POST /snapshot": "enregistre l'image courante sur le disque",
            },
        }

    @app.get("/")
    def index() -> Response:
        """Interface web pour un navigateur, découverte JSON pour le reste."""
        best = request.accept_mimetypes.best_match(["text/html", "application/json"])
        if best == "text/html" and (
            request.accept_mimetypes["text/html"] > request.accept_mimetypes["application/json"]
        ):
            return _page()
        return _json(_discovery())

    @app.get("/ui")
    def ui() -> Response:
        return _page()

    @app.get("/api")
    def api_index() -> Response:
        return _json(_discovery())

    def _page() -> Response:
        response = Response(render_template("index.html", version=__version__), mimetype="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response

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
    # Aperçu
    # ------------------------------------------------------------------ #
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
