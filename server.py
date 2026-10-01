#!/usr/bin/env python3
"""Serveur du mat de vision : API REST + détection ArUco.

Exemples
--------
Démarrage simple (détection lancée automatiquement) ::

    python server.py

Caméra et port personnalisés, sans fenêtre ::

    python server.py --device 0 --port 5000 --headless --autostart

Depuis un autre poste du réseau local ::

    curl http://<ip-lattepanda>:5000/objects
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matvision.api import create_app  # noqa: E402
from matvision.config import load_config  # noqa: E402
from matvision.vision import VisionEngine  # noqa: E402

log = logging.getLogger("server")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mat de vision - API REST + détection ArUco.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.json", help="Fichier de configuration JSON.")
    parser.add_argument("--host", default=None, help="Adresse d'écoute de l'API.")
    parser.add_argument("--port", type=int, default=None, help="Port de l'API.")
    parser.add_argument("--device", default=None, help="Index ou chemin de la caméra.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--autostart", action="store_true", help="Démarre la détection sans attendre /start.")
    parser.add_argument("--display", action="store_true", help="Affiche la fenêtre de prévisualisation locale.")
    parser.add_argument("--no-cors", action="store_true", help="Désactive l'en-tête CORS.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)

    if args.host:
        config.api_host = args.host
    if args.port:
        config.api_port = args.port
    if args.device is not None:
        config.camera.device = int(args.device) if str(args.device).isdigit() else args.device
    if args.width:
        config.camera.width = args.width
    if args.height:
        config.camera.height = args.height

    for warning in config.validate():
        log.warning("configuration : %s", warning)

    engine = VisionEngine(config, display=args.display)
    engine.start()
    if args.autostart:
        engine.start_detection()

    app = create_app(engine, cors=not args.no_cors)

    log.info("API disponible sur http://%s:%d/", config.api_host, config.api_port)
    log.info("Prévisualisation : http://%s:%d/preview", config.api_host, config.api_port)

    stopping = threading.Event()

    def _handle_signal(_signum, _frame) -> None:
        if stopping.is_set():
            return
        stopping.set()
        log.info("Arrêt demandé...")
        engine.shutdown()
        # werkzeug n'expose pas d'arrêt propre : on termine le processus.
        threading.Thread(target=lambda: sys.exit(0), daemon=True).start()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        app.run(
            host=config.api_host,
            port=config.api_port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
