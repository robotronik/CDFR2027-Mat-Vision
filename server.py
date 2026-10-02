#!/usr/bin/env python3
"""Serveur du mat de vision : API REST + détection ArUco.

Exemples
--------
Démarrage simple (détection lancée automatiquement) ::

    python server.py

Caméra et port personnalisés, avec fenêtre locale ::

    python server.py --device 1 --port 5000 --display

Mode test — vérifier la détection des tags sans les tags de coin de la table ::

    python server.py --test

Mode match — performance maximale, sans interface web ni aperçu image
(détection lancée automatiquement, repère table via les 4 tags de coin) ::

    python server.py --match

Depuis un autre poste du réseau local ::

    curl http://<ip-lattepanda>:5000/objects

Arrêt du processus
------------------
Trois moyens équivalents, tous propres (moteur arrêté, caméra libérée) :

* ``Ctrl+C`` dans le terminal qui a lancé le serveur ;
* ``kill <pid>`` (SIGTERM) — le PID est affiché au démarrage ;
* ``curl -X POST http://<ip>:5000/shutdown``, ou le bouton « Arrêter le serveur »
  de l'interface web.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from werkzeug.serving import make_server  # noqa: E402

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
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Mode test : relève tous les tags en pixels, sans exiger les tags de coin "
            "de la table (détection lancée automatiquement)."
        ),
    )
    parser.add_argument(
        "--match",
        action="store_true",
        help=(
            "Mode match : désactive l'interface web et l'aperçu/flux image ainsi que "
            "les annotations et le HUD, pour accélérer le traitement. Seules les routes "
            "JSON (/objects, /status…) restent servies. La détection démarre "
            "automatiquement : recherche des 4 tags de coin de la table, puis relevé "
            "des objets."
        ),
    )
    parser.add_argument("--display", action="store_true", help="Affiche la fenêtre de prévisualisation locale.")
    parser.add_argument("--no-cors", action="store_true", help="Désactive l'en-tête CORS.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def autostart_requested(args: argparse.Namespace) -> bool:
    """La détection doit-elle démarrer sans attendre ``POST /start`` ?

    Vrai avec ``--autostart``, ``--test`` ou ``--match``. En mode match le moteur
    reste en détection **normale** : il cherche d'abord les 4 tags de coin pour se
    repérer dans le repère table, puis relève les objets.
    """
    return bool(args.autostart or args.test or args.match)


def main(argv: list[str] | None = None) -> int:
    boot_started = time.monotonic()
    args = parse_args(argv)
    # Journalisation sur la console (stderr) uniquement : aucun fichier de log n'est écrit.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)

    log.info(
        "Configuration : %s (%d objets, %d tags de coin, dessin=%s)",
        config_path if config_path.exists() else "valeurs par défaut",
        len(config.objects),
        len(config.table.markers),
        "oui" if config.detection.draw else "non",
    )

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

    if args.test:
        log.info(
            "Mode test : repère table ignoré, tous les tags sont relevés en pixels "
            "(les tags de coin ne sont pas nécessaires)."
        )

    if args.match:
        # Allège la boucle de traitement : plus d'annotation ni de HUD sur l'image.
        config.detection.draw = False
        log.info(
            "Mode match : interface web, aperçu/flux et annotations désactivés "
            "(API JSON seule), détection lancée automatiquement."
        )

    engine = VisionEngine(config, display=args.display, test_mode=args.test)
    engine.start()
    if autostart_requested(args):
        # En mode match, la détection « normale » cherche les 4 tags de coin pour
        # construire le repère table avant de relever les objets.
        engine.start_detection()
        log.info("Détection lancée automatiquement (sans attendre /start)")

    stopping = threading.Event()
    shutdown_thread: threading.Thread | None = None

    def run_shutdown() -> None:
        """Arrête le serveur HTTP, puis le moteur (donc la caméra)."""
        server.shutdown()          # rend la main quand serve_forever() s'arrête
        engine.shutdown()

    def request_stop(origin: str) -> None:
        """Déclenche l'arrêt, une seule fois, depuis un signal ou l'API."""
        nonlocal shutdown_thread
        if stopping.is_set():
            return
        stopping.set()
        log.info("Arrêt demandé (%s)", origin)
        shutdown_thread = threading.Thread(target=run_shutdown, name="shutdown", daemon=True)
        shutdown_thread.start()

    app = create_app(
        engine,
        cors=not args.no_cors,
        on_shutdown=lambda: request_stop("POST /shutdown"),
        web_ui=not args.match,
    )
    server = make_server(config.api_host, config.api_port, app, threaded=True)

    signal.signal(signal.SIGINT, lambda *_: request_stop("Ctrl+C"))
    signal.signal(signal.SIGTERM, lambda *_: request_stop("SIGTERM"))

    log.info(
        "API disponible sur http://%s:%d/ (démarrage en %.0f ms)",
        config.api_host,
        config.api_port,
        (time.monotonic() - boot_started) * 1000.0,
    )
    if args.match:
        log.info("Mode match        : aucune interface web, aperçu ni annotation")
    else:
        log.info("Interface web     : http://%s:%d/ui", config.api_host, config.api_port)
    log.info(
        "Arrêt             : Ctrl+C, « kill %d » ou « curl -X POST http://%s:%d/shutdown »",
        os.getpid(),
        config.api_host,
        config.api_port,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:  # filet de sécurité si le gestionnaire n'a pas pris
        request_stop("KeyboardInterrupt")

    if shutdown_thread is not None:
        shutdown_thread.join(timeout=5.0)
    else:
        engine.shutdown()
    log.info("Serveur arrêté")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
