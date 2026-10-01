#!/usr/bin/env python3
"""Calibration automatique de la caméra du mat de vision.

Deux usages :

* **intrinsèques** (par défaut) — présentez un damier devant la caméra : le
  script capture automatiquement les vues exploitables, calcule la matrice
  intrinsèque et les coefficients de distorsion, puis les enregistre.

  .. code-block:: bash

     python calibrate.py --device 0 --frames 20

* **vérification du repère table** (``--check-table``) — positionnez les
  4 tags de coin, le script contrôle qu'ils sont tous vus et affiche la qualité
  de l'homographie (résidu en mm) ainsi que la pose de la caméra.

  .. code-block:: bash

     python calibrate.py --check-table
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from matvision.aruco import ArucoDetector, draw_markers  # noqa: E402
from matvision.calibration import (  # noqa: E402
    AutoCalibrator,
    load_intrinsics,
    save_intrinsics,
)
from matvision.camera import Camera, CameraError, list_video_devices  # noqa: E402
from matvision.config import Config, load_config  # noqa: E402
from matvision.table import localize_table  # noqa: E402

log = logging.getLogger("calibrate")


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration automatique de la caméra (damier) du mat de vision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.json", help="Fichier de configuration JSON.")
    parser.add_argument("--device", default=None, help="Index ou chemin du périphérique vidéo.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None, help="Nombre de vues à capturer.")
    parser.add_argument("--grid", type=int, nargs=2, metavar=("COLS", "ROWS"), default=None,
                        help="Nombre de coins internes du damier.")
    parser.add_argument("--square-size", type=float, default=None, help="Taille d'une case du damier (mm).")
    parser.add_argument("--output", default=None, help="Fichier de sortie des intrinsèques.")
    parser.add_argument("--timeout", type=float, default=None, help="Durée maximale de la session (s).")
    parser.add_argument("--no-display", action="store_true", help="Mode sans fenêtre (headless).")
    parser.add_argument("--check-table", action="store_true",
                        help="Contrôle des 4 tags de coin au lieu de la calibration intrinsèque.")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Durée d'observation pour --check-table (0 = jusqu'à 'q').")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Applique les surcharges de la ligne de commande à la configuration."""
    if args.device is not None:
        config.camera.device = int(args.device) if str(args.device).isdigit() else args.device
    if args.width:
        config.camera.width = args.width
    if args.height:
        config.camera.height = args.height
    if args.frames:
        config.calibration.target_frames = args.frames
    if args.grid:
        config.calibration.chessboard_cols, config.calibration.chessboard_rows = args.grid
    if args.square_size:
        config.calibration.square_size_mm = args.square_size
    if args.output:
        config.calibration.intrinsics_file = args.output
    if args.timeout:
        config.calibration.max_capture_seconds = args.timeout
    return config


# --------------------------------------------------------------------------- #
def run_intrinsics(config: Config, display: bool) -> int:
    """Session de calibration intrinsèque."""
    calibrator = AutoCalibrator(config.calibration)
    calibrator.start()

    print(
        f"Calibration automatique : {config.calibration.chessboard_cols}x"
        f"{config.calibration.chessboard_rows} coins internes, "
        f"case de {config.calibration.square_size_mm:.1f} mm, "
        f"{config.calibration.target_frames} vues à capturer."
    )
    print("Présentez le damier sous différents angles et distances.")
    if display:
        print("Appuyez sur 'q' pour interrompre.")

    try:
        camera = Camera(config.camera).open()
    except CameraError as exc:
        log.error("%s", exc)
        return 2

    last_message = ""
    try:
        while calibrator.running:
            ok, frame = camera.read_latest(drain=2)
            if not ok or frame is None:
                log.warning("Image non reçue...")
                continue

            calibrator.process(frame)
            if calibrator.state.message != last_message:
                last_message = calibrator.state.message
                print(f"  [{calibrator.state.captured}/{calibrator.state.target}] {last_message}")

            if display:
                cv2.imshow("Calibration - matvision", calibrator.draw_overlay(frame.copy()))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    calibrator.cancel()
                    break
    except KeyboardInterrupt:
        calibrator.cancel()
    finally:
        camera.close()
        if display:
            cv2.destroyAllWindows()

    state = calibrator.state
    intrinsics = state.intrinsics
    if intrinsics is None:
        print(f"\nÉchec de la calibration : {state.message}")
        return 1

    path = save_intrinsics(config.intrinsics_path, intrinsics)
    print("\nCalibration terminée")
    print(f"  vues utilisées    : {intrinsics.frames}")
    print(f"  erreur moyenne    : {intrinsics.rms:.4f} px")
    print(f"  fx, fy            : {intrinsics.fx:.1f}, {intrinsics.fy:.1f}")
    print(f"  taille image      : {intrinsics.image_size[0]}x{intrinsics.image_size[1]}")
    print(f"  enregistré dans   : {path}")
    if intrinsics.rms > 1.0:
        print("  ⚠ erreur élevée : multipliez et variez les vues (angles, distances).")
    return 0


# --------------------------------------------------------------------------- #
def run_check_table(config: Config, display: bool, seconds: float) -> int:
    """Vérification du repère table à partir des 4 tags de coin."""
    detector = ArucoDetector(config.aruco["dictionary"], config.aruco["params"])
    intrinsics = load_intrinsics(config.intrinsics_path)
    if intrinsics is None:
        print("(pas de calibration intrinsèque : la pose caméra ne sera pas calculée)")

    try:
        camera = Camera(config.camera).open()
    except CameraError as exc:
        log.error("%s", exc)
        return 2

    print("Vérification du repère table - 'q' pour quitter.")
    corner_ids = config.table.marker_ids
    started = time.time()
    exit_code = 1

    try:
        while True:
            ok, frame = camera.read_latest(drain=2)
            if not ok or frame is None:
                continue

            markers, rejected = detector.detect_by_id(frame)
            corners_found = {mid: c for mid, c in markers.items() if mid in corner_ids}
            localization = localize_table(corners_found, config.table, intrinsics)

            missing = sorted(corner_ids - set(corners_found))
            color = (0, 200, 0) if localization.ok else (0, 0, 255)
            cv2.putText(
                frame,
                f"tags {len(corners_found)}/{len(corner_ids)}"
                + (f" | manquants: {missing}" if missing else "")
                + (f" | residu {localization.residual_mm:.1f} mm | inliers {localization.inliers}"
                   if localization.ok else f" | {localization.reason}"),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            draw_markers(frame, corners_found, color=(255, 128, 0))
            draw_markers(frame, {k: v for k, v in markers.items() if k not in corner_ids}, color=(0, 220, 255))

            if localization.ok and localization.camera_position is not None:
                cam = localization.camera_position
                cv2.putText(
                    frame,
                    f"camera x={cam.x:.0f} y={cam.y:.0f} z={cam.z:.0f} a={cam.a:.1f} "
                    f"err={localization.pose_error_px:.2f}px",
                    (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if display:
                cv2.imshow("Vérification repère table - matvision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif seconds and (time.time() - started) >= seconds:
                break

            if localization.ok:
                exit_code = 0
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        if display:
            cv2.destroyAllWindows()

    if exit_code == 0:
        print("Repère table OK : les 4 tags sont vus et l'homographie est stable.")
    else:
        print("Repère table incomplet : vérifiez les identifiants et la visibilité des tags.")
    return exit_code


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    config = apply_overrides(config, args)

    for warning in config.validate():
        log.warning("configuration : %s", warning)

    if not list_video_devices():
        log.warning("aucun périphérique /dev/video* détecté")

    display = not args.no_display
    if args.check_table:
        return run_check_table(config, display, args.seconds)
    return run_intrinsics(config, display)


if __name__ == "__main__":
    raise SystemExit(main())
