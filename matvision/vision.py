"""Moteur de vision : une seule boucle possède la caméra.

Le moteur fonctionne en tâche de fond et expose un état thread-safe consommé par
l'API REST. Trois modes :

* ``idle``      — la caméra tourne, aucune analyse (aperçu disponible) ;
* ``detect``    — repère table + détection des objets ;
* ``calibrate`` — calibration automatique des intrinsèques (damier).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .aruco import ArucoDetector, marker_center
from .calibration import AutoCalibrator, Intrinsics, load_intrinsics, save_intrinsics
from .camera import Camera, CameraError
from .config import Config, ObjectConfig
from .geometry import Position, build_roi_mask
from .overlay import annotate, draw_hud
from .table import TableLocalization, localize_table, table_polygon_image

log = logging.getLogger(__name__)

__all__ = ["VisionEngine", "VisionMode"]


class VisionMode(str, Enum):
    IDLE = "idle"
    DETECT = "detect"
    CALIBRATE = "calibrate"


class VisionEngine:
    """Boucle d'acquisition + état partagé."""

    def __init__(self, config: Config, *, display: bool = False) -> None:
        self.config = config
        self.display = bool(display)

        self.detector = ArucoDetector(
            config.aruco.get("dictionary", "DICT_4X4_50"),
            config.aruco.get("params"),
        )
        self.objects_config = {obj.id: obj for obj in config.objects}
        self.camera = Camera(config.camera)

        self._intrinsics: Intrinsics | None = load_intrinsics(config.intrinsics_path)
        if self._intrinsics is not None:
            self._intrinsics.source = str(config.intrinsics_path)
            log.info(
                "Calibration intrinsèque chargée (%s, %d vues)",
                config.intrinsics_path,
                self._intrinsics.frames,
            )

        self._calibrator: AutoCalibrator | None = None
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._mode = VisionMode.IDLE
        self._localization = TableLocalization(reason="en attente")
        self._latest_frame: np.ndarray | None = None
        self._objects: list[dict] = []
        self._started_at = 0.0
        self._error = ""
        self._intrinsics_warning = ""
        self._frame_size_checked = False
        self._stats: dict[str, Any] = {
            "frames": 0,
            "failed": 0,
            "detections": 0,
            "fps": 0.0,
            "last_frame_ts": 0.0,
        }

    # ------------------------------------------------------------------ #
    # Cycle de vie
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Démarre la boucle d'acquisition (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="matvision", daemon=True)
        self._thread.start()
        log.info("Moteur de vision démarré")

    def shutdown(self) -> None:
        """Arrête la boucle et libère la caméra."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.camera.close()
        if self.display:
            cv2.destroyAllWindows()
        log.info("Moteur de vision arrêté")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Commandes
    # ------------------------------------------------------------------ #
    def start_detection(self) -> dict:
        with self._lock:
            self._objects = []
            self._stats["detections"] = 0
            self._mode = VisionMode.DETECT
        log.info("Détection démarrée")
        return {"message": "détection démarrée", "mode": VisionMode.DETECT.value}

    def stop(self) -> dict:
        with self._lock:
            self._mode = VisionMode.IDLE
        log.info("Détection arrêtée")
        return {"message": "détection arrêtée", "mode": VisionMode.IDLE.value}

    def reset(self) -> dict:
        with self._lock:
            self._objects = []
            self._localization = TableLocalization(reason="en attente")
            self._stats["detections"] = 0
        return {"message": "relevés réinitialisés"}

    def start_calibration(self) -> dict:
        with self._lock:
            self._calibrator = AutoCalibrator(self.config.calibration)
            self._calibrator.start()
            self._mode = VisionMode.CALIBRATE
        log.info("Calibration automatique démarrée")
        return {"message": "calibration démarrée", "mode": VisionMode.CALIBRATE.value}

    def cancel_calibration(self) -> dict:
        with self._lock:
            if self._calibrator is not None:
                self._calibrator.cancel()
            self._mode = VisionMode.IDLE
        return {"message": "calibration annulée"}

    def reload_intrinsics(self) -> Intrinsics | None:
        """Recharge la calibration intrinsèque depuis le disque."""
        intrinsics = load_intrinsics(self.config.intrinsics_path)
        with self._lock:
            self._intrinsics = intrinsics
        return intrinsics

    def save_frame(self, path: str) -> bool:
        """Enregistre l'image courante (aperçu) sur le disque."""
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return False
        return bool(cv2.imwrite(path, frame))

    # ------------------------------------------------------------------ #
    # Lectures d'état
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        with self._lock:
            mode = self._mode
            stats = dict(self._stats)
            localization = self._localization
            error = self._error
            intrinsics = self._intrinsics
            intrinsics_warning = self._intrinsics_warning
            objects = len(self._objects)
        return {
            "running": self.running,
            "mode": mode.value,
            "camera": self.camera.info(),
            "error": error,
            "objects_count": objects,
            "table": localization.to_dict(),
            "intrinsics": intrinsics.summary() if intrinsics else None,
            "intrinsics_warning": intrinsics_warning,
            "stats": stats,
            "uptime_s": round(time.monotonic() - self._started_at, 1) if self._started_at else 0.0,
        }

    def objects(self, label: str | None = None) -> dict:
        """Objets relevés sur la dernière image, dans le repère de la table.

        Une entrée par tag détecté : un élément de jeu vu par plusieurs de ses
        tags apparaît donc plusieurs fois, à des positions très proches.
        """
        with self._lock:
            items = list(self._objects)
        if label:
            items = [item for item in items if item["label"].lower() == label.lower()]

        by_label: dict[str, list[dict]] = {}
        for item in items:
            by_label.setdefault(item["label"], []).append(item)
        return {"count": len(items), "objects": items, "by_label": by_label}

    def camera_position(self) -> dict:
        with self._lock:
            localization = self._localization
            mode = self._mode
        payload: dict[str, Any] = {
            "mode": mode.value,
            "table_locked": bool(localization.ok),
            "used_ids": localization.used_ids,
        }
        if localization.camera_position is not None:
            payload["position"] = localization.camera_position.to_dict()
        else:
            payload["position"] = None
            payload["message"] = localization.reason or "pose caméra indisponible (calibration ?)"
        return payload

    def table_info(self) -> dict:
        table = self.config.table
        with self._lock:
            localization = self._localization
        return {
            "width_mm": table.width_mm,
            "height_mm": table.height_mm,
            "markers": [
                {"id": m.id, "x": m.x, "y": m.y, "a": m.a, "size": m.size} for m in table.markers
            ],
            "objects": [
                {"id": o.id, "label": o.label or str(o.id),
                 "angle_offset": o.angle_offset, "offset": list(o.offset)}
                for o in self.config.objects
            ],
            "localization": localization.to_dict(),
        }

    def calibration_status(self) -> dict:
        with self._lock:
            calibrator = self._calibrator
            intrinsics = self._intrinsics
        payload = {
            "config": {
                "cols": self.config.calibration.chessboard_cols,
                "rows": self.config.calibration.chessboard_rows,
                "square_size_mm": self.config.calibration.square_size_mm,
                "target_frames": self.config.calibration.target_frames,
                "file": str(self.config.intrinsics_path),
            },
            "intrinsics": intrinsics.summary() if intrinsics else None,
        }
        payload.update(calibrator.state.to_dict() if calibrator else {"status": "idle"})
        return payload

    def preview_jpeg(self, quality: int | None = None) -> bytes | None:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        quality = int(quality if quality is not None else self.config.preview_quality)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buffer.tobytes() if ok else None

    # ------------------------------------------------------------------ #
    # Boucle principale
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        frames_window = 0
        window_start = time.monotonic()

        while not self._stop_event.is_set():
            if not self.camera.is_open:
                try:
                    self.camera.open()
                    with self._lock:
                        self._error = ""
                except CameraError as exc:
                    with self._lock:
                        self._error = str(exc)
                    log.warning("Caméra indisponible : %s", exc)
                    self._stop_event.wait(2.0)
                    continue

            ok, frame = self.camera.read_latest(drain=2)
            if not ok or frame is None:
                with self._lock:
                    self._stats["failed"] += 1
                self._stop_event.wait(0.05)
                continue

            now = time.monotonic()
            if not self._frame_size_checked:
                self._frame_size_checked = True
                self._check_intrinsics_resolution(frame.shape[1], frame.shape[0])
            annotated = frame
            try:
                annotated = self._handle_frame(frame)
            except Exception:  # pragma: no cover - garde-fou de la boucle
                log.exception("Erreur pendant le traitement de l'image")
                with self._lock:
                    self._error = "erreur de traitement (voir les logs)"

            with self._lock:
                self._latest_frame = annotated
                self._stats["frames"] += 1
                self._stats["last_frame_ts"] = time.time()

            frames_window += 1
            elapsed = now - window_start
            if elapsed >= 1.0:
                with self._lock:
                    self._stats["fps"] = round(frames_window / elapsed, 1)
                frames_window = 0
                window_start = now

            if self.display:
                cv2.imshow("Mat de vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    with self._lock:
                        self._mode = VisionMode.IDLE

        if self.display:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    def _handle_frame(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            mode = self._mode
            calibrator = self._calibrator

        if mode is VisionMode.CALIBRATE and calibrator is not None:
            calibrator.process(frame)
            if calibrator.finished:
                self._finish_calibration()
            return calibrator.draw_overlay(frame.copy()) if self.config.detection.draw else frame

        if mode is VisionMode.DETECT:
            return self._detect(frame)

        if self.config.detection.draw:
            canvas = frame.copy()
            draw_hud(canvas, self._hud_lines())
            return canvas
        return frame

    # ------------------------------------------------------------------ #
    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Détecte les 4 tags de coin, construit la ROI, puis relève les objets."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner_ids = self.config.table.marker_ids

        found, _rejected = self.detector.detect_by_id(gray, gray=True)
        corner_markers = {mid: corners for mid, corners in found.items() if mid in corner_ids}

        localization = localize_table(
            corner_markers,
            self.config.table,
            self._intrinsics,
            min_inliers=self.config.detection.min_homography_inliers,
        )

        objects: list[dict] = []
        object_markers: dict[int, np.ndarray] = {}
        roi_mask: np.ndarray | None = None

        if localization.ok and localization.homography is not None:
            # On ne traite que la surface de jeu (cf. table.roi_mode) : soit tout
            # le tapis reconstruit par homographie, soit seulement le
            # quadrilatère des tags de coin.
            groups: list[np.ndarray] = list(corner_markers.values())
            if self.config.table.roi_mode == "table":
                table_quad = table_polygon_image(localization, self.config.table)
                if table_quad is not None:
                    groups.append(table_quad)
            roi_mask = build_roi_mask(gray.shape, groups, self.config.table.roi_margin_px)
            if roi_mask is not None:
                object_markers, _ = self.detector.detect_by_id(gray, roi_mask, gray=True)
            else:
                object_markers = found

            for marker_id, corners in object_markers.items():
                object_config = self.objects_config.get(marker_id)
                if object_config is not None:
                    objects.append(self._locate(object_config, corners, localization))

        with self._lock:
            self._localization = localization
            self._objects = objects
            self._stats["detections"] = len(objects)

        if not self.config.detection.draw:
            return frame

        canvas = frame.copy()
        annotate(canvas, corner_markers, object_markers, localization, objects)
        draw_hud(canvas, self._hud_lines())
        return canvas

    def _locate(
        self, config: ObjectConfig, corners: np.ndarray, localization: TableLocalization
    ) -> dict:
        """Position d'un objet dans le repère de la table (mm et degrés).

        Par défaut c'est le centre du tag. ``offset`` permet de viser un point
        particulié de l'objet : le décalage est exprimé dans le repère du tag et
        tourne avec l'orientation mesurée.
        """
        table_xy = localization.image_to_table(marker_center(corners))[0]
        x, y = float(table_xy[0]), float(table_xy[1])
        angle = localization.marker_angle_deg(corners, config.angle_offset)

        offset_x, offset_y = config.offset
        if offset_x or offset_y:
            radians = math.radians(angle)
            x += offset_x * math.cos(radians) - offset_y * math.sin(radians)
            y += offset_x * math.sin(radians) + offset_y * math.cos(radians)

        position = Position(x=x, y=y, z=0.0, a=angle).to_dict()
        position.update({"id": config.id, "label": config.label or str(config.id)})
        return position

    def _finish_calibration(self) -> None:
        with self._lock:
            calibrator = self._calibrator
            self._mode = VisionMode.IDLE
        if calibrator is None:
            return
        intrinsics = calibrator.state.intrinsics
        if intrinsics is None:
            log.error("Calibration non aboutie : %s", calibrator.state.message)
            return
        intrinsics.source = str(self.config.intrinsics_path)
        save_intrinsics(self.config.intrinsics_path, intrinsics)
        with self._lock:
            self._intrinsics = intrinsics
            self._intrinsics_warning = ""
        log.info("Calibration enregistrée dans %s", self.config.intrinsics_path)

    def _check_intrinsics_resolution(self, width: int, height: int) -> None:
        """Avertit si la calibration ne correspond pas à la résolution courante.

        Les paramètres intrinsèques sont exprimés en pixels : une calibration
        faite en 1920x1080 est fausse si le flux est ensuite lu en 3840x2160.
        Les positions des objets (issues de l'homographie) ne sont pas
        affectées, mais la pose caméra de ``/position`` le serait.
        """
        intrinsics = self._intrinsics
        if intrinsics is None or not intrinsics.image_size or not intrinsics.image_size[0]:
            return
        if tuple(intrinsics.image_size) == (width, height):
            return

        message = (
            f"calibration faite en {intrinsics.image_size[0]}x{intrinsics.image_size[1]} "
            f"mais la caméra fournit du {width}x{height} : la pose caméra sera fausse. "
            f"Relancez « python calibrate.py » à cette résolution."
        )
        log.warning("%s", message)
        with self._lock:
            self._intrinsics_warning = message

    # ------------------------------------------------------------------ #
    # Aide à l'affichage
    # ------------------------------------------------------------------ #
    def _hud_lines(self) -> list[str]:
        """Lignes d'état écrites sur l'image d'aperçu."""
        with self._lock:
            mode = self._mode
            stats = dict(self._stats)
            localization = self._localization
            error = self._error

        lines = [
            f"mode: {mode.value}   fps: {stats['fps']:.1f}   frames: {stats['frames']}",
            f"objets: {stats['detections']}   inliers: {localization.inliers}/{localization.total_points}",
        ]
        if localization.camera_position is not None:
            camera = localization.camera_position
            lines.append(
                f"caméra: x={camera.x:.1f} y={camera.y:.1f} z={camera.z:.1f} a={camera.a:.1f}"
            )
        if error:
            lines.append(f"erreur: {error}")
        return lines
