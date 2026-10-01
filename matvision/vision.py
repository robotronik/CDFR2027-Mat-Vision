"""Moteur de vision : une seule boucle possède la caméra.

Le moteur fonctionne en tâche de fond et expose un état thread-safe consommé par
l'API REST. Trois modes :

* ``idle``      — la caméra tourne, aucune analyse (aperçu disponible) ;
* ``detect``    — repère table + détection des objets ;
* ``calibrate`` — calibration automatique des intrinsèques (damier).
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .aruco import ArucoDetector, draw_markers, marker_center
from .calibration import AutoCalibrator, Intrinsics, load_intrinsics, save_intrinsics
from .camera import Camera, CameraError
from .config import Config
from .geometry import Position, build_roi_mask, polygon_hull
from .table import TableLocalization, localize_table, table_polygon_image
from .tracker import ObjectDetection, ObjectTracker

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
        self.tracker = ObjectTracker(
            config.objects,
            smoothing=config.detection.smoothing,
            max_age_s=config.detection.max_age_s,
            match_distance_mm=config.detection.match_distance_mm,
        )
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
        self._stable_frames = 0
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
            self.tracker.clear()
            self._stable_frames = 0
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
            self.tracker.clear()
            self._localization = TableLocalization(reason="en attente")
            self._stable_frames = 0
            self._stats["detections"] = 0
        return {"message": "suivi réinitialisé"}

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
            objects = self.tracker.count
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

    def objects(self) -> dict:
        with self._lock:
            return {"objects": self.tracker.snapshot(), "list": self.tracker.flat()}

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
                {"id": o.id, "label": o.label or str(o.id), "size": o.size,
                 "angle_offset": o.angle_offset}
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
                annotated = self._handle_frame(frame, now)
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
    def _handle_frame(self, frame: np.ndarray, now: float) -> np.ndarray:
        with self._lock:
            mode = self._mode
            calibrator = self._calibrator

        if mode is VisionMode.CALIBRATE and calibrator is not None:
            calibrator.process(frame)
            if calibrator.finished:
                self._finish_calibration()
            return calibrator.draw_overlay(frame.copy()) if self.config.detection.draw else frame

        if mode is VisionMode.DETECT:
            return self._detect(frame, now)

        if self.config.detection.draw:
            overlay = frame.copy()
            self._draw_hud(overlay)
            return overlay
        return frame

    # ------------------------------------------------------------------ #
    def _detect(self, frame: np.ndarray, now: float) -> np.ndarray:
        """Détecte les 4 tags de coin, construit la ROI, puis les objets."""
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

        detections: list[ObjectDetection] = []
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
                if marker_id in corner_ids:
                    continue
                object_config = self.tracker.objects.get(marker_id)
                if object_config is None:
                    continue
                table_xy = localization.image_to_table(marker_center(corners))[0]
                angle = localization.marker_angle_deg(corners, object_config.angle_offset)
                detections.append(
                    ObjectDetection(
                        marker_id=marker_id,
                        label=object_config.label or str(marker_id),
                        position=Position(x=float(table_xy[0]), y=float(table_xy[1]), z=0.0, a=angle),
                        size=object_config.size,
                    )
                )

        with self._lock:
            self._localization = localization
            if localization.ok:
                self._stable_frames += 1
                if self._stable_frames >= max(1, self.config.detection.stabilize_frames):
                    self.tracker.update(detections, now)
                    self._stats["detections"] = len(detections)
            else:
                self._stable_frames = 0
                self._stats["detections"] = 0

        if not self.config.detection.draw:
            return frame

        overlay = frame.copy()
        self._annotate(overlay, corner_markers, object_markers, localization, roi_mask)
        self._draw_hud(overlay)
        return overlay

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
    # Affichage
    # ------------------------------------------------------------------ #
    def _annotate(
        self,
        frame: np.ndarray,
        corner_markers: dict[int, np.ndarray],
        object_markers: dict[int, np.ndarray],
        localization: TableLocalization,
        roi_mask: np.ndarray | None,
    ) -> None:
        if roi_mask is not None:
            hull = polygon_hull(corner_markers.values())
            if hull is not None:
                cv2.polylines(frame, [np.round(hull).astype(np.int32)], True, (0, 200, 0), 2)

        draw_markers(frame, corner_markers, color=(255, 128, 0))
        draw_markers(frame, object_markers, color=(0, 220, 255))

        self._draw_table_axes(frame, localization)

        for marker_id, corners in object_markers.items():
            if marker_id in self.config.table.marker_ids:
                continue
            object_config = self.tracker.objects.get(marker_id)
            if object_config is None or localization.homography is None:
                continue
            table_xy = localization.image_to_table(marker_center(corners))[0]
            angle = localization.marker_angle_deg(corners, object_config.angle_offset)
            center = marker_center(corners)
            label = f"{object_config.label or marker_id} ({table_xy[0]:.0f},{table_xy[1]:.0f}) {angle:.0f}deg"
            cv2.putText(
                frame,
                label,
                (int(center[0]) - 80, int(center[1]) - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

        if not localization.ok:
            cv2.putText(
                frame,
                localization.reason,
                (10, frame.shape[0] - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_table_axes(self, frame: np.ndarray, localization: TableLocalization) -> None:
        """Dessine l'origine et les axes de la table dans l'image."""
        if localization.homography is None:
            return
        try:
            inverse = np.linalg.inv(localization.homography)
        except np.linalg.LinAlgError:
            return
        axis_length = 300.0
        points = np.array(
            [[0.0, 0.0], [axis_length, 0.0], [0.0, axis_length]], dtype=np.float64
        )
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        origin, x_axis, y_axis = projected
        if not np.all(np.isfinite(projected)):
            return
        origin_px = tuple(np.round(origin).astype(int))
        cv2.arrowedLine(
            frame, origin_px, tuple(np.round(x_axis).astype(int)), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2
        )
        cv2.arrowedLine(
            frame, origin_px, tuple(np.round(y_axis).astype(int)), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.2
        )
        cv2.circle(frame, origin_px, 5, (0, 255, 255), -1)

    def _draw_hud(self, frame: np.ndarray) -> None:
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
            cam = localization.camera_position
            lines.append(f"caméra: x={cam.x:.1f} y={cam.y:.1f} z={cam.z:.1f} a={cam.a:.1f}")
        if error:
            lines.append(f"erreur: {error}")

        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (10, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
