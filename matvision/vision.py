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
from dataclasses import asdict
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .aruco import ArucoDetector, marker_center
from .calibration import AutoCalibrator, Intrinsics, load_intrinsics, save_intrinsics
from .camera import Camera, CameraError
from .config import Config, ObjectConfig
from .geometry import Position, build_roi_mask, normalize_angle_deg
from .overlay import annotate, annotate_test, draw_hud
from .table import TableLocalization, localize_table, table_polygon_image

log = logging.getLogger(__name__)

__all__ = ["VisionEngine", "VisionMode"]

#: Intervalle (secondes) entre deux bilans de performance écrits sur la console.
PERF_LOG_INTERVAL_S = 5.0


class VisionMode(str, Enum):
    IDLE = "idle"
    DETECT = "detect"
    CALIBRATE = "calibrate"


class VisionEngine:
    """Boucle d'acquisition + état partagé.

    En mode test (``test_mode=True``) le repère table n'est pas utilisé : tous
    les tags de l'image sont relevés en pixels, sans exiger les tags de coin.
    """

    def __init__(
        self, config: Config, *, display: bool = False, test_mode: bool = False
    ) -> None:
        self.config = config
        self.display = bool(display)
        self.test_mode = bool(test_mode)

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
            "proc_ms": 0.0,       # durée moyenne de traitement (ms) sur la dernière seconde
            "proc_ms_max": 0.0,   # pic de traitement (ms) depuis le dernier bilan
            "last_proc_ms": 0.0,  # durée de la dernière image traitée (ms)
        }
        self._last_table_ok = False

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
            self._last_table_ok = False
        log.info("Relevés réinitialisés")
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
        log.info("Calibration annulée")
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
            "test_mode": self.test_mode,
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
        return {
            "count": len(items),
            "objects": items,
            "by_label": by_label,
            "test_mode": self.test_mode,
            "frame": "image" if self.test_mode else "table",
        }

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
            "test_mode": self.test_mode,
            "markers": [
                {"id": m.id, "x": m.x, "y": m.y, "a": m.a, "size": m.size} for m in table.markers
            ],
            "objects": [
                {"id": o.id, "label": o.label or str(o.id),
                 "angle_offset": o.angle_offset, "offset": list(o.offset),
                 "size": o.size, "box": asdict(o.box) if o.box else None}
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
        last_perf_log = window_start
        proc_ms_sum = 0.0
        proc_ms_peak = 0.0

        while not self._stop_event.is_set():
            if not self.camera.is_open:
                opened_at = time.perf_counter()
                try:
                    self.camera.open()
                    with self._lock:
                        self._error = ""
                    log.info(
                        "Caméra ouverte en %.0f ms", (time.perf_counter() - opened_at) * 1000.0
                    )
                except CameraError as exc:
                    with self._lock:
                        self._error = str(exc)
                    log.warning("Caméra indisponible : %s", exc)
                    self._stop_event.wait(2.0)
                    continue

            read_started = time.perf_counter()
            ok, frame = self.camera.read_latest(drain=2)
            read_ms = (time.perf_counter() - read_started) * 1000.0
            if not ok or frame is None:
                with self._lock:
                    self._stats["failed"] += 1
                log.debug("Lecture caméra en échec (%.1f ms)", read_ms)
                self._stop_event.wait(0.05)
                continue

            if not self._frame_size_checked:
                self._frame_size_checked = True
                log.info("Première image reçue : %dx%d", frame.shape[1], frame.shape[0])
                self._check_intrinsics_resolution(frame.shape[1], frame.shape[0])

            proc_started = time.perf_counter()
            annotated = frame
            try:
                annotated = self._handle_frame(frame)
            except Exception:  # pragma: no cover - garde-fou de la boucle
                log.exception("Erreur pendant le traitement de l'image")
                with self._lock:
                    self._error = "erreur de traitement (voir les logs)"
            proc_ms = (time.perf_counter() - proc_started) * 1000.0
            proc_ms_sum += proc_ms
            proc_ms_peak = max(proc_ms_peak, proc_ms)

            with self._lock:
                self._latest_frame = annotated
                self._stats["frames"] += 1
                self._stats["last_frame_ts"] = time.time()
                self._stats["last_proc_ms"] = round(proc_ms, 2)
                frames = self._stats["frames"]

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "image %d traitée en %.1f ms (lecture %.1f ms)", frames, proc_ms, read_ms
                )

            frames_window += 1
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= 1.0:
                with self._lock:
                    self._stats["fps"] = round(frames_window / elapsed, 1)
                    self._stats["proc_ms"] = round(proc_ms_sum / frames_window, 2)
                    self._stats["proc_ms_max"] = round(proc_ms_peak, 2)
                    fps = self._stats["fps"]
                    proc_avg = self._stats["proc_ms"]
                    mode = self._mode.value
                    objects = len(self._objects)
                if now - last_perf_log >= PERF_LOG_INTERVAL_S:
                    log.info(
                        "perf: %.1f fps | traitement %.1f ms/image (max %.1f ms) | "
                        "mode=%s | objets=%d | frames=%d",
                        fps,
                        proc_avg,
                        proc_ms_peak,
                        mode,
                        objects,
                        frames,
                    )
                    last_perf_log = now
                    proc_ms_peak = 0.0
                frames_window = 0
                proc_ms_sum = 0.0
                window_start = now

            if self.display:
                cv2.imshow("Mat de vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    with self._lock:
                        self._mode = VisionMode.IDLE
                    log.info("Arrêt de la détection demandé depuis la fenêtre (touche « q »)")

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
            return self._detect_test(frame) if self.test_mode else self._detect(frame)

        if self.config.detection.draw:
            canvas = frame.copy()
            draw_hud(canvas, self._hud_lines())
            return canvas
        return frame

    # ------------------------------------------------------------------ #
    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Détecte les 4 tags de coin, construit la ROI, puis relève les objets."""
        started = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner_ids = self.config.table.marker_ids
        gray_done = time.perf_counter()

        found, _rejected = self.detector.detect_by_id(gray, gray=True)
        corner_markers = {mid: corners for mid, corners in found.items() if mid in corner_ids}
        corners_done = time.perf_counter()

        localization = localize_table(
            corner_markers,
            self.config.table,
            self._intrinsics,
            min_inliers=self.config.detection.min_homography_inliers,
        )
        locate_done = time.perf_counter()
        self._log_localization(localization)

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

        objects_done = time.perf_counter()

        with self._lock:
            self._localization = localization
            self._objects = objects
            self._stats["detections"] = len(objects)

        if self.config.detection.draw:
            canvas = frame.copy()
            annotate(canvas, corner_markers, object_markers, localization, objects)
            draw_hud(canvas, self._hud_lines())
        else:
            canvas = frame

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "détection : gris %.1f ms | tags %.1f ms (%d trouvés, %d de coin) | "
                "repère %.1f ms (%s) | objets %.1f ms (%d) | total %.1f ms",
                (gray_done - started) * 1000.0,
                (corners_done - gray_done) * 1000.0,
                len(found),
                len(corner_markers),
                (locate_done - corners_done) * 1000.0,
                "ok" if localization.ok else (localization.reason or "non localisé"),
                (objects_done - locate_done) * 1000.0,
                len(objects),
                (time.perf_counter() - started) * 1000.0,
            )
        return canvas

    def _detect_test(self, frame: np.ndarray) -> np.ndarray:
        """Mode test : tous les tags de l'image, sans repère table.

        Les tags de coin ne sont pas requis. Les positions sont exprimées en
        pixels (repère image, ``+y`` vers le bas) et non en millimètres.
        """
        started = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _rejected = self.detector.detect_by_id(gray, gray=True)
        detected = time.perf_counter()

        objects = [self._locate_test(marker_id, corners) for marker_id, corners in found.items()]

        with self._lock:
            self._localization = TableLocalization(
                used_ids=sorted(found), reason="mode test : repère table non utilisé"
            )
            self._objects = objects
            self._stats["detections"] = len(objects)

        if self.config.detection.draw:
            canvas = frame.copy()
            annotate_test(canvas, found, objects, self._intrinsics)
            draw_hud(canvas, self._hud_lines())
        else:
            canvas = frame

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "mode test : gris+tags %.1f ms (%d tags) | relevé %.1f ms | total %.1f ms",
                (detected - started) * 1000.0,
                len(found),
                (time.perf_counter() - detected) * 1000.0,
                (time.perf_counter() - started) * 1000.0,
            )
        return canvas

    def _log_localization(self, localization: TableLocalization) -> None:
        """Trace les transitions d'état du repère table (acquis / perdu)."""
        if localization.ok and not self._last_table_ok:
            log.info(
                "Repère table acquis (tags %s, %d inliers, résidu %.1f mm)",
                localization.used_ids,
                localization.inliers,
                localization.residual_mm,
            )
        elif not localization.ok and self._last_table_ok:
            log.warning("Repère table perdu : %s", localization.reason or "raison inconnue")
        self._last_table_ok = localization.ok

    def _locate_test(self, marker_id: int, corners: np.ndarray) -> dict:
        """Relevé d'un tag en pixels : centre, angle image, taille apparente.

        L'angle suit la convention image (``+x`` vers la droite, ``+y`` vers le
        bas), et non celle du repère table.
        """
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        center = marker_center(corners)
        edges = np.array([points[1] - points[0], points[2] - points[3]])
        mean_edge = edges.mean(axis=0)
        angle = normalize_angle_deg(float(np.degrees(np.arctan2(mean_edge[1], mean_edge[0]))))

        sides = np.linalg.norm(
            [
                points[1] - points[0],
                points[2] - points[1],
                points[3] - points[2],
                points[0] - points[3],
            ],
            axis=1,
        )
        size_px = float(np.mean(sides))

        object_config = self.objects_config.get(marker_id)
        size_mm = object_config.size if object_config else 100.0
        px_per_mm = size_px / size_mm if size_mm else 0.0

        item = Position(x=float(center[0]), y=float(center[1]), z=0.0, a=angle).to_dict()
        item.update(
            {
                "id": marker_id,
                "label": (
                    object_config.label
                    if object_config is not None and object_config.label
                    else str(marker_id)
                ),
                "declared": object_config is not None,
                "angle_image": True,
                "size_mm": round(float(size_mm), 2),
                "size_px": round(size_px, 2),
                "px_per_mm": round(px_per_mm, 4),
            }
        )
        if object_config is not None and object_config.box is not None:
            box = object_config.box
            item["box_mm"] = [box.length_mm, box.width_mm, box.height_mm]
        return item

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

        if self.test_mode:
            lines = [
                f"MODE TEST (repère image)   fps: {stats['fps']:.1f}   frames: {stats['frames']}",
                f"tags détectés: {stats['detections']}",
            ]
        else:
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
