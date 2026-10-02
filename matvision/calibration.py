"""Calibration automatique des paramètres intrinsèques de la caméra.

Le principe : on présente un damier devant la caméra, le module détecte
automatiquement les poses suffisamment différentes, nettes et complètes, puis
lance ``cv2.calibrateCamera`` sans aucune action clavier.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CalibrationConfig

log = logging.getLogger(__name__)

__all__ = [
    "Intrinsics",
    "AutoCalibrator",
    "load_intrinsics",
    "save_intrinsics",
    "calibration_file_kind",
]


# --------------------------------------------------------------------------- #
# Modèle de calibration
# --------------------------------------------------------------------------- #
@dataclass
class Intrinsics:
    """Résultat de la calibration intrinsèque."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms: float = 0.0
    frames: int = 0
    calibrated_at: str = ""
    source: str = ""

    # -- sérialisation ------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "camera_matrix": np.asarray(self.camera_matrix).tolist(),
            "dist_coeffs": np.asarray(self.dist_coeffs).reshape(-1).tolist(),
            "rms": float(self.rms),
            "frames": int(self.frames),
            "calibrated_at": self.calibrated_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "") -> "Intrinsics":
        size = data.get("image_size") or [0, 0]
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            image_size=(int(size[0]), int(size[1])),
            rms=float(data.get("rms", 0.0)),
            frames=int(data.get("frames", 0)),
            calibrated_at=str(data.get("calibrated_at", "")),
            source=source,
        )

    # -- utilitaires --------------------------------------------------- #
    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    def summary(self) -> dict[str, Any]:
        info = self.to_dict()
        info["fov_x_deg"] = round(
            2.0 * np.degrees(np.arctan(self.image_size[0] / (2.0 * self.fx))), 2
        ) if self.fx else None
        info["fov_y_deg"] = round(
            2.0 * np.degrees(np.arctan(self.image_size[1] / (2.0 * self.fy))), 2
        ) if self.fy else None
        return info


def calibration_file_kind(path: str | Path) -> str:
    """Détecte le format d'un fichier de calibration : ``json``, ``yaml`` ou ``inconnu``."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return "json"
    if p.suffix.lower() in {".yml", ".yaml"}:
        return "yaml"
    return "inconnu"


def save_intrinsics(path: str | Path, intrinsics: Intrinsics) -> Path:
    """Écrit la calibration au format JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = intrinsics.to_dict()
    payload["format"] = "matvision.intrinsics/1"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_intrinsics(path: str | Path) -> Intrinsics | None:
    """Charge une calibration JSON ou (compatibilité) un ``.yml`` OpenCV.

    Le fichier ``.yml`` est celui produit par les scripts ``calibrate_camera.py``
    d'origine.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        if calibration_file_kind(target) == "yaml":
            storage = cv2.FileStorage(str(target), cv2.FILE_STORAGE_READ)
            if not storage.isOpened():
                return None
            try:
                matrix = storage.getNode("camera_matrix").mat()
                dist = storage.getNode("dist_coeffs").mat()
            finally:
                storage.release()
            if matrix is None or dist is None:
                return None
            return Intrinsics(
                camera_matrix=np.asarray(matrix, dtype=np.float64),
                dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
                image_size=(0, 0),
                source=str(target),
            )
        data = json.loads(target.read_text(encoding="utf-8"))
        return Intrinsics.from_dict(data, source=str(target))
    except Exception as exc:  # pragma: no cover - dépend du contenu du fichier
        log.error("Calibration illisible (%s) : %s", target, exc)
        return None


# --------------------------------------------------------------------------- #
# Calibration automatique
# --------------------------------------------------------------------------- #
_CALIB_CB_FLAGS = (
    getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 1)
    | getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 2)
    | getattr(cv2, "CALIB_CB_FAST_CHECK", 8)
)


@dataclass
class CalibrationState:
    """État d'une session de calibration (exposé par l'API)."""

    status: str = "idle"          # idle | running | done | failed | cancelled
    found: bool = False
    captured: int = 0
    target: int = 0
    attempts: int = 0
    sharpness: float = 0.0
    area_ratio: float = 0.0
    movement: float = 0.0
    message: str = "présentez le damier devant la caméra"
    rms: float | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    intrinsics: Intrinsics | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "found": self.found,
            "captured": self.captured,
            "target": self.target,
            "attempts": self.attempts,
            "sharpness": round(self.sharpness, 1),
            "area_ratio": round(self.area_ratio, 4),
            "movement": round(self.movement, 3),
            "message": self.message,
            "rms": None if self.rms is None else round(self.rms, 4),
            "elapsed_s": round(
                (self.finished_at or time.time()) - self.started_at, 1
            ) if self.started_at else 0.0,
        }


class AutoCalibrator:
    """Collecte automatiquement des vues d'un damier puis calcule les intrinsèques.

    Usage ::

        calib = AutoCalibrator(cfg.calibration)
        calib.start()
        while not calib.finished:
            calib.process(frame)
        intrinsics = calib.state.intrinsics
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self.grid_size = (int(self.config.chessboard_cols), int(self.config.chessboard_rows))
        self._object_template = self._build_object_points()
        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self._last_corners: np.ndarray | None = None
        self._image_size: tuple[int, int] = (0, 0)
        self.state = CalibrationState()

    # ------------------------------------------------------------------ #
    def start(self) -> CalibrationState:
        """Démarre (ou redémarre) une session."""
        self._object_points.clear()
        self._image_points.clear()
        self._last_corners = None
        self._image_size = (0, 0)
        self.state = CalibrationState(
            status="running",
            target=int(self.config.target_frames),
            started_at=time.time(),
            message="présentez le damier devant la caméra",
        )
        log.info("Session de calibration démarrée (objectif : %d vues)", self.state.target)
        return self.state

    def cancel(self) -> CalibrationState:
        if self.state.status == "running":
            self.state.status = "cancelled"
            self.state.finished_at = time.time()
            self.state.message = "session annulée"
        return self.state

    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self.state.status == "running"

    @property
    def finished(self) -> bool:
        return self.state.status in {"done", "failed", "cancelled"}

    @property
    def progress(self) -> float:
        if not self.state.target:
            return 0.0
        return min(1.0, self.state.captured / float(self.state.target))

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> CalibrationState:
        """Analyse une image ; capture la vue si elle est exploitable.

        À appeler pour **chaque** image tant que :attr:`running` est vrai.
        """
        state = self.state
        if not self.running:
            return state

        state.attempts += 1
        if self._image_size == (0, 0):
            self._image_size = (int(frame.shape[1]), int(frame.shape[0]))

        elapsed = time.time() - state.started_at
        if elapsed > float(self.config.max_capture_seconds):
            self._finalize_timeout()
            return state

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.grid_size, _CALIB_CB_FLAGS)
        state.found = bool(found)

        if not found:
            state.message = "damier non détecté"
            return state

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        state.area_ratio = self._area_ratio(corners, gray.shape[:2])
        if not (self.config.min_board_area_ratio <= state.area_ratio <= self.config.max_board_area_ratio):
            state.message = (
                f"damier trop {'petit' if state.area_ratio < self.config.min_board_area_ratio else 'grand'} "
                f"({state.area_ratio * 100:.0f} % de l'image)"
            )
            return state

        if not self._inside_border(corners, gray.shape[:2], int(self.config.border_margin_px)):
            state.message = "damier trop près du bord - centrez-le"
            return state

        state.sharpness = self._sharpness(gray, corners)
        if state.sharpness < float(self.config.min_sharpness):
            state.message = f"image floue ({state.sharpness:.0f} < {self.config.min_sharpness:.0f})"
            return state

        state.movement = self._movement_ratio(corners, gray.shape[:2])
        if self._last_corners is not None and state.movement < float(self.config.min_move_ratio):
            state.message = "position trop proche de la vue précédente"
            return state

        # Vue acceptée.
        self._object_points.append(self._object_template.copy())
        self._image_points.append(corners.copy())
        self._last_corners = corners.copy()
        state.captured += 1
        state.message = f"vue {state.captured}/{state.target} capturée"
        log.info("Vue de calibration capturée (%d/%d)", state.captured, state.target)

        if state.captured >= state.target:
            self.compute()
        return state

    # ------------------------------------------------------------------ #
    def compute(self) -> Intrinsics | None:
        """Calcule les intrinsèques à partir des vues collectées."""
        state = self.state
        if len(self._object_points) < 5:
            state.status = "failed"
            state.finished_at = time.time()
            state.message = (
                f"échec : {len(self._object_points)} vue(s) exploitable(s), 5 minimum requises"
            )
            log.error(state.message)
            return None

        size = self._image_size or (0, 0)
        flags = self._resolve_flags()
        log.info("Calibration sur %d vues (%dx%d)...", len(self._object_points), size[0], size[1])
        compute_started = time.perf_counter()
        try:
            rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
                self._object_points, self._image_points, size, None, None, flags=flags
            )
        except cv2.error as exc:  # pragma: no cover - dépend des données
            state.status = "failed"
            state.finished_at = time.time()
            state.message = f"échec de calibrateCamera : {exc}"
            log.exception("calibrateCamera a échoué")
            return None

        mean_error = _mean_error(
            matrix, dist, rvecs, tvecs, self._object_points, self._image_points
        )
        intrinsics = Intrinsics(
            camera_matrix=np.asarray(matrix, dtype=np.float64),
            dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
            image_size=(int(size[0]), int(size[1])),
            rms=float(mean_error if mean_error else rms),
            frames=len(self._object_points),
            calibrated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        )
        state.intrinsics = intrinsics
        state.rms = intrinsics.rms
        state.status = "done"
        state.finished_at = time.time()
        state.message = f"calibration terminée (erreur de reprojection {intrinsics.rms:.3f} px)"
        log.info(state.message)
        log.info(
            "Calibration calculée en %.0f ms (%d vues, rms=%.3f px)",
            (time.perf_counter() - compute_started) * 1000.0,
            len(self._object_points),
            intrinsics.rms,
        )
        return intrinsics

    # ------------------------------------------------------------------ #
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Annote l'image avec la progression (damier détecté ou non)."""
        state = self.state
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.grid_size, _CALIB_CB_FLAGS)
        if found:
            cv2.drawChessboardCorners(frame, self.grid_size, corners, found)

        height = frame.shape[0]
        color = (0, 200, 0) if state.found else (0, 165, 255)
        bar_width = int(frame.shape[1] * self.progress)
        cv2.rectangle(frame, (0, height - 12), (bar_width, height), color, -1)
        cv2.putText(
            frame,
            f"Calibration {state.captured}/{state.target} - {state.message}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _finalize_timeout(self) -> None:
        state = self.state
        if len(self._object_points) >= 5:
            self.compute()
        else:
            state.status = "failed"
            state.finished_at = time.time()
            state.message = (
                f"délai dépassé : seulement {len(self._object_points)} vue(s) capturée(s)"
            )

    # ------------------------------------------------------------------ #
    def _build_object_points(self) -> np.ndarray:
        cols, rows = self.grid_size
        grid = np.zeros((cols * rows, 3), np.float32)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return grid * float(self.config.square_size_mm)

    def _resolve_flags(self) -> int:
        flags = 0
        for name in (self.config.flags or "").replace(",", " ").split():
            if hasattr(cv2, name):
                flags |= int(getattr(cv2, name))
            else:
                log.warning("Option de calibration inconnue ignorée : %s", name)
        return flags

    @staticmethod
    def _area_ratio(corners: np.ndarray, shape: tuple[int, int]) -> float:
        points = np.asarray(corners).reshape(-1, 2)
        rect = cv2.boundingRect(points.astype(np.float32))
        image_area = float(shape[0] * shape[1])
        return float(rect[2] * rect[3]) / image_area if image_area else 0.0

    @staticmethod
    def _inside_border(corners: np.ndarray, shape: tuple[int, int], margin: int = 0) -> bool:
        points = np.asarray(corners).reshape(-1, 2)
        height, width = shape
        if margin <= 0:
            margin = 1
        return bool(
            points[:, 0].min() >= margin
            and points[:, 1].min() >= margin
            and points[:, 0].max() <= width - margin
            and points[:, 1].max() <= height - margin
        )

    @staticmethod
    def _sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
        points = np.asarray(corners).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(points.astype(np.float32))
        if w <= 2 or h <= 2:
            return 0.0
        patch = gray[y : y + h, x : x + w]
        return float(cv2.Laplacian(patch, cv2.CV_64F).var())

    def _movement_ratio(self, corners: np.ndarray, shape: tuple[int, int]) -> float:
        if self._last_corners is None:
            return 1.0
        current = np.asarray(corners).reshape(-1, 2)
        previous = np.asarray(self._last_corners).reshape(-1, 2)
        if current.shape != previous.shape:
            return 1.0
        diagonal = float(np.hypot(shape[0], shape[1])) or 1.0
        return float(np.mean(np.linalg.norm(current - previous, axis=1)) / diagonal)


def _mean_error(
    matrix: np.ndarray,
    dist: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> float:
    """Erreur de reprojection moyenne, en pixels.

    Le calcul est fait en NumPy (et non avec ``cv2.norm``) car la disposition et
    le type des tableaux renvoyés par ``cv2.projectPoints`` varient selon les
    versions d'OpenCV.
    """
    total = 0.0
    for index in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[index], rvecs[index], tvecs[index], matrix, dist
        )
        observed = np.asarray(image_points[index], dtype=np.float64).reshape(-1, 2)
        expected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        total += float(np.mean(np.linalg.norm(observed - expected, axis=1)))
    return total / max(1, len(object_points))
