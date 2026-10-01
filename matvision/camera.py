"""Accès à la webcam USB (Logitech 4K Stream Edition sur LattePanda Delta)."""

from __future__ import annotations

import glob
import logging
import time
from typing import Any

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)

__all__ = ["Camera", "CameraError", "list_video_devices"]

#: Backends OpenCV utilisables ; toute autre valeur = détection automatique.
_BACKENDS: dict[str, int] = {
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
}


class CameraError(RuntimeError):
    """Erreur d'accès à la caméra."""


def list_video_devices() -> list[str]:
    """Liste les périphériques vidéo disponibles (Linux : ``/dev/video*``)."""
    return sorted(glob.glob("/dev/video*"))


def _fourcc_to_code(fourcc: str) -> int:
    code = (fourcc or "").strip().upper()
    if len(code) != 4:
        return 0
    return cv2.VideoWriter_fourcc(*code)


class Camera:
    """Fine enveloppe autour de ``cv2.VideoCapture`` avec configuration UVC."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None
        self._size: tuple[int, int] = (0, 0)
        self._fourcc: str = ""
        self._fps: float = 0.0

    # ------------------------------------------------------------------ #
    def open(self) -> "Camera":
        """Ouvre la caméra et applique la configuration demandée."""
        self.close()
        device = self._normalize_device(self.config.device)
        backend = _BACKENDS.get(str(self.config.backend).lower(), cv2.CAP_ANY)

        log.info("Ouverture de la caméra %s (backend=%s)", device, self.config.backend)
        capture = cv2.VideoCapture(device, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(device)
        if not capture.isOpened():
            raise CameraError(
                f"impossible d'ouvrir la caméra {device!r} "
                f"(périphériques visibles : {list_video_devices() or 'aucun'})"
            )

        if self.config.buffer_size > 0:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config.buffer_size))
        if self.config.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, _fourcc_to_code(self.config.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height))
        if self.config.fps:
            capture.set(cv2.CAP_PROP_FPS, int(self.config.fps))

        self._capture = capture
        self._read_actual_settings()

        for _ in range(max(0, int(self.config.warmup_frames))):
            capture.grab()
        return self

    # ------------------------------------------------------------------ #
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Lit une image ; ``(False, None)`` en cas d'échec."""
        if self._capture is None:
            raise CameraError("caméra non ouverte")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def read_latest(self, drain: int = 3) -> tuple[bool, np.ndarray | None]:
        """Lit l'image la plus récente.

        La LattePanda Delta a un CPU limité : vider le tampon V4L2 évite de
        traiter des images déjà périmées et de prendre du retard.
        """
        if self._capture is None:
            raise CameraError("caméra non ouverte")
        for _ in range(max(0, drain)):
            self._capture.grab()
        return self.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def size(self) -> tuple[int, int]:
        """Résolution réellement négociée ``(largeur, hauteur)``."""
        return self._size

    def info(self) -> dict[str, Any]:
        return {
            "device": self.config.device,
            "opened": self.is_open,
            "width": self._size[0],
            "height": self._size[1],
            "fourcc": self._fourcc,
            "fps": round(self._fps, 2),
            "requested": {
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "fourcc": self.config.fourcc,
            },
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_device(device: int | str) -> Any:
        """Accepte ``0``, ``"/dev/video0"`` ou ``"http://..."``."""
        if isinstance(device, str):
            stripped = device.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return int(device)

    def _read_actual_settings(self) -> None:
        assert self._capture is not None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._size = (width, height)

        code = int(self._capture.get(cv2.CAP_PROP_FOURCC) or 0)
        self._fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) if code else ""

        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

        if not self.config.auto_exposure and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if not self.config.auto_white_balance and hasattr(cv2, "CAP_PROP_AUTO_WB"):
            self._capture.set(cv2.CAP_PROP_AUTO_WB, 0)

        log.info(
            "Caméra prête : %dx%d @ %.1f fps (%s)",
            self._size[0],
            self._size[1],
            self._fps,
            self._fourcc or "?",
        )


def probe(device: int | str = 0, timeout_s: float = 5.0) -> dict[str, Any]:
    """Ouvre brièvement la caméra pour vérifier qu'elle fonctionne."""
    camera = Camera(CameraConfig(device=device, warmup_frames=3))
    started = time.time()
    try:
        camera.open()
        ok, frame = camera.read_latest(drain=2)
        return {
            "ok": bool(ok and frame is not None),
            "info": camera.info(),
            "elapsed_s": round(time.time() - started, 3),
            "frame_shape": list(frame.shape) if frame is not None else None,
        }
    except CameraError as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.time() - started, 3)}
    finally:
        camera.close()
