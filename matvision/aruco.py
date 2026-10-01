"""Détection des tags ArUco.

Le module reste volontairement fin : il encapsule la création du dictionnaire et
des paramètres du détecteur, la détection (avec masque ROI optionnel) et
quelques utilitaires géométriques sur les coins détectés.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "ArucoDetector",
    "default_detector_parameters",
    "markers_by_id",
    "marker_center",
    "draw_markers",
]

DEFAULT_DICTIONARY = "DICT_4X4_50"


# --------------------------------------------------------------------------- #
# Paramètres du détecteur
# --------------------------------------------------------------------------- #
def default_detector_parameters() -> Any:
    """Crée un ``DetectorParameters`` compatible avec les versions d'OpenCV."""
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()  # OpenCV < 4.7


def _resolve_parameter_value(value: Any) -> Any:
    """Résout les constantes nommées (``"CORNER_REFINE_SUBPIX"`` -> ``cv2.aruco....``)."""
    if isinstance(value, str):
        if hasattr(cv2.aruco, value):
            return getattr(cv2.aruco, value)
        if hasattr(cv2, value):
            return getattr(cv2, value)
    return value


def apply_parameters(params: Any, values: dict[str, Any]) -> list[str]:
    """Applique un dictionnaire de paramètres ; renvoie les clés inconnues."""
    unknown: list[str] = []
    for key, value in (values or {}).items():
        if not hasattr(params, key):
            unknown.append(key)
            continue
        setattr(params, key, _resolve_parameter_value(value))
    return unknown


def _build_detector(dictionary: Any, parameters: Any) -> Any | None:
    """Instancie ``cv2.aruco.ArucoDetector`` si la version d'OpenCV le propose.

    OpenCV 5.0 a supprimé la fonction libre ``cv2.aruco.detectMarkers`` : seule
    la classe subsiste. On construit donc la classe quand elle existe et on
    retombe sur la fonction libre pour OpenCV 4.6 et antérieurs.
    """
    detector_cls = getattr(cv2.aruco, "ArucoDetector", None)
    if detector_cls is None:
        return None
    refine_cls = getattr(cv2.aruco, "RefineParameters", None)
    try:
        if refine_cls is not None:
            return detector_cls(dictionary, parameters, refine_cls())
        return detector_cls(dictionary, parameters)
    except Exception as exc:  # pragma: no cover - dépend de la version
        log.debug("ArucoDetector indisponible (%s), repli sur detectMarkers", exc)
        return None


# --------------------------------------------------------------------------- #
# Détecteur
# --------------------------------------------------------------------------- #
class ArucoDetector:
    """Encapsule dictionnaire + paramètres + appel ``detectMarkers``."""

    def __init__(self, dictionary: str = DEFAULT_DICTIONARY, params: dict[str, Any] | None = None) -> None:
        self.dictionary_name = dictionary
        dictionary_id = getattr(cv2.aruco, dictionary, None)
        if dictionary_id is None:
            raise ValueError(f"dictionnaire ArUco inconnu : {dictionary!r}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

        self.parameters = default_detector_parameters()
        unknown = apply_parameters(self.parameters, params or {})
        if unknown:
            log.warning(
                "Paramètres ArUco ignorés (absents de cette version d'OpenCV) : %s",
                ", ".join(sorted(unknown)),
            )
        self._detector = _build_detector(self.dictionary, self.parameters)
        log.debug(
            "Détecteur ArUco prêt (dictionnaire=%s, api=%s)",
            dictionary,
            "ArucoDetector" if self._detector is not None else "detectMarkers",
        )

    # ------------------------------------------------------------------ #
    def detect(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        gray: bool = False,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        """Détecte les marqueurs.

        :param image: image BGR (ou niveaux de gris si ``gray=True``).
        :param mask: masque binaire ROI ; ce qui est hors ROI est neutralisé,
            ce qui évite de traiter toute l'image.
        :returns: ``(corners, ids, rejected)``.
        """
        if gray:
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        working = gray_image
        if mask is not None:
            working = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        if self._detector is not None:
            corners, ids, rejected = self._detector.detectMarkers(working)
        else:  # OpenCV < 4.7
            corners, ids, rejected = cv2.aruco.detectMarkers(
                working, self.dictionary, parameters=self.parameters
            )
        return list(corners), ids, list(rejected)

    # ------------------------------------------------------------------ #
    def detect_by_id(self, image: np.ndarray, mask: np.ndarray | None = None, *, gray: bool = False):
        """Comme :meth:`detect` mais renvoie ``dict[id -> coins]``."""
        corners, ids, rejected = self.detect(image, mask, gray=gray)
        return markers_by_id(corners, ids), rejected


# --------------------------------------------------------------------------- #
# Utilitaires sur les détections
# --------------------------------------------------------------------------- #
def markers_by_id(
    corners: Sequence[np.ndarray], ids: np.ndarray | None
) -> dict[int, np.ndarray]:
    """Convertit ``(corners, ids)`` en ``{id: array(4, 2)}``."""
    result: dict[int, np.ndarray] = {}
    if ids is None or len(ids) == 0:
        return result
    for index, marker_id in enumerate(np.asarray(ids).flatten()):
        result[int(marker_id)] = np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
    return result


def marker_center(corners: np.ndarray) -> np.ndarray:
    """Centre du marqueur (moyenne des 4 coins)."""
    return np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0)


# --------------------------------------------------------------------------- #
# Affichage
# --------------------------------------------------------------------------- #
def draw_markers(
    frame: np.ndarray,
    markers: dict[int, np.ndarray],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    rejected: Iterable[np.ndarray] | None = None,
    show_ids: bool = True,
    thickness: int = 2,
) -> None:
    """Dessine les marqueurs détectés et rejetés (modifie ``frame`` en place).

    Le tracé est fait « à la main » (``polylines`` + ``putText``) plutôt qu'avec
    ``cv2.aruco.drawDetectedMarkers``, dont la signature varie selon les
    versions d'OpenCV.
    """
    for marker_id, corners in markers.items():
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2).astype(np.int32)
        cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
        if show_ids:
            center = points.mean(axis=0).astype(int)
            label = str(marker_id)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.putText(
                frame,
                label,
                (int(center[0]) - text_width // 2, int(center[1]) + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    if rejected:
        for corners in rejected:
            points = np.asarray(corners, dtype=np.float64).reshape(4, 2).astype(np.int32)
            cv2.polylines(frame, [points], True, (100, 0, 255), 1, cv2.LINE_AA)
