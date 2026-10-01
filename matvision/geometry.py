"""Géométrie du plan de jeu.

Conventions
-----------
Repère **table** (millimètres) :

* origine au centre de la table ;
* ``+x`` vers la droite (largeur, 2000 mm) ;
* ``+y`` vers le fond (longueur, 3000 mm) ;
* ``+z`` vers le haut, perpendiculaire au tapis ;
* angle ``a`` = lacet en degrés, sens trigonométrique vu du dessus,
  ``0°`` = axe ``+x`` de la table, normalisé dans ``]-180, 180]``.

Repère **marqueur** (comme OpenCV) : origine au centre du tag, ``+x`` vers la
droite, ``+y`` vers le bas, coins renvoyés dans l'ordre haut-gauche,
haut-droite, bas-droite, bas-gauche.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Position",
    "normalize_angle_deg",
    "angle_difference_deg",
    "average_angle_deg",
    "marker_corners_local",
    "marker_corners_table",
    "apply_homography",
    "apply_homography_direction",
    "build_roi_mask",
    "polygon_hull",
]


# --------------------------------------------------------------------------- #
# Angles
# --------------------------------------------------------------------------- #
def normalize_angle_deg(angle: float) -> float:
    """Ramène un angle en degrés dans l'intervalle ``]-180, 180]``."""
    value = (float(angle) + 180.0) % 360.0 - 180.0
    if value <= -180.0:
        value += 360.0
    return value


def angle_difference_deg(a: float, b: float) -> float:
    """Plus courte différence angulaire ``a - b`` en degrés."""
    return normalize_angle_deg(float(a) - float(b))


def average_angle_deg(angles: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Moyenne circulaire d'angles en degrés (robuste au passage par ±180°)."""
    if not angles:
        return 0.0
    if weights is None:
        weights = [1.0] * len(angles)
    sin_sum = sum(math.sin(math.radians(a)) * w for a, w in zip(angles, weights))
    cos_sum = sum(math.cos(math.radians(a)) * w for a, w in zip(angles, weights))
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return normalize_angle_deg(angles[0])
    return normalize_angle_deg(math.degrees(math.atan2(sin_sum, cos_sum)))


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    """Équivalent Python de ``position_t`` (x, y, a) + hauteur ``z``."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    a: float = 0.0

    def to_dict(self, digits: int = 2) -> dict:
        """Représentation JSON arrondie (mm et degrés)."""
        return {
            "x": round(float(self.x), digits),
            "y": round(float(self.y), digits),
            "z": round(float(self.z), digits),
            "a": round(normalize_angle_deg(self.a), digits),
        }

    def lerp(self, other: "Position", ratio: float) -> "Position":
        """Interpolation linéaire (l'angle suit le plus court chemin)."""
        ratio = max(0.0, min(1.0, float(ratio)))
        return Position(
            x=self.x + (other.x - self.x) * ratio,
            y=self.y + (other.y - self.y) * ratio,
            z=self.z + (other.z - self.z) * ratio,
            a=normalize_angle_deg(self.a + angle_difference_deg(other.a, self.a) * ratio),
        )

    def distance_to(self, other: "Position") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


# --------------------------------------------------------------------------- #
# Coins de marqueurs
# --------------------------------------------------------------------------- #
def marker_corners_local(size: float) -> np.ndarray:
    """Coins du marqueur dans son repère local, ordre OpenCV."""
    half = float(size) / 2.0
    return np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float64,
    )


def marker_corners_table(cx: float, cy: float, angle_deg: float, size: float) -> np.ndarray:
    """Coins d'un marqueur (mm) exprimés dans le repère table.

    Le ``+y`` du marqueur pointe vers le bas dans l'image ; comme le ``+y`` de la
    table pointe vers le fond (haut de l'image pour une caméra en visée
    plongeante), on inverse l'axe ``y`` avant la rotation.
    """
    local = marker_corners_local(size).copy()
    local[:, 1] *= -1.0  # repère marqueur (y vers le bas) -> repère table (y vers le fond)
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.array([cx, cy], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Homographie
# --------------------------------------------------------------------------- #
def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Applique ``H`` à des points ``(N, 2)`` (coordonnées euclidiennes)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([pts, ones]) @ np.asarray(homography, dtype=np.float64).T
    w = homogeneous[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return homogeneous[:, :2] / w


def apply_homography_direction(homography: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Applique la partie linéaire de ``H`` à des vecteurs directions ``(N, 2)``."""
    vecs = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    zeros = np.zeros((vecs.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([vecs, zeros]) @ np.asarray(homography, dtype=np.float64).T
    return homogeneous[:, :2]


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def polygon_hull(corners_groups: Iterable[np.ndarray]) -> np.ndarray | None:
    """Enveloppe convexe de plusieurs jeux de coins ``(N, 2)``."""
    import cv2

    points = [np.asarray(group, dtype=np.float32).reshape(-1, 2) for group in corners_groups]
    points = [p for p in points if p.size]
    if not points:
        return None
    stacked = np.vstack(points)
    if stacked.shape[0] < 3:
        return None
    return cv2.convexHull(stacked)


def build_roi_mask(
    shape: tuple[int, int],
    corners_groups: Iterable[np.ndarray],
    margin_px: int = 20,
) -> np.ndarray | None:
    """Masque binaire de la zone de travail définie par les tags de coin.

    Renvoie ``None`` si aucune zone exploitable (moins de 3 points).
    """
    import cv2

    hull = polygon_hull(corners_groups)
    if hull is None:
        return None

    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(hull).astype(np.int32)], 255)

    margin = int(max(0, margin_px))
    if margin:
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    return mask
