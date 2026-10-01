"""Repère de la table construit à partir des 4 tags ArUco de coin.

Deux sorties complémentaires :

* une **homographie** image -> table (millimètres), robuste et adaptée aux objets
  posés à plat ; elle ne nécessite aucune calibration intrinsèque ;
* la **pose de la caméra** dans le repère table (``solvePnP``), disponible si la
  calibration intrinsèque a été faite — utile pour le diagnostic / le suivi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import cv2
import numpy as np

from .calibration import Intrinsics
from .config import TableConfig
from .geometry import (
    Position,
    apply_homography,
    apply_homography_direction,
    marker_corners_table,
    normalize_angle_deg,
)

log = logging.getLogger(__name__)

__all__ = ["TableLocalization", "localize_table", "table_polygon_image"]

#: Nombre minimal de tags de coin détectés pour construire le repère.
MIN_MARKERS = 2


@dataclass
class TableLocalization:
    """Résultat de la localisation du repère table sur une image."""

    ok: bool = False
    homography: np.ndarray | None = None
    camera_position: Position | None = None
    camera_rvec: np.ndarray | None = None
    camera_tvec: np.ndarray | None = None
    used_ids: list[int] = field(default_factory=list)
    total_points: int = 0
    inliers: int = 0
    residual_mm: float = 0.0
    pose_error_px: float = 0.0
    reason: str = ""

    # ------------------------------------------------------------------ #
    def image_to_table(self, points: np.ndarray) -> np.ndarray:
        """Convertit un ou plusieurs points image (pixels) en coordonnées table (mm)."""
        if self.homography is None:
            raise ValueError("aucune homographie disponible")
        return apply_homography(self.homography, points)

    def image_direction_to_table(self, vectors: np.ndarray) -> np.ndarray:
        if self.homography is None:
            raise ValueError("aucune homographie disponible")
        return apply_homography_direction(self.homography, vectors)

    def marker_angle_deg(self, corners: np.ndarray, angle_offset: float = 0.0) -> float:
        """Lacet table (degrés) d'un marqueur à partir de ses 4 coins image."""
        pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        # Moyenne des deux arêtes parallèles qui portent l'axe +x du marqueur.
        edges = np.array([pts[1] - pts[0], pts[2] - pts[3]])
        if self.homography is None:
            raise ValueError("aucune homographie disponible")
        mapped = self.image_direction_to_table(edges)
        mean_edge = mapped.mean(axis=0)
        if float(np.linalg.norm(mean_edge)) < 1e-9:
            return normalize_angle_deg(angle_offset)
        return normalize_angle_deg(np.degrees(np.arctan2(mean_edge[1], mean_edge[0])) + angle_offset)

    def to_dict(self) -> dict:
        data = {
            "ok": self.ok,
            "used_ids": self.used_ids,
            "points": self.total_points,
            "inliers": self.inliers,
            "residual_mm": round(self.residual_mm, 2),
            "reason": self.reason,
        }
        if self.camera_position is not None:
            data["camera_position"] = self.camera_position.to_dict()
            data["pose_error_px"] = round(self.pose_error_px, 3)
        return data


# --------------------------------------------------------------------------- #
def localize_table(
    markers: Mapping[int, np.ndarray],
    table: TableConfig,
    intrinsics: Intrinsics | None = None,
    *,
    min_inliers: int = 12,
    ransac_threshold_mm: float = 8.0,
) -> TableLocalization:
    """Construit le repère table à partir des tags de coin détectés.

    :param markers: ``{id: coins(4, 2)}`` issus de la détection ArUco.
    :param table: configuration des tags de coin (positions connues en mm).
    :param intrinsics: calibration intrinsèque optionnelle (pour la pose caméra).
    :param min_inliers: nombre minimal de coins cohérents avec l'homographie.
    """
    image_points: list[np.ndarray] = []
    table_points: list[np.ndarray] = []
    used_ids: list[int] = []

    for marker in table.markers:
        corners = markers.get(marker.id)
        if corners is None:
            continue
        used_ids.append(marker.id)
        image_points.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
        table_points.append(marker_corners_table(marker.x, marker.y, marker.a, marker.size))

    if len(used_ids) < MIN_MARKERS:
        return TableLocalization(
            used_ids=used_ids,
            reason=(
                f"{len(used_ids)} tag(s) de coin détecté(s) sur {len(table.markers)} "
                f"({MIN_MARKERS} minimum)"
            ),
        )

    src = np.vstack(image_points).astype(np.float64)
    dst = np.vstack(table_points).astype(np.float64)

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold_mm, maxIters=5000)
    if homography is None:
        return TableLocalization(used_ids=used_ids, reason="homographie non calculable")

    inliers = int(mask.sum()) if mask is not None else int(src.shape[0])
    projected = apply_homography(homography, src)
    residual = float(np.mean(np.linalg.norm(projected - dst, axis=1)))

    result = TableLocalization(
        ok=True,
        homography=np.asarray(homography, dtype=np.float64),
        used_ids=used_ids,
        total_points=int(src.shape[0]),
        inliers=inliers,
        residual_mm=residual,
    )

    if inliers < min_inliers:
        result.ok = False
        result.reason = f"homographie instable ({inliers} inliers < {min_inliers})"
        return result

    if intrinsics is not None and intrinsics.camera_matrix is not None:
        pose = _estimate_camera_pose(dst, src, intrinsics)
        if pose is not None:
            rvec, tvec, error = pose
            result.camera_rvec = rvec
            result.camera_tvec = tvec
            result.pose_error_px = error
            result.camera_position = _camera_position_in_table(rvec, tvec)

    return result


# --------------------------------------------------------------------------- #
def _estimate_camera_pose(
    table_points: np.ndarray, image_points: np.ndarray, intrinsics: Intrinsics
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """``solvePnP`` table -> caméra ; renvoie ``(rvec, tvec, erreur_px)``."""
    object_points = np.hstack(
        [table_points, np.zeros((table_points.shape[0], 1))]
    ).astype(np.float64)
    image_pts = image_points.reshape(-1, 1, 2).astype(np.float64)

    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_pts,
                np.asarray(intrinsics.camera_matrix, dtype=np.float64),
                np.asarray(intrinsics.dist_coeffs, dtype=np.float64),
                flags=flag,
            )
        except cv2.error as exc:  # pragma: no cover - dépend de la version d'OpenCV
            log.debug("solvePnP (flag=%s) a échoué : %s", flag, exc)
            continue
        if ok:
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
            )
            error = float(
                np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1))
            )
            return rvec.reshape(3), tvec.reshape(3), error
    return None


def _camera_position_in_table(rvec: np.ndarray, tvec: np.ndarray) -> Position:
    """Position / orientation de la caméra exprimée dans le repère table."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    position = -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    # Axe +x de la caméra exprimé dans le repère table = première ligne de R.
    yaw = np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))
    return Position(
        x=float(position[0, 0]),
        y=float(position[1, 0]),
        z=float(position[2, 0]),
        a=normalize_angle_deg(float(yaw)),
    )


def table_polygon_image(localization: TableLocalization, table: TableConfig) -> np.ndarray | None:
    """Emprise de la table dans l'image (4 coins projetés), ou ``None``.

    Sert à construire la région d'intérêt : contrairement au quadrilatère des
    tags de coin (qui peut être nettement plus petit que le tapis), cette
    emprise couvre toute la surface de jeu.
    """
    if localization.homography is None:
        return None
    try:
        inverse = np.linalg.inv(np.asarray(localization.homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None

    half_width = float(table.width_mm) / 2.0
    half_height = float(table.height_mm) / 2.0
    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return None
    return projected
