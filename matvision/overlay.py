"""Annotations de l'image d'aperçu : tags, axes de la table, positions, HUD.

Fonctions pures, sans état : elles ne reçoivent que les données à dessiner. Le
moteur (:mod:`matvision.vision`) reste ainsi centré sur l'acquisition.
"""

from __future__ import annotations

import cv2
import numpy as np

from .aruco import draw_markers, marker_center
from .table import TableLocalization

__all__ = ["annotate", "draw_hud"]

CORNER_COLOR = (255, 128, 0)     # BGR, bleu clair
OBJECT_COLOR = (0, 220, 255)     # BGR, jaune
WARNING_COLOR = (0, 165, 255)    # BGR, orange


def annotate(
    frame: np.ndarray,
    corner_markers: dict[int, np.ndarray],
    object_markers: dict[int, np.ndarray],
    localization: TableLocalization,
    objects: list[dict],
) -> None:
    """Dessine les tags de coin, les objets relevés et les axes de la table."""
    draw_markers(frame, corner_markers, color=CORNER_COLOR)
    draw_markers(frame, object_markers, color=OBJECT_COLOR)
    _draw_table_axes(frame, localization)

    for item in objects:
        corners = object_markers.get(item["id"])
        if corners is None:
            continue
        center = marker_center(corners)
        label = f"{item['label']} ({item['x']:.0f},{item['y']:.0f}) {item['a']:.0f}deg"
        cv2.putText(
            frame,
            label,
            (int(center[0]) - 80, int(center[1]) - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            OBJECT_COLOR,
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
            WARNING_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Écrit quelques lignes d'état en haut à gauche de l'image."""
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


def _draw_table_axes(frame: np.ndarray, localization: TableLocalization) -> None:
    """Dessine l'origine et les axes de la table (x en rouge, y en bleu)."""
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
    if not np.all(np.isfinite(projected)):
        return

    origin, x_axis, y_axis = projected
    origin_px = tuple(np.round(origin).astype(int))
    cv2.arrowedLine(
        frame, origin_px, tuple(np.round(x_axis).astype(int)),
        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2,
    )
    cv2.arrowedLine(
        frame, origin_px, tuple(np.round(y_axis).astype(int)),
        (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.2,
    )
    cv2.circle(frame, origin_px, 5, (0, 255, 255), -1)
