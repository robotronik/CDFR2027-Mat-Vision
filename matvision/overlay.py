"""Annotations de l'image d'aperçu : tags, axes de la table, positions, HUD.

Fonctions pures, sans état : elles ne reçoivent que les données à dessiner. Le
moteur (:mod:`matvision.vision`) reste ainsi centré sur l'acquisition.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .aruco import draw_markers, marker_center
from .table import TableLocalization

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - uniquement pour les annotations de type
    from .calibration import Intrinsics

__all__ = [
    "annotate",
    "annotate_test",
    "draw_box_edges",
    "box_face_points",
    "box_points_3d",
    "draw_hud",
]

CORNER_COLOR = (255, 128, 0)     # BGR, bleu clair
OBJECT_COLOR = (0, 220, 255)     # BGR, jaune
WARNING_COLOR = (0, 165, 255)    # BGR, orange
TEST_COLOR = (0, 255, 160)       # BGR, vert menthe (tags non déclarés)
BOX_COLOR = (255, 160, 0)        # BGR, bleu-vert clair (arêtes de la boîte)

#: Arêtes de la boîte (paires d'indices dans :func:`box_points_3d`).
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),          # face avant (portant le tag)
    (4, 5), (5, 6), (6, 7), (7, 4),          # face arrière
    (0, 4), (1, 5), (2, 6), (3, 7),          # montants
]


def annotate(
    frame: np.ndarray,
    corner_markers: dict[int, np.ndarray],
    object_markers: dict[int, np.ndarray],
    localization: TableLocalization,
    objects: list[dict],
) -> None:
    """Dessine les tags de coin, les objets relevés et les axes de la table."""
    log.debug(
        "annotate : %d tags de coin, %d tags objets, %d relevés",
        len(corner_markers),
        len(object_markers),
        len(objects),
    )
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


def annotate_test(
    frame: np.ndarray,
    markers: dict[int, np.ndarray],
    objects: list[dict],
    intrinsics: "Intrinsics | None" = None,
) -> None:
    """Aperçu du mode test : tous les tags détectés et arêtes des boîtes.

    Contrairement à :func:`annotate`, il n'y a ni repère table ni tags de coin :
    chaque tag est étiqueté par son identifiant (et son libellé s'il est
    déclaré), et les éléments de jeu voient leurs arêtes tracées.
    """
    declared = {item["id"] for item in objects if item.get("declared")}
    log.debug("annotate_test : %d tags détectés, %d déclarés", len(markers), len(declared))
    draw_markers(frame, {m: c for m, c in markers.items() if m not in declared}, color=TEST_COLOR)
    draw_markers(frame, {m: c for m, c in markers.items() if m in declared}, color=OBJECT_COLOR)

    for item in objects:
        corners = markers.get(item["id"])
        if corners is None:
            continue
        center = marker_center(corners)
        color = OBJECT_COLOR if item.get("declared") else TEST_COLOR
        label = f"{item['id']} {item['label']}" if item.get("declared") else str(item["id"])
        cv2.putText(
            frame,
            label,
            (int(center[0]) - 24, int(center[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        if item.get("box_mm"):
            draw_box_edges(frame, corners, item, intrinsics)


def box_face_points(length: float, width: float) -> np.ndarray:
    """Sommets (mm) de la face taguée, repère local du tag (``+y`` vers le haut)."""
    half_length, half_width = length / 2.0, width / 2.0
    return np.array(
        [
            [-half_length, half_width],
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
        ],
        dtype=np.float32,
    )


def box_points_3d(length: float, width: float, height: float) -> np.ndarray:
    """8 sommets (mm) de la boîte, repère local du tag (``z=0`` sur le tag)."""
    half_length, half_width = length / 2.0, width / 2.0
    front = [
        [-half_length, half_width, 0.0],
        [half_length, half_width, 0.0],
        [half_length, -half_width, 0.0],
        [-half_length, -half_width, 0.0],
    ]
    back = [
        [-half_length, half_width, -height],
        [half_length, half_width, -height],
        [half_length, -half_width, -height],
        [-half_length, -half_width, -height],
    ]
    return np.array(front + back, dtype=np.float64)


def draw_box_edges(
    frame: np.ndarray,
    corners: np.ndarray,
    item: dict,
    intrinsics: "Intrinsics | None" = None,
    *,
    color: tuple[int, int, int] = BOX_COLOR,
    thickness: int = 2,
) -> None:
    """Trace les arêtes de la boîte d'un élément de jeu (``item["box_mm"]``).

    La face portant le tag (``length_mm × width_mm``) est projetée
    **exactement** via l'homographie locale du tag : aucune calibration
    intrinsèque n'est requise. Si les intrinsèques correspondent à la
    résolution de l'image, le volume complet (12 arêtes) est en plus projeté
    par ``solvePnP`` — la boîte s'étend derrière la face taguée.
    """
    length, width, height = (float(value) for value in item["box_mm"])
    tag_size = float(item.get("size_mm") or 100.0)
    half = tag_size / 2.0

    source = np.array(
        [[-half, half], [half, half], [half, -half], [-half, -half]], dtype=np.float32
    )
    target = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    homography = cv2.getPerspectiveTransform(source, target)

    face = box_face_points(length, width)
    projected = cv2.perspectiveTransform(face.reshape(-1, 1, 2), homography).reshape(-1, 2)
    cv2.polylines(
        frame, [np.round(projected).astype(np.int32)], True, color, thickness, cv2.LINE_AA
    )

    # Une calibration d'une autre résolution fausserait le tracé 3D : on s'abstient.
    if intrinsics is not None and tuple(intrinsics.image_size) != (
        frame.shape[1],
        frame.shape[0],
    ):
        return

    pose = _tag_pose(source, target, intrinsics)
    if pose is None:
        return
    rvec, tvec = pose
    volume = box_points_3d(length, width, height)
    image_points, _ = cv2.projectPoints(
        volume,
        rvec,
        tvec,
        np.asarray(intrinsics.camera_matrix, dtype=np.float64),
        np.asarray(intrinsics.dist_coeffs, dtype=np.float64),
    )
    image_points = image_points.reshape(-1, 2)
    for start, end in BOX_EDGES:
        cv2.line(
            frame,
            tuple(np.round(image_points[start]).astype(int)),
            tuple(np.round(image_points[end]).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _tag_pose(
    source: np.ndarray, target: np.ndarray, intrinsics: "Intrinsics | None"
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pose du tag (repère tag -> caméra) par ``solvePnP``, ou ``None``."""
    if intrinsics is None or intrinsics.camera_matrix is None:
        return None
    object_points = np.hstack(
        [np.asarray(source, dtype=np.float64).reshape(4, 2), np.zeros((4, 1))]
    )
    image_points = np.asarray(target, dtype=np.float64).reshape(-1, 1, 2)
    camera_matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics.dist_coeffs, dtype=np.float64)
    for flags in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs, flags=flags
            )
        except cv2.error:  # pragma: no cover - dépend de la version d'OpenCV
            continue
        if ok:
            return rvec.reshape(3), tvec.reshape(3)
    return None


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
