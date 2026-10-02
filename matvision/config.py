"""Configuration du mat de vision.

La configuration vit dans un fichier JSON (``config/default.json``) ; chaque
champ possède une valeur par défaut, il n'est donc jamais obligatoire d'écrire
le fichier. Les scripts offrent des surcharges en ligne de commande.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"
DATA_DIR = PROJECT_ROOT / "data"

#: Paramètres du détecteur ArUco (noms d'attributs ``cv2.aruco.DetectorParameters``).
DEFAULT_ARUCO_PARAMS: dict[str, Any] = {
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 23,
    "adaptiveThreshWinSizeStep": 10,
    "adaptiveThreshConstant": 7,
    "minMarkerPerimeterRate": 0.02,
    "maxMarkerPerimeterRate": 4.0,
    "polygonalApproxAccuracyRate": 0.03,
    "minCornerDistanceRate": 0.05,
    "minDistanceToBorder": 3,
    "minMarkerDistanceRate": 0.03,
    "cornerRefinementMethod": "CORNER_REFINE_SUBPIX",
    "cornerRefinementWinSize": 5,
    "relativeCornerRefinmentWinSize": 0.01,
    "cornerRefinementMaxIterations": 30,
    "cornerRefinementMinAccuracy": 0.1,
    "markerBorderBits": 1,
    "perspectiveRemovePixelPerCell": 4,
    "perspectiveRemoveIgnoredMarginPerCell": 0.13,
    "maxErroneousBitsInBorderRate": 0.35,
    "minOtsuStdDev": 5.0,
    "errorCorrectionRate": 0.6,
    "detectInvertedMarker": False,
    "useAruco3Detection": True,
}


def _subset(cls, data: dict | None) -> dict:
    """Ne conserve que les clés connues de la dataclass ``cls``."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in allowed}


@dataclass
class CameraConfig:
    """Webcam USB (Logitech 4K Stream Edition) branchée sur la LattePanda Delta."""

    device: int | str = 0
    backend: str = "v4l2"          # "" = auto, "v4l2" sous Linux, "dshow" sous Windows
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"           # MJPG indispensable pour le 4K sur USB
    buffer_size: int = 1           # 1 = latence minimale
    warmup_frames: int = 5
    auto_exposure: bool = True
    auto_white_balance: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "CameraConfig":
        return cls(**_subset(cls, data))


def _default_corner_markers() -> list["CornerMarker"]:
    """Tags ArUco des 4 coins de la zone de travail (repère table, mm)."""
    return [
        CornerMarker(id=20, x=-400.0, y=-900.0, a=0.0),
        CornerMarker(id=21, x=-400.0, y=900.0, a=0.0),
        CornerMarker(id=22, x=400.0, y=-900.0, a=0.0),
        CornerMarker(id=23, x=400.0, y=900.0, a=0.0),
    ]


@dataclass
class CornerMarker:
    """Tag de coin : position connue dans le repère table."""

    id: int
    x: float
    y: float
    a: float = 0.0
    size: float = 100.0

    @classmethod
    def from_dict(cls, data: dict) -> "CornerMarker":
        return cls(**_subset(cls, data))


@dataclass
class TableConfig:
    """Géométrie de la table et tags de coin."""

    width_mm: float = 2000.0       # axe x
    height_mm: float = 3000.0      # axe y
    markers: list[CornerMarker] = field(default_factory=_default_corner_markers)
    roi_mode: str = "table"        # "table" = tout le tapis, "markers" = quadrilatère des tags
    roi_margin_px: int = 25        # marge (pixels) autour de la zone analysée

    @classmethod
    def from_dict(cls, data: dict | None) -> "TableConfig":
        data = data or {}
        kwargs = _subset(cls, data)
        if "markers" in data and data["markers"] is not None:
            kwargs["markers"] = [CornerMarker.from_dict(m) for m in data["markers"]]
        return cls(**kwargs)

    @property
    def marker_ids(self) -> set[int]:
        return {m.id for m in self.markers}


@dataclass
class BoxConfig:
    """Boîte d'un élément de jeu, décrite autour de son tag.

    Le tag est centré sur une face ``length_mm × width_mm`` (``length_mm`` dans
    l'axe ``+x`` local du tag, ``width_mm`` dans son axe ``+y``) ; ``height_mm``
    est la profondeur de la boîte, perpendiculaire à cette face. Sert au tracé
    des arêtes dans l'aperçu (mode ``--test``).
    """

    length_mm: float = 320.0
    width_mm: float = 110.0
    height_mm: float = 110.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "BoxConfig":
        return cls(**_subset(cls, data))


@dataclass
class ObjectConfig:
    """Objet de la table identifié par un tag ArUco (robot ou élément de jeu).

    La position renvoyée est celle du centre du tag (ou du centre de l'objet
    si ``offset`` est renseigné) dans le repère de la table. ``size`` est la
    taille physique du tag (mm) : elle sert d'échelle apparente en mode test et
    au tracé de la boîte (``box``).
    """

    id: int
    label: str = ""
    angle_offset: float = 0.0      # correction si le « devant » n'est pas l'axe +x du tag
    offset: tuple[float, float] = (0.0, 0.0)   # décalage (mm) dans le repère du tag
    size: float = 100.0            # côté du tag imprimé (mm)
    box: BoxConfig | None = None   # géométrie de la boîte (éléments de jeu)

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectConfig":
        data = dict(data)
        if isinstance(data.get("offset"), (list, tuple)):
            data["offset"] = tuple(float(v) for v in data["offset"])
        box = data.pop("box", None)
        config = cls(**_subset(cls, data))
        if box:
            config.box = BoxConfig.from_dict(box)
        return config


@dataclass
class DetectionConfig:
    """Réglages de la détection en continu."""

    min_homography_inliers: int = 12
    draw: bool = True              # annote l'image d'aperçu

    @classmethod
    def from_dict(cls, data: dict | None) -> "DetectionConfig":
        return cls(**_subset(cls, data))


@dataclass
class CalibrationConfig:
    """Calibration automatique des intrinsèques (damier)."""

    intrinsics_file: str = "data/camera_calibration.json"
    chessboard_cols: int = 7       # coins internes en largeur
    chessboard_rows: int = 7       # coins internes en hauteur
    square_size_mm: float = 25.0
    target_frames: int = 20        # nb de poses à capturer automatiquement
    min_board_area_ratio: float = 0.04   # aire minimale du damier / image
    max_board_area_ratio: float = 0.95
    min_sharpness: float = 60.0    # variance du Laplacien (netteté)
    min_move_ratio: float = 0.12   # déplacement minimal entre 2 captures
    border_margin_px: int = 10
    max_capture_seconds: float = 120.0
    flags: str = "CALIB_RATIONAL_MODEL"

    @classmethod
    def from_dict(cls, data: dict | None) -> "CalibrationConfig":
        return cls(**_subset(cls, data))


@dataclass
class Config:
    """Configuration complète du mat de vision."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    aruco: dict[str, Any] = field(
        default_factory=lambda: {
            "dictionary": "DICT_4X4_50",
            "params": dict(DEFAULT_ARUCO_PARAMS),
        }
    )
    table: TableConfig = field(default_factory=TableConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    objects: list[ObjectConfig] = field(default_factory=list)
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    preview_quality: int = 75

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict | None) -> "Config":
        data = data or {}
        config = cls()
        if "camera" in data:
            config.camera = CameraConfig.from_dict(data["camera"])
        if "aruco" in data:
            aruco = dict(data["aruco"])
            params = dict(DEFAULT_ARUCO_PARAMS)
            params.update(aruco.get("params") or {})
            config.aruco = {
                "dictionary": aruco.get("dictionary", "DICT_4X4_50"),
                "params": params,
            }
        if "table" in data:
            config.table = TableConfig.from_dict(data["table"])
        if "detection" in data:
            config.detection = DetectionConfig.from_dict(data["detection"])
        if "calibration" in data:
            config.calibration = CalibrationConfig.from_dict(data["calibration"])
        if "objects" in data and data["objects"] is not None:
            config.objects = [ObjectConfig.from_dict(o) for o in data["objects"]]
        for key in ("api_host", "api_port", "preview_quality"):
            if data.get(key) is not None:
                setattr(config, key, data[key])
        return config

    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Retourne la liste des avertissements / incohérences détectées."""
        problems: list[str] = []
        if len(self.table.markers) < 4:
            problems.append(
                "moins de 4 tags de coin configurés : le repère table ne pourra pas "
                "être construit de façon robuste"
            )
        ids = [m.id for m in self.table.markers]
        if len(ids) != len(set(ids)):
            problems.append("identifiants de tags de coin dupliqués")
        overlap = set(ids) & {o.id for o in self.objects}
        if overlap:
            problems.append(f"tags utilisés à la fois comme coin et comme objet : {sorted(overlap)}")
        if self.table.roi_mode not in {"table", "markers"}:
            problems.append("table.roi_mode doit valoir 'table' ou 'markers'")
        if self.calibration.chessboard_cols < 3 or self.calibration.chessboard_rows < 3:
            problems.append("damier trop petit (>= 3x3 coins internes)")
        return problems

    # ------------------------------------------------------------------ #
    @property
    def intrinsics_path(self) -> Path:
        path = Path(self.calibration.intrinsics_file)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(path: str | Path | None = None) -> Config:
    """Charge la configuration depuis un fichier JSON (défauts si absent)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    return Config.from_dict(data)
