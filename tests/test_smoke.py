#!/usr/bin/env python3
"""Tests de fumée du mat de vision — fonctionnent sans matériel.

Une scène synthétique est rendue (caméra en visée plongeante au-dessus d'une
table de 2000x3000 mm, tags aux coins, objets sur le tapis), puis :

* détection ArUco ;
* construction du repère table (homographie + pose caméra) ;
* projection des objets en millimètres et en degrés ;
* suivi temporel ;
* calibration automatique sur des vues synthétiques de damier ;
* API Flask (client de test) avec une caméra factice.

Exécution ::

    python tests/test_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from matvision.api import create_app  # noqa: E402
from matvision.aruco import ArucoDetector, marker_center  # noqa: E402
from matvision.calibration import (  # noqa: E402
    AutoCalibrator,
    Intrinsics,
    load_intrinsics,
    save_intrinsics,
)
from matvision.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    Config,
    ObjectConfig,
    load_config,
)
from matvision.geometry import marker_corners_table, normalize_angle_deg  # noqa: E402
from matvision.table import localize_table, table_polygon_image  # noqa: E402
from matvision.vision import VisionEngine  # noqa: E402

# --------------------------------------------------------------------------- #
# Caméra synthétique
# --------------------------------------------------------------------------- #
WIDTH, HEIGHT = 2560, 1440
FX = FY = 1450.0
IMAGE_SIZE = (WIDTH, HEIGHT)
CAMERA_EYE = np.array([150.0, -120.0, 2600.0])
CAMERA_TARGET = np.array([0.0, 0.0, 0.0])

# Objets posés sur la table (id, x mm, y mm, angle deg)
SCENE_OBJECTS = [
    (1, 300.0, 250.0, 30.0),      # robot bleu
    (6, -250.0, -400.0, -75.0),   # robot jaune
    (13, 520.0, -700.0, 120.0),   # élément de jeu
]

MARKER_RENDER_PX = 480


def camera_matrix() -> np.ndarray:
    return np.array([[FX, 0.0, WIDTH / 2.0], [0.0, FY, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def look_at(eye: np.ndarray, target: np.ndarray, up: tuple[float, float, float] = (0.0, 1.0, 0.0)):
    """Matrice monde->caméra (``Xc`` droite, ``Yc`` bas, ``Zc`` direction de visée)."""
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    up_vector = np.asarray(up, dtype=np.float64)
    x_axis = np.cross(z_axis, up_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.vstack([x_axis, y_axis, z_axis])
    translation = -rotation @ eye
    return rotation, translation


def project(points_world: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    in_camera = points @ rotation.T + translation
    uv = in_camera[:, :2] / in_camera[:, 2:3]
    return np.column_stack([uv[:, 0] * FX + WIDTH / 2.0, uv[:, 1] * FY + HEIGHT / 2.0])


def render_marker(
    canvas: np.ndarray,
    dictionary,
    marker_id: int,
    marker_size: float,
    x: float,
    y: float,
    angle: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """Peint un tag du repère table sur la toile (niveaux de gris)."""
    corners_table = marker_corners_table(x, y, angle, marker_size)
    world = np.column_stack([corners_table, np.zeros(4)])
    image_corners = project(world, rotation, translation).astype(np.float32)

    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_RENDER_PX)
    source = np.array(
        [[0, 0], [MARKER_RENDER_PX, 0], [MARKER_RENDER_PX, MARKER_RENDER_PX], [0, MARKER_RENDER_PX]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, image_corners)

    layer = np.full(canvas.shape, 255, dtype=np.uint8)
    cv2.warpPerspective(
        marker, transform, (canvas.shape[1], canvas.shape[0]),
        dst=layer, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    np.minimum(canvas, layer, out=canvas)


def build_scene(config: Config) -> np.ndarray:
    """Rend une image de la table vue par la caméra synthétique."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config.aruco["dictionary"])
    )
    rotation, translation = look_at(CAMERA_EYE, CAMERA_TARGET)
    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)

    for marker in config.table.markers:
        render_marker(canvas, dictionary, marker.id, marker.size, marker.x, marker.y,
                      marker.a, rotation, translation)
    for marker_id, x, y, angle in SCENE_OBJECTS:
        render_marker(canvas, dictionary, marker_id, 100.0, x, y, angle, rotation, translation)

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# Configuration de test
# --------------------------------------------------------------------------- #
class FakeCamera:
    """Caméra factice renvoyant toujours la même image."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> "FakeCamera":
        self._open = True
        return self

    def read(self):
        return True, self.frame.copy()

    def read_latest(self, drain: int = 0):
        return True, self.frame.copy()

    def close(self) -> None:
        self._open = False

    def info(self) -> dict:
        return {
            "device": "fake", "opened": self._open, "width": WIDTH, "height": HEIGHT,
            "fourcc": "FAKE", "fps": 30.0,
            "requested": {"width": WIDTH, "height": HEIGHT, "fps": 30, "fourcc": "MJPG"},
        }


def make_config(tmpdir: Path) -> Config:
    config = load_config(DEFAULT_CONFIG_PATH)
    config.camera.width = WIDTH
    config.camera.height = HEIGHT
    config.calibration.intrinsics_file = str(tmpdir / "camera_calibration.json")
    config.detection.draw = True
    save_intrinsics(
        config.intrinsics_path,
        Intrinsics(
            camera_matrix=camera_matrix(),
            dist_coeffs=np.zeros((5, 1)),
            image_size=IMAGE_SIZE,
            rms=0.05,
            frames=20,
        ),
    )
    return config


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_geometry_units() -> None:
    assert normalize_angle_deg(190.0) == -170.0
    assert normalize_angle_deg(-180.0) == 180.0
    assert normalize_angle_deg(0.0) == 0.0

    # a = 90 deg : l'axe +x du tag part vers +y de la table
    corners = marker_corners_table(0.0, 0.0, 90.0, 100.0)
    assert np.allclose(corners[0], (-50.0, -50.0), atol=1e-9), corners[0]
    assert np.allclose(corners[1], (-50.0, 50.0), atol=1e-9), corners[1]
    assert np.allclose(corners[1] - corners[0], (0.0, 100.0), atol=1e-9)
    print("  ok  géométrie (angles, coins de marqueur)")


def test_aruco_and_table() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        frame = build_scene(config)
        detector = ArucoDetector(config.aruco["dictionary"], config.aruco["params"])
        markers, rejected = detector.detect_by_id(frame)

        corner_ids = config.table.marker_ids
        found_corners = {mid: c for mid, c in markers.items() if mid in corner_ids}
        assert found_corners.keys() == corner_ids, f"tags de coin manquants : {found_corners.keys()}"
        assert set(markers) >= {1, 6, 13}, f"objets manquants : {set(markers)}"

        intrinsics = load_intrinsics(config.intrinsics_path)
        localization = localize_table(found_corners, config.table, intrinsics)
        assert localization.ok, localization.reason
        assert localization.inliers >= 14, localization.inliers
        assert localization.residual_mm < 3.0, localization.residual_mm

        assert localization.camera_position is not None
        camera = localization.camera_position
        error = np.linalg.norm([camera.x - CAMERA_EYE[0], camera.y - CAMERA_EYE[1], camera.z - CAMERA_EYE[2]])
        assert error < 40.0, f"pose caméra à {error:.1f} mm de la vérité"
        assert abs(camera.a) < 3.0, camera.a
        assert localization.pose_error_px < 1.0, localization.pose_error_px

        print(
            f"  ok  détection ArUco : {len(markers)} tags, {len(rejected)} rejetés ; "
            f"repere table (residu {localization.residual_mm:.2f} mm, "
            f"pose a {error:.1f} mm, erreur {localization.pose_error_px:.3f} px)"
        )


def test_object_positions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        frame = build_scene(config)
        detector = ArucoDetector(config.aruco["dictionary"], config.aruco["params"])
        markers, _ = detector.detect_by_id(frame)

        corner_ids = config.table.marker_ids
        found_corners = {mid: c for mid, c in markers.items() if mid in corner_ids}
        localization = localize_table(found_corners, config.table, load_intrinsics(config.intrinsics_path))
        assert localization.ok, localization.reason

        for marker_id, expected_x, expected_y, expected_a in SCENE_OBJECTS:
            corners = markers[marker_id]
            xy = localization.image_to_table(marker_center(corners))[0]
            angle = localization.marker_angle_deg(corners)
            assert abs(xy[0] - expected_x) < 6.0, f"tag {marker_id} : x={xy[0]:.2f} != {expected_x}"
            assert abs(xy[1] - expected_y) < 6.0, f"tag {marker_id} : y={xy[1]:.2f} != {expected_y}"
            delta = abs(normalize_angle_deg(angle - expected_a))
            assert delta < 3.0, f"tag {marker_id} : a={angle:.2f} != {expected_a}"

        print("  ok  projection des objets en mm et en degrés (erreur < 6 mm, < 3°)")


def test_roi_modes() -> None:
    """La ROI doit couvrir toute la table, pas seulement le quadrilatère des tags.

    Les tags de coin sont à ±400 mm en x alors que la table fait 2000 mm : un
    objet posé près du bord (ici le tag 13, à x = 520 mm) sortirait du
    quadrilatère des tags et deviendrait invisible.
    """
    from matvision.geometry import build_roi_mask

    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        frame = build_scene(config)
        detector = ArucoDetector(config.aruco["dictionary"], config.aruco["params"])
        markers, _ = detector.detect_by_id(frame)
        corner_ids = config.table.marker_ids
        found_corners = {mid: c for mid, c in markers.items() if mid in corner_ids}
        localization = localize_table(
            found_corners, config.table, load_intrinsics(config.intrinsics_path)
        )
        assert localization.ok, localization.reason
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- mode "markers" : quadrilatère des tags uniquement -----------------
        markers_only = build_roi_mask(
            gray.shape, found_corners.values(), config.table.roi_margin_px
        )
        assert markers_only is not None
        ratio_markers = float(np.count_nonzero(markers_only)) / markers_only.size
        seen_markers, _ = detector.detect_by_id(gray, markers_only, gray=True)
        assert {1, 6} <= set(seen_markers), f"robots perdus : {set(seen_markers)}"
        assert 13 not in seen_markers, (
            "le tag 13 est hors du quadrilatère des tags : il ne doit pas être vu "
            "dans ce mode"
        )

        # --- mode "table" (défaut) : toute la surface de jeu -------------------
        quad = table_polygon_image(localization, config.table)
        assert quad is not None
        table_mask = build_roi_mask(
            gray.shape, [*found_corners.values(), quad], config.table.roi_margin_px
        )
        assert table_mask is not None
        ratio_table = float(np.count_nonzero(table_mask)) / table_mask.size
        seen_table, _ = detector.detect_by_id(gray, table_mask, gray=True)
        assert {1, 6, 13} <= set(seen_table), f"objets perdus : {set(seen_table)}"
        assert ratio_table < 0.6, f"la ROI couvre {ratio_table:.0%} de l'image"

        print(
            f"  ok  ROI : mode « markers » {ratio_markers:.0%} de l'image (tag 13 hors zone), "
            f"mode « table » {ratio_table:.0%} (tag 13 détecté)"
        )


def test_tag_configuration() -> None:
    """Les tags déclarés existent dans le dictionnaire et se décodent correctement.

    Garde-fou : un identifiant hors dictionnaire (ou une plage de couleurs
    modifiée par erreur) passe inaperçu au démarrage et rend un robot invisible.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    name = config.aruco["dictionary"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    capacity = int(dictionary.bytesList.shape[0])

    used = sorted(config.table.marker_ids | {o.id for o in config.objects})
    assert all(0 <= marker_id < capacity for marker_id in used), (
        f"identifiant hors du dictionnaire {name} (0..{capacity - 1}) : "
        f"{[i for i in used if not 0 <= i < capacity]}"
    )
    assert len(used) == len(config.table.marker_ids) + len(config.objects), (
        "un identifiant est déclaré deux fois"
    )

    labels = {o.id: o.label for o in config.objects}
    assert all(labels.get(i) == "blue" for i in range(1, 6)), f"plage bleue incorrecte : {labels}"
    assert all(labels.get(i) == "yellow" for i in range(6, 11)), f"plage jaune incorrecte : {labels}"

    detector = ArucoDetector(name, config.aruco["params"])
    for marker_id in used:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
        canvas = np.full((200, 200), 255, dtype=np.uint8)
        canvas[60:140, 60:140] = marker
        found, _ = detector.detect_by_id(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
        assert set(found) == {marker_id}, f"tag {marker_id} décodé comme {sorted(found)}"

    print(
        f"  ok  tags : {len(used)} identifiants valides et décodés "
        f"(bleu 1-5, jaune 6-10, élément {labels.get(13)})"
    )


def test_intrinsics_resolution_mismatch() -> None:
    """Une calibration faite à une autre résolution doit être signalée.

    Les intrinsèques sont en pixels : une calibration en 1920x1080 est fausse
    pour un flux 2560x1440. Les positions d'objets, elles, viennent de
    l'homographie et restent justes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        save_intrinsics(
            config.intrinsics_path,
            Intrinsics(camera_matrix(), np.zeros((5, 1)), (1920, 1080), rms=0.1, frames=20),
        )
        frame = build_scene(config)

        engine = VisionEngine(config, display=False)
        engine.camera = FakeCamera(frame)
        engine.start()
        try:
            engine.start_detection()
            deadline = time.time() + 15.0
            while time.time() < deadline and not engine.status()["intrinsics_warning"]:
                time.sleep(0.05)

            status = engine.status()
            warning = status["intrinsics_warning"]
            assert warning, "l'écart de résolution n'a pas été signalé"
            assert "1920x1080" in warning and f"{WIDTH}x{HEIGHT}" in warning, warning

            deadline = time.time() + 30.0
            while time.time() < deadline and len(engine.objects()["objects"]) < 3:
                time.sleep(0.05)
            objects = engine.objects()["objects"]
            assert len(objects) == 3, objects
            for item in objects:
                assert -1000.0 < item["x"] < 1000.0, item
                assert -1500.0 < item["y"] < 1500.0, item

            print(
                "  ok  alerte de résolution (calibration 1920x1080 / flux "
                f"{WIDTH}x{HEIGHT}) — positions d'objets toujours justes"
            )
        finally:
            engine.shutdown()


def test_object_offset() -> None:
    """Le champ ``offset`` est exprimé dans le repère du tag et tourne avec lui."""
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        config.objects = [ObjectConfig(id=13, label="element", offset=(100.0, 0.0))]
        frame = build_scene(config)

        engine = VisionEngine(config, display=False)
        engine.camera = FakeCamera(frame)
        engine.start()
        try:
            engine.start_detection()
            deadline = time.time() + 25.0
            while time.time() < deadline and not engine.objects()["objects"]:
                time.sleep(0.05)

            items = engine.objects()["objects"]
            assert len(items) == 1, items
            item = items[0]
            assert item["id"] == 13 and item["label"] == "element", item

            # Tag posé en (520, -700) avec un angle de 120° : un offset de
            # (100, 0) mm dans le repère du tag devient (-50, +86.6) mm
            # dans le repère de la table.
            assert abs(item["a"] - 120.0) < 2.0, item
            assert abs(item["x"] - (520.0 - 50.0)) < 8.0, item
            assert abs(item["y"] - (-700.0 + 86.6)) < 8.0, item

            print(
                f"  ok  offset par tag : tag (520, -700) à 120° + offset (100, 0) mm "
                f"-> centre ({item['x']:.0f}, {item['y']:.0f})"
            )
        finally:
            engine.shutdown()


def test_calibration_intrinsics_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calib.json"
        original = Intrinsics(camera_matrix(), np.zeros((5, 1)), IMAGE_SIZE, rms=0.12, frames=17)
        save_intrinsics(path, original)
        loaded = load_intrinsics(path)
        assert loaded is not None
        assert np.allclose(loaded.camera_matrix, original.camera_matrix)
        assert loaded.frames == 17 and loaded.image_size == IMAGE_SIZE
        print("  ok  sérialisation / relecture de la calibration (JSON)")


def synth_chessboard_views(views: int = 12):
    """Génère des vues d'un damier vues par une caméra pinhole synthétique."""
    cols, rows = 7, 7
    square_mm = 20.0
    pixels_per_mm = 4.0
    square_px = int(square_mm * pixels_per_mm)
    board = np.full((square_px * (rows + 1), square_px * (cols + 1)), 255, dtype=np.uint8)
    for row in range(rows + 1):
        for column in range(cols + 1):
            if (row + column) % 2 == 0:
                board[row * square_px : (row + 1) * square_px,
                      column * square_px : (column + 1) * square_px] = 0

    truth = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]])
    image_width, image_height = 1280, 960
    frames = []

    for index in range(views):
        angle = np.radians(-28.0 + 56.0 * index / max(1, views - 1))
        roll = np.radians(18.0 * np.sin(index * 1.7))
        pitch = np.radians(18.0 * np.cos(index * 1.1))
        rot_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        rot_y = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
        rot_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        rotation = rot_z @ rot_y @ rot_x
        translation = np.array([0.0, 0.0, 430.0 + 40.0 * np.sin(index)])

        homography_mm = truth @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
        homography_px = homography_mm @ np.diag([1.0 / pixels_per_mm, 1.0 / pixels_per_mm, 1.0])

        canvas = np.full((image_height, image_width), 255, dtype=np.uint8)
        cv2.warpPerspective(
            board, homography_px, (image_width, image_height), dst=canvas,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    return frames, truth


def test_automatic_calibration() -> None:
    frames, truth = synth_chessboard_views()
    config = load_config(DEFAULT_CONFIG_PATH).calibration
    config.chessboard_cols, config.chessboard_rows = 7, 7
    config.square_size_mm = 20.0
    config.target_frames = 8
    config.min_move_ratio = 0.01
    config.min_sharpness = 20.0

    calibrator = AutoCalibrator(config)
    calibrator.start()
    for frame in frames:
        calibrator.process(frame)
        if calibrator.finished:
            break

    assert calibrator.state.status == "done", f"{calibrator.state.status} : {calibrator.state.message}"
    assert calibrator.state.intrinsics is not None
    intrinsics = calibrator.state.intrinsics

    fx_error = abs(intrinsics.fx - truth[0, 0]) / truth[0, 0]
    cx_error = abs(intrinsics.camera_matrix[0, 2] - truth[0, 2])
    assert fx_error < 0.12, f"fx={intrinsics.fx:.1f} (attendu {truth[0, 0]:.0f})"
    assert cx_error < 15.0, f"cx={intrinsics.camera_matrix[0, 2]:.1f} (attendu {truth[0, 2]:.0f})"
    assert intrinsics.rms < 1.0, intrinsics.rms

    print(
        f"  ok  calibration automatique : {intrinsics.frames} vues, "
        f"fx={intrinsics.fx:.1f} (vérité {truth[0, 0]:.0f}), "
        f"cx={intrinsics.camera_matrix[0, 2]:.1f}, rms={intrinsics.rms:.3f} px"
    )


def test_engine_and_api() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        frame = build_scene(config)

        engine = VisionEngine(config, display=False)
        engine.camera = FakeCamera(frame)  # caméra factice : aucun matériel requis
        engine.start()
        try:
            engine.start_detection()
            deadline = time.time() + 30.0
            while time.time() < deadline:
                status = engine.status()
                if len(engine.objects()["objects"]) >= 3 and status["stats"]["fps"] > 0:
                    break
                time.sleep(0.05)

            status = engine.status()
            assert status["running"], status
            assert status["mode"] == "detect", status
            assert status["camera"]["opened"], status
            assert status["table"]["ok"], status["table"]
            assert status["stats"]["frames"] > 0, status["stats"]
            assert status["stats"]["fps"] > 0, status["stats"]

            objects = engine.objects()
            labels = {item["label"] for item in objects["objects"]}
            assert labels >= {"blue", "yellow", "element"}, objects
            for item in objects["objects"]:
                assert -1000.0 < item["x"] < 1000.0, item
                assert -1500.0 < item["y"] < 1500.0, item

            app = create_app(engine)
            client = app.test_client()

            assert client.get("/health").get_json()["ok"] is True
            assert client.get("/status").get_json()["mode"] == "detect"
            assert client.get("/table").get_json()["markers"][0]["id"] == 20

            objects_response = client.get("/objects")
            assert objects_response.status_code == 200
            payload = objects_response.get_json()
            assert "blue" in payload["by_label"], payload
            assert payload["count"] == len(payload["objects"]), payload

            single = client.get("/objects/1")
            assert single.status_code == 200 and single.get_json()["objects"], single.get_json()
            assert client.get("/objects/inexistant").status_code == 404

            preview = client.get("/preview")
            assert preview.status_code == 200, preview.status_code
            assert preview.mimetype == "image/jpeg", preview.mimetype
            assert len(preview.data) > 1000, len(preview.data)

            position = client.get("/position").get_json()
            assert position["table_locked"] is True, position
            assert position["position"] is not None, position

            assert client.post("/stop").status_code == 200
            assert client.get("/status").get_json()["mode"] == "idle"
            assert client.post("/start").get_json()["mode"] == "detect"
            assert client.post("/reset").status_code == 200
            assert client.get("/").get_json()["name"] == "matvision"
            assert client.get("/api").get_json()["name"] == "matvision"
            assert client.get("/inexistante").status_code == 404

            calibration = client.get("/calibration/status").get_json()
            assert "config" in calibration, calibration

            print(
                f"  ok  moteur + API : {payload['count']} objets relevés, "
                f"aperçu JPEG de {len(preview.data)} octets, fps {status['stats']['fps']}"
            )
        finally:
            engine.shutdown()


def test_shutdown_route() -> None:
    """``POST /shutdown`` déclenche l'arrêt, ou répond 501 s'il n'est pas câblé."""
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        engine = VisionEngine(config, display=False)
        try:
            without_callback = create_app(engine).test_client()
            assert without_callback.post("/shutdown").status_code == 501

            called = threading.Event()
            app = create_app(engine, on_shutdown=called.set)
            response = app.test_client().post("/shutdown")
            assert response.status_code == 200, response.get_json()
            assert called.wait(timeout=3.0), "le rappel d'arrêt n'a pas été appelé"

            print("  ok  POST /shutdown : 501 sans rappel, 200 et arrêt déclenché avec")
        finally:
            engine.shutdown()


def test_web_interface() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        engine = VisionEngine(config, display=False)
        engine.camera = FakeCamera(build_scene(config))
        engine.start()
        try:
            engine.start_detection()
            client = create_app(engine).test_client()

            # Page HTML
            page = client.get("/ui")
            assert page.status_code == 200, page.status_code
            assert page.mimetype == "text/html", page.mimetype
            html = page.get_data(as_text=True)
            for token in ("Mat de vision", "app.js", "style.css", "Plan de la table"):
                assert token in html, f"absent de la page : {token}"

            # Ressources statiques
            for asset in ("/static/app.js", "/static/style.css"):
                response = client.get(asset)
                assert response.status_code == 200, f"{asset} -> {response.status_code}"
                assert response.data, asset

            # Négociation de contenu sur "/"
            browser = client.get("/", headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
            assert browser.mimetype == "text/html", browser.mimetype
            curl = client.get("/", headers={"Accept": "*/*"})
            assert curl.mimetype == "application/json", curl.mimetype

            # Flux MJPEG borné (2 images) : vérifie l'en-tête et le contenu
            stream = client.get("/stream?frames=2&fps=60", buffered=False)
            assert stream.status_code == 200
            assert stream.mimetype == "multipart/x-mixed-replace", stream.mimetype
            chunks = []
            for chunk in stream.response:
                chunks.append(chunk)
                if len(chunks) >= 2:
                    break
            stream.close()
            payload = b"".join(chunks)
            assert b"Content-Type: image/jpeg" in payload, payload[:120]

            print(
                f"  ok  interface web : page HTML, ressources statiques, "
                f"négociation de contenu, flux MJPEG ({len(payload)} octets)"
            )
        finally:
            engine.shutdown()


def test_match_mode() -> None:
    """Mode match : interface web et sorties image désactivées, API JSON seule."""
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        config.detection.draw = False
        engine = VisionEngine(config, display=False)
        engine.camera = FakeCamera(build_scene(config))
        engine.start()
        try:
            engine.start_detection()
            deadline = time.time() + 25.0
            while time.time() < deadline and not engine.objects()["objects"]:
                time.sleep(0.05)

            client = create_app(engine, web_ui=False).test_client()

            # Aucune route web ni aperçu : tout répond 404 (même les statiques).
            for route in ("/", "/ui", "/preview", "/preview.jpg", "/stream", "/static/app.js"):
                assert client.get(route).status_code == 404, route

            # La découverte ne référence plus l'interface web ni les sorties image.
            discovery = client.get("/api").get_json()
            assert discovery["ui"] is None, discovery
            assert discovery["match_mode"] is True, discovery
            assert not any(
                key.endswith(("/preview", "/preview.jpg", "/stream", "/ui", "/"))
                for key in discovery["endpoints"]
            ), discovery["endpoints"]

            # L'API JSON reste pleinement fonctionnelle.
            assert client.get("/health").get_json()["ok"] is True
            assert client.get("/status").status_code == 200
            objects = client.get("/objects")
            assert objects.status_code == 200 and objects.get_json()["count"] > 0, objects

            # La détection « normale » a bien repéré la table via les 4 tags de coin
            # (la caméra « se place ») avant de relever les objets.
            assert engine.status()["table"]["ok"] is True, engine.status()["table"]

            # L'option de ligne de commande est bien câblée : --match démarre la
            # détection sans attendre /start (comme --test et --autostart).
            import server as server_module

            assert server_module.parse_args(["--match"]).match is True
            assert server_module.parse_args([]).match is False
            assert server_module.autostart_requested(server_module.parse_args(["--match"])) is True
            assert server_module.autostart_requested(server_module.parse_args(["--test"])) is True
            assert server_module.autostart_requested(
                server_module.parse_args(["--autostart"])
            ) is True
            assert server_module.autostart_requested(server_module.parse_args([])) is False

            print("  ok  mode match : API JSON seule, interface web et aperçu absents")
        finally:
            engine.shutdown()


def build_object_scene(
    config: Config, marker_id: int = 13, x: float = 0.0, y: float = 0.0, angle: float = 0.0
) -> np.ndarray:
    """Rend uniquement un tag objet — aucun tag de coin (scène du mode test)."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config.aruco["dictionary"])
    )
    rotation, translation = look_at(CAMERA_EYE, CAMERA_TARGET)
    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    render_marker(canvas, dictionary, marker_id, 100.0, x, y, angle, rotation, translation)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_test_mode() -> None:
    """Mode test : tous les tags relevés en pixels, sans tags de coin.

    Le repère table n'est pas construit ; les tags de coin présents dans
    l'image sont relevés comme les autres, et l'élément de jeu (tag 13) expose
    la géométrie de sa boîte (320x110x110 mm).
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        engine = VisionEngine(config, display=False, test_mode=True)
        engine.camera = FakeCamera(build_scene(config))
        engine.start()
        try:
            engine.start_detection()
            deadline = time.time() + 25.0
            while time.time() < deadline and (
                len(engine.objects()["objects"]) < 7
                or engine.status()["stats"]["frames"] < 1
            ):
                time.sleep(0.05)

            payload = engine.objects()
            assert payload["test_mode"] is True, payload
            assert payload["frame"] == "image", payload
            items = {item["id"]: item for item in payload["objects"]}
            assert {1, 6, 13, 20, 21, 22, 23} <= set(items), items

            for item in items.values():
                assert 0.0 <= item["x"] <= WIDTH and 0.0 <= item["y"] <= HEIGHT, item

            assert items[13]["declared"] is True and items[13]["label"] == "element", items[13]
            assert items[13]["box_mm"] == [320.0, 110.0, 110.0], items[13]
            assert items[13]["size_px"] > 10.0, items[13]
            assert items[1]["declared"] is True and "box_mm" not in items[1], items[1]
            assert items[20]["declared"] is False, items[20]

            assert engine.status()["test_mode"] is True, engine.status()
            assert engine.preview_jpeg() is not None

            # Sans tags de coin ET sans intrinsèques : la détection et le tracé
            # de la face de la boîte doivent fonctionner malgré tout.
            plain = load_config(DEFAULT_CONFIG_PATH)
            plain.camera.width, plain.camera.height = WIDTH, HEIGHT
            plain.calibration.intrinsics_file = str(Path(tmp) / "absent.json")
            blind = VisionEngine(plain, display=False, test_mode=True)
            blind.camera = FakeCamera(build_object_scene(plain))
            blind.start()
            try:
                blind.start_detection()
                deadline = time.time() + 20.0
                while time.time() < deadline and not blind.objects()["objects"]:
                    time.sleep(0.05)
                solo = blind.objects()["objects"]
                assert solo and solo[0]["id"] == 13, solo
            finally:
                blind.shutdown()

            print(
                f"  ok  mode test : {len(items)} tags relevés en pixels, "
                f"sans tags de coin (aperçu JPEG produit)"
            )
        finally:
            engine.shutdown()


# --------------------------------------------------------------------------- #
TESTS = [
    test_geometry_units,
    test_aruco_and_table,
    test_object_positions,
    test_roi_modes,
    test_tag_configuration,
    test_object_offset,
    test_calibration_intrinsics_roundtrip,
    test_intrinsics_resolution_mismatch,
    test_automatic_calibration,
    test_engine_and_api,
    test_shutdown_route,
    test_web_interface,
    test_match_mode,
    test_test_mode,
]


def main() -> int:
    print(f"OpenCV {cv2.__version__} — tests de fumée du mat de vision\n")
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  ÉCHEC {test.__name__} : {exc}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            import traceback

            print(f"  ERREUR {test.__name__} : {exc}")
            traceback.print_exc()
    print()
    if failures:
        print(f"{failures} test(s) en échec sur {len(TESTS)}")
        return 1
    print(f"{len(TESTS)} tests réussis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
