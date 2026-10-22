#pragma once

/// Configuration du mat de vision.
///
/// La configuration vit dans un fichier JSON (`config/default.json`) ; chaque
/// champ possède une valeur par défaut, il n'est donc jamais obligatoire
/// d'écrire le fichier.

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace matvision {

/// Webcam USB (Logitech 4K Stream Edition sur LattePanda Delta).
struct CameraConfig {
  std::string device = "0";  ///< index (\"0\") ou chemin (/dev/video0, URL)
  std::string backend = "v4l2";
  int width = 1920;
  int height = 1080;
  int fps = 30;
  std::string fourcc = "MJPG";
  int buffer_size = 1;
  /// Images jetées à l'ouverture pour laisser l'exposition se stabiliser.
  int warmup_frames = 5;
  bool auto_exposure = true;
  bool auto_white_balance = true;

  bool device_is_index() const;
  int device_index() const;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_ONLY_SERIALIZE(CameraConfig, device, backend, width, height, fps,
                                                fourcc, buffer_size, warmup_frames,
                                                auto_exposure, auto_white_balance)
};

/// `device` accepte un index numérique ou un chemin.
void from_json(const nlohmann::json& data, CameraConfig& camera);

/// Tag de coin : position connue dans le repère table (mm).
struct CornerMarker {
  int id = 0;
  double x = 0.0;
  double y = 0.0;
  double a = 0.0;
  double size = 100.0;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(CornerMarker, id, x, y, a, size)
};

/// Géométrie de la table et tags de coin.
struct TableConfig {
  double width_mm = 2000.0;   ///< axe x
  double height_mm = 3000.0;  ///< axe y
  std::vector<CornerMarker> markers = {
      CornerMarker{20, -400.0, -900.0, 0.0, 100.0},
      CornerMarker{21, -400.0, 900.0, 0.0, 100.0},
      CornerMarker{22, 400.0, -900.0, 0.0, 100.0},
      CornerMarker{23, 400.0, 900.0, 0.0, 100.0},
  };
  std::string roi_mode = "table";  ///< "table" = tout le tapis, "markers" = quadrilatère des tags
  int roi_margin_px = 25;

  std::vector<int> marker_ids() const;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(TableConfig, width_mm, height_mm, markers, roi_mode,
                                              roi_margin_px)
};

/// Boîte d'un élément de jeu, décrite autour de son tag.
struct BoxConfig {
  double length_mm = 320.0;
  double width_mm = 110.0;
  double height_mm = 110.0;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(BoxConfig, length_mm, width_mm, height_mm)
};

/// Objet de la table identifié par un tag ArUco (robot ou élément de jeu).
struct ObjectConfig {
  int id = 0;
  std::string label;
  double angle_offset = 0.0;
  std::array<double, 2> offset{0.0, 0.0};
  double size = 100.0;
  std::optional<BoxConfig> box;
};

void to_json(nlohmann::json& data, const ObjectConfig& object);
void from_json(const nlohmann::json& data, ObjectConfig& object);

/// Réglages de la détection en continu.
struct DetectionConfig {
  bool draw = true;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(DetectionConfig, draw)
};

/// Calibration automatique des intrinsèques (damier).
struct CalibrationConfig {
  std::string intrinsics_file = "data/camera_calibration.json";
  int chessboard_cols = 7;
  int chessboard_rows = 7;
  double square_size_mm = 25.0;
  int target_frames = 20;
  double min_board_area_ratio = 0.04;
  double max_board_area_ratio = 0.95;
  double min_sharpness = 60.0;
  double min_move_ratio = 0.12;
  int border_margin_px = 10;
  double max_capture_seconds = 120.0;
  std::string flags = "CALIB_RATIONAL_MODEL";

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      CalibrationConfig, intrinsics_file, chessboard_cols, chessboard_rows, square_size_mm,
      target_frames, min_board_area_ratio, max_board_area_ratio, min_sharpness, min_move_ratio,
      border_margin_px, max_capture_seconds, flags)
};

/// Un robot pilotable par le mat via son API REST embarquée.
struct RobotConfig {
  std::string name;
  std::string host;
  int port = 80;
  bool enabled = true;
  std::optional<int> tag;

  std::string base_url() const;
};

/// Flotte du mat : robot principal, chasseur et essaim de petits robots.
struct RobotsConfig {
  std::string our_color;
  double request_timeout_s = 0.8;
  double path_window_s = 30.0;
  std::string map_image;
  std::optional<RobotConfig> main;
  std::optional<RobotConfig> hunter;
  std::vector<RobotConfig> swarm;

  std::vector<std::pair<std::string, const RobotConfig*>> targets() const;
};

void to_json(nlohmann::json& data, const RobotConfig& robot);
void from_json(const nlohmann::json& data, RobotConfig& robot);
void to_json(nlohmann::json& data, const RobotsConfig& robots);
void from_json(const nlohmann::json& data, RobotsConfig& robots);

struct ArucoConfig {
  std::string dictionary = "DICT_4X4_50";
  nlohmann::json params;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(ArucoConfig, dictionary, params)
};

/// Configuration complète du mat de vision.
struct Config {
  CameraConfig camera;
  ArucoConfig aruco;
  TableConfig table;
  DetectionConfig detection;
  CalibrationConfig calibration;
  RobotsConfig robots;
  std::vector<ObjectConfig> objects;
  std::string api_host = "0.0.0.0";
  int api_port = 5000;
  int preview_quality = 75;

  /// Chemin absolu du fichier de calibration intrinsèque.
  std::string intrinsics_path() const;
  /// Liste des avertissements / incohérences détectées.
  std::vector<std::string> validate() const;
  nlohmann::json to_json() const;
};

/// Applique une nouvelle adresse à un robot de la flotte, en mémoire.
///
/// :param key: `main`, `hunter` ou `swarm/<index>`.
/// :returns: `false` si la clé ne correspond à aucun robot.
bool apply_robot_host(RobotsConfig& robots, const std::string& key, const std::string& host,
                      int port);

/// Réécrit `robots.<key>.host/port` dans le fichier de configuration JSON.
///
/// Les autres clés du fichier sont préservées. Le fichier est créé s'il n'existe
/// pas encore. :returns: `false` (avec un message journalisé) en cas d'échec.
bool set_robot_host_in_config(const std::string& path, const std::string& key,
                              const std::string& host, int port);

/// Chemin de la configuration par défaut (défauts si le fichier est absent).
std::string default_config_path();
/// Charge la configuration depuis un fichier JSON (`path` vide = défaut).
Config load_config(const std::string& path = "");
/// Racine du projet (résolue à la compilation).
std::string project_root();
/// Répertoire des ressources de l'interface web.
std::string web_root();
/// Répertoire des données (calibration, captures).
std::string data_dir();

}  // namespace matvision
