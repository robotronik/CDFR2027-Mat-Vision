#pragma once

/// Moteur de vision : une seule boucle possède la caméra.
///
/// Le moteur fonctionne en tâche de fond et expose un état thread-safe consommé
/// par l'API REST. Deux modes de traitement :
///
/// * `detect`    — pose caméra via les 4 tags de coin, puis relevé des objets ;
/// * `calibrate` — calibration automatique des intrinsèques (damier).
///
/// L'état `idle` (caméra ouverte, aucune analyse) subsiste en interne entre deux
/// commandes. Les positions sont calculées par changement de base : les
/// intrinsèques sont donc obligatoires.

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "matvision/aruco.hpp"
#include "matvision/calibration.hpp"
#include "matvision/camera.hpp"
#include "matvision/config.hpp"
#include "matvision/geometry.hpp"
#include "matvision/json.hpp"
#include "matvision/table.hpp"

namespace matvision {

enum class VisionMode { Idle, Detect, Calibrate };

std::string to_string(VisionMode mode);

class VisionEngine {
 public:
  /// `camera` permet d'injecter une source factice (tests) ; sinon une webcam
  /// est créée d'après la configuration.
  explicit VisionEngine(const Config& config, bool display = false,
                        std::unique_ptr<IFrameSource> camera = nullptr);
  ~VisionEngine();

  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  // -- cycle de vie ---------------------------------------------------
  void start();
  void shutdown();
  bool running() const { return thread_.joinable(); }

  // -- commandes ------------------------------------------------------
  nlohmann::json start_detection();
  nlohmann::json stop();
  nlohmann::json reset();
  nlohmann::json start_calibration();
  nlohmann::json cancel_calibration();
  nlohmann::json calibration_status();
  void reload_intrinsics();
  bool save_frame(const std::string& path);
  /// Met à jour l'adresse d'un robot dans la configuration effective (pour que
  /// `GET /config` reste cohérent avec les changements faits depuis l'interface).
  void update_robot_host(const std::string& key, const std::string& host, int port);

  // -- lectures d'état ------------------------------------------------
  nlohmann::json status();
  nlohmann::json objects(const std::string& label = "") ;
  nlohmann::json camera_position();
  nlohmann::json table_info();
  std::optional<std::vector<unsigned char>> preview_jpeg(int quality = -1);

  const Config& config() const { return config_; }

 private:
  void run();
  cv::Mat handle_frame(const cv::Mat& frame);
  cv::Mat detect(const cv::Mat& frame);
  nlohmann::json locate_object(const ObjectConfig& object, const Marker& corners,
                               const TableLocalization& localization);
  void log_localization(const TableLocalization& localization);
  void check_intrinsics_resolution(int width, int height);
  void finish_calibration();
  std::vector<std::string> hud_lines();
  const Intrinsics* intrinsics() const { return intrinsics_ ? &intrinsics_.value() : nullptr; }

  Config config_;
  bool display_ = false;

  std::unique_ptr<ArucoDetector> detector_;
  std::map<int, ObjectConfig> objects_config_;
  std::unique_ptr<IFrameSource> camera_;

  std::optional<Intrinsics> intrinsics_;
  std::unique_ptr<AutoCalibrator> calibrator_;

  mutable std::mutex mutex_;
  std::thread thread_;
  std::atomic<bool> stop_requested_{false};

  VisionMode mode_ = VisionMode::Idle;
  TableLocalization localization_;
  cv::Mat latest_frame_;
  std::vector<nlohmann::json> objects_;
  double started_at_ = 0.0;
  std::string error_;
  std::string intrinsics_warning_;
  bool frame_size_checked_ = false;
  bool last_table_ok_ = false;
  double last_table_fail_log_s_ = 0.0;
  bool shut_down_ = false;

  nlohmann::json stats_;
};

}  // namespace matvision
