#include "matvision/vision.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "matvision/logger.hpp"
#include "matvision/overlay.hpp"
#include "matvision/util.hpp"

namespace matvision {

namespace {

/// Intervalle (secondes) entre deux bilans de performance écrits sur la console.
constexpr double kPerfLogIntervalS = 5.0;

/// Intervalle minimal entre deux avertissements « repère non défini ».
constexpr double kTableWarnIntervalS = 5.0;

}  // namespace

std::string to_string(VisionMode mode) {
  switch (mode) {
    case VisionMode::Idle:
      return "idle";
    case VisionMode::Detect:
      return "detect";
    case VisionMode::Calibrate:
      return "calibrate";
  }
  return "idle";
}

nlohmann::json FrameStats::to_json() const {
  return {{"frames", frames},
          {"failed", failed},
          {"detections", detections},
          {"fps", fps},
          {"last_frame_ts", last_frame_ts},
          {"proc_ms", proc_ms},
          {"proc_ms_max", proc_ms_max},
          {"last_proc_ms", last_proc_ms},
          {"aruco_ms", aruco_ms}};
}

VisionEngine::VisionEngine(const Config& config, bool display,
                           std::unique_ptr<IFrameSource> camera)
    : config_(config), display_(display) {
  detector_ = std::make_unique<ArucoDetector>(config_.aruco.dictionary, config_.aruco.params);
  corner_ids_ = config_.table.marker_ids();
  for (const auto& object : config_.objects) {
    objects_config_[object.id] = object;
  }
  camera_ = camera ? std::move(camera) : std::make_unique<Camera>(config_.camera);

  intrinsics_ = load_intrinsics(config_.intrinsics_path());
  if (intrinsics_) {
    intrinsics_->source = config_.intrinsics_path();
    MV_LOGI("calibration intrinsèque chargée (" << config_.intrinsics_path() << ", "
                                                << intrinsics_->frames << " vues)");
  }
}

VisionEngine::~VisionEngine() { shutdown(); }

void VisionEngine::start() {
  if (thread_.joinable()) {
    return;
  }
  stop_requested_.store(false);
  started_at_ = steady_seconds();
  thread_ = std::thread([this]() { run(); });
  MV_LOGI("moteur de vision démarré");
}

void VisionEngine::shutdown() {
  if (shut_down_) {
    return;
  }
  shut_down_ = true;
  stop_requested_.store(true);
  if (thread_.joinable()) {
    thread_.join();
  }
  if (camera_) {
    camera_->close();
  }
  if (display_) {
    cv::destroyAllWindows();
  }
  MV_LOGI("moteur de vision arrêté");
}

nlohmann::json VisionEngine::start_detection() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    objects_.clear();
    stats_.detections = 0;
    mode_ = VisionMode::Detect;
  }
  MV_LOGI("détection démarrée");
  return {{"message", "détection démarrée"}, {"mode", "detect"}};
}

nlohmann::json VisionEngine::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = VisionMode::Idle;
  }
  MV_LOGI("détection arrêtée");
  return {{"message", "détection arrêtée"}, {"mode", "idle"}};
}

nlohmann::json VisionEngine::reset() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    objects_.clear();
    localization_ = TableLocalization();
    localization_.reason = "en attente";
    stats_.detections = 0;
    last_table_ok_ = false;
  }
  MV_LOGI("relevés réinitialisés");
  return {{"message", "relevés réinitialisés"}};
}

nlohmann::json VisionEngine::start_calibration() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    calibrator_ = std::make_unique<AutoCalibrator>(config_.calibration);
    calibrator_->start();
    mode_ = VisionMode::Calibrate;
  }
  MV_LOGI("calibration automatique démarrée");
  return {{"message", "calibration démarrée"}, {"mode", "calibrate"}};
}

nlohmann::json VisionEngine::cancel_calibration() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (calibrator_) {
      calibrator_->cancel();
    }
    mode_ = VisionMode::Idle;
  }
  MV_LOGI("calibration annulée");
  return {{"message", "calibration annulée"}};
}

nlohmann::json VisionEngine::calibration_status() {
  std::lock_guard<std::mutex> lock(mutex_);
  nlohmann::json payload;
  payload["config"] = {{"cols", config_.calibration.chessboard_cols},
                       {"rows", config_.calibration.chessboard_rows},
                       {"square_size_mm", config_.calibration.square_size_mm},
                       {"target_frames", config_.calibration.target_frames},
                       {"file", config_.intrinsics_path()}};
  payload["intrinsics"] = intrinsics_ ? intrinsics_->summary() : nlohmann::json(nullptr);
  payload.update(calibrator_ ? calibrator_->state().to_json()
                             : nlohmann::json{{"status", "idle"}});
  return payload;
}

void VisionEngine::reload_intrinsics() {
  auto loaded = load_intrinsics(config_.intrinsics_path());
  std::lock_guard<std::mutex> lock(mutex_);
  intrinsics_ = loaded;
}

bool VisionEngine::update_robot_host(const std::string& key, const std::string& host,
                                     int port) {
  std::lock_guard<std::mutex> lock(mutex_);
  return apply_robot_host(config_.robots, key, host, port);
}

nlohmann::json VisionEngine::config_json() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.to_json();
}

std::string VisionEngine::intrinsics_path() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.intrinsics_path();
}

RobotsConfig VisionEngine::robots_config() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.robots;
}

bool VisionEngine::save_frame(const std::string& path) {
  cv::Mat frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_frame_.empty()) {
      return false;
    }
    frame = latest_frame_.clone();
  }
  return cv::imwrite(path, frame);
}

nlohmann::json VisionEngine::status() {
  std::lock_guard<std::mutex> lock(mutex_);
  nlohmann::json data;
  data["running"] = running();
  data["mode"] = to_string(mode_);
  data["camera"] = camera_->info();
  data["error"] = error_;
  data["objects_count"] = objects_.size();
  data["table"] = localization_.to_json();
  data["intrinsics"] = intrinsics_ ? intrinsics_->summary() : nlohmann::json(nullptr);
  data["intrinsics_warning"] = intrinsics_warning_;
  data["stats"] = stats_.to_json();
  data["uptime_s"] = started_at_ > 0.0 ? round_to(steady_seconds() - started_at_, 1) : 0.0;
  return data;
}

nlohmann::json VisionEngine::objects(const std::string& label) {
  std::vector<nlohmann::json> items;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    items = objects_;
  }
  if (!label.empty()) {
    const std::string wanted = lowered(label);
    items.erase(std::remove_if(items.begin(), items.end(),
                               [&](const nlohmann::json& item) {
                                 return lowered(item.value("label", std::string())) != wanted;
                               }),
                items.end());
  }

  nlohmann::json by_label = nlohmann::json::object();
  for (const auto& item : items) {
    by_label[item.value("label", std::string())].push_back(item);
  }
  return {{"count", items.size()},
          {"objects", items},
          {"by_label", by_label},
          {"frame", "table"}};
}

nlohmann::json VisionEngine::camera_position() {
  std::lock_guard<std::mutex> lock(mutex_);
  nlohmann::json payload;
  payload["mode"] = to_string(mode_);
  payload["table_locked"] = localization_.ok;
  payload["used_ids"] = localization_.used_ids;
  if (localization_.has_camera_position) {
    payload["position"] = localization_.camera_position.to_json();
  } else {
    payload["position"] = nullptr;
    payload["message"] = localization_.reason.empty()
                             ? "pose caméra indisponible (calibration ?)"
                             : localization_.reason;
  }
  return payload;
}

nlohmann::json VisionEngine::table_info() {
  std::lock_guard<std::mutex> lock(mutex_);
  nlohmann::json markers = config_.table.markers;
  nlohmann::json objects = nlohmann::json::array();
  for (const auto& object : config_.objects) {
    nlohmann::json item = object;
    if (item["label"].get<std::string>().empty()) {
      item["label"] = std::to_string(object.id);
    }
    if (!item.contains("box")) {
      item["box"] = nullptr;
    }
    objects.push_back(item);
  }
  return {{"width_mm", config_.table.width_mm},
          {"height_mm", config_.table.height_mm},
          {"roi_mode", config_.table.roi_mode},
          {"roi_margin_px", config_.table.roi_margin_px},
          {"markers", markers},
          {"objects", objects},
          {"localization", localization_.to_json()}};
}

std::optional<std::vector<unsigned char>> VisionEngine::preview_jpeg(int quality) {
  cv::Mat preview;
  int effective_quality = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_frame_.empty()) {
      return std::nullopt;
    }
    // Réduction sous verrou : elle évite une copie pleine résolution et reste
    // bien plus courte que l'encodage JPEG.
    if (config_.preview_max_width > 0 && latest_frame_.cols > config_.preview_max_width) {
      const double scale =
          static_cast<double>(config_.preview_max_width) / latest_frame_.cols;
      cv::resize(latest_frame_, preview, cv::Size(), scale, scale, cv::INTER_LINEAR);
    } else {
      preview = latest_frame_.clone();
    }
    effective_quality = quality >= 0 ? quality : config_.preview_quality;
  }
  std::vector<unsigned char> buffer;
  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, effective_quality};
  if (!cv::imencode(".jpg", preview, buffer, params)) {
    return std::nullopt;
  }
  return buffer;
}

void VisionEngine::run() {
  int frames_window = 0;
  double window_start = steady_seconds();
  double last_perf_log = window_start;
  double proc_ms_sum = 0.0;
  double proc_ms_peak = 0.0;
  double aruco_ms_sum = 0.0;

  while (!stop_requested_.load()) {
    if (!camera_->is_open()) {
      try {
        camera_->open();
        std::lock_guard<std::mutex> lock(mutex_);
        error_.clear();
        MV_LOGI("caméra ouverte");
      } catch (const CameraError& exc) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          error_ = exc.what();
        }
        MV_LOGW("caméra indisponible : " << exc.what());
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        continue;
      }
    }

    cv::Mat frame;
    bool ok = false;
    try {
      ok = camera_->read_latest(frame, config_.camera.drain_frames);
    } catch (const CameraError& exc) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = exc.what();
      stats_.failed += 1;
      continue;
    }
    if (!ok || frame.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.failed += 1;
      continue;
    }

    if (!frame_size_checked_) {
      frame_size_checked_ = true;
      MV_LOGI("première image reçue : " << frame.cols << 'x' << frame.rows);
      check_intrinsics_resolution(frame.cols, frame.rows);
    }

    const double proc_started = steady_seconds();
    cv::Mat annotated = frame;
    try {
      annotated = handle_frame(frame);
    } catch (const std::exception& exc) {
      MV_LOGE("erreur pendant le traitement de l'image : " << exc.what());
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = "erreur de traitement (voir les logs)";
    }
    const double proc_ms = (steady_seconds() - proc_started) * 1000.0;
    proc_ms_sum += proc_ms;
    aruco_ms_sum += last_aruco_ms_;
    last_aruco_ms_ = 0.0;
    proc_ms_peak = std::max(proc_ms_peak, proc_ms);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_frame_ = annotated;
      stats_.frames += 1;
      stats_.last_frame_ts = wall_seconds();
      stats_.last_proc_ms = round_to(proc_ms, 2);
    }

    ++frames_window;
    const double now = steady_seconds();
    const double elapsed = now - window_start;
    if (elapsed >= 1.0) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.fps = round_to(frames_window / elapsed, 1);
        stats_.proc_ms = round_to(proc_ms_sum / frames_window, 2);
        stats_.proc_ms_max = round_to(proc_ms_peak, 2);
        stats_.aruco_ms = round_to(aruco_ms_sum / frames_window, 2);
      }
      if (now - last_perf_log >= kPerfLogIntervalS) {
        std::lock_guard<std::mutex> lock(mutex_);
        MV_LOGI("perf: " << stats_.fps << " fps | traitement " << stats_.proc_ms
                         << " ms/image (max " << stats_.proc_ms_max << " ms, arUco "
                         << stats_.aruco_ms << " ms) | mode=" << to_string(mode_)
                         << " | objets=" << objects_.size());
        last_perf_log = now;
        proc_ms_peak = 0.0;
      }
      frames_window = 0;
      proc_ms_sum = 0.0;
      aruco_ms_sum = 0.0;
      window_start = now;
    }

    if (display_) {
      cv::imshow("Mat de vision", annotated);
      if ((cv::waitKey(1) & 0xFF) == 'q') {
        std::lock_guard<std::mutex> lock(mutex_);
        mode_ = VisionMode::Idle;
      }
    }
  }

  if (display_) {
    cv::destroyAllWindows();
  }
}

cv::Mat VisionEngine::handle_frame(cv::Mat frame) {
  VisionMode mode;
  AutoCalibrator* calibrator = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode = mode_;
    calibrator = calibrator_.get();
  }

  if (mode == VisionMode::Calibrate && calibrator != nullptr) {
    calibrator->process(frame);
    if (calibrator->finished()) {
      finish_calibration();
    }
    if (config_.detection.draw) {
      return calibrator->draw_overlay(frame);
    }
    return frame;
  }

  if (mode == VisionMode::Detect) {
    return detect(frame);
  }

  if (config_.detection.draw) {
    draw_hud(frame, hud_lines());
  }
  return frame;
}

cv::Mat VisionEngine::detect(cv::Mat frame) {
  // `frame` est un tampon privé (copie faite par la caméra) : l'annotation est
  // faite en place, sans clone.
  cv::cvtColor(frame, gray_, cv::COLOR_BGR2GRAY);

  const double detect_started = steady_seconds();
  detected_ids_.clear();
  detected_corners_.clear();
  detector_->detect(gray_, detected_ids_, detected_corners_);
  last_aruco_ms_ = (steady_seconds() - detect_started) * 1000.0;

  MarkerMap corner_markers;
  for (std::size_t i = 0; i < detected_ids_.size(); ++i) {
    if (std::find(corner_ids_.begin(), corner_ids_.end(), detected_ids_[i]) !=
        corner_ids_.end()) {
      corner_markers.emplace(detected_ids_[i], detected_corners_[i]);
    }
  }

  std::optional<Intrinsics> intrinsics_copy;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (intrinsics_) {
      intrinsics_copy = *intrinsics_;
    }
  }
  const Intrinsics* intr = intrinsics_copy ? &*intrinsics_copy : nullptr;

  const TableLocalization localization =
      localize_table(corner_markers, config_.table, intr);
  log_localization(localization);

  std::vector<nlohmann::json> objects;
  std::vector<ObjectAnnotation> annotations;
  if (localization.ok && intr != nullptr) {
    // La ROI est un simple polygone : les détections hors zone sont écartées
    // après coup, ce qui évite une seconde passe complète du détecteur ArUco.
    std::vector<std::vector<cv::Point2f>> groups;
    groups.reserve(corner_markers.size() + 1);
    for (const auto& [id, corners] : corner_markers) {
      groups.push_back(corners);
    }
    if (config_.table.roi_mode == "table") {
      const auto quad = table_polygon_image(localization, config_.table, *intr);
      if (!quad.empty()) {
        groups.push_back(quad);
      }
    }
    const std::vector<cv::Point2f> roi = build_roi_polygon(groups);

    // La détection brute conserve les doublons : un même identifiant peut
    // apparaître plusieurs fois (ex. plusieurs éléments de jeu taggés 13).
    annotations.reserve(detected_ids_.size());
    for (std::size_t i = 0; i < detected_ids_.size(); ++i) {
      if (!point_in_roi(roi, marker_center(detected_corners_[i]), config_.table.roi_margin_px)) {
        continue;
      }
      ObjectAnnotation annotation;
      annotation.id = detected_ids_[i];
      annotation.corners = detected_corners_[i];
      const auto it = objects_config_.find(annotation.id);
      if (it != objects_config_.end()) {
        nlohmann::json item =
            locate_object(it->second, annotation.corners, localization, intr);
        if (!item.is_null()) {
          objects.push_back(item);
          annotation.object = item;
        }
      }
      annotations.push_back(std::move(annotation));
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    localization_ = localization;
    objects_ = objects;
    stats_.detections = static_cast<int>(objects.size());
  }

  if (config_.detection.draw) {
    annotate(frame, corner_markers, annotations, localization);
    draw_hud(frame, hud_lines());
  }
  return frame;
}

nlohmann::json VisionEngine::locate_object(const ObjectConfig& object, const Marker& corners,
                                           const TableLocalization& localization,
                                           const Intrinsics* intrinsics) {
  if (intrinsics == nullptr) {
    return nullptr;
  }
  const auto pose = solve_tag_pose(corners, object.size, *intrinsics);
  if (!pose.has_value()) {
    return nullptr;
  }

  Position position = localization.tag_position(pose->first, pose->second, object.angle_offset);
  double x = position.x;
  double y = position.y;
  if (object.offset[0] != 0.0 || object.offset[1] != 0.0) {
    const double radians = position.a * kPi / 180.0;
    x += object.offset[0] * std::cos(radians) - object.offset[1] * std::sin(radians);
    y += object.offset[0] * std::sin(radians) + object.offset[1] * std::cos(radians);
  }
  position.x = x;
  position.y = y;

  nlohmann::json item = position.to_json();
  item["id"] = object.id;
  item["label"] = object.label.empty() ? std::to_string(object.id) : object.label;
  return item;
}

void VisionEngine::log_localization(const TableLocalization& localization) {
  if (localization.ok) {
    if (!last_table_ok_) {
      MV_LOGI("repère table acquis (tags=" << localization.used_ids.size() << ", résidu "
                                           << localization.residual_px << " px)");
    }
    last_table_ok_ = true;
    return;
  }

  // Tant que le repère n'est pas défini, on répète la cause (au plus une fois
  // toutes les 5 s) avec le nombre de tags de coin vus, pour diagnostiquer.
  const double now = steady_seconds();
  if (last_table_ok_ || now - last_table_fail_log_s_ >= kTableWarnIntervalS) {
    MV_LOGW("repère table non défini : "
            << (localization.reason.empty() ? "raison inconnue" : localization.reason)
            << " (tags de coin vus " << localization.used_ids.size() << '/'
            << config_.table.markers.size() << ')');
    last_table_fail_log_s_ = now;
  }
  last_table_ok_ = false;
}

void VisionEngine::check_intrinsics_resolution(int width, int height) {
  std::optional<Intrinsics> intrinsics_copy;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (intrinsics_) {
      intrinsics_copy = *intrinsics_;
    }
  }
  if (!intrinsics_copy || intrinsics_copy->image_size.width == 0) {
    return;
  }
  if (intrinsics_copy->image_size.width == width && intrinsics_copy->image_size.height == height) {
    return;
  }
  std::ostringstream oss;
  oss << "calibration faite en " << intrinsics_copy->image_size.width << 'x'
      << intrinsics_copy->image_size.height
      << " mais la caméra fournit du " << width << 'x' << height
      << " : les positions seront fausses. Relancez « matvision-calibrate » à cette résolution.";
  MV_LOGW(oss.str());
  std::lock_guard<std::mutex> lock(mutex_);
  intrinsics_warning_ = oss.str();
}

void VisionEngine::finish_calibration() {
  std::shared_ptr<Intrinsics> intrinsics;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (calibrator_) {
      intrinsics = calibrator_->state().intrinsics;
    }
    mode_ = VisionMode::Idle;
  }
  if (!intrinsics) {
    MV_LOGE("calibration non aboutie");
    return;
  }
  intrinsics->source = config_.intrinsics_path();
  if (!save_intrinsics(config_.intrinsics_path(), *intrinsics)) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  intrinsics_ = *intrinsics;
  intrinsics_warning_.clear();
  MV_LOGI("calibration enregistrée dans " << config_.intrinsics_path());
}

std::vector<std::string> VisionEngine::hud_lines() {
  VisionMode mode;
  FrameStats stats;
  TableLocalization localization;
  std::string error;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode = mode_;
    stats = stats_;
    localization = localization_;
    error = error_;
  }

  std::ostringstream line1;
  line1 << "mode: " << to_string(mode) << "   fps: " << stats.fps
        << "   frames: " << stats.frames;
  std::ostringstream line2;
  line2 << "objets: " << stats.detections << "   résidu: "
        << static_cast<int>(localization.residual_px) << " px   tags: "
        << localization.used_ids.size();
  std::vector<std::string> lines{line1.str(), line2.str()};
  if (localization.has_camera_position) {
    std::ostringstream oss;
    oss << "caméra: x=" << static_cast<int>(localization.camera_position.x)
        << " y=" << static_cast<int>(localization.camera_position.y)
        << " z=" << static_cast<int>(localization.camera_position.z)
        << " a=" << static_cast<int>(localization.camera_position.a);
    lines.push_back(oss.str());
  }
  if (!error.empty()) {
    lines.push_back("erreur: " + error);
  }
  return lines;
}

}  // namespace matvision
