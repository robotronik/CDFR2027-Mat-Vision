#include "matvision/vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "matvision/logger.hpp"
#include "matvision/overlay.hpp"

namespace matvision {

namespace {

/// Intervalle (secondes) entre deux bilans de performance écrits sur la console.
constexpr double kPerfLogIntervalS = 5.0;

double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

nlohmann::json default_stats() {
  return {{"frames", 0},      {"failed", 0},      {"detections", 0},
          {"fps", 0.0},       {"last_frame_ts", 0.0},
          {"proc_ms", 0.0},   {"proc_ms_max", 0.0}, {"last_proc_ms", 0.0}};
}

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

VisionEngine::VisionEngine(const Config& config, bool display,
                           std::unique_ptr<IFrameSource> camera)
    : config_(config), display_(display), stats_(default_stats()) {
  detector_ = std::make_unique<ArucoDetector>(config_.aruco.dictionary, config_.aruco.params);
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
    stats_["detections"] = 0;
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
    stats_["detections"] = 0;
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
  data["stats"] = stats_;
  data["uptime_s"] = started_at_ > 0.0
                         ? std::round((steady_seconds() - started_at_) * 10.0) / 10.0
                         : 0.0;
  return data;
}

nlohmann::json VisionEngine::objects(const std::string& label) {
  std::vector<nlohmann::json> items;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    items = objects_;
  }
  if (!label.empty()) {
    std::string lowered = label;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    items.erase(std::remove_if(items.begin(), items.end(),
                               [&](const nlohmann::json& item) {
                                 std::string value =
                                     item.value("label", std::string());
                                 std::transform(
                                     value.begin(), value.end(), value.begin(),
                                     [](unsigned char c) {
                                       return static_cast<char>(std::tolower(c));
                                     });
                                 return value != lowered;
                               }),
                items.end());
  }

  nlohmann::json by_label = nlohmann::json::object();
  for (const auto& item : items) {
    by_label[item.value("label", std::string())].push_back(item);
  }
  nlohmann::json array = nlohmann::json::array();
  for (const auto& item : items) {
    array.push_back(item);
  }
  return {{"count", items.size()},
          {"objects", array},
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
  nlohmann::json markers = nlohmann::json::array();
  for (const auto& marker : config_.table.markers) {
    markers.push_back({{"id", marker.id},
                       {"x", marker.x},
                       {"y", marker.y},
                       {"a", marker.a},
                       {"size", marker.size}});
  }
  nlohmann::json objects = nlohmann::json::array();
  for (const auto& object : config_.objects) {
    nlohmann::json item = {{"id", object.id},
                           {"label", object.label.empty() ? std::to_string(object.id)
                                                          : object.label},
                           {"angle_offset", object.angle_offset},
                           {"offset", {object.offset[0], object.offset[1]}},
                           {"size", object.size}};
    if (object.box.has_value()) {
      item["box"] = {{"length_mm", object.box->length_mm},
                     {"width_mm", object.box->width_mm},
                     {"height_mm", object.box->height_mm}};
    } else {
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
  cv::Mat frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_frame_.empty()) {
      return std::nullopt;
    }
    frame = latest_frame_.clone();
  }
  const int effective = quality >= 0 ? quality : config_.preview_quality;
  std::vector<unsigned char> buffer;
  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, effective};
  if (!cv::imencode(".jpg", frame, buffer, params)) {
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
      ok = camera_->read_latest(frame, 2);
    } catch (const CameraError& exc) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = exc.what();
      stats_["failed"] = stats_["failed"].get<int>() + 1;
      continue;
    }
    if (!ok || frame.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_["failed"] = stats_["failed"].get<int>() + 1;
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
    proc_ms_peak = std::max(proc_ms_peak, proc_ms);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_frame_ = annotated;
      stats_["frames"] = stats_["frames"].get<int>() + 1;
      stats_["last_frame_ts"] = wall_seconds();
      stats_["last_proc_ms"] = std::round(proc_ms * 100.0) / 100.0;
    }

    ++frames_window;
    const double now = steady_seconds();
    const double elapsed = now - window_start;
    if (elapsed >= 1.0) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_["fps"] = std::round(frames_window / elapsed * 10.0) / 10.0;
        stats_["proc_ms"] = std::round(proc_ms_sum / frames_window * 100.0) / 100.0;
        stats_["proc_ms_max"] = std::round(proc_ms_peak * 100.0) / 100.0;
      }
      if (now - last_perf_log >= kPerfLogIntervalS) {
        std::lock_guard<std::mutex> lock(mutex_);
        MV_LOGI("perf: " << stats_["fps"] << " fps | traitement " << stats_["proc_ms"]
                         << " ms/image (max " << stats_["proc_ms_max"] << " ms) | mode="
                         << to_string(mode_) << " | objets=" << objects_.size());
        last_perf_log = now;
        proc_ms_peak = 0.0;
      }
      frames_window = 0;
      proc_ms_sum = 0.0;
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

cv::Mat VisionEngine::handle_frame(const cv::Mat& frame) {
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
    cv::Mat canvas = frame.clone();
    draw_hud(canvas, hud_lines());
    return canvas;
  }
  return frame;
}

cv::Mat VisionEngine::detect(const cv::Mat& frame) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  const MarkerMap found = detector_->detect_by_id(gray, cv::Mat(), true);
  const std::vector<int> corner_ids = config_.table.marker_ids();
  MarkerMap corner_markers;
  for (const auto& [id, corners] : found) {
    if (std::find(corner_ids.begin(), corner_ids.end(), id) != corner_ids.end()) {
      corner_markers[id] = corners;
    }
  }

  const Intrinsics* intr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    intr = intrinsics();
  }

  const TableLocalization localization =
      localize_table(corner_markers, config_.table, intr);
  log_localization(localization);

  std::vector<nlohmann::json> objects;
  MarkerMap object_markers;
  if (localization.ok && intr != nullptr) {
    std::vector<std::vector<cv::Point2f>> groups;
    for (const auto& [id, corners] : corner_markers) {
      groups.push_back(corners);
    }
    if (config_.table.roi_mode == "table") {
      const auto quad = table_polygon_image(localization, config_.table, *intr);
      if (!quad.empty()) {
        groups.push_back(quad);
      }
    }
    const cv::Mat mask =
        build_roi_mask(gray.size(), groups, config_.table.roi_margin_px);
    object_markers = mask.empty() ? found : detector_->detect_by_id(gray, mask, true);

    for (const auto& [id, corners] : object_markers) {
      const auto it = objects_config_.find(id);
      if (it != objects_config_.end()) {
        nlohmann::json item = locate_object(it->second, corners, localization);
        if (!item.is_null()) {
          objects.push_back(item);
        }
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    localization_ = localization;
    objects_ = objects;
    stats_["detections"] = static_cast<int>(objects.size());
  }

  if (config_.detection.draw) {
    cv::Mat canvas = frame.clone();
    annotate(canvas, corner_markers, object_markers, localization, objects);
    draw_hud(canvas, hud_lines());
    return canvas;
  }
  return frame;
}

nlohmann::json VisionEngine::locate_object(const ObjectConfig& object, const Marker& corners,
                                           const TableLocalization& localization) {
  const Intrinsics* intr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    intr = intrinsics();
  }
  if (intr == nullptr) {
    return nullptr;
  }
  const auto pose = solve_tag_pose(corners, object.size, *intr);
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
  if (localization.ok && !last_table_ok_) {
    MV_LOGI("repère table acquis (tags=" << localization.used_ids.size() << ", résidu "
                                         << localization.residual_px << " px)");
  } else if (!localization.ok && last_table_ok_) {
    MV_LOGW("repère table perdu : "
            << (localization.reason.empty() ? "raison inconnue" : localization.reason));
  }
  last_table_ok_ = localization.ok;
}

void VisionEngine::check_intrinsics_resolution(int width, int height) {
  const Intrinsics* intr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    intr = intrinsics();
  }
  if (intr == nullptr || intr->image_size.width == 0) {
    return;
  }
  if (intr->image_size.width == width && intr->image_size.height == height) {
    return;
  }
  std::ostringstream oss;
  oss << "calibration faite en " << intr->image_size.width << 'x' << intr->image_size.height
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
  nlohmann::json stats;
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
  line1 << "mode: " << to_string(mode) << "   fps: " << stats.value("fps", 0.0)
        << "   frames: " << stats.value("frames", 0);
  std::ostringstream line2;
  line2 << "objets: " << stats.value("detections", 0) << "   résidu: "
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
