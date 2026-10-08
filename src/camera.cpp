#include "matvision/camera.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <glob.h>

#include "matvision/logger.hpp"

namespace matvision {

namespace {

/// Code FOURCC OpenCV à partir d'une chaîne de 4 caractères.
int fourcc_to_code(const std::string& fourcc) {
  std::string code = fourcc;
  std::transform(code.begin(), code.end(), code.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (code.size() != 4) {
    return 0;
  }
  return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
}

int backend_to_flag(const std::string& backend) {
  std::string lowered = backend;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "v4l2") {
    return cv::CAP_V4L2;
  }
  if (lowered == "dshow") {
    return cv::CAP_DSHOW;
  }
  return cv::CAP_ANY;
}

}  // namespace

Camera::Camera(const CameraConfig& config) : config_(config) {}

bool Camera::open() {
  close();
  const int backend = backend_to_flag(config_.backend);
  const bool by_index = config_.device_is_index();

  MV_LOGI("ouverture de la caméra " << config_.device << " (backend=" << config_.backend
                                    << ")");
  if (by_index) {
    capture_.open(config_.device_index(), backend);
  } else {
    capture_.open(config_.device, backend);
  }
  if (!capture_.isOpened()) {
    std::string devices;
    for (const auto& device : list_video_devices()) {
      devices += (devices.empty() ? "" : ", ") + device;
    }
    throw CameraError("impossible d'ouvrir la caméra '" + config_.device +
                      "' (périphériques visibles : " + (devices.empty() ? "aucun" : devices) +
                      ")");
  }

  if (config_.buffer_size > 0) {
    capture_.set(cv::CAP_PROP_BUFFERSIZE, config_.buffer_size);
  }
  if (!config_.fourcc.empty()) {
    capture_.set(cv::CAP_PROP_FOURCC, fourcc_to_code(config_.fourcc));
  }
  capture_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
  capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
  if (config_.fps > 0) {
    capture_.set(cv::CAP_PROP_FPS, config_.fps);
  }

  read_actual_settings();

  for (int i = 0; i < std::max(0, config_.warmup_frames); ++i) {
    capture_.grab();
  }

  MV_LOGI("caméra prête : " << size_.width << 'x' << size_.height << " @ " << fps_ << " fps ("
                            << (fourcc_.empty() ? "?" : fourcc_) << ")");
  return true;
}

bool Camera::read(cv::Mat& frame) {
  if (!capture_.isOpened()) {
    throw CameraError("caméra non ouverte");
  }
  return capture_.read(frame) && !frame.empty();
}

bool Camera::read_latest(cv::Mat& frame, int drain) {
  if (!capture_.isOpened()) {
    throw CameraError("caméra non ouverte");
  }
  for (int i = 0; i < std::max(0, drain); ++i) {
    capture_.grab();
  }
  return read(frame);
}

void Camera::close() {
  if (capture_.isOpened()) {
    capture_.release();
    MV_LOGD("caméra fermée");
  }
}

bool Camera::is_open() const { return capture_.isOpened(); }

nlohmann::json Camera::info() const {
  return {{"device", config_.device},
          {"opened", is_open()},
          {"width", size_.width},
          {"height", size_.height},
          {"fourcc", fourcc_},
          {"fps", std::round(fps_ * 100.0) / 100.0},
          {"requested",
           {{"width", config_.width},
            {"height", config_.height},
            {"fps", config_.fps},
            {"fourcc", config_.fourcc}}}};
}

void Camera::read_actual_settings() {
  size_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
  size_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));

  const int code = static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC));
  if (code != 0) {
    fourcc_.clear();
    for (int i = 0; i < 4; ++i) {
      fourcc_.push_back(static_cast<char>((code >> (8 * i)) & 0xFF));
    }
  } else {
    fourcc_.clear();
  }

  fps_ = capture_.get(cv::CAP_PROP_FPS);
  if (!(fps_ >= 0.0)) {
    fps_ = 0.0;
  }

  if (!config_.auto_exposure) {
    capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
  }
  if (!config_.auto_white_balance) {
    capture_.set(cv::CAP_PROP_AUTO_WB, 0);
  }

  MV_LOGD("réglages négociés : " << size_.width << 'x' << size_.height << " @ " << fps_
                                 << " fps (" << (fourcc_.empty() ? "?" : fourcc_) << ")");
}

std::vector<std::string> list_video_devices() {
  std::vector<std::string> devices;
  glob_t results{};
  if (glob("/dev/video*", 0, nullptr, &results) == 0) {
    for (std::size_t i = 0; i < results.gl_pathc; ++i) {
      devices.emplace_back(results.gl_pathv[i]);
    }
  }
  globfree(&results);
  std::sort(devices.begin(), devices.end());
  return devices;
}

nlohmann::json probe(const std::string& device, double /*timeout_s*/) {
  CameraConfig config;
  config.device = device;
  config.warmup_frames = 3;
  Camera camera(config);
  const auto started = std::chrono::steady_clock::now();
  try {
    camera.open();
    cv::Mat frame;
    const bool ok = camera.read_latest(frame, 2);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    nlohmann::json rows = nullptr;
    if (!frame.empty()) {
      rows = {frame.rows, frame.cols, frame.channels()};
    }
    return {{"ok", ok && !frame.empty()},
            {"info", camera.info()},
            {"elapsed_s", std::round(elapsed * 1000.0) / 1000.0},
            {"frame_shape", rows}};
  } catch (const CameraError& exc) {
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return {{"ok", false},
            {"error", exc.what()},
            {"elapsed_s", std::round(elapsed * 1000.0) / 1000.0}};
  }
}

}  // namespace matvision
