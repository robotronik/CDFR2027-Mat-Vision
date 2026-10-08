#include "matvision/calibration.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "matvision/geometry.hpp"
#include "matvision/logger.hpp"

namespace matvision {

namespace {

constexpr int kCalibCbFlags =
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;

double wall_seconds() {
  const auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

std::string iso_now() {
  const std::time_t seconds = std::time(nullptr);
  std::tm parts{};
#if defined(_WIN32)
  localtime_s(&parts, &seconds);
#else
  localtime_r(&seconds, &parts);
#endif
  std::ostringstream oss;
  oss << std::put_time(&parts, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}

cv::Mat json_flat_to_mat(const nlohmann::json& data) {
  std::vector<double> values;
  for (const auto& item : data) {
    values.push_back(item.get<double>());
  }
  cv::Mat mat(static_cast<int>(values.size()), 1, CV_64F);
  for (std::size_t i = 0; i < values.size(); ++i) {
    mat.at<double>(static_cast<int>(i), 0) = values[i];
  }
  return mat;
}

cv::Mat json_matrix_to_mat(const nlohmann::json& data) {
  const int rows = static_cast<int>(data.size());
  const int cols = rows > 0 ? static_cast<int>(data[0].size()) : 0;
  cv::Mat mat = cv::Mat::zeros(rows, cols, CV_64F);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      mat.at<double>(r, c) = data[r][c].get<double>();
    }
  }
  return mat;
}

nlohmann::json mat_to_json(const cv::Mat& mat) {
  nlohmann::json data = nlohmann::json::array();
  for (int r = 0; r < mat.rows; ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (int c = 0; c < mat.cols; ++c) {
      row.push_back(mat.at<double>(r, c));
    }
    data.push_back(row);
  }
  return data;
}

int flag_from_name(const std::string& name) {
  static const std::pair<const char*, int> kFlags[] = {
      {"CALIB_USE_INTRINSIC_GUESS", cv::CALIB_USE_INTRINSIC_GUESS},
      {"CALIB_FIX_ASPECT_RATIO", cv::CALIB_FIX_ASPECT_RATIO},
      {"CALIB_FIX_PRINCIPAL_POINT", cv::CALIB_FIX_PRINCIPAL_POINT},
      {"CALIB_ZERO_TANGENT_DIST", cv::CALIB_ZERO_TANGENT_DIST},
      {"CALIB_FIX_FOCAL_LENGTH", cv::CALIB_FIX_FOCAL_LENGTH},
      {"CALIB_FIX_K1", cv::CALIB_FIX_K1},
      {"CALIB_FIX_K2", cv::CALIB_FIX_K2},
      {"CALIB_FIX_K3", cv::CALIB_FIX_K3},
      {"CALIB_FIX_K4", cv::CALIB_FIX_K4},
      {"CALIB_FIX_K5", cv::CALIB_FIX_K5},
      {"CALIB_FIX_K6", cv::CALIB_FIX_K6},
      {"CALIB_RATIONAL_MODEL", cv::CALIB_RATIONAL_MODEL},
      {"CALIB_THIN_PRISM_MODEL", cv::CALIB_THIN_PRISM_MODEL},
      {"CALIB_FIX_S1_S2_S3_S4", cv::CALIB_FIX_S1_S2_S3_S4},
      {"CALIB_TILTED_MODEL", cv::CALIB_TILTED_MODEL},
      {"CALIB_FIX_TAUX_TAUY", cv::CALIB_FIX_TAUX_TAUY},
  };
  for (const auto& [key, value] : kFlags) {
    if (name == key) {
      return value;
    }
  }
  return 0;
}

double mean_reprojection_error(const cv::Mat& matrix, const cv::Mat& dist,
                               const std::vector<cv::Mat>& rvecs,
                               const std::vector<cv::Mat>& tvecs,
                               const std::vector<cv::Mat>& object_points,
                               const std::vector<cv::Mat>& image_points) {
  double total = 0.0;
  for (std::size_t i = 0; i < object_points.size(); ++i) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points[i], rvecs[i], tvecs[i], matrix, dist, projected);
    cv::Mat observed = image_points[i];
  double sum = 0.0;
    for (int j = 0; j < observed.rows; ++j) {
      const cv::Point2f point = observed.at<cv::Point2f>(j, 0);
      sum += std::hypot(projected[static_cast<std::size_t>(j)].x - point.x,
                        projected[static_cast<std::size_t>(j)].y - point.y);
    }
    total += sum / std::max(1, observed.rows);
  }
  return total / std::max<std::size_t>(1, object_points.size());
}

}  // namespace

double Intrinsics::fx() const { return camera_matrix.empty() ? 0.0 : camera_matrix.at<double>(0, 0); }

double Intrinsics::fy() const { return camera_matrix.empty() ? 0.0 : camera_matrix.at<double>(1, 1); }

nlohmann::json Intrinsics::to_json() const {
  nlohmann::json data;
  data["image_size"] = {image_size.width, image_size.height};
  data["camera_matrix"] = mat_to_json(camera_matrix);
  data["dist_coeffs"] = mat_to_json(dist_coeffs.reshape(1, 1))[0];
  data["rms"] = rms;
  data["frames"] = frames;
  data["calibrated_at"] = calibrated_at;
  data["source"] = source;
  return data;
}

nlohmann::json Intrinsics::summary() const {
  nlohmann::json info = to_json();
  const double fx = this->fx();
  const double fy = this->fy();
  info["fov_x_deg"] =
      fx != 0.0 ? nlohmann::json(std::round(2.0 * std::atan(image_size.width / (2.0 * fx)) *
                                            180.0 / kPi * 100.0) /
                                 100.0)
                : nlohmann::json(nullptr);
  info["fov_y_deg"] =
      fy != 0.0 ? nlohmann::json(std::round(2.0 * std::atan(image_size.height / (2.0 * fy)) *
                                            180.0 / kPi * 100.0) /
                                 100.0)
                : nlohmann::json(nullptr);
  return info;
}

std::optional<Intrinsics> Intrinsics::from_json(const nlohmann::json& data,
                                                const std::string& source) {
  if (!data.contains("camera_matrix") || !data.contains("dist_coeffs")) {
    return std::nullopt;
  }
  Intrinsics intrinsics;
  intrinsics.camera_matrix = json_matrix_to_mat(data["camera_matrix"]);
  intrinsics.dist_coeffs = json_flat_to_mat(data["dist_coeffs"]);
  const auto size = data.value("image_size", std::vector<int>{0, 0});
  if (size.size() >= 2) {
    intrinsics.image_size = cv::Size(size[0], size[1]);
  }
  intrinsics.rms = data.value("rms", 0.0);
  intrinsics.frames = data.value("frames", 0);
  intrinsics.calibrated_at = data.value("calibrated_at", std::string());
  intrinsics.source = source;
  return intrinsics;
}

std::string calibration_file_kind(const std::string& path) {
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) {
    return "inconnu";
  }
  std::string ext = path.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (ext == ".json") {
    return "json";
  }
  if (ext == ".yml" || ext == ".yaml") {
    return "yaml";
  }
  return "inconnu";
}

bool save_intrinsics(const std::string& path, const Intrinsics& intrinsics) {
  try {
    const std::filesystem::path target(path);
    if (target.has_parent_path()) {
      std::filesystem::create_directories(target.parent_path());
    }
    nlohmann::json payload = intrinsics.to_json();
    payload["format"] = "matvision.intrinsics/1";
    std::ofstream output(path);
    if (!output) {
      return false;
    }
    output << payload.dump(2) << '\n';
    return true;
  } catch (const std::exception& exc) {
    MV_LOGE("écriture de la calibration impossible (" << path << ") : " << exc.what());
    return false;
  }
}

std::optional<Intrinsics> load_intrinsics(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    return std::nullopt;
  }
  try {
    if (calibration_file_kind(path) == "yaml") {
      cv::FileStorage storage(path, cv::FileStorage::READ);
      if (!storage.isOpened()) {
        return std::nullopt;
      }
      cv::Mat matrix;
      cv::Mat dist;
      storage["camera_matrix"] >> matrix;
      storage["dist_coeffs"] >> dist;
      storage.release();
      if (matrix.empty() || dist.empty()) {
        return std::nullopt;
      }
      Intrinsics intrinsics;
      intrinsics.camera_matrix = matrix.clone();
      intrinsics.camera_matrix.convertTo(intrinsics.camera_matrix, CV_64F);
      intrinsics.dist_coeffs = dist.reshape(1).clone();
      intrinsics.dist_coeffs.convertTo(intrinsics.dist_coeffs, CV_64F);
      intrinsics.source = path;
      return intrinsics;
    }

    std::ifstream input(path);
    if (!input) {
      return std::nullopt;
    }
    nlohmann::json data;
    input >> data;
    return Intrinsics::from_json(data, path);
  } catch (const std::exception& exc) {
    MV_LOGE("calibration illisible (" << path << ") : " << exc.what());
    return std::nullopt;
  }
}

nlohmann::json CalibrationState::to_json() const {
  const double elapsed =
      started_at > 0.0 ? (finished_at > 0.0 ? finished_at : wall_seconds()) - started_at : 0.0;
  return {{"status", status},
          {"found", found},
          {"captured", captured},
          {"target", target},
          {"attempts", attempts},
          {"sharpness", std::round(sharpness * 10.0) / 10.0},
          {"area_ratio", std::round(area_ratio * 10000.0) / 10000.0},
          {"movement", std::round(movement * 1000.0) / 1000.0},
          {"message", message},
          {"rms", rms.has_value() ? nlohmann::json(std::round(rms.value() * 10000.0) / 10000.0)
                                  : nlohmann::json(nullptr)},
          {"elapsed_s", std::round(elapsed * 10.0) / 10.0}};
}

AutoCalibrator::AutoCalibrator(const CalibrationConfig& config) : config_(config) {
  grid_size_ = cv::Size(config_.chessboard_cols, config_.chessboard_rows);
  object_template_ = build_object_points();
}

CalibrationState AutoCalibrator::start() {
  object_points_.clear();
  image_points_.clear();
  last_corners_.reset();
  image_size_ = cv::Size(0, 0);
  state_ = CalibrationState();
  state_.status = "running";
  state_.target = config_.target_frames;
  state_.started_at = wall_seconds();
  MV_LOGI("session de calibration démarrée (objectif : " << state_.target << " vues)");
  return state_;
}

CalibrationState AutoCalibrator::cancel() {
  if (state_.status == "running") {
    state_.status = "cancelled";
    state_.finished_at = wall_seconds();
    state_.message = "calibration annulée";
    MV_LOGI("calibration annulée");
  }
  return state_;
}

bool AutoCalibrator::finished() const {
  return state_.status == "done" || state_.status == "failed" || state_.status == "cancelled";
}

double AutoCalibrator::progress() const {
  if (state_.target <= 0) {
    return 0.0;
  }
  return std::min(1.0, state_.captured / static_cast<double>(state_.target));
}

void AutoCalibrator::process(const cv::Mat& frame) {
  if (!running()) {
    return;
  }
  state_.attempts += 1;
  if (image_size_.width == 0) {
    image_size_ = cv::Size(frame.cols, frame.rows);
  }

  const double elapsed = wall_seconds() - state_.started_at;
  if (elapsed > config_.max_capture_seconds) {
    finalize_timeout();
    return;
  }

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  std::vector<cv::Point2f> corners;
  const bool found = cv::findChessboardCorners(gray, grid_size_, corners, kCalibCbFlags);
  state_.found = found;
  if (!found) {
    state_.message = "damier non détecté";
    return;
  }

  cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                                    0.001));

  const cv::Rect bounds = cv::boundingRect(corners);
  const double image_area = static_cast<double>(gray.rows) * gray.cols;
  state_.area_ratio =
      image_area > 0.0 ? static_cast<double>(bounds.width) * bounds.height / image_area : 0.0;
  if (state_.area_ratio < config_.min_board_area_ratio ||
      state_.area_ratio > config_.max_board_area_ratio) {
    const bool too_small = state_.area_ratio < config_.min_board_area_ratio;
    std::ostringstream oss;
    oss << "damier trop " << (too_small ? "petit" : "grand") << " ("
        << static_cast<int>(state_.area_ratio * 100.0) << " % de l'image)";
    state_.message = oss.str();
    return;
  }

  const int margin = std::max(1, config_.border_margin_px);
  if (bounds.x < margin || bounds.y < margin || bounds.x + bounds.width > gray.cols - margin ||
      bounds.y + bounds.height > gray.rows - margin) {
    state_.message = "damier trop près du bord - centrez-le";
    return;
  }

  const cv::Mat patch = gray(bounds);
  cv::Mat laplacian;
  cv::Laplacian(patch, laplacian, CV_64F);
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  state_.sharpness = stddev[0] * stddev[0];
  if (state_.sharpness < config_.min_sharpness) {
    std::ostringstream oss;
    oss << "image floue (" << static_cast<int>(state_.sharpness) << " < "
        << static_cast<int>(config_.min_sharpness) << ")";
    state_.message = oss.str();
    return;
  }

  if (last_corners_.has_value() && last_corners_->rows == static_cast<int>(corners.size())) {
    const double diagonal = std::hypot(static_cast<double>(gray.rows), gray.cols);
    double sum = 0.0;
    for (int i = 0; i < last_corners_->rows; ++i) {
      const cv::Point2f previous = last_corners_->at<cv::Point2f>(i, 0);
      sum += std::hypot(corners[static_cast<std::size_t>(i)].x - previous.x,
                        corners[static_cast<std::size_t>(i)].y - previous.y);
    }
    state_.movement = diagonal > 0.0 ? (sum / corners.size()) / diagonal : 0.0;
    if (state_.movement < config_.min_move_ratio) {
      state_.message = "position trop proche de la vue précédente";
      return;
    }
  } else {
    state_.movement = 1.0;
  }

  cv::Mat corners_mat(corners, true);
  object_points_.push_back(object_template_.clone());
  image_points_.push_back(corners_mat.clone());
  last_corners_ = corners_mat.clone();
  state_.captured += 1;
  state_.message = "vue " + std::to_string(state_.captured) + "/" +
                   std::to_string(state_.target) + " capturée";
  MV_LOGI("vue de calibration capturée (" << state_.captured << '/' << state_.target << ")");

  if (state_.captured >= state_.target) {
    compute();
  }
}

std::shared_ptr<Intrinsics> AutoCalibrator::compute() {
  if (object_points_.size() < 5) {
    state_.status = "failed";
    state_.finished_at = wall_seconds();
    state_.message = "échec : " + std::to_string(object_points_.size()) +
                     " vue(s) exploitable(s), 5 minimum requises";
    MV_LOGE(state_.message);
    return nullptr;
  }

  const int flags = resolve_flags();
  MV_LOGI("calibration sur " << object_points_.size() << " vues (" << image_size_.width << 'x'
                             << image_size_.height << ")...");
  cv::Mat matrix;
  cv::Mat dist;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  double rms = 0.0;
  try {
    rms = cv::calibrateCamera(object_points_, image_points_, image_size_, matrix, dist, rvecs,
                              tvecs, flags);
  } catch (const cv::Exception& exc) {
    state_.status = "failed";
    state_.finished_at = wall_seconds();
    state_.message = std::string("échec de calibrateCamera : ") + exc.what();
    MV_LOGE(state_.message);
    return nullptr;
  }

  const double mean_error =
      mean_reprojection_error(matrix, dist, rvecs, tvecs, object_points_, image_points_);
  auto intrinsics = std::make_shared<Intrinsics>();
  intrinsics->camera_matrix = matrix.clone();
  intrinsics->dist_coeffs = dist.reshape(1).clone();
  intrinsics->image_size = image_size_;
  intrinsics->rms = mean_error > 0.0 ? mean_error : rms;
  intrinsics->frames = static_cast<int>(object_points_.size());
  intrinsics->calibrated_at = iso_now();

  state_.intrinsics = intrinsics;
  state_.rms = intrinsics->rms;
  state_.status = "done";
  state_.finished_at = wall_seconds();
  std::ostringstream oss;
  oss << "calibration terminée (erreur de reprojection " << std::fixed << std::setprecision(3)
      << intrinsics->rms << " px)";
  state_.message = oss.str();
  MV_LOGI(state_.message);
  return intrinsics;
}

cv::Mat AutoCalibrator::draw_overlay(const cv::Mat& frame) const {
  cv::Mat canvas = frame.clone();
  cv::Mat gray;
  if (canvas.channels() == 3) {
    cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = canvas;
  }
  std::vector<cv::Point2f> corners;
  if (cv::findChessboardCorners(gray, grid_size_, corners, kCalibCbFlags)) {
    cv::drawChessboardCorners(canvas, grid_size_, corners, true);
  }
  const int height = canvas.rows;
  const cv::Scalar color = state_.found ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 165, 255);
  const int bar_width = static_cast<int>(canvas.cols * progress());
  if (bar_width > 0) {
    cv::rectangle(canvas, cv::Rect(0, height - 12, bar_width, 12), color, cv::FILLED);
  }
  cv::putText(canvas,
              "Calibration " + std::to_string(state_.captured) + "/" +
                  std::to_string(state_.target) + " - " + state_.message,
              cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2,
              cv::LINE_AA);
  return canvas;
}

void AutoCalibrator::finalize_timeout() {
  if (object_points_.size() >= 5) {
    compute();
  } else {
    state_.status = "failed";
    state_.finished_at = wall_seconds();
    state_.message = "délai dépassé : seulement " + std::to_string(object_points_.size()) +
                     " vue(s) capturée(s)";
  }
}

cv::Mat AutoCalibrator::build_object_points() const {
  const int cols = grid_size_.width;
  const int rows = grid_size_.height;
  cv::Mat points(cols * rows, 1, CV_32FC3);
  int index = 0;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      points.at<cv::Vec3f>(index++) = cv::Vec3f(static_cast<float>(col * config_.square_size_mm),
                                                static_cast<float>(row * config_.square_size_mm),
                                                0.0f);
    }
  }
  return points;
}

int AutoCalibrator::resolve_flags() const {
  int flags = 0;
  std::string text = config_.flags;
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream stream(text);
  std::string token;
  while (stream >> token) {
    const int flag = flag_from_name(token);
    if (flag != 0) {
      flags |= flag;
    } else {
      MV_LOGW("option de calibration inconnue ignorée : " << token);
    }
  }
  return flags;
}

}  // namespace matvision
