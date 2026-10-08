#pragma once

/// Calibration intrinsèque : modèle, sérialisation et capture automatique (damier).

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "matvision/config.hpp"
#include "matvision/json.hpp"

namespace matvision {

/// Résultat de la calibration intrinsèque.
struct Intrinsics {
  cv::Mat camera_matrix;  ///< 3x3 CV_64F
  cv::Mat dist_coeffs;    ///< Nx1 CV_64F
  cv::Size image_size{0, 0};
  double rms = 0.0;
  int frames = 0;
  std::string calibrated_at;
  std::string source;

  bool valid() const { return !camera_matrix.empty(); }
  double fx() const;
  double fy() const;
  nlohmann::json to_json() const;
  /// `to_json` enrichi de la focale et du champ de vision.
  nlohmann::json summary() const;
  static std::optional<Intrinsics> from_json(const nlohmann::json& data,
                                             const std::string& source);
};

/// Format d'un fichier de calibration : "json", "yaml" ou "inconnu".
std::string calibration_file_kind(const std::string& path);

/// Écrit la calibration au format JSON.
bool save_intrinsics(const std::string& path, const Intrinsics& intrinsics);

/// Charge une calibration JSON ou (compatibilité) un `.yml` OpenCV.
std::optional<Intrinsics> load_intrinsics(const std::string& path);

/// État d'une session de calibration (exposé par l'API).
struct CalibrationState {
  std::string status = "idle";  // idle | running | done | failed | cancelled
  bool found = false;
  int captured = 0;
  int target = 0;
  int attempts = 0;
  double sharpness = 0.0;
  double area_ratio = 0.0;
  double movement = 0.0;
  std::string message = "présentez le damier devant la caméra";
  std::optional<double> rms;
  double started_at = 0.0;
  double finished_at = 0.0;
  std::shared_ptr<Intrinsics> intrinsics;

  nlohmann::json to_json() const;
};

/// Collecte automatiquement des vues d'un damier puis calcule les intrinsèques.
class AutoCalibrator {
 public:
  explicit AutoCalibrator(const CalibrationConfig& config = CalibrationConfig());

  CalibrationState start();
  CalibrationState cancel();
  bool running() const { return state_.status == "running"; }
  bool finished() const;
  double progress() const;
  const CalibrationState& state() const { return state_; }

  /// Analyse une image ; capture la vue si elle est exploitable.
  void process(const cv::Mat& frame);

  /// Calcule les intrinsèques à partir des vues collectées.
  std::shared_ptr<Intrinsics> compute();

  /// Annote l'image avec la progression (damier détecté ou non).
  cv::Mat draw_overlay(const cv::Mat& frame) const;

 private:
  void finalize_timeout();
  cv::Mat build_object_points() const;
  int resolve_flags() const;

  CalibrationConfig config_;
  cv::Size grid_size_;
  cv::Mat object_template_;
  std::vector<cv::Mat> object_points_;
  std::vector<cv::Mat> image_points_;
  std::optional<cv::Mat> last_corners_;
  cv::Size image_size_{0, 0};
  CalibrationState state_;
};

}  // namespace matvision
