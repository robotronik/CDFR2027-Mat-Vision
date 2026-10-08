#pragma once

/// Détection des tags ArUco : dictionnaire, paramètres et utilitaires de tracé.

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include "matvision/geometry.hpp"
#include "matvision/json.hpp"

namespace matvision {

/// Paramètres par défaut du détecteur ArUco.
cv::aruco::DetectorParameters default_detector_parameters();

/// Encapsule dictionnaire + paramètres + appel `detectMarkers`.
class ArucoDetector {
 public:
  /// `params` est un objet JSON dont les clés sont des champs de
  /// `DetectorParameters` (les clés inconnues sont signalées en debug).
  explicit ArucoDetector(const std::string& dictionary = "DICT_4X4_50",
                         const nlohmann::json& params = nlohmann::json());

  /// Détecte les marqueurs. `mask` (optionnel) restreint la recherche à la ROI.
  void detect(const cv::Mat& image, std::vector<int>& ids, std::vector<Marker>& corners,
              std::vector<Marker>* rejected = nullptr, const cv::Mat& mask = cv::Mat(),
              bool gray = false) const;

  /// Comme `detect` mais renvoie une table `id -> coins`.
  MarkerMap detect_by_id(const cv::Mat& image, const cv::Mat& mask = cv::Mat(),
                         bool gray = false) const;

  const std::string& dictionary_name() const { return dictionary_name_; }

 private:
  std::string dictionary_name_;
  cv::aruco::Dictionary dictionary_;
  cv::aruco::DetectorParameters parameters_;
  cv::aruco::ArucoDetector detector_;
};

/// Dessine les marqueurs détectés (modifie `frame` en place).
void draw_markers(cv::Mat& frame, const MarkerMap& markers, const cv::Scalar& color,
                  bool show_ids = true, int thickness = 2);

}  // namespace matvision
