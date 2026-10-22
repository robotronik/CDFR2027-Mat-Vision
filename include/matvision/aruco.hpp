#pragma once

/// Détection des tags ArUco : dictionnaire, paramètres et utilitaires de tracé.

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include "matvision/geometry.hpp"
#include <nlohmann/json.hpp>

namespace matvision {

/// Encapsule dictionnaire + paramètres + appel `detectMarkers`.
class ArucoDetector {
 public:
  /// `params` est un objet JSON dont les clés sont des champs de
  /// `DetectorParameters` (les clés inconnues sont signalées en debug).
  explicit ArucoDetector(const std::string& dictionary = "DICT_4X4_50",
                         const nlohmann::json& params = nlohmann::json());

  /// Détecte les marqueurs. `mask` (optionnel) restreint la recherche à la ROI.
  /// Une image couleur est convertie en niveaux de gris automatiquement.
  void detect(const cv::Mat& image, std::vector<int>& ids, std::vector<Marker>& corners,
              const cv::Mat& mask = cv::Mat()) const;

  /// Comme `detect` mais renvoie une table `id -> coins`.
  MarkerMap detect_by_id(const cv::Mat& image, const cv::Mat& mask = cv::Mat()) const;

 private:
  cv::aruco::ArucoDetector detector_;
};

/// Dessine les marqueurs détectés (modifie `frame` en place).
void draw_markers(cv::Mat& frame, const MarkerMap& markers, const cv::Scalar& color,
                  bool show_ids = true, int thickness = 2);

}  // namespace matvision
