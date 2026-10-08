#pragma once

/// Repère table construit à partir des 4 tags ArUco de coin.
///
/// Aucune homographie : la pose de la caméra dans le repère table est estimée
/// par `solvePnP` (les positions des tags de coin sont connues en mm), puis la
/// position de chaque objet est obtenue par un **changement de base**
/// (composition de la pose caméra-table et de la pose caméra-objet).
///
/// Les paramètres intrinsèques sont donc **obligatoires** : sans calibration,
/// aucune position ne peut être calculée.

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "matvision/calibration.hpp"
#include "matvision/config.hpp"
#include "matvision/geometry.hpp"
#include "matvision/json.hpp"

namespace matvision {

/// Nombre minimal de tags de coin détectés pour estimer la pose caméra.
inline constexpr int kMinCornerMarkers = 2;

/// Résultat de la localisation du repère table sur une image.
struct TableLocalization {
  bool ok = false;
  std::vector<int> used_ids;
  int total_points = 0;
  double residual_px = 0.0;
  bool has_pose = false;
  cv::Mat rvec;         ///< pose table -> caméra (3x1)
  cv::Mat tvec;         ///< pose table -> caméra (3x1)
  cv::Mat rotation;     ///< rotation table -> caméra (3x3)
  cv::Mat rotation_t;   ///< rotation caméra -> table (3x3) = rotation^T
  cv::Vec3d translation;  ///< translation table -> caméra

  bool has_camera_position = false;
  Position camera_position;
  std::string reason;

  /// Change de base caméra -> table d'un point 3D.
  cv::Vec3d camera_to_table(const cv::Vec3d& point_camera) const;

  /// Position (repère table) du centre d'un tag dont la pose caméra est connue.
  Position tag_position(const cv::Mat& rvec_tag, const cv::Mat& tvec_tag,
                        double angle_offset = 0.0) const;

  nlohmann::json to_json() const;
};

/// Estime la pose caméra (repère table -> caméra) à partir des tags de coin.
///
/// :param markers: `{id: coins}` issus de la détection ArUco.
/// :param intrinsics: calibration intrinsèque (obligatoire).
/// :param max_residual_px: erreur de reprojection maximale tolérée.
TableLocalization localize_table(const MarkerMap& markers, const TableConfig& table,
                                 const Intrinsics* intrinsics,
                                 double max_residual_px = 50.0);

/// Pose d'un tag (repère tag -> caméra) par `solvePnP`, taille connue en mm.
std::optional<std::pair<cv::Mat, cv::Mat>> solve_tag_pose(const Marker& corners,
                                                          double size,
                                                          const Intrinsics& intrinsics);

/// Emprise de la table dans l'image (coins 3D projetés via la pose caméra).
std::vector<cv::Point2f> table_polygon_image(const TableLocalization& localization,
                                             const TableConfig& table,
                                             const Intrinsics& intrinsics);

}  // namespace matvision
