#include "matvision/table.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include <opencv2/calib3d.hpp>

#include "matvision/logger.hpp"

namespace matvision {

namespace {

std::vector<cv::Point3d> marker_object_points(const CornerMarker& marker) {
  const auto corners = marker_corners_table(marker.x, marker.y, marker.a, marker.size);
  std::vector<cv::Point3d> points;
  points.reserve(corners.size());
  for (const auto& corner : corners) {
    points.emplace_back(corner.x, corner.y, 0.0);
  }
  return points;
}

std::vector<cv::Point3d> tag_object_points(double size) {
  const double half = size / 2.0;
  return {cv::Point3d(-half, half, 0.0), cv::Point3d(half, half, 0.0),
          cv::Point3d(half, -half, 0.0), cv::Point3d(-half, -half, 0.0)};
}

double reprojection_error(const std::vector<cv::Point3d>& object_points,
                          const std::vector<cv::Point2f>& image_points, const cv::Mat& rvec,
                          const cv::Mat& tvec, const Intrinsics& intrinsics) {
  const std::vector<cv::Point2f> projected = project_points_3d(
      object_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs);
  double total = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i) {
    total += std::hypot(projected[i].x - image_points[i].x, projected[i].y - image_points[i].y);
  }
  return projected.empty() ? 0.0 : total / projected.size();
}

}  // namespace

namespace {
/// Produit d'une matrice 3x3 par un vecteur 3D (résultat explicite).
cv::Vec3d mat3_times(const cv::Mat& matrix, const cv::Vec3d& vector) {
  const cv::Mat result = matrix * cv::Mat(vector);
  return cv::Vec3d(result.at<double>(0), result.at<double>(1), result.at<double>(2));
}
}  // namespace

cv::Vec3d TableLocalization::camera_to_table(const cv::Vec3d& point_camera) const {
  return mat3_times(rotation_t, point_camera - translation);
}

Position TableLocalization::tag_position(const cv::Mat& rvec_tag, const cv::Mat& tvec_tag,
                                         double angle_offset) const {
  cv::Mat rotation_tag;
  cv::Rodrigues(rvec_tag, rotation_tag);
  const cv::Mat t_tag = tvec_tag.reshape(1, 3);

  const cv::Vec3d center_table =
      camera_to_table(cv::Vec3d(t_tag.at<double>(0), t_tag.at<double>(1), t_tag.at<double>(2)));
  const cv::Vec3d x_axis_camera(rotation_tag.at<double>(0, 0), rotation_tag.at<double>(1, 0),
                                rotation_tag.at<double>(2, 0));
  const cv::Vec3d x_axis_table = mat3_times(rotation_t, x_axis_camera);
  const double yaw = std::atan2(x_axis_table[1], x_axis_table[0]) * 180.0 / kPi;

  Position position;
  position.x = center_table[0];
  position.y = center_table[1];
  position.z = center_table[2];
  position.a = normalize_angle_deg(yaw + angle_offset);
  return position;
}

nlohmann::json TableLocalization::to_json() const {
  nlohmann::json data = {{"ok", ok},
                         {"used_ids", used_ids},
                         {"points", total_points},
                         {"inliers", static_cast<int>(used_ids.size()) * 4},
                         {"residual_px", std::round(residual_px * 100.0) / 100.0},
                         {"reason", reason}};
  if (has_camera_position) {
    data["camera_position"] = camera_position.to_json();
  }
  return data;
}

TableLocalization localize_table(const MarkerMap& markers, const TableConfig& table,
                                 const Intrinsics* intrinsics, double max_residual_px) {
  TableLocalization result;

  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2f> image_points;
  for (const auto& marker : table.markers) {
    const auto it = markers.find(marker.id);
    if (it == markers.end()) {
      continue;
    }
    result.used_ids.push_back(marker.id);
    const auto points = marker_object_points(marker);
    object_points.insert(object_points.end(), points.begin(), points.end());
    image_points.insert(image_points.end(), it->second.begin(), it->second.end());
  }

  if (intrinsics == nullptr || !intrinsics->valid()) {
    result.reason = "calibration intrinsèque requise pour la pose caméra";
    MV_LOGD("localize_table : " << result.reason);
    return result;
  }

  if (static_cast<int>(result.used_ids.size()) < kMinCornerMarkers) {
    std::ostringstream oss;
    oss << result.used_ids.size() << " tag(s) de coin détecté(s) sur " << table.markers.size()
        << " (" << kMinCornerMarkers << " minimum)";
    result.reason = oss.str();
    MV_LOGD("localize_table : " << result.reason);
    return result;
  }

  result.total_points = static_cast<int>(image_points.size());

  cv::Mat rvec;
  cv::Mat tvec;
  bool solved = false;
  for (const int flag : {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE}) {
    try {
      solved = cv::solvePnP(object_points, image_points, intrinsics->camera_matrix,
                            intrinsics->dist_coeffs, rvec, tvec, false, flag);
    } catch (const cv::Exception& exc) {
      MV_LOGD("solvePnP (flag=" << flag << ") a échoué : " << exc.what());
      solved = false;
    }
    if (solved) {
      break;
    }
  }
  if (solved) {
    // Affinage Levenberg-Marquardt à partir de la solution analytique.
    try {
      cv::solvePnP(object_points, image_points, intrinsics->camera_matrix,
                   intrinsics->dist_coeffs, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    } catch (const cv::Exception&) {
      // On garde la solution non affinée.
    }
  }
  if (!solved) {
    result.reason = "pose caméra non calculable (solvePnP sans solution)";
    MV_LOGD("localize_table : " << result.reason);
    return result;
  }

  result.residual_px = reprojection_error(object_points, image_points, rvec, tvec, *intrinsics);
  if (result.residual_px > max_residual_px) {
    std::ostringstream oss;
    oss << "pose caméra imprécise (résidu " << std::fixed << std::setprecision(1)
        << result.residual_px << " px > " << max_residual_px << " px)";
    result.reason = oss.str();
    MV_LOGD("localize_table : " << result.reason);
    return result;
  }

  result.has_pose = true;
  result.rvec = rvec.reshape(1, 3).clone();
  result.tvec = tvec.reshape(1, 3).clone();
  cv::Rodrigues(result.rvec, result.rotation);
  result.rotation_t = result.rotation.t();
  result.translation = cv::Vec3d(result.tvec.at<double>(0), result.tvec.at<double>(1),
                                 result.tvec.at<double>(2));

  // Position de la caméra dans le repère table : -R^T * t.
  const cv::Vec3d origin(0.0, 0.0, 0.0);
  const cv::Vec3d t_table(result.tvec.at<double>(0), result.tvec.at<double>(1),
                          result.tvec.at<double>(2));
  const cv::Vec3d camera = mat3_times(result.rotation_t, origin - t_table);
  const double yaw =
      std::atan2(result.rotation.at<double>(0, 1), result.rotation.at<double>(0, 0)) * 180.0 / kPi;
  result.camera_position = Position{camera[0], camera[1], camera[2], normalize_angle_deg(yaw)};
  result.has_camera_position = true;
  result.ok = true;

  MV_LOGD("localize_table : ok, tags=" << result.used_ids.size() << ", résidu "
                                       << result.residual_px << " px");
  return result;
}

std::optional<std::pair<cv::Mat, cv::Mat>> solve_tag_pose(const Marker& corners, double size,
                                                          const Intrinsics& intrinsics) {
  if (corners.size() < 4 || !intrinsics.valid()) {
    return std::nullopt;
  }
  const std::vector<cv::Point3d> object_points = tag_object_points(size);
  std::vector<cv::Point2f> image_points(corners.begin(), corners.begin() + 4);

  cv::Mat rvec;
  cv::Mat tvec;
  for (const int flag : {cv::SOLVEPNP_IPPE_SQUARE, cv::SOLVEPNP_ITERATIVE}) {
    try {
      if (cv::solvePnP(object_points, image_points, intrinsics.camera_matrix,
                       intrinsics.dist_coeffs, rvec, tvec, false, flag)) {
        // Affinage Levenberg-Marquardt à partir de la solution analytique.
        try {
          cv::solvePnP(object_points, image_points, intrinsics.camera_matrix,
                       intrinsics.dist_coeffs, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
        } catch (const cv::Exception&) {
        }
        return std::make_pair(rvec.reshape(1, 3).clone(), tvec.reshape(1, 3).clone());
      }
    } catch (const cv::Exception& exc) {
      MV_LOGD("solvePnP tag (flag=" << flag << ") a échoué : " << exc.what());
    }
  }
  return std::nullopt;
}

std::vector<cv::Point2f> table_polygon_image(const TableLocalization& localization,
                                             const TableConfig& table,
                                             const Intrinsics& intrinsics) {
  if (!localization.has_pose || !intrinsics.valid()) {
    return {};
  }
  const double half_width = table.width_mm / 2.0;
  const double half_height = table.height_mm / 2.0;
  const std::vector<cv::Point3d> corners = {
      cv::Point3d(-half_width, -half_height, 0.0), cv::Point3d(half_width, -half_height, 0.0),
      cv::Point3d(half_width, half_height, 0.0), cv::Point3d(-half_width, half_height, 0.0)};
  return project_points_3d(corners, localization.rvec, localization.tvec,
                           intrinsics.camera_matrix, intrinsics.dist_coeffs);
}

}  // namespace matvision
