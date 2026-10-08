#include "matvision/overlay.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "matvision/aruco.hpp"
#include "matvision/logger.hpp"

namespace matvision {

namespace {

const cv::Scalar kCornerColor(255, 128, 0);   // BGR, bleu clair
const cv::Scalar kObjectColor(0, 220, 255);   // BGR, jaune
const cv::Scalar kWarningColor(0, 165, 255);  // BGR, orange

/// Convertit un point 2D de la table en pixel, via la pose caméra + intrinsèques.
cv::Point2f table_to_image(const cv::Point3d& point, const TableLocalization& localization,
                           const Intrinsics& intrinsics) {
  const std::vector<cv::Point2f> projected =
      project_points_3d({point}, localization.rvec, localization.tvec,
                        intrinsics.camera_matrix, intrinsics.dist_coeffs);
  return projected.empty() ? cv::Point2f(0.0f, 0.0f) : projected[0];
}

}  // namespace

void draw_hud(cv::Mat& frame, const std::vector<std::string>& lines) {
  for (std::size_t index = 0; index < lines.size(); ++index) {
    cv::putText(frame, lines[index],
                cv::Point(10, 24 + static_cast<int>(index) * 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

void draw_table_axes(cv::Mat& frame, const TableLocalization& localization,
                     const Intrinsics* intrinsics) {
  if (!localization.has_pose || intrinsics == nullptr || !intrinsics->valid()) {
    return;
  }
  const double axis_length = 300.0;
  const cv::Point2f origin_px =
      table_to_image(cv::Point3d(0.0, 0.0, 0.0), localization, *intrinsics);
  const cv::Point2f x_axis_px =
      table_to_image(cv::Point3d(axis_length, 0.0, 0.0), localization, *intrinsics);
  const cv::Point2f y_axis_px =
      table_to_image(cv::Point3d(0.0, axis_length, 0.0), localization, *intrinsics);

  cv::arrowedLine(frame, origin_px, x_axis_px, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);
  cv::arrowedLine(frame, origin_px, y_axis_px, cv::Scalar(255, 0, 0), 2, cv::LINE_AA, 0, 0.2);
  cv::circle(frame, origin_px, 5, cv::Scalar(0, 255, 255), cv::FILLED);
}

void annotate(cv::Mat& frame, const MarkerMap& corner_markers, const MarkerMap& object_markers,
              const TableLocalization& localization, const std::vector<nlohmann::json>& objects) {
  MV_LOGD("annotate : " << corner_markers.size() << " tags de coin, " << object_markers.size()
                        << " tags objets, " << objects.size() << " relevés");
  draw_markers(frame, corner_markers, kCornerColor);

  for (const auto& item : objects) {
    const int id = item.value("id", -1);
    const auto it = object_markers.find(id);
    if (it == object_markers.end()) {
      continue;
    }
    const cv::Point2f center = marker_center(it->second);
    std::ostringstream oss;
    oss << item.value("label", std::string()) << " (" << static_cast<int>(item.value("x", 0.0))
        << ',' << static_cast<int>(item.value("y", 0.0)) << ") "
        << static_cast<int>(item.value("a", 0.0)) << "deg";
    cv::putText(frame, oss.str(), cv::Point(cvRound(center.x) - 80, cvRound(center.y) - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, kObjectColor, 1, cv::LINE_AA);
  }

  draw_markers(frame, object_markers, kObjectColor);

  if (!localization.ok) {
    cv::putText(frame, localization.reason, cv::Point(10, frame.rows - 44),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, kWarningColor, 2, cv::LINE_AA);
  }
}

}  // namespace matvision
