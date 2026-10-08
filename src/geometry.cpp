#include "matvision/geometry.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace matvision {

nlohmann::json Position::to_json(int digits) const {
  nlohmann::json data;
  data["x"] = std::round(x * std::pow(10.0, digits)) / std::pow(10.0, digits);
  data["y"] = std::round(y * std::pow(10.0, digits)) / std::pow(10.0, digits);
  data["z"] = std::round(z * std::pow(10.0, digits)) / std::pow(10.0, digits);
  data["a"] = std::round(normalize_angle_deg(a) * std::pow(10.0, digits)) /
              std::pow(10.0, digits);
  return data;
}

double normalize_angle_deg(double angle) {
  double value = std::fmod(angle + 180.0, 360.0) - 180.0;
  if (value <= -180.0) {
    value += 360.0;
  }
  return value;
}

std::array<cv::Point2d, 4> marker_corners_table(double cx, double cy, double angle_deg,
                                                double size) {
  const double half = size / 2.0;
  const std::array<cv::Point2d, 4> local = {
      cv::Point2d(-half, half), cv::Point2d(half, half), cv::Point2d(half, -half),
      cv::Point2d(-half, -half)};
  const double radians = angle_deg * kPi / 180.0;
  const double c = std::cos(radians);
  const double s = std::sin(radians);

  std::array<cv::Point2d, 4> out{};
  for (std::size_t i = 0; i < local.size(); ++i) {
    // Rotation par l'angle `a` puis translation au centre du marqueur.
    out[i].x = c * local[i].x - s * local[i].y + cx;
    out[i].y = s * local[i].x + c * local[i].y + cy;
  }
  return out;
}

cv::Point2f marker_center(const Marker& corners) {
  cv::Point2f sum(0.0f, 0.0f);
  for (const auto& point : corners) {
    sum += point;
  }
  const float count = static_cast<float>(corners.empty() ? 1 : corners.size());
  return sum / count;
}

cv::Mat build_roi_mask(const cv::Size& shape,
                       const std::vector<std::vector<cv::Point2f>>& groups, int margin_px) {
  std::vector<cv::Point2f> stacked;
  for (const auto& group : groups) {
    stacked.insert(stacked.end(), group.begin(), group.end());
  }
  if (stacked.size() < 3) {
    return cv::Mat();
  }

  std::vector<cv::Point2f> hull;
  cv::convexHull(stacked, hull);
  if (hull.size() < 3) {
    return cv::Mat();
  }

  cv::Mat mask = cv::Mat::zeros(shape, CV_8UC1);
  std::vector<cv::Point> polygon;
  polygon.reserve(hull.size());
  for (const auto& point : hull) {
    polygon.emplace_back(cv::Point(cvRound(point.x), cvRound(point.y)));
  }
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255));

  const int margin = std::max(0, margin_px);
  if (margin > 0) {
    const int kernel_size = margin * 2 + 1;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::dilate(mask, mask, kernel);
  }
  return mask;
}

std::vector<cv::Point2f> project_points_3d(const std::vector<cv::Point3d>& points,
                                           const cv::Mat& rvec, const cv::Mat& tvec,
                                           const cv::Mat& camera_matrix,
                                           const cv::Mat& dist_coeffs) {
  std::vector<cv::Point2f> out;
  if (points.empty()) {
    return out;
  }
  cv::Mat object_points(static_cast<int>(points.size()), 1, CV_64FC3);
  for (std::size_t i = 0; i < points.size(); ++i) {
    object_points.at<cv::Vec3d>(static_cast<int>(i)) =
        cv::Vec3d(points[i].x, points[i].y, points[i].z);
  }
  cv::Mat projected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected);
  cv::Mat projected64;
  projected.convertTo(projected64, CV_64FC2);
  out.resize(points.size());
  for (int i = 0; i < projected64.rows; ++i) {
    const cv::Vec2d value = projected64.at<cv::Vec2d>(i, 0);
    out[static_cast<std::size_t>(i)] =
        cv::Point2f(static_cast<float>(value[0]), static_cast<float>(value[1]));
  }
  return out;
}

}  // namespace matvision
