#include "matvision/aruco.hpp"

#include <opencv2/imgproc.hpp>

#include "matvision/logger.hpp"

namespace matvision {

namespace {

/// Table des dictionnaires ArUco prédéfinis exposés par la configuration.
bool dictionary_from_name(const std::string& name, cv::aruco::PredefinedDictionaryType& out) {
  static const std::pair<const char*, cv::aruco::PredefinedDictionaryType> kTable[] = {
      {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
      {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
      {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
      {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
      {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
      {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
      {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
      {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
      {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
      {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
      {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
      {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
      {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
      {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
      {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
      {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
      {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
      {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
      {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
      {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
      {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
  };
  for (const auto& [key, value] : kTable) {
    if (name == key) {
      out = value;
      return true;
    }
  }
  return false;
}

void apply_corner_refinement(cv::aruco::DetectorParameters& params, const std::string& name) {
  if (name == "CORNER_REFINE_SUBPIX") {
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  } else if (name == "CORNER_REFINE_CONTOUR") {
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
  } else if (name == "CORNER_REFINE_APRILTAG") {
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
  } else if (name == "CORNER_REFINE_NONE") {
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
  } else {
    MV_LOGW("cornerRefinementMethod inconnu ignoré : " << name);
  }
}

/// Renseigne un champ du détecteur si la clé existe et a le bon type.
template <typename T>
void set_param(const nlohmann::json& params, const char* key, T& field) {
  if (!params.contains(key) || params[key].is_null()) {
    return;
  }
  try {
    field = params[key].get<T>();
  } catch (const std::exception&) {
    MV_LOGW("paramètre ArUco ignoré (type invalide) : " << key);
  }
}

void apply_parameters(cv::aruco::DetectorParameters& params, const nlohmann::json& values) {
  if (!values.is_object()) {
    return;
  }
  set_param(values, "adaptiveThreshWinSizeMin", params.adaptiveThreshWinSizeMin);
  set_param(values, "adaptiveThreshWinSizeMax", params.adaptiveThreshWinSizeMax);
  set_param(values, "adaptiveThreshWinSizeStep", params.adaptiveThreshWinSizeStep);
  set_param(values, "adaptiveThreshConstant", params.adaptiveThreshConstant);
  set_param(values, "minMarkerPerimeterRate", params.minMarkerPerimeterRate);
  set_param(values, "maxMarkerPerimeterRate", params.maxMarkerPerimeterRate);
  set_param(values, "polygonalApproxAccuracyRate", params.polygonalApproxAccuracyRate);
  set_param(values, "minCornerDistanceRate", params.minCornerDistanceRate);
  set_param(values, "minDistanceToBorder", params.minDistanceToBorder);
  set_param(values, "minMarkerDistanceRate", params.minMarkerDistanceRate);
  set_param(values, "cornerRefinementWinSize", params.cornerRefinementWinSize);
  set_param(values, "relativeCornerRefinmentWinSize",
            params.relativeCornerRefinmentWinSize);
  set_param(values, "cornerRefinementMaxIterations",
            params.cornerRefinementMaxIterations);
  set_param(values, "cornerRefinementMinAccuracy", params.cornerRefinementMinAccuracy);
  set_param(values, "markerBorderBits", params.markerBorderBits);
  set_param(values, "perspectiveRemovePixelPerCell", params.perspectiveRemovePixelPerCell);
  set_param(values, "perspectiveRemoveIgnoredMarginPerCell",
            params.perspectiveRemoveIgnoredMarginPerCell);
  set_param(values, "maxErroneousBitsInBorderRate", params.maxErroneousBitsInBorderRate);
  set_param(values, "minOtsuStdDev", params.minOtsuStdDev);
  set_param(values, "errorCorrectionRate", params.errorCorrectionRate);
  set_param(values, "detectInvertedMarker", params.detectInvertedMarker);
  set_param(values, "useAruco3Detection", params.useAruco3Detection);

  if (values.contains("cornerRefinementMethod") &&
      values["cornerRefinementMethod"].is_string()) {
    apply_corner_refinement(params, values["cornerRefinementMethod"].get<std::string>());
  }
}

}  // namespace

cv::aruco::DetectorParameters default_detector_parameters() {
  return cv::aruco::DetectorParameters();
}

ArucoDetector::ArucoDetector(const std::string& dictionary, const nlohmann::json& params)
    : dictionary_name_(dictionary) {
  cv::aruco::PredefinedDictionaryType type;
  if (!dictionary_from_name(dictionary, type)) {
    throw std::invalid_argument("dictionnaire ArUco inconnu : " + dictionary);
  }
  dictionary_ = cv::aruco::getPredefinedDictionary(type);
  parameters_ = default_detector_parameters();
  apply_parameters(parameters_, params);
  detector_ = cv::aruco::ArucoDetector(dictionary_, parameters_);
  MV_LOGD("détecteur ArUco prêt (dictionnaire=" << dictionary << ")");
}

void ArucoDetector::detect(const cv::Mat& image, std::vector<int>& ids,
                           std::vector<Marker>& corners, std::vector<Marker>* rejected,
                           const cv::Mat& mask, bool gray) const {
  cv::Mat gray_image;
  if (gray) {
    gray_image = image;
  } else {
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  }

  cv::Mat working = gray_image;
  if (!mask.empty()) {
    cv::bitwise_and(gray_image, gray_image, working, mask);
  }

  corners.clear();
  ids.clear();
  std::vector<Marker> rejected_local;
  if (rejected != nullptr) {
    rejected->clear();
    detector_.detectMarkers(working, corners, ids, *rejected);
  } else {
    detector_.detectMarkers(working, corners, ids, rejected_local);
  }
  MV_LOGD("detectMarkers : " << ids.size() << " tag(s) (masque=" << (mask.empty() ? "non" : "oui")
                             << ")");
}

MarkerMap ArucoDetector::detect_by_id(const cv::Mat& image, const cv::Mat& mask,
                                      bool gray) const {
  std::vector<int> ids;
  std::vector<Marker> corners;
  detect(image, ids, corners, nullptr, mask, gray);
  MarkerMap result;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    result[ids[i]] = corners[i];
  }
  return result;
}

void draw_markers(cv::Mat& frame, const MarkerMap& markers, const cv::Scalar& color,
                  bool show_ids, int thickness) {
  for (const auto& [id, corners] : markers) {
    std::vector<cv::Point> points;
    points.reserve(corners.size());
    for (const auto& corner : corners) {
      points.emplace_back(cv::Point(cvRound(corner.x), cvRound(corner.y)));
    }
    cv::polylines(frame, std::vector<std::vector<cv::Point>>{points}, true, color, thickness,
                  cv::LINE_AA);
    if (show_ids) {
      const cv::Point2f center = marker_center(corners);
      const std::string label = std::to_string(id);
      int baseline = 0;
      const cv::Size text_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
      cv::putText(frame, label,
                  cv::Point(cvRound(center.x) - text_size.width / 2,
                            cvRound(center.y) + text_size.height / 2),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }
  }
}

}  // namespace matvision
