#pragma once

/// Géométrie du plan de jeu (repère table en millimètres, origine au centre).
///
/// Repère **table** : `+x` vers la droite (largeur 2000 mm), `+y` vers le fond
/// (longueur 3000 mm), `+z` vers le haut. L'angle `a` est un lacet en degrés,
/// sens trigonométrique vu du dessus, `0°` = axe `+x`, normalisé dans `]-180, 180]`.
///
/// Repère **marqueur** : origine au centre du tag, coins renvoyés dans l'ordre
/// haut-gauche, haut-droite, bas-droite, bas-gauche (comme OpenCV).

#include <array>
#include <map>
#include <vector>

#include <opencv2/core.hpp>

#include "matvision/json.hpp"

namespace matvision {

/// Constante pi (M_PI n'est pas garanti en mode C++ strict).
inline constexpr double kPi = 3.14159265358979323846;

/// Coins d'un marqueur détecté (4 points, ordre OpenCV).
using Marker = std::vector<cv::Point2f>;
/// Marqueurs détectés indexés par identifiant de tag.
using MarkerMap = std::map<int, Marker>;

/// Position (x, y, z) en millimètres et lacet `a` en degrés, repère table.
struct Position {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double a = 0.0;

  nlohmann::json to_json(int digits = 2) const;
};

/// Ramène un angle en degrés dans l'intervalle `]-180, 180]`.
double normalize_angle_deg(double angle);

/// Coins d'un marqueur (mm) exprimés dans le repère table, ordre OpenCV.
std::array<cv::Point2d, 4> marker_corners_table(double cx, double cy, double angle_deg,
                                                double size);

/// Centre du marqueur (moyenne des 4 coins).
cv::Point2f marker_center(const Marker& corners);

/// Masque binaire de la zone de travail (enveloppe convexe des groupes de coins).
/// Renvoie un `cv::Mat` vide si aucune zone exploitable.
cv::Mat build_roi_mask(const cv::Size& shape,
                       const std::vector<std::vector<cv::Point2f>>& groups, int margin_px);

/// Projette des points 3D (mm) en pixels.
///
/// Enveloppe `cv::projectPoints` qui contourne une limite d'OpenCV 4.11 : avec
/// des points objet `Point3d` et une sortie `std::vector<Point2f>`, l'allocation
/// de la sortie échoue. On passe donc par une matrice intermédiaire.
std::vector<cv::Point2f> project_points_3d(const std::vector<cv::Point3d>& points,
                                           const cv::Mat& rvec, const cv::Mat& tvec,
                                           const cv::Mat& camera_matrix,
                                           const cv::Mat& dist_coeffs);

}  // namespace matvision
