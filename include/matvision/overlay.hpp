#pragma once

/// Annotations de l'image d'aperçu : tags, axes de la table, positions, HUD.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "matvision/calibration.hpp"
#include "matvision/config.hpp"
#include "matvision/geometry.hpp"
#include <nlohmann/json.hpp>
#include "matvision/table.hpp"

namespace matvision {

/// Tag objet détecté dans l'image, avec son relevé de position.
///
/// Plusieurs détections peuvent partager le même `id` (ex. plusieurs éléments
/// de jeu taggés `13`) : chaque occurrence a donc sa propre annotation.
struct ObjectAnnotation {
  int id = -1;
  Marker corners;
  nlohmann::json object;  ///< relevé `{id, label, x, y, z, a}`, ou `null` si non situé.
};

/// Écrit quelques lignes d'état en haut à gauche de l'image.
void draw_hud(cv::Mat& frame, const std::vector<std::string>& lines);

/// Dessine les tags de coin, les tags objets et les axes de la table.
void annotate(cv::Mat& frame, const MarkerMap& corner_markers,
              const std::vector<ObjectAnnotation>& objects,
              const TableLocalization& localization);

/// Dessine l'origine et les axes de la table (x en rouge, y en bleu).
void draw_table_axes(cv::Mat& frame, const TableLocalization& localization,
                     const Intrinsics* intrinsics);

}  // namespace matvision
