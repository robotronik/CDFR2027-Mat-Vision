#pragma once

/// Annotations de l'image d'aperçu : tags, axes de la table, positions, HUD.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "matvision/calibration.hpp"
#include "matvision/config.hpp"
#include "matvision/geometry.hpp"
#include "matvision/json.hpp"
#include "matvision/table.hpp"

namespace matvision {

/// Écrit quelques lignes d'état en haut à gauche de l'image.
void draw_hud(cv::Mat& frame, const std::vector<std::string>& lines);

/// Dessine les tags de coin, les objets relevés et les axes de la table.
void annotate(cv::Mat& frame, const MarkerMap& corner_markers, const MarkerMap& object_markers,
              const TableLocalization& localization, const std::vector<nlohmann::json>& objects);

/// Dessine l'origine et les axes de la table (x en rouge, y en bleu).
void draw_table_axes(cv::Mat& frame, const TableLocalization& localization,
                     const Intrinsics* intrinsics);

}  // namespace matvision
