#pragma once

/// Petits utilitaires partagés (temps, chaînes, arrondis).

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <string>

namespace matvision {

/// Secondes écoulées depuis l'époque de `steady_clock` (mesures de durée).
inline double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// Secondes écoulées depuis l'époque Unix (horodatage).
inline double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/// Copie de `text` en minuscules ASCII.
inline std::string lowered(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

/// Arrondit `value` à `digits` décimales.
inline double round_to(double value, int digits) {
  const double factor = std::pow(10.0, digits);
  return std::round(value * factor) / factor;
}

}  // namespace matvision
