#pragma once

/// Analyse d'arguments commune aux exécutables.

#include <cstdlib>
#include <iostream>
#include <string>

namespace matvision::cli {

/// Valeur qui suit l'option courante, ou arrêt avec le code 2.
inline std::string require_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    std::cerr << "option " << argv[index] << " : valeur manquante\n";
    std::exit(2);
  }
  return argv[++index];
}

inline int to_int(const std::string& option, const std::string& text) {
  try {
    return std::stoi(text);
  } catch (const std::exception&) {
    std::cerr << "option " << option << " : entier invalide (" << text << ")\n";
    std::exit(2);
  }
}

inline double to_double(const std::string& option, const std::string& text) {
  try {
    return std::stod(text);
  } catch (const std::exception&) {
    std::cerr << "option " << option << " : nombre invalide (" << text << ")\n";
    std::exit(2);
  }
}

}  // namespace matvision::cli
