#pragma once

/// API REST exposant le mat de vision sur le réseau local (cpp-httplib).
///
/// L'interface web est servie sur `/` (négociation de contenu : un navigateur
/// reçoit la page HTML, `curl` reçoit le JSON de découverte) et sur `/ui`.
/// Le reste répond en JSON, sauf `/preview` (JPEG) et `/stream` (MJPEG).

#include <functional>
#include <memory>
#include <string>

#include <httplib.h>

#include "matvision/config.hpp"
#include "matvision/vision.hpp"

namespace matvision {

struct ApiOptions {
  bool cors = true;
  /// Appelée par `POST /shutdown` pour arrêter le serveur (sans rappel : 501).
  std::function<void()> on_shutdown;
  /// Répertoire des ressources de l'interface web (défaut : `web/`).
  std::string web_dir;
  /// Fichier de configuration à réécrire quand l'adresse d'un robot est changée
  /// depuis l'interface (vide : modification en mémoire seulement).
  std::string config_path;
};

/// Construit le serveur HTTP autour d'un :class:`VisionEngine`.
std::unique_ptr<httplib::Server> create_api(VisionEngine& engine,
                                            const ApiOptions& options = ApiOptions());

}  // namespace matvision
