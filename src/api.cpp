#include "matvision/api.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

#include "matvision/fleet.hpp"
#include "matvision/logger.hpp"
#include "matvision/version.hpp"

namespace matvision {

namespace fs = std::filesystem;

namespace {

/// Délai avant l'arrêt effectif, pour laisser la réponse HTTP partir.
constexpr double kShutdownDelayS = 0.3;

double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

nlohmann::json parse_body(const httplib::Request& req) {
  if (req.body.empty()) {
    return nullptr;
  }
  try {
    return nlohmann::json::parse(req.body);
  } catch (const std::exception&) {
    return nullptr;
  }
}

/// Corps JSON de la requête, ou paramètres d'URL en repli.
nlohmann::json body_or_params(const httplib::Request& req) {
  nlohmann::json body = parse_body(req);
  if (body.is_object()) {
    return body;
  }
  nlohmann::json params = nlohmann::json::object();
  for (const auto& [key, value] : req.params) {
    params[key] = value;
  }
  return params;
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

}  // namespace

std::unique_ptr<httplib::Server> create_api(VisionEngine& engine, const ApiOptions& options) {
  auto server = std::make_unique<httplib::Server>();
  const Config& config = engine.config();
  const std::string web_dir = options.web_dir.empty() ? web_root() : options.web_dir;
  const std::string config_path = options.config_path;
  auto fleet = std::make_shared<Fleet>(config.robots, engine);

  if (options.cors) {
    server->set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
      res.set_header("Access-Control-Allow-Origin", "*");
      res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
    server->Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
      res.status = 204;
    });
  }

  auto json_response = [](httplib::Response& res, const nlohmann::json& payload, int status) {
    res.status = status;
    res.set_content(payload.dump(), "application/json");
    res.set_header("Cache-Control", "no-store");
  };
  auto json_ok = [json_response](httplib::Response& res, const nlohmann::json& payload) {
    json_response(res, payload, 200);
  };
  auto fleet_call = [json_response](httplib::Response& res,
                                    const std::function<nlohmann::json()>& action) {
    try {
      json_response(res, action(), 200);
    } catch (const FleetError& exc) {
      json_response(res, {{"message", exc.what()}}, exc.status());
    }
  };

  auto discovery = [web_dir]() {
    nlohmann::json endpoints = {
        {"GET /health", "test de vie"},
        {"GET /status", "état complet du moteur"},
        {"GET|POST /start", "démarre la détection"},
        {"GET|POST /stop", "arrête la détection"},
        {"GET|POST /reset", "réinitialise le relevé"},
        {"GET /objects", "objets détectés (repère table, mm)"},
        {"GET /objects/<clé>", "objets d'un tag ou d'un libellé"},
        {"GET /position", "pose de la caméra dans le repère table"},
        {"GET /table", "géométrie de la table et tags de coin"},
        {"GET /config", "configuration effective"},
        {"POST /calibration/start", "calibration automatique (damier)"},
        {"GET /calibration/status", "progression de la calibration"},
        {"POST /calibration/stop", "annule la calibration"},
        {"GET /calibration/result", "intrinsèques courantes"},
        {"POST /snapshot", "enregistre l'image courante sur le disque"},
        {"POST /shutdown", "arrête le serveur et le processus"},
        {"GET /fleet", "état des robots (principal, chasseur, essaim)"},
        {"GET /fleet/strategies", "stratégies disponibles par robot"},
        {"POST /fleet/<cible>/strategy", "change la stratégie (main|hunter|swarm|swarm/<i>)"},
        {"POST /fleet/<cible>/color", "change la couleur (mêmes cibles)"},
        {"GET|POST /fleet/<cible>/host", "change l'adresse d'un robot (host, port)"},
        {"GET /fleet/live", "positions et trajectoires pour la table live"},
        {"POST /fleet/report", "position déclarée d'un robot sans tag + objets de jeu"},
        {"GET /", "interface web (navigateur) ou ce JSON (curl)"},
        {"GET /ui", "interface web de pilotage"},
        {"GET /preview", "image annotée (JPEG)"},
        {"GET /stream", "flux MJPEG temps réel"}};
    return nlohmann::json{{"name", "matvision"},
                          {"version", MATVISION_VERSION},
                          {"description",
                           "Mat de vision ArUco (LattePanda Delta + Logitech 4K Stream)"},
                          {"ui", "/ui"},
                          {"endpoints", endpoints}};
  };

  server->Get("/api", [json_ok, discovery](const httplib::Request&, httplib::Response& res) {
    json_ok(res, discovery());
  });

  auto page = [web_dir]() {
    std::string html = read_file((fs::path(web_dir) / "index.html").string());
    const std::string token = "__MATVISION_VERSION__";
    const std::string version = MATVISION_VERSION;
    std::size_t position = html.find(token);
    while (position != std::string::npos) {
      html.replace(position, token.size(), version);
      position = html.find(token, position + version.size());
    }
    return html;
  };

  server->Get("/", [json_ok, discovery, page](const httplib::Request& req,
                                              httplib::Response& res) {
    const std::string accept = req.get_header_value("Accept");
    if (accept.find("text/html") != std::string::npos) {
      res.set_content(page(), "text/html");
      res.set_header("Cache-Control", "no-store");
      return;
    }
    json_ok(res, discovery());
  });

  server->Get("/ui", [page](const httplib::Request&, httplib::Response& res) {
    res.set_content(page(), "text/html");
    res.set_header("Cache-Control", "no-store");
  });

  server->Get("/health", [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
    const nlohmann::json status = engine.status();
    json_ok(res, {{"ok", status.value("running", false) &&
                             status["camera"].value("opened", false)},
                  {"mode", status.value("mode", std::string())},
                  {"uptime_s", status.value("uptime_s", 0.0)},
                  {"error", status.value("error", std::string())}});
  });

  server->Get("/status", [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, engine.status());
  });

  server->Get("/config", [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, engine.config().to_json());
  });

  server->Get("/table", [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, engine.table_info());
  });

  server->Get("/position", [&engine, json_ok](const httplib::Request&,
                                              httplib::Response& res) {
    json_ok(res, engine.camera_position());
  });

  auto command_handler = [&engine, json_ok](const httplib::Request&, httplib::Response& res,
                                            nlohmann::json (VisionEngine::*method)()) {
    json_ok(res, (engine.*method)());
  };
  server->Get("/start", [command_handler](const httplib::Request& req,
                                          httplib::Response& res) {
    command_handler(req, res, &VisionEngine::start_detection);
  });
  server->Post("/start", [command_handler](const httplib::Request& req,
                                           httplib::Response& res) {
    command_handler(req, res, &VisionEngine::start_detection);
  });
  server->Get("/stop",
              [command_handler](const httplib::Request& req, httplib::Response& res) {
                command_handler(req, res, &VisionEngine::stop);
              });
  server->Post("/stop",
               [command_handler](const httplib::Request& req, httplib::Response& res) {
                 command_handler(req, res, &VisionEngine::stop);
               });
  server->Get("/reset",
              [command_handler](const httplib::Request& req, httplib::Response& res) {
                command_handler(req, res, &VisionEngine::reset);
              });
  server->Post("/reset",
               [command_handler](const httplib::Request& req, httplib::Response& res) {
                 command_handler(req, res, &VisionEngine::reset);
               });
  server->Get("/reset_tracking",
              [command_handler](const httplib::Request& req, httplib::Response& res) {
                command_handler(req, res, &VisionEngine::reset);
              });
  server->Post("/reset_tracking",
               [command_handler](const httplib::Request& req, httplib::Response& res) {
                 command_handler(req, res, &VisionEngine::reset);
               });

  server->Get("/objects", [&engine, json_ok](const httplib::Request& req,
                                             httplib::Response& res) {
    const std::string label = req.has_param("label") ? req.get_param_value("label") : "";
    json_ok(res, engine.objects(label));
  });

  server->Get(R"(/objects/([^/]+))", [&engine, json_ok](const httplib::Request& req,
                                                        httplib::Response& res) {
    const std::string key = req.matches[1].str();
    const nlohmann::json payload = engine.objects();
    nlohmann::json matches = nlohmann::json::array();
    const nlohmann::json by_label = payload.value("by_label", nlohmann::json::object());
    if (by_label.contains(key) && by_label[key].is_array()) {
      matches = by_label[key];
    } else {
      for (const auto& item : payload.value("objects", nlohmann::json::array())) {
        if (std::to_string(item.value("id", -1)) == key) {
          matches.push_back(item);
        }
      }
    }
    if (matches.empty()) {
      json_ok(res, {{"message", "aucun objet pour '" + key + "'"}});
      res.status = 404;
      return;
    }
    json_ok(res, {{"count", matches.size()}, {"objects", matches}});
  });

  // -- flotte ---------------------------------------------------------
  server->Get("/fleet", [fleet, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, fleet->status());
  });
  server->Get("/fleet/strategies", [fleet, json_ok](const httplib::Request&,
                                                    httplib::Response& res) {
    json_ok(res, fleet->strategies());
  });
  server->Get("/fleet/live", [fleet, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, fleet->live());
  });
  server->Get(R"(/fleet/(.+)/strategy)", [fleet, fleet_call](const httplib::Request& req,
                                                             httplib::Response& res) {
    const std::string target = req.matches[1].str();
    const nlohmann::json body = body_or_params(req);
    std::string name = body.value("strat", std::string());
    if (name.empty()) {
      name = body.value("strategy", std::string());
    }
    if (name.empty()) {
      fleet_call(res, []() -> nlohmann::json {
        throw FleetError("stratégie manquante", 400);
      });
      return;
    }
    fleet_call(res, [fleet, target, name]() { return fleet->set_strategy(target, name); });
  });
  server->Post(R"(/fleet/(.+)/strategy)", [fleet, fleet_call](const httplib::Request& req,
                                                              httplib::Response& res) {
    const std::string target = req.matches[1].str();
    const nlohmann::json body = body_or_params(req);
    std::string name = body.value("strat", std::string());
    if (name.empty()) {
      name = body.value("strategy", std::string());
    }
    if (name.empty()) {
      fleet_call(res, []() -> nlohmann::json {
        throw FleetError("stratégie manquante", 400);
      });
      return;
    }
    fleet_call(res, [fleet, target, name]() { return fleet->set_strategy(target, name); });
  });
  server->Post(R"(/fleet/(.+)/color)", [fleet, fleet_call](const httplib::Request& req,
                                                           httplib::Response& res) {
    const std::string target = req.matches[1].str();
    const nlohmann::json body = body_or_params(req);
    if (!body.contains("color")) {
      fleet_call(res, []() -> nlohmann::json { throw FleetError("couleur manquante", 400); });
      return;
    }
    const nlohmann::json color = body["color"];
    fleet_call(res, [fleet, target, color]() { return fleet->set_color(target, color); });
  });

  // Changement d'adresse d'un robot (persisté dans le fichier de configuration).
  auto set_host_handler = [fleet, &engine, config_path, json_response](
                              const httplib::Request& req, httplib::Response& res) {
    const std::string target = req.matches[1].str();
    const nlohmann::json body = body_or_params(req);
    if (!body.contains("host") || !body["host"].is_string()) {
      json_response(res, {{"message", "adresse (host) manquante"}}, 400);
      return;
    }
    const std::string host = body["host"].get<std::string>();
    int port = 0;
    try {
      if (body.contains("port") && !body["port"].is_null()) {
        port = body["port"].is_string() ? std::stoi(body["port"].get<std::string>())
                                        : body["port"].get<int>();
      }
    } catch (const std::exception&) {
      json_response(res, {{"message", "port invalide"}}, 400);
      return;
    }
    try {
      nlohmann::json result = fleet->set_host(target, host, port);
      const std::string key = result.value("key", std::string());
      const int applied = result.value("port", port);
      engine.update_robot_host(key, host, applied);
      result["saved"] = !config_path.empty() &&
                        set_robot_host_in_config(config_path, key, host, applied);
      json_response(res, result, 200);
    } catch (const FleetError& exc) {
      json_response(res, {{"message", exc.what()}}, exc.status());
    }
  };
  server->Get(R"(/fleet/(.+)/host)", set_host_handler);
  server->Post(R"(/fleet/(.+)/host)", set_host_handler);
  auto report_handler = [fleet, fleet_call](const httplib::Request& req,
                                            httplib::Response& res) {
    const nlohmann::json body = body_or_params(req);
    std::string key = body.value("robot", std::string());
    if (key.empty()) {
      key = body.value("key", std::string());
    }
    if (key.empty()) {
      fleet_call(res, []() -> nlohmann::json { throw FleetError("robot manquant", 400); });
      return;
    }
    if (!body.contains("x") || !body.contains("y")) {
      fleet_call(res, []() -> nlohmann::json {
        throw FleetError("position invalide (x, y requis)", 400);
      });
      return;
    }
    try {
      const double x = body["x"].get<double>();
      const double y = body["y"].get<double>();
      const double a = body.contains("a") ? body["a"].get<double>() : 0.0;
      fleet_call(res, [fleet, key, x, y, a]() { return fleet->report(key, x, y, a); });
    } catch (const nlohmann::json::exception&) {
      fleet_call(res, []() -> nlohmann::json {
        throw FleetError("position invalide (x, y requis)", 400);
      });
    }
  };
  server->Get("/fleet/report", report_handler);
  server->Post("/fleet/report", report_handler);

  // -- aperçu ---------------------------------------------------------
  server->Get("/preview", [&engine, json_response](const httplib::Request& req,
                                                   httplib::Response& res) {
    const int quality = req.has_param("quality") ? std::stoi(req.get_param_value("quality"))
                                                 : -1;
    const auto data = engine.preview_jpeg(quality);
    if (!data) {
      json_response(res, {{"message", "aucune image disponible"}}, 503);
      return;
    }
    res.status = 200;
    res.set_content(reinterpret_cast<const char*>(data->data()), data->size(), "image/jpeg");
    res.set_header("Cache-Control", "no-store, max-age=0");
  });
  server->Get("/preview.jpg", [&engine, json_response](const httplib::Request& req,
                                                       httplib::Response& res) {
    const int quality = req.has_param("quality") ? std::stoi(req.get_param_value("quality"))
                                                 : -1;
    const auto data = engine.preview_jpeg(quality);
    if (!data) {
      json_response(res, {{"message", "aucune image disponible"}}, 503);
      return;
    }
    res.status = 200;
    res.set_content(reinterpret_cast<const char*>(data->data()), data->size(), "image/jpeg");
    res.set_header("Cache-Control", "no-store, max-age=0");
  });

  server->Get("/stream", [&engine](const httplib::Request& req, httplib::Response& res) {
    const int quality = req.has_param("quality") ? std::stoi(req.get_param_value("quality"))
                                                 : -1;
    const double fps = req.has_param("fps") ? std::stod(req.get_param_value("fps")) : 10.0;
    const int max_frames =
        req.has_param("frames") ? std::stoi(req.get_param_value("frames")) : 0;
    const double max_seconds =
        req.has_param("max_seconds") ? std::stod(req.get_param_value("max_seconds")) : 300.0;
    const double interval = 1.0 / std::max(1.0, std::min(fps, 60.0));

    struct StreamState {
      double started = steady_seconds();
      int sent = 0;
    };
    auto state = std::make_shared<StreamState>();

    res.set_chunked_content_provider(
        "multipart/x-mixed-replace; boundary=frame",
        [&engine, quality, interval, max_frames, max_seconds,
         state](std::size_t, httplib::DataSink& sink) -> bool {
          const double elapsed = steady_seconds() - state->started;
          if ((max_frames > 0 && state->sent >= max_frames) ||
              (max_seconds > 0.0 && elapsed > max_seconds)) {
            // Termine proprement le transfert chunked (sinon le client voit une
            // connexion coupée au lieu de la fin du flux).
            sink.done();
            return true;
          }
          const auto data = engine.preview_jpeg(quality);
          if (!data) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return true;
          }
          const std::string header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     std::to_string(data->size()) + "\r\n\r\n";
          if (!sink.write(header.data(), header.size())) {
            return false;
          }
          if (!sink.write(reinterpret_cast<const char*>(data->data()), data->size())) {
            return false;
          }
          if (!sink.write("\r\n", 2)) {
            return false;
          }
          state->sent += 1;
          std::this_thread::sleep_for(std::chrono::duration<double>(interval));
          return true;
        });
  });

  server->Post("/snapshot", [&engine, json_response](const httplib::Request& req,
                                                     httplib::Response& res) {
    std::string target = req.has_param("path") ? req.get_param_value("path") : std::string();
    fs::create_directories(data_dir());
    if (target.empty()) {
      target = (fs::path(data_dir()) /
                ("snapshot_" + std::to_string(static_cast<long long>(steady_seconds())) + ".jpg"))
                   .string();
    }
    if (!engine.save_frame(target)) {
      json_response(res, {{"message", "aucune image à enregistrer"}}, 503);
      return;
    }
    json_response(res, {{"message", "image enregistrée"}, {"path", target}}, 200);
  });

  // -- calibration ----------------------------------------------------
  auto calibration_start = [&engine, json_ok](const httplib::Request&,
                                              httplib::Response& res) {
    json_ok(res, engine.start_calibration());
  };
  server->Get("/calibration/start", calibration_start);
  server->Post("/calibration/start", calibration_start);
  server->Get("/calibration/status", [&engine, json_ok](const httplib::Request&,
                                                        httplib::Response& res) {
    json_ok(res, engine.calibration_status());
  });
  auto calibration_stop = [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, engine.cancel_calibration());
  };
  server->Get("/calibration/stop", calibration_stop);
  server->Post("/calibration/stop", calibration_stop);
  server->Get("/calibration/cancel", calibration_stop);
  server->Post("/calibration/cancel", calibration_stop);
  server->Get("/calibration/result", [&engine, json_response](const httplib::Request&,
                                                              httplib::Response& res) {
    engine.reload_intrinsics();
    const nlohmann::json status = engine.status();
    if (status["intrinsics"].is_null()) {
      json_response(res,
                    {{"message", "aucune calibration disponible"},
                     {"file", engine.config().intrinsics_path()}},
                    404);
      return;
    }
    json_response(res, status["intrinsics"], 200);
  });

  // -- arrêt ----------------------------------------------------------
  auto on_shutdown = options.on_shutdown;
  server->Post("/shutdown", [json_response, on_shutdown](const httplib::Request&,
                                                         httplib::Response& res) {
    if (!on_shutdown) {
      json_response(res, {{"message", "arrêt non câblé sur ce serveur"}}, 501);
      return;
    }
    std::thread([on_shutdown]() {
      std::this_thread::sleep_for(std::chrono::duration<double>(kShutdownDelayS));
      on_shutdown();
    }).detach();
    MV_LOGI("arrêt demandé via POST /shutdown");
    json_response(res, {{"message", "arrêt du serveur en cours"}}, 200);
  });

  // -- fichiers statiques ---------------------------------------------
  server->set_mount_point("/static", (fs::path(web_dir) / "static").string());

  server->set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.status == 404) {
      res.set_content(R"({"message": "route inconnue"})", "application/json");
    } else if (res.body.empty()) {
      res.set_content(R"({"message": "erreur interne"})", "application/json");
    }
  });

  MV_LOGI("API créée (interface web " << web_dir << ", CORS " << (options.cors ? "activé"
                                                                               : "désactivé")
                                      << ")");
  return server;
}

}  // namespace matvision
