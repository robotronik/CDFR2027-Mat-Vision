#include "matvision/api.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

#include "matvision/fleet.hpp"
#include "matvision/logger.hpp"
#include "matvision/util.hpp"
#include "matvision/version.hpp"

namespace matvision {

namespace fs = std::filesystem;

namespace {

/// Délai avant l'arrêt effectif, pour laisser la réponse HTTP partir.
constexpr double kShutdownDelayS = 0.3;

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
  const std::string web_dir = options.web_dir.empty() ? web_root() : options.web_dir;
  const std::string config_path = options.config_path;
  auto fleet = std::make_shared<Fleet>(engine);
  auto endpoints = std::make_shared<nlohmann::json>(nlohmann::json::object());

  enum class Methods { Get, Post, Both };
  auto route = [server_ptr = server.get(), endpoints](const std::string& pattern,
                                                      Methods methods,
                                                      const std::string& description,
                                                      const httplib::Server::Handler& handler) {
    std::string label;
    if (methods != Methods::Post) {
      server_ptr->Get(pattern, handler);
      label = "GET";
    }
    if (methods != Methods::Get) {
      server_ptr->Post(pattern, handler);
      label += label.empty() ? "POST" : "|POST";
    }
    (*endpoints)[label + " " + pattern] = description;
  };

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

  auto discovery = [endpoints]() {
    return nlohmann::json{{"name", "matvision"},
                          {"version", MATVISION_VERSION},
                          {"description",
                           "Mat de vision ArUco (LattePanda Delta + Logitech 4K Stream)"},
                          {"ui", "/ui"},
                          {"endpoints", *endpoints}};
  };

  route("/api", Methods::Get, "découverte JSON (version + routes)",
        [json_ok, discovery](const httplib::Request&, httplib::Response& res) {
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

  route("/", Methods::Get, "interface web (navigateur) ou découverte JSON (curl)",
        [json_ok, discovery, page](const httplib::Request& req, httplib::Response& res) {
          const std::string accept = req.get_header_value("Accept");
          if (accept.find("text/html") != std::string::npos) {
            res.set_content(page(), "text/html");
            res.set_header("Cache-Control", "no-store");
            return;
          }
          json_ok(res, discovery());
        });

  route("/ui", Methods::Get, "interface web de pilotage",
        [page](const httplib::Request&, httplib::Response& res) {
          res.set_content(page(), "text/html");
          res.set_header("Cache-Control", "no-store");
        });

  route("/health", Methods::Get, "test de vie",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          const nlohmann::json status = engine.status();
          json_ok(res, {{"ok", status.value("running", false) &&
                                 status["camera"].value("opened", false)},
                        {"mode", status.value("mode", std::string())},
                        {"uptime_s", status.value("uptime_s", 0.0)},
                        {"error", status.value("error", std::string())}});
        });

  route("/status", Methods::Get, "état complet du moteur",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, engine.status());
        });

  route("/config", Methods::Get, "configuration effective",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, engine.config_json());
        });

  route("/table", Methods::Get, "géométrie de la table et tags de coin",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, engine.table_info());
        });

  route("/position", Methods::Get, "pose de la caméra dans le repère table",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, engine.camera_position());
        });

  auto command = [&engine, json_ok](nlohmann::json (VisionEngine::*method)()) {
    return [&engine, json_ok, method](const httplib::Request&, httplib::Response& res) {
      json_ok(res, (engine.*method)());
    };
  };
  route("/start", Methods::Both, "démarre la détection", command(&VisionEngine::start_detection));
  route("/stop", Methods::Both, "arrête la détection", command(&VisionEngine::stop));

  auto reset_handler = [&engine, fleet, json_ok](const httplib::Request&,
                                                 httplib::Response& res) {
    fleet->clear();
    json_ok(res, engine.reset());
  };
  route("/reset", Methods::Both, "réinitialise le suivi", reset_handler);
  route("/reset_tracking", Methods::Both, "alias de /reset", reset_handler);

  route("/objects", Methods::Get, "objets détectés (repère table, mm)",
        [&engine, json_ok](const httplib::Request& req, httplib::Response& res) {
          const std::string label = req.has_param("label") ? req.get_param_value("label") : "";
          json_ok(res, engine.objects(label));
        });

  route(R"(/objects/([^/]+))", Methods::Get, "objets d'un tag ou d'un libellé",
        [&engine, json_response](const httplib::Request& req, httplib::Response& res) {
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
            json_response(res, {{"message", "aucun objet pour '" + key + "'"}}, 404);
            return;
          }
          json_response(res, {{"count", matches.size()}, {"objects", matches}}, 200);
        });

  route("/fleet", Methods::Get, "état des robots (principal, chasseur, essaim)",
        [fleet, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, fleet->status());
        });
  route("/fleet/strategies", Methods::Get, "stratégies disponibles par robot",
        [fleet, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, fleet->strategies());
        });
  route("/fleet/live", Methods::Get, "positions et trajectoires pour la table live",
        [fleet, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, fleet->live());
        });

  auto strategy_handler = [fleet, fleet_call](const httplib::Request& req,
                                              httplib::Response& res) {
    const std::string target = req.matches[1].str();
    const nlohmann::json body = body_or_params(req);
    std::string name = body.value("strat", std::string());
    if (name.empty()) {
      name = body.value("strategy", std::string());
    }
    if (name.empty()) {
      fleet_call(res, []() -> nlohmann::json { throw FleetError("stratégie manquante", 400); });
      return;
    }
    fleet_call(res, [fleet, target, name]() { return fleet->set_strategy(target, name); });
  };
  route(R"(/fleet/(.+)/strategy)", Methods::Both,
        "change la stratégie (main|hunter|swarm|swarm/<i>)", strategy_handler);

  route(R"(/fleet/(.+)/color)", Methods::Post, "change la couleur (mêmes cibles)",
        [fleet, fleet_call](const httplib::Request& req, httplib::Response& res) {
          const std::string target = req.matches[1].str();
          const nlohmann::json body = body_or_params(req);
          if (!body.contains("color")) {
            fleet_call(res, []() -> nlohmann::json { throw FleetError("couleur manquante", 400); });
            return;
          }
          const nlohmann::json color = body["color"];
          fleet_call(res, [fleet, target, color]() { return fleet->set_color(target, color); });
        });

  auto set_host_handler = [fleet, config_path, json_response](const httplib::Request& req,
                                                              httplib::Response& res) {
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
      result["saved"] =
          !config_path.empty() && set_robot_host_in_config(config_path, key, host, applied);
      json_response(res, result, 200);
    } catch (const FleetError& exc) {
      json_response(res, {{"message", exc.what()}}, exc.status());
    }
  };
  route(R"(/fleet/(.+)/host)", Methods::Both, "change l'adresse d'un robot (host, port)",
        set_host_handler);

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
  route("/fleet/report", Methods::Both,
        "position déclarée d'un robot sans tag + objets de jeu", report_handler);

  auto preview_handler = [&engine, json_response](const httplib::Request& req,
                                                  httplib::Response& res) {
    const int quality = req.has_param("quality") ? std::stoi(req.get_param_value("quality")) : -1;
    const auto data = engine.preview_jpeg(quality);
    if (!data) {
      json_response(res, {{"message", "aucune image disponible"}}, 503);
      return;
    }
    res.status = 200;
    res.set_content(reinterpret_cast<const char*>(data->data()), data->size(), "image/jpeg");
    res.set_header("Cache-Control", "no-store, max-age=0");
  };
  route("/preview", Methods::Get, "image annotée (JPEG)", preview_handler);
  route("/preview.jpg", Methods::Get, "alias de /preview", preview_handler);

  route("/stream", Methods::Get, "flux MJPEG temps réel",
        [&engine](const httplib::Request& req, httplib::Response& res) {
          const int quality =
              req.has_param("quality") ? std::stoi(req.get_param_value("quality")) : -1;
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

  route("/snapshot", Methods::Post, "enregistre l'image courante sur le disque",
        [&engine, json_response](const httplib::Request& req, httplib::Response& res) {
          std::string target =
              req.has_param("path") ? req.get_param_value("path") : std::string();
          fs::create_directories(data_dir());
          if (target.empty()) {
            target = (fs::path(data_dir()) /
                      ("snapshot_" + std::to_string(static_cast<long long>(steady_seconds())) +
                       ".jpg"))
                         .string();
          }
          if (!engine.save_frame(target)) {
            json_response(res, {{"message", "aucune image à enregistrer"}}, 503);
            return;
          }
          json_response(res, {{"message", "image enregistrée"}, {"path", target}}, 200);
        });

  route("/calibration/start", Methods::Both, "calibration automatique (damier)",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, engine.start_calibration());
        });
  route("/calibration/status", Methods::Get, "progression de la calibration",
        [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
          json_ok(res, engine.calibration_status());
        });
  auto calibration_stop = [&engine, json_ok](const httplib::Request&, httplib::Response& res) {
    json_ok(res, engine.cancel_calibration());
  };
  route("/calibration/stop", Methods::Both, "annule la calibration", calibration_stop);
  route("/calibration/cancel", Methods::Both, "alias de /calibration/stop", calibration_stop);
  route("/calibration/result", Methods::Get, "intrinsèques courantes",
        [&engine, json_response](const httplib::Request&, httplib::Response& res) {
          engine.reload_intrinsics();
          const nlohmann::json status = engine.status();
          if (status["intrinsics"].is_null()) {
            json_response(res,
                          {{"message", "aucune calibration disponible"},
                           {"file", engine.intrinsics_path()}},
                          404);
            return;
          }
          json_response(res, status["intrinsics"], 200);
        });

  const auto on_shutdown = options.on_shutdown;
  route("/shutdown", Methods::Post, "arrête le serveur et le processus",
        [json_response, on_shutdown](const httplib::Request&, httplib::Response& res) {
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

  server->set_mount_point("/static", (fs::path(web_dir) / "static").string());

  server->set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                    std::exception_ptr error) {
    try {
      std::rethrow_exception(error);
    } catch (const std::exception& exc) {
      MV_LOGE("exception dans un handler API : " << exc.what());
    } catch (...) {
      MV_LOGE("exception inconnue dans un handler API");
    }
    res.status = 500;
  });

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
