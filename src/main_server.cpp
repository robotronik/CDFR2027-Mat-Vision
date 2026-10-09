/// Serveur du mat de vision : API REST + détection ArUco.
///
/// Exemples
/// --------
///    matvision-server                       # API sur 0.0.0.0:5000
///    matvision-server --autostart           # détection lancée immédiatement
///    matvision-server --display             # + fenêtre locale d'aperçu
///    matvision-server --device 1 --width 3840 --height 2160
///    curl http://<ip-lattepanda>:5000/objects
///
/// Arrêt : Ctrl+C, `kill <pid>` (SIGTERM) ou `curl -X POST .../shutdown`.

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

#include <pthread.h>

#include "matvision/api.hpp"
#include "matvision/config.hpp"
#include "matvision/logger.hpp"
#include "matvision/vision.hpp"

namespace {

struct Args {
  std::string config = "config/default.json";
  std::string host;
  int port = 0;
  std::string device;
  int width = 0;
  int height = 0;
  bool autostart = false;
  bool display = false;
  bool no_cors = false;
  bool verbose = false;
  bool help = false;
};

void print_usage() {
  std::cout << "Usage : matvision-server [options]\n"
               "  --config FILE      fichier de configuration JSON (défaut config/default.json)\n"
               "  --host ADDR        adresse d'écoute de l'API\n"
               "  --port N           port de l'API\n"
               "  --device D         index ou chemin de la caméra\n"
               "  --width N          largeur d'image demandée\n"
               "  --height N         hauteur d'image demandée\n"
               "  --autostart        démarre la détection sans attendre /start\n"
               "  --display          affiche la fenêtre d'aperçu locale\n"
               "  --no-cors          désactive l'en-tête CORS\n"
               "  -v, --verbose      journalisation détaillée\n"
               "  -h, --help         affiche cette aide\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  auto require_value = [&](int& index) -> std::string {
    if (index + 1 >= argc) {
      std::cerr << "option " << argv[index] << " : valeur manquante\n";
      std::exit(2);
    }
    return argv[++index];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      args.config = require_value(i);
    } else if (arg == "--host") {
      args.host = require_value(i);
    } else if (arg == "--port") {
      args.port = std::stoi(require_value(i));
    } else if (arg == "--device") {
      args.device = require_value(i);
    } else if (arg == "--width") {
      args.width = std::stoi(require_value(i));
    } else if (arg == "--height") {
      args.height = std::stoi(require_value(i));
    } else if (arg == "--autostart") {
      args.autostart = true;
    } else if (arg == "--display") {
      args.display = true;
    } else if (arg == "--no-cors") {
      args.no_cors = true;
    } else if (arg == "--verbose" || arg == "-v") {
      args.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      args.help = true;
    } else {
      std::cerr << "option inconnue : " << arg << '\n';
      std::exit(2);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (args.help) {
    print_usage();
    return 0;
  }
  matvision::set_log_level(args.verbose ? matvision::LogLevel::Debug : matvision::LogLevel::Info);

  // Les signaux sont bloqués puis attendus par `sigwait` (thread principal) :
  // l'arrêt reste sûr, y compris déclenché par POST /shutdown.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  sigaddset(&signals, SIGUSR1);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  const pthread_t main_thread = pthread_self();

  const matvision::Config config_file = matvision::load_config(args.config);
  matvision::Config config = config_file;
  if (!args.host.empty()) {
    config.api_host = args.host;
  }
  if (args.port > 0) {
    config.api_port = args.port;
  }
  if (!args.device.empty()) {
    config.camera.device = args.device;
  }
  if (args.width > 0) {
    config.camera.width = args.width;
  }
  if (args.height > 0) {
    config.camera.height = args.height;
  }

  MV_LOGI("configuration : " << args.config << " (" << config.objects.size() << " objets, "
                             << config.table.markers.size() << " tags de coin, dessin="
                             << (config.detection.draw ? "oui" : "non") << ")");
  for (const auto& warning : config.validate()) {
    MV_LOGW("configuration : " << warning);
  }

  auto engine = std::make_unique<matvision::VisionEngine>(config, args.display);
  engine->start();
  if (args.autostart) {
    engine->start_detection();
    MV_LOGI("détection lancée automatiquement (sans attendre /start)");
  }

  matvision::ApiOptions options;
  options.cors = !args.no_cors;
  options.config_path = args.config;
  options.on_shutdown = [main_thread]() { pthread_kill(main_thread, SIGUSR1); };
  auto server = matvision::create_api(*engine, options);

  std::thread server_thread([&server, &config]() {
    if (!server->listen(config.api_host, config.api_port)) {
      MV_LOGE("impossible d'écouter sur " << config.api_host << ':' << config.api_port);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  if (!server->is_running()) {
    server_thread.join();
    MV_LOGE("le serveur n'a pas démarré");
    return 1;
  }

  MV_LOGI("API disponible sur http://" << config.api_host << ':' << config.api_port << '/');
  MV_LOGI("Interface web     : http://" << config.api_host << ':' << config.api_port << "/ui");
  MV_LOGI("Arrêt             : Ctrl+C, kill " << ::getpid()
                                              << " ou curl -X POST http://" << config.api_host
                                              << ':' << config.api_port << "/shutdown");

  int signal_number = 0;
  sigwait(&signals, &signal_number);
  MV_LOGI("arrêt demandé (signal " << signal_number << ")");

  server->stop();
  engine->shutdown();
  server_thread.join();
  MV_LOGI("serveur arrêté");
  return 0;
}
