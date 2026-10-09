/// Calibration automatique de la caméra du mat de vision.
///
/// Deux usages :
///   * intrinsèques (défaut) — présentez un damier : le script capture
///     automatiquement les vues exploitables, calcule la matrice intrinsèque et
///     les coefficients de distorsion, puis les enregistre ;
///   * vérification du repère table (`--check-table`) — contrôle que les 4 tags
///     de coin sont vus et affiche la pose de la caméra (solvePnP).

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "matvision/aruco.hpp"
#include "matvision/calibration.hpp"
#include "matvision/camera.hpp"
#include "matvision/config.hpp"
#include "matvision/logger.hpp"
#include "matvision/table.hpp"

namespace {

struct Args {
  std::string config = "config/default.json";
  std::string device;
  int width = 0;
  int height = 0;
  int frames = 0;
  int grid_cols = 0;
  int grid_rows = 0;
  double square_size = 0.0;
  std::string output;
  double timeout = 0.0;
  bool no_display = false;
  bool check_table = false;
  double seconds = 0.0;
  bool verbose = false;
  bool help = false;
};

void print_usage() {
  std::cout << "Usage : matvision-calibrate [options]\n"
               "  --config FILE       fichier de configuration JSON\n"
               "  --device D          index ou chemin du périphérique vidéo\n"
               "  --width N / --height N  résolution demandée\n"
               "  --frames N          nombre de vues à capturer\n"
               "  --grid COLS ROWS    nombre de coins internes du damier\n"
               "  --square-size MM    taille d'une case du damier\n"
               "  --output FILE       fichier de sortie des intrinsèques\n"
               "  --timeout S         durée maximale de la session\n"
               "  --no-display        mode sans fenêtre (headless)\n"
               "  --check-table       vérifie le repère table au lieu de calibrer\n"
               "  --seconds S         durée d'observation pour --check-table\n"
               "  -v, --verbose       journalisation détaillée\n"
               "  -h, --help          affiche cette aide\n";
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
    } else if (arg == "--device") {
      args.device = require_value(i);
    } else if (arg == "--width") {
      args.width = std::stoi(require_value(i));
    } else if (arg == "--height") {
      args.height = std::stoi(require_value(i));
    } else if (arg == "--frames") {
      args.frames = std::stoi(require_value(i));
    } else if (arg == "--grid") {
      args.grid_cols = std::stoi(require_value(i));
      args.grid_rows = std::stoi(require_value(i));
    } else if (arg == "--square-size") {
      args.square_size = std::stod(require_value(i));
    } else if (arg == "--output") {
      args.output = require_value(i);
    } else if (arg == "--timeout") {
      args.timeout = std::stod(require_value(i));
    } else if (arg == "--seconds") {
      args.seconds = std::stod(require_value(i));
    } else if (arg == "--no-display") {
      args.no_display = true;
    } else if (arg == "--check-table") {
      args.check_table = true;
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

void apply_overrides(matvision::Config& config, const Args& args) {
  if (!args.device.empty()) {
    config.camera.device = args.device;
  }
  if (args.width > 0) {
    config.camera.width = args.width;
  }
  if (args.height > 0) {
    config.camera.height = args.height;
  }
  if (args.frames > 0) {
    config.calibration.target_frames = args.frames;
  }
  if (args.grid_cols > 0 && args.grid_rows > 0) {
    config.calibration.chessboard_cols = args.grid_cols;
    config.calibration.chessboard_rows = args.grid_rows;
  }
  if (args.square_size > 0.0) {
    config.calibration.square_size_mm = args.square_size;
  }
  if (!args.output.empty()) {
    config.calibration.intrinsics_file = args.output;
  }
  if (args.timeout > 0.0) {
    config.calibration.max_capture_seconds = args.timeout;
  }
}

int run_intrinsics(matvision::Config& config, bool display) {
  matvision::AutoCalibrator calibrator(config.calibration);
  calibrator.start();

  std::cout << "Calibration automatique : " << config.calibration.chessboard_cols << 'x'
            << config.calibration.chessboard_rows << " coins internes, case de "
            << config.calibration.square_size_mm << " mm, " << config.calibration.target_frames
            << " vues à capturer.\n"
            << "Présentez le damier sous différents angles et distances.\n";
  if (display) {
    std::cout << "Appuyez sur 'q' pour interrompre.\n";
  }

  matvision::Camera camera(config.camera);
  try {
    camera.open();
  } catch (const matvision::CameraError& exc) {
    MV_LOGE(exc.what());
    return 2;
  }

  std::string last_message;
  while (calibrator.running()) {
    cv::Mat frame;
    if (!camera.read_latest(frame, 2)) {
      MV_LOGW("image non reçue...");
      continue;
    }
    calibrator.process(frame);
    const std::string message = calibrator.state().message;
    if (message != last_message) {
      last_message = message;
      std::cout << "  [" << calibrator.state().captured << '/' << calibrator.state().target
                << "] " << message << '\n';
    }
    if (display) {
      cv::imshow("Calibration - matvision", calibrator.draw_overlay(frame));
      if ((cv::waitKey(1) & 0xFF) == 'q') {
        calibrator.cancel();
        break;
      }
    }
  }
  camera.close();
  if (display) {
    cv::destroyAllWindows();
  }

  const auto intrinsics = calibrator.state().intrinsics;
  if (!intrinsics) {
    std::cout << "\nÉchec de la calibration : " << calibrator.state().message << '\n';
    return 1;
  }
  matvision::Intrinsics saved = *intrinsics;
  saved.source = config.intrinsics_path();
  if (!save_intrinsics(config.intrinsics_path(), saved)) {
    std::cout << "\nImpossible d'enregistrer la calibration dans " << config.intrinsics_path()
              << '\n';
    return 1;
  }
  std::cout << "\nCalibration terminée\n"
            << "  vues utilisées  : " << saved.frames << '\n'
            << "  erreur moyenne  : " << saved.rms << " px\n"
            << "  fx, fy          : " << saved.fx() << ", " << saved.fy() << '\n'
            << "  taille image    : " << saved.image_size.width << 'x' << saved.image_size.height
            << '\n'
            << "  enregistré dans : " << config.intrinsics_path() << '\n';
  if (saved.rms > 1.0) {
    std::cout << "  ⚠ erreur élevée : multipliez et variez les vues (angles, distances).\n";
  }
  return 0;
}

int run_check_table(matvision::Config& config, bool display, double seconds) {
  matvision::ArucoDetector detector(config.aruco.dictionary, config.aruco.params);
  const auto intrinsics = matvision::load_intrinsics(config.intrinsics_path());
  if (!intrinsics) {
    std::cout << "(pas de calibration intrinsèque : la pose caméra ne sera pas calculée)\n";
  }

  matvision::Camera camera(config.camera);
  try {
    camera.open();
  } catch (const matvision::CameraError& exc) {
    MV_LOGE(exc.what());
    return 2;
  }

  std::cout << "Vérification du repère table - 'q' pour quitter.\n";
  std::vector<int> corner_ids = config.table.marker_ids();
  const auto started = std::chrono::steady_clock::now();
  int exit_code = 1;
  bool running = true;
  while (running) {
    cv::Mat frame;
    if (!camera.read_latest(frame, 2)) {
      continue;
    }
    const matvision::MarkerMap markers = detector.detect_by_id(frame);
    matvision::MarkerMap corners_found;
    for (const auto& [id, corners] : markers) {
      if (std::find(corner_ids.begin(), corner_ids.end(), id) != corner_ids.end()) {
        corners_found[id] = corners;
      }
    }
    const matvision::TableLocalization localization = matvision::localize_table(
        corners_found, config.table, intrinsics ? &intrinsics.value() : nullptr);

    std::vector<int> missing;
    for (const int id : corner_ids) {
      if (corners_found.find(id) == corners_found.end()) {
        missing.push_back(id);
      }
    }
    const cv::Scalar color = localization.ok ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
    std::string text = "tags " + std::to_string(corners_found.size()) + "/" +
                       std::to_string(corner_ids.size());
    if (!missing.empty()) {
      text += " | manquants:";
      for (const int id : missing) {
        text += " " + std::to_string(id);
      }
    }
    if (localization.ok) {
      text += " | residu " + std::to_string(static_cast<int>(localization.residual_px)) + " px";
    } else {
      text += " | " + localization.reason;
    }
    cv::putText(frame, text, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv::LINE_AA);
    matvision::draw_markers(frame, corners_found, cv::Scalar(255, 128, 0));
    matvision::MarkerMap others;
    for (const auto& [id, corners] : markers) {
      if (std::find(corner_ids.begin(), corner_ids.end(), id) == corner_ids.end()) {
        others[id] = corners;
      }
    }
    matvision::draw_markers(frame, others, cv::Scalar(0, 220, 255));

    if (localization.has_camera_position) {
      const matvision::Position& cam = localization.camera_position;
      std::string pose = "camera x=" + std::to_string(static_cast<int>(cam.x)) +
                         " y=" + std::to_string(static_cast<int>(cam.y)) +
                         " z=" + std::to_string(static_cast<int>(cam.z)) +
                         " a=" + std::to_string(static_cast<int>(cam.a));
      cv::putText(frame, pose, cv::Point(10, 54), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

    if (display) {
      cv::imshow("Vérification repère table - matvision", frame);
      if ((cv::waitKey(1) & 0xFF) == 'q') {
        running = false;
      }
    } else if (seconds > 0.0) {
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      if (elapsed >= seconds) {
        running = false;
      }
    } else {
      running = false;  // une seule mesure en mode headless sans durée
    }

    if (localization.ok) {
      exit_code = 0;
    }
  }
  camera.close();
  if (display) {
    cv::destroyAllWindows();
  }

  if (exit_code == 0) {
    std::cout << "Repère table OK : les tags de coin sont vus et la pose caméra est stable.\n";
  } else {
    std::cout << "Repère table incomplet : vérifiez les identifiants et la visibilité des "
                 "tags.\n";
  }
  return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (args.help) {
    print_usage();
    return 0;
  }
  matvision::set_log_level(args.verbose ? matvision::LogLevel::Debug : matvision::LogLevel::Info);

  matvision::Config config = matvision::load_config(args.config);
  apply_overrides(config, args);
  for (const auto& warning : config.validate()) {
    MV_LOGW("configuration : " << warning);
  }

  const bool display = !args.no_display;
  if (args.check_table) {
    return run_check_table(config, display, args.seconds);
  }
  return run_intrinsics(config, display);
}
