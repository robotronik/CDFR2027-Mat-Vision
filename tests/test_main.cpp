// Tests d'intégration du mat de vision — fonctionnent sans matériel.
//
// Une scène synthétique est rendue (caméra en visée plongeante au-dessus d'une
// table de 2000x3000 mm, tags aux coins, objets sur le tapis), puis :
//   * détection ArUco et pose caméra (solvePnP, sans homographie) ;
//   * positions des objets en mm/deg par changement de base ;
//   * calibration automatique sur des vues synthétiques de damier ;
//   * API HTTP complète (cpp-httplib) avec une caméra factice ;
//   * flotte de robots pilotée contre un petit serveur HTTP simulé.
//
// Exécution : ./build/matvision-tests

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <httplib.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include "matvision/api.hpp"
#include "matvision/aruco.hpp"
#include "matvision/calibration.hpp"
#include "matvision/config.hpp"
#include "matvision/geometry.hpp"
#include "matvision/table.hpp"
#include "matvision/vision.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                                  \
  do {                                                                                    \
    ++g_checks;                                                                           \
    if (!(condition)) {                                                                   \
      ++g_failures;                                                                       \
      std::cerr << "FAIL " << __LINE__ << " : " << #condition << '\n';                    \
    }                                                                                     \
  } while (0)

#define CHECK_NEAR(value, expected, tolerance)                                            \
  do {                                                                                    \
    ++g_checks;                                                                           \
    const double mv_value = static_cast<double>(value);                                   \
    const double mv_expected = static_cast<double>(expected);                             \
    if (!(std::fabs(mv_value - mv_expected) <= (tolerance))) {                            \
      ++g_failures;                                                                       \
      std::cerr << "FAIL " << __LINE__ << " : " << #value << " = " << mv_value             \
                << " (attendu " << mv_expected << " +/- " << (tolerance) << ")\n";        \
    }                                                                                     \
  } while (0)

constexpr int kWidth = 2560;
constexpr int kHeight = 1440;
constexpr double kFx = 1450.0;
constexpr double kFy = 1450.0;
constexpr int kMarkerRenderPx = 480;

struct SceneObject {
  int id;
  double x;
  double y;
  double angle;
};

const std::vector<SceneObject> kSceneObjects = {
    {1, 300.0, 250.0, 30.0},
    {6, -250.0, -400.0, -75.0},
    {13, 520.0, -700.0, 120.0},
};

cv::Mat camera_matrix() {
  return (cv::Mat_<double>(3, 3) << kFx, 0.0, kWidth / 2.0, 0.0, kFy, kHeight / 2.0, 0.0, 0.0,
          1.0);
}

std::pair<cv::Mat, cv::Mat> look_at(const cv::Vec3d& eye, const cv::Vec3d& target,
                                    const cv::Vec3d& up = cv::Vec3d(0.0, 1.0, 0.0)) {
  cv::Vec3d z = target - eye;
  z /= cv::norm(z);
  cv::Vec3d x = z.cross(up);
  x /= cv::norm(x);
  cv::Vec3d y = z.cross(x);
  cv::Mat rotation = (cv::Mat_<double>(3, 3) << x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1],
                      z[2]);
  const cv::Mat translation = -rotation * (cv::Mat_<double>(3, 1) << eye[0], eye[1], eye[2]);
  return {rotation, translation};
}

std::vector<cv::Point2f> project(const std::vector<cv::Point3d>& points, const cv::Mat& rotation,
                                 const cv::Mat& translation) {
  std::vector<cv::Point2f> out;
  for (const auto& point : points) {
    const cv::Mat world = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
    const cv::Mat camera = rotation * world + translation;
    const double depth = camera.at<double>(2, 0);
    out.emplace_back(static_cast<float>(camera.at<double>(0, 0) / depth * kFx + kWidth / 2.0),
                     static_cast<float>(camera.at<double>(1, 0) / depth * kFy + kHeight / 2.0));
  }
  return out;
}

void render_marker(cv::Mat& canvas, const cv::aruco::Dictionary& dictionary, int marker_id,
                   double marker_size, double x, double y, double angle, const cv::Mat& rotation,
                   const cv::Mat& translation) {
  const auto corners = matvision::marker_corners_table(x, y, angle, marker_size);
  std::vector<cv::Point3d> world;
  for (const auto& corner : corners) {
    world.emplace_back(corner.x, corner.y, 0.0);
  }
  const std::vector<cv::Point2f> image_corners = project(world, rotation, translation);

  cv::Mat marker;
  cv::aruco::generateImageMarker(dictionary, marker_id, kMarkerRenderPx, marker, 1);
  const std::vector<cv::Point2f> source = {{0.0f, 0.0f},
                                           {static_cast<float>(kMarkerRenderPx), 0.0f},
                                           {static_cast<float>(kMarkerRenderPx),
                                            static_cast<float>(kMarkerRenderPx)},
                                           {0.0f, static_cast<float>(kMarkerRenderPx)}};
  const cv::Mat transform = cv::getPerspectiveTransform(source, image_corners);

  cv::Mat layer(canvas.size(), CV_8UC1, cv::Scalar(255));
  cv::warpPerspective(marker, layer, transform, canvas.size(), cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT, cv::Scalar(255));
  cv::min(canvas, layer, canvas);
}

cv::Mat build_scene(const matvision::Config& config) {
  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  const auto pose = look_at(cv::Vec3d(150.0, -120.0, 2600.0), cv::Vec3d(0.0, 0.0, 0.0));
  cv::Mat canvas(kHeight, kWidth, CV_8UC1, cv::Scalar(255));
  for (const auto& marker : config.table.markers) {
    render_marker(canvas, dictionary, marker.id, marker.size, marker.x, marker.y, marker.a,
                  pose.first, pose.second);
  }
  for (const auto& object : kSceneObjects) {
    render_marker(canvas, dictionary, object.id, 100.0, object.x, object.y, object.angle,
                  pose.first, pose.second);
  }
  cv::Mat color;
  cv::cvtColor(canvas, color, cv::COLOR_GRAY2BGR);
  return color;
}

/// Caméra factice renvoyant toujours la même image.
class FakeFrameSource : public matvision::IFrameSource {
 public:
  explicit FakeFrameSource(cv::Mat frame) : frame_(std::move(frame)) {}
  bool open() override {
    open_ = true;
    return true;
  }
  bool read(cv::Mat& frame) override {
    if (!open_) {
      return false;
    }
    frame = frame_.clone();
    return true;
  }
  bool read_latest(cv::Mat& frame, int /*drain*/) override { return read(frame); }
  void close() override { open_ = false; }
  bool is_open() const override { return open_; }
  cv::Size size() const override { return frame_.size(); }
  nlohmann::json info() const override {
    return {{"device", "fake"},     {"opened", open_},    {"width", frame_.cols},
            {"height", frame_.rows}, {"fourcc", "FAKE"},   {"fps", 30.0}};
  }

 private:
  cv::Mat frame_;
  bool open_ = false;
};

std::string temp_dir() {
  const char* buffer = std::getenv("TMPDIR");
  return std::string(buffer != nullptr ? buffer : "/tmp") + "/matvision-tests";
}

matvision::Config make_config(const std::string& dir) {
  matvision::Config config = matvision::load_config(matvision::default_config_path());
  config.camera.width = kWidth;
  config.camera.height = kHeight;
  config.calibration.intrinsics_file = dir + "/camera_calibration.json";
  config.detection.draw = true;
  matvision::Intrinsics intrinsics;
  intrinsics.camera_matrix = camera_matrix();
  intrinsics.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
  intrinsics.image_size = cv::Size(kWidth, kHeight);
  intrinsics.rms = 0.05;
  intrinsics.frames = 20;
  save_intrinsics(config.intrinsics_path(), intrinsics);
  return config;
}

bool wait_for_objects(matvision::VisionEngine& engine, int expected, double timeout_s) {
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
         timeout_s) {
    if (engine.objects()["count"].get<int>() >= expected) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return engine.objects()["count"].get<int>() >= expected;
}

// --------------------------------------------------------------------------- //

void test_geometry() {
  CHECK_NEAR(matvision::normalize_angle_deg(190.0), -170.0, 1e-9);
  CHECK_NEAR(matvision::normalize_angle_deg(-180.0), 180.0, 1e-9);
  CHECK_NEAR(matvision::normalize_angle_deg(0.0), 0.0, 1e-9);
  const auto corners = matvision::marker_corners_table(0.0, 0.0, 0.0, 100.0);
  CHECK_NEAR(corners[0].x, -50.0, 1e-6);
  CHECK_NEAR(corners[0].y, 50.0, 1e-6);
  const auto rotated = matvision::marker_corners_table(0.0, 0.0, 90.0, 100.0);
  CHECK_NEAR(rotated[0].x, -50.0, 1e-6);
  CHECK_NEAR(rotated[0].y, -50.0, 1e-6);
}

void test_table_pose(const matvision::Config& config, const cv::Mat& scene) {
  matvision::ArucoDetector detector(config.aruco.dictionary, config.aruco.params);
  const matvision::MarkerMap markers = detector.detect_by_id(scene);

  matvision::MarkerMap corner_markers;
  const std::vector<int> corner_ids = config.table.marker_ids();
  for (int id : corner_ids) {
    if (markers.count(id) != 0) {
      corner_markers[id] = markers.at(id);
    }
  }
  CHECK(corner_markers.size() == 4);

  const auto intrinsics = matvision::load_intrinsics(config.intrinsics_path());
  CHECK(intrinsics.has_value());
  const matvision::TableLocalization localization =
      matvision::localize_table(corner_markers, config.table, &intrinsics.value());
  CHECK(localization.ok);
  CHECK(localization.used_ids.size() == 4);
  CHECK(localization.residual_px < 2.0);
  CHECK(localization.has_camera_position);
  CHECK_NEAR(localization.camera_position.x, 150.0, 25.0);
  CHECK_NEAR(localization.camera_position.y, -120.0, 25.0);
  CHECK_NEAR(localization.camera_position.z, 2600.0, 40.0);

  // Sans intrinsèques, aucune pose (et donc aucune position) n'est possible.
  const matvision::TableLocalization none =
      matvision::localize_table(corner_markers, config.table, nullptr);
  CHECK(!none.ok);
  CHECK(!none.reason.empty());
}

void test_engine_positions(matvision::Config config, const cv::Mat& scene) {
  auto engine = std::make_unique<matvision::VisionEngine>(
      config, false, std::make_unique<FakeFrameSource>(scene));
  engine->start();
  engine->start_detection();
  CHECK(wait_for_objects(*engine, 3, 5.0));

  const nlohmann::json payload = engine->objects();
  CHECK(payload["count"].get<int>() >= 3);
  CHECK(payload.value("frame", std::string()) == "table");

  for (const auto& expected : kSceneObjects) {
    nlohmann::json found = nullptr;
    for (const auto& item : payload["objects"]) {
      if (item.value("id", -1) == expected.id) {
        found = item;
        break;
      }
    }
    CHECK(!found.is_null());
    if (!found.is_null()) {
      // Précision attendue : ~10 mm en x/y (le z d'un petit tag est plus bruité).
      CHECK_NEAR(found.value("x", 0.0), expected.x, 12.0);
      CHECK_NEAR(found.value("y", 0.0), expected.y, 12.0);
      CHECK_NEAR(found.value("a", 0.0), expected.angle, 6.0);
    }
  }

  CHECK(engine->objects("blue")["count"].get<int>() >= 1);
  CHECK(engine->table_info().value("width_mm", 0.0) == 2000.0);
  CHECK(!engine->camera_position()["position"].is_null());
  engine->shutdown();
}

void test_intrinsics_roundtrip(const std::string& dir) {
  matvision::Intrinsics intrinsics;
  intrinsics.camera_matrix = camera_matrix();
  intrinsics.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
  intrinsics.image_size = cv::Size(kWidth, kHeight);
  intrinsics.rms = 0.123;
  intrinsics.frames = 14;
  const std::string path = dir + "/roundtrip.json";
  CHECK(save_intrinsics(path, intrinsics));

  const auto loaded = matvision::load_intrinsics(path);
  CHECK(loaded.has_value());
  if (loaded) {
    CHECK_NEAR(loaded->fx(), kFx, 1e-6);
    CHECK_NEAR(loaded->rms, 0.123, 1e-9);
    CHECK(loaded->frames == 14);
    CHECK(loaded->image_size.width == kWidth);
    CHECK(matvision::calibration_file_kind(path) == "json");
  }
}

void test_intrinsics_resolution_warning(matvision::Config config, const cv::Mat& scene) {
  matvision::Intrinsics intrinsics;
  intrinsics.camera_matrix = camera_matrix();
  intrinsics.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
  intrinsics.image_size = cv::Size(1920, 1080);  // volontairement différent du flux
  save_intrinsics(config.intrinsics_path(), intrinsics);

  auto engine = std::make_unique<matvision::VisionEngine>(
      config, false, std::make_unique<FakeFrameSource>(scene));
  engine->start();
  engine->start_detection();

  const auto start = std::chrono::steady_clock::now();
  bool warned = false;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 3.0) {
    if (!engine->status().value("intrinsics_warning", std::string()).empty()) {
      warned = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  CHECK(warned);
  engine->shutdown();
}

cv::Mat chessboard_image(int cols, int rows, int cell) {
  cv::Mat board((rows + 1) * cell, (cols + 1) * cell, CV_8UC1, cv::Scalar(255));
  for (int row = 0; row < rows + 1; ++row) {
    for (int col = 0; col < cols + 1; ++col) {
      if ((row + col) % 2 == 0) {
        cv::rectangle(board, cv::Rect(col * cell, row * cell, cell, cell), cv::Scalar(0),
                      cv::FILLED);
      }
    }
  }
  return board;
}

std::vector<cv::Mat> synth_chessboard_views(int views, int cols, int rows, int cell) {
  const cv::Mat board = chessboard_image(cols, rows, cell);
  const double board_width = board.cols;
  const double board_height = board.rows;
  const std::vector<cv::Point2f> source = {{0.0f, 0.0f},
                                           {static_cast<float>(board_width), 0.0f},
                                           {static_cast<float>(board_width),
                                            static_cast<float>(board_height)},
                                           {0.0f, static_cast<float>(board_height)}};
  const std::vector<cv::Point3d> board_corners = {{0.0, 0.0, 0.0},
                                                  {board_width, 0.0, 0.0},
                                                  {board_width, board_height, 0.0},
                                                  {0.0, board_height, 0.0}};
  const cv::Mat camera = camera_matrix();
  std::vector<cv::Mat> out;
  for (int i = 0; i < views; ++i) {
    const double t = views > 1 ? static_cast<double>(i) / (views - 1) : 0.0;
    // Vues réellement hors plan (rotations 3D) : indispensable pour contraindre
    // la focale, une simple rotation dans le plan la laisse indéterminée.
    const double rx = (-18.0 + 36.0 * t) * matvision::kPi / 180.0;
    const double ry = (18.0 - 36.0 * t) * matvision::kPi / 180.0;
    const double rz = (-20.0 + 40.0 * t) * matvision::kPi / 180.0;
    const double tx = (t - 0.5) * 100.0;
    const double ty = (t - 0.5) * 80.0;
    const double tz = 500.0 + 150.0 * (0.5 + 0.5 * std::sin(t * 3.0));

    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx, ry, rz);
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    const cv::Mat translation = (cv::Mat_<double>(3, 1) << tx, ty, tz);

    std::vector<cv::Point2f> target;
    for (const auto& corner : board_corners) {
      const cv::Mat point =
          (cv::Mat_<double>(3, 1) << corner.x - board_width / 2.0,
           corner.y - board_height / 2.0, 0.0);
      const cv::Mat camera_point = rotation * point + translation;
      const double depth = camera_point.at<double>(2, 0);
      target.emplace_back(static_cast<float>(camera_point.at<double>(0, 0) / depth * kFx +
                                             kWidth / 2.0),
                          static_cast<float>(camera_point.at<double>(1, 0) / depth * kFy +
                                             kHeight / 2.0));
    }
    const cv::Mat transform = cv::getPerspectiveTransform(source, target);
    cv::Mat view;
    cv::warpPerspective(board, view, transform, cv::Size(kWidth, kHeight), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(255));
    out.push_back(view);
  }
  return out;
}

void test_automatic_calibration() {
  matvision::CalibrationConfig config;
  config.chessboard_cols = 7;
  config.chessboard_rows = 7;
  config.target_frames = 10;
  config.min_move_ratio = 0.02;
  config.min_sharpness = 30.0;

  const auto views =
      synth_chessboard_views(16, config.chessboard_cols, config.chessboard_rows, 40);
  matvision::AutoCalibrator calibrator(config);
  calibrator.start();
  for (const auto& view : views) {
    calibrator.process(view);
    if (calibrator.finished()) {
      break;
    }
  }
  CHECK(calibrator.state().captured >= 5);
  const auto intrinsics =
      calibrator.finished() ? calibrator.state().intrinsics : calibrator.compute();
  CHECK(intrinsics != nullptr);
  if (intrinsics) {
    CHECK(intrinsics->frames >= 5);
    CHECK(intrinsics->rms < 3.0);
    CHECK_NEAR(intrinsics->fx(), kFx, 0.25 * kFx);
    CHECK(intrinsics->image_size.width == kWidth);
  }
}

/// Petit serveur HTTP simulant l'API embarquée d'un robot.
class FakeRobot {
 public:
  explicit FakeRobot(int port) : port_(port) {
    server_.Get("/get_robot", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"team":1,"strategy":"Match","status":"idle","score":0})",
                      "application/json");
    });
    server_.Get("/get_strategies", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"strategies":["Match","Test"]})", "application/json");
    });
    server_.Post("/set_strat", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"message":"ok"})", "application/json");
    });
    server_.Post("/set_color", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"message":"ok"})", "application/json");
    });
    thread_ = std::thread([this]() { server_.listen("127.0.0.1", port_); });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  }

  ~FakeRobot() {
    server_.stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  int port_;
  httplib::Server server_;
  std::thread thread_;
};

void test_api(matvision::Config config, const cv::Mat& scene, int robot_port) {
  config.robots.main = matvision::RobotConfig{"Principal", "127.0.0.1", robot_port};
  config.robots.hunter = matvision::RobotConfig{"Chasseur", "127.0.0.1", robot_port};

  auto engine = std::make_unique<matvision::VisionEngine>(
      config, false, std::make_unique<FakeFrameSource>(scene));
  engine->start();
  engine->start_detection();
  CHECK(wait_for_objects(*engine, 3, 5.0));

  matvision::ApiOptions options;
  options.web_dir = matvision::web_root();
  auto server = matvision::create_api(*engine, options);
  const int port = 5311;
  std::thread thread([&server]() { server->listen("127.0.0.1", 5311); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CHECK(server->is_running());

  httplib::Client client("http://127.0.0.1:" + std::to_string(port));
  const auto get = [&client](const std::string& path) {
    auto res = client.Get(path);
    return res ? nlohmann::json::parse(res->body) : nlohmann::json(nullptr);
  };

  const auto api = client.Get("/api");
  CHECK(api && api->status == 200 && api->body.find("matvision") != std::string::npos);

  const auto health = get("/health");
  CHECK(health.value("ok", false));
  CHECK(health.value("mode", std::string()) == "detect");

  const auto status = get("/status");
  CHECK(status.value("mode", std::string()) == "detect");
  CHECK(!status["intrinsics"].is_null());

  const auto objects = get("/objects");
  CHECK(objects.value("count", 0) >= 3);
  CHECK(objects.value("frame", std::string()) == "table");

  const auto by_id = get("/objects/1");
  CHECK(by_id.value("count", 0) == 1);
  const auto by_label = get("/objects/blue");
  CHECK(by_label.value("count", 0) >= 1);

  const auto table = get("/table");
  CHECK(table.value("width_mm", 0.0) == 2000.0);
  CHECK(table["markers"].size() == 4);

  const auto position = get("/position");
  CHECK(!position["position"].is_null());

  CHECK(!get("/config").is_null());

  auto ui = client.Get("/ui");
  CHECK(ui && ui->status == 200 &&
        ui->get_header_value("Content-Type").find("text/html") != std::string::npos);
  auto static_file = client.Get("/static/app.js");
  CHECK(static_file && static_file->status == 200);
  auto missing = client.Get("/route-inexistante");
  CHECK(missing && missing->status == 404);

  auto preview = client.Get("/preview");
  CHECK(preview && preview->status == 200 &&
        preview->get_header_value("Content-Type") == "image/jpeg");
  httplib::Client stream_client("http://127.0.0.1:" + std::to_string(port));
  auto stream = stream_client.Get("/stream?frames=1&fps=60");
  CHECK(stream && stream->status == 200 && stream->body.find("--frame") != std::string::npos);

  const auto calibration = get("/calibration/status");
  CHECK(calibration.value("status", std::string()) == "idle");

  // -- flotte ---------------------------------------------------------
  const auto fleet = get("/fleet");
  CHECK(fleet["robots"].size() >= 2);
  CHECK(fleet.value("our_color", std::string()) == "blue");
  CHECK(fleet.value("opponent_color", std::string()) == "yellow");

  const auto strategies = get("/fleet/strategies");
  CHECK(strategies["robots"][0]["strategies"].size() == 2);

  auto set_strategy =
      client.Post("/fleet/main/strategy", R"({"strat":"Match"})", "application/json");
  CHECK(set_strategy && set_strategy->status == 200);
  auto set_color = client.Post("/fleet/all/color", R"({"color":1})", "application/json");
  CHECK(set_color && set_color->status == 200);

  const auto live = get("/fleet/live");
  CHECK(live.value("our_color", std::string()) == "blue");
  CHECK(live["robots"].size() >= 2);

  auto report = client.Post("/fleet/report",
                            R"({"robot":"hunter","x":-320.0,"y":810.0,"a":90.0})",
                            "application/json");
  CHECK(report && report->status == 200);
  if (report) {
    const nlohmann::json payload = nlohmann::json::parse(report->body);
    CHECK(payload.value("robot", std::string()) == "hunter");
    CHECK(payload.contains("objects"));
    CHECK(payload.contains("opponents"));
  }
  auto bad_report =
      client.Post("/fleet/report", R"({"robot":"main","x":0,"y":0})", "application/json");
  CHECK(bad_report && bad_report->status == 400);

  auto shutdown = client.Post("/shutdown", "", "application/json");
  CHECK(shutdown && shutdown->status == 501);

  server->stop();
  thread.join();
  engine->shutdown();
}

void test_shutdown_route(matvision::Config config, const cv::Mat& scene) {
  auto engine = std::make_unique<matvision::VisionEngine>(
      config, false, std::make_unique<FakeFrameSource>(scene));
  engine->start();

  auto stopping = std::make_shared<std::atomic<bool>>(false);
  matvision::ApiOptions options;
  options.on_shutdown = [stopping]() { stopping->store(true); };
  auto server = matvision::create_api(*engine, options);
  std::thread thread([&server]() { server->listen("127.0.0.1", 5313); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  httplib::Client client("http://127.0.0.1:5313");
  auto response = client.Post("/shutdown", "", "application/json");
  CHECK(response && response->status == 200);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  CHECK(stopping->load());

  server->stop();
  thread.join();
  engine->shutdown();
}

}  // namespace

int main() {
  cv::setNumThreads(1);
  const std::string dir = temp_dir();
  std::system(("mkdir -p " + dir).c_str());

  matvision::Config config = make_config(dir);
  const cv::Mat scene = build_scene(config);

  std::cout << "== géométrie ==\n";
  test_geometry();
  std::cout << "== pose caméra (solvePnP) ==\n";
  test_table_pose(config, scene);
  std::cout << "== moteur & positions ==\n";
  test_engine_positions(config, scene);
  std::cout << "== intrinsèques (sérialisation) ==\n";
  test_intrinsics_roundtrip(dir);
  std::cout << "== avertissement de résolution ==\n";
  test_intrinsics_resolution_warning(config, scene);
  std::cout << "== calibration automatique ==\n";
  test_automatic_calibration();
  std::cout << "== API HTTP & flotte ==\n";
  {
    FakeRobot robot(5312);
    test_api(make_config(dir), scene, 5312);
  }
  std::cout << "== route /shutdown ==\n";
  test_shutdown_route(make_config(dir), scene);

  std::cout << "\n" << (g_checks - g_failures) << '/' << g_checks << " vérifications réussies\n";
  if (g_failures > 0) {
    std::cerr << g_failures << " échec(s)\n";
    std::system(("rm -rf " + dir).c_str());
    return 1;
  }
  std::system(("rm -rf " + dir).c_str());
  std::cout << "OK\n";
  return 0;
}
