#include "matvision/config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "matvision/logger.hpp"

namespace matvision {

namespace fs = std::filesystem;

namespace {

bool is_all_digits(const std::string& text) {
  return !text.empty() &&
         std::all_of(text.begin(), text.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; });
}

/// Renseigne `target` si la clé existe et n'est pas nulle.
template <typename T>
void set_if(const nlohmann::json& obj, const char* key, T& target) {
  if (obj.contains(key) && !obj[key].is_null()) {
    target = obj[key].get<T>();
  }
}

std::string join_path(const std::string& base, const std::string& name) {
  return (fs::path(base) / name).string();
}

CameraConfig parse_camera(const nlohmann::json& data) {
  CameraConfig camera;
  if (data.is_null()) {
    return camera;
  }
  if (data.contains("device") && !data["device"].is_null()) {
    if (data["device"].is_string()) {
      camera.device = data["device"].get<std::string>();
    } else {
      camera.device = std::to_string(data["device"].get<long long>());
    }
  }
  set_if(data, "backend", camera.backend);
  set_if(data, "width", camera.width);
  set_if(data, "height", camera.height);
  set_if(data, "fps", camera.fps);
  set_if(data, "fourcc", camera.fourcc);
  set_if(data, "buffer_size", camera.buffer_size);
  set_if(data, "warmup_frames", camera.warmup_frames);
  set_if(data, "auto_exposure", camera.auto_exposure);
  set_if(data, "auto_white_balance", camera.auto_white_balance);
  return camera;
}

TableConfig parse_table(const nlohmann::json& data) {
  TableConfig table;
  if (data.is_null()) {
    return table;
  }
  set_if(data, "width_mm", table.width_mm);
  set_if(data, "height_mm", table.height_mm);
  set_if(data, "roi_mode", table.roi_mode);
  set_if(data, "roi_margin_px", table.roi_margin_px);
  if (data.contains("markers") && data["markers"].is_array()) {
    table.markers.clear();
    for (const auto& item : data["markers"]) {
      CornerMarker marker;
      set_if(item, "id", marker.id);
      set_if(item, "x", marker.x);
      set_if(item, "y", marker.y);
      set_if(item, "a", marker.a);
      set_if(item, "size", marker.size);
      table.markers.push_back(marker);
    }
  }
  return table;
}

ObjectConfig parse_object(const nlohmann::json& data) {
  ObjectConfig config;
  set_if(data, "id", config.id);
  set_if(data, "label", config.label);
  set_if(data, "angle_offset", config.angle_offset);
  set_if(data, "size", config.size);
  if (data.contains("offset") && data["offset"].is_array() && data["offset"].size() >= 2) {
    config.offset[0] = data["offset"][0].get<double>();
    config.offset[1] = data["offset"][1].get<double>();
  }
  if (data.contains("box") && data["box"].is_object()) {
    BoxConfig box;
    set_if(data["box"], "length_mm", box.length_mm);
    set_if(data["box"], "width_mm", box.width_mm);
    set_if(data["box"], "height_mm", box.height_mm);
    config.box = box;
  }
  return config;
}

RobotConfig parse_robot(const nlohmann::json& data) {
  RobotConfig robot;
  if (data.is_null()) {
    return robot;
  }
  set_if(data, "name", robot.name);
  set_if(data, "host", robot.host);
  set_if(data, "port", robot.port);
  set_if(data, "enabled", robot.enabled);
  if (data.contains("tag") && !data["tag"].is_null()) {
    robot.tag = data["tag"].get<int>();
  }
  return robot;
}

RobotsConfig parse_robots(const nlohmann::json& data) {
  RobotsConfig config;
  if (data.is_null()) {
    return config;
  }
  set_if(data, "our_color", config.our_color);
  set_if(data, "request_timeout_s", config.request_timeout_s);
  set_if(data, "path_window_s", config.path_window_s);
  set_if(data, "map_image", config.map_image);
  if (data.contains("main") && data["main"].is_object()) {
    config.main = parse_robot(data["main"]);
  }
  if (data.contains("hunter") && data["hunter"].is_object()) {
    config.hunter = parse_robot(data["hunter"]);
  }
  if (data.contains("swarm") && data["swarm"].is_array()) {
    config.swarm.clear();
    for (const auto& item : data["swarm"]) {
      config.swarm.push_back(parse_robot(item));
    }
  }
  return config;
}

CalibrationConfig parse_calibration(const nlohmann::json& data) {
  CalibrationConfig config;
  if (data.is_null()) {
    return config;
  }
  set_if(data, "intrinsics_file", config.intrinsics_file);
  set_if(data, "chessboard_cols", config.chessboard_cols);
  set_if(data, "chessboard_rows", config.chessboard_rows);
  set_if(data, "square_size_mm", config.square_size_mm);
  set_if(data, "target_frames", config.target_frames);
  set_if(data, "min_board_area_ratio", config.min_board_area_ratio);
  set_if(data, "max_board_area_ratio", config.max_board_area_ratio);
  set_if(data, "min_sharpness", config.min_sharpness);
  set_if(data, "min_move_ratio", config.min_move_ratio);
  set_if(data, "border_margin_px", config.border_margin_px);
  set_if(data, "max_capture_seconds", config.max_capture_seconds);
  set_if(data, "flags", config.flags);
  return config;
}

}  // namespace

bool CameraConfig::device_is_index() const { return is_all_digits(device); }

int CameraConfig::device_index() const {
  return device_is_index() ? std::stoi(device) : 0;
}

TableConfig::TableConfig() {
  markers = {
      CornerMarker{20, -400.0, -900.0, 0.0, 100.0},
      CornerMarker{21, -400.0, 900.0, 0.0, 100.0},
      CornerMarker{22, 400.0, -900.0, 0.0, 100.0},
      CornerMarker{23, 400.0, 900.0, 0.0, 100.0},
  };
}

std::vector<int> TableConfig::marker_ids() const {
  std::vector<int> ids;
  ids.reserve(markers.size());
  for (const auto& marker : markers) {
    ids.push_back(marker.id);
  }
  return ids;
}

std::string RobotConfig::base_url() const {
  std::ostringstream oss;
  oss << "http://" << host << ':' << port;
  return oss.str();
}

std::vector<std::pair<std::string, const RobotConfig*>> RobotsConfig::targets() const {
  std::vector<std::pair<std::string, const RobotConfig*>> items;
  if (main.has_value()) {
    items.emplace_back("main", &main.value());
  }
  if (hunter.has_value()) {
    items.emplace_back("hunter", &hunter.value());
  }
  for (std::size_t index = 0; index < swarm.size(); ++index) {
    items.emplace_back("swarm/" + std::to_string(index), &swarm[index]);
  }
  return items;
}

bool apply_robot_host(RobotsConfig& robots, const std::string& key, const std::string& host,
                      int port) {
  if (key == "main") {
    if (!robots.main.has_value()) {
      return false;
    }
    robots.main->host = host;
    robots.main->port = port;
    return true;
  }
  if (key == "hunter") {
    if (!robots.hunter.has_value()) {
      return false;
    }
    robots.hunter->host = host;
    robots.hunter->port = port;
    return true;
  }
  const std::string prefix = "swarm/";
  if (key.rfind(prefix, 0) == 0) {
    const std::string index_text = key.substr(prefix.size());
    if (!is_all_digits(index_text)) {
      return false;
    }
    const std::size_t index = static_cast<std::size_t>(std::stoul(index_text));
    if (index >= robots.swarm.size()) {
      return false;
    }
    robots.swarm[index].host = host;
    robots.swarm[index].port = port;
    return true;
  }
  return false;
}

bool set_robot_host_in_config(const std::string& path, const std::string& key,
                              const std::string& host, int port) {
  if (path.empty()) {
    return false;
  }
  nlohmann::json data = nlohmann::json::object();
  if (fs::exists(path)) {
    std::ifstream input(path);
    if (!input) {
      MV_LOGE("configuration illisible pour écriture (" << path << ")");
      return false;
    }
    try {
      input >> data;
    } catch (const std::exception& exc) {
      MV_LOGE("configuration JSON invalide, adresse non enregistrée (" << path
                                                                       << ") : " << exc.what());
      return false;
    }
  }
  if (!data.is_object()) {
    data = nlohmann::json::object();
  }
  nlohmann::json& robots = data["robots"];
  if (!robots.is_object()) {
    robots = nlohmann::json::object();
  }
  nlohmann::json* target = nullptr;
  if (key == "main" || key == "hunter") {
    target = &robots[key];
  } else {
    const std::string prefix = "swarm/";
    if (key.rfind(prefix, 0) != 0 || !is_all_digits(key.substr(prefix.size()))) {
      MV_LOGE("clé de robot inconnue pour l'enregistrement : " << key);
      return false;
    }
    const std::size_t index = static_cast<std::size_t>(std::stoul(key.substr(prefix.size())));
    nlohmann::json& swarm = robots["swarm"];
    if (!swarm.is_array()) {
      swarm = nlohmann::json::array();
    }
    while (swarm.size() <= index) {
      swarm.push_back(nlohmann::json::object());
    }
    target = &swarm[index];
  }
  if (!target->is_object()) {
    *target = nlohmann::json::object();
  }
  (*target)["host"] = host;
  (*target)["port"] = port;

  std::ofstream output(path);
  if (!output) {
    MV_LOGE("impossible d'écrire la configuration (" << path << ")");
    return false;
  }
  output << data.dump(2) << '\n';
  MV_LOGI("adresse de " << key << " enregistrée dans " << path << " : " << host << ':' << port);
  return true;
}

std::string project_root() {
#ifdef MATVISION_PROJECT_ROOT
  return MATVISION_PROJECT_ROOT;
#else
  return ".";
#endif
}

std::string web_root() { return join_path(project_root(), "web"); }

std::string data_dir() { return join_path(project_root(), "data"); }

std::string default_config_path() { return join_path(project_root(), "config/default.json"); }

std::string Config::intrinsics_path() const {
  const fs::path path(calibration.intrinsics_file);
  if (path.is_absolute()) {
    return path.string();
  }
  return (fs::path(project_root()) / path).string();
}

std::vector<std::string> Config::validate() const {
  std::vector<std::string> problems;
  if (table.markers.size() < 4) {
    problems.emplace_back(
        "moins de 4 tags de coin configurés : la pose caméra ne pourra pas être "
        "estimée de façon robuste");
  }
  std::vector<int> ids = table.marker_ids();
  std::set<int> unique(ids.begin(), ids.end());
  if (unique.size() != ids.size()) {
    problems.emplace_back("identifiants de tags de coin dupliqués");
  }
  std::vector<int> overlap;
  for (const auto& object : objects) {
    if (unique.count(object.id) != 0) {
      overlap.push_back(object.id);
    }
  }
  if (!overlap.empty()) {
    std::ostringstream oss;
    oss << "tags utilisés à la fois comme coin et comme objet : [";
    for (std::size_t i = 0; i < overlap.size(); ++i) {
      oss << (i ? ", " : "") << overlap[i];
    }
    oss << ']';
    problems.push_back(oss.str());
  }
  if (table.roi_mode != "table" && table.roi_mode != "markers") {
    problems.emplace_back("table.roi_mode doit valoir 'table' ou 'markers'");
  }
  if (calibration.chessboard_cols < 3 || calibration.chessboard_rows < 3) {
    problems.emplace_back("damier trop petit (>= 3x3 coins internes)");
  }
  if (!robots.our_color.empty() && robots.our_color != "blue" && robots.our_color != "yellow") {
    problems.emplace_back("robots.our_color doit valoir 'blue' ou 'yellow'");
  }
  for (const auto& [key, robot] : robots.targets()) {
    if (robot->enabled && robot->host.empty()) {
      problems.push_back("robots." + key + " : aucune adresse (host) renseignée");
    }
  }
  return problems;
}

nlohmann::json Config::to_json() const {
  nlohmann::json data;
  data["camera"] = {
      {"device", camera.device},       {"backend", camera.backend},
      {"width", camera.width},         {"height", camera.height},
      {"fps", camera.fps},             {"fourcc", camera.fourcc},
      {"buffer_size", camera.buffer_size},
      {"warmup_frames", camera.warmup_frames},
      {"auto_exposure", camera.auto_exposure},
      {"auto_white_balance", camera.auto_white_balance}};
  data["aruco"] = {{"dictionary", aruco.dictionary}, {"params", aruco.params}};
  nlohmann::json markers = nlohmann::json::array();
  for (const auto& marker : table.markers) {
    markers.push_back({{"id", marker.id},
                       {"x", marker.x},
                       {"y", marker.y},
                       {"a", marker.a},
                       {"size", marker.size}});
  }
  data["table"] = {{"width_mm", table.width_mm},
                   {"height_mm", table.height_mm},
                   {"roi_mode", table.roi_mode},
                   {"roi_margin_px", table.roi_margin_px},
                   {"markers", markers}};
  data["detection"] = {{"draw", detection.draw}};
  data["calibration"] = {
      {"intrinsics_file", calibration.intrinsics_file},
      {"chessboard_cols", calibration.chessboard_cols},
      {"chessboard_rows", calibration.chessboard_rows},
      {"square_size_mm", calibration.square_size_mm},
      {"target_frames", calibration.target_frames},
      {"min_board_area_ratio", calibration.min_board_area_ratio},
      {"max_board_area_ratio", calibration.max_board_area_ratio},
      {"min_sharpness", calibration.min_sharpness},
      {"min_move_ratio", calibration.min_move_ratio},
      {"border_margin_px", calibration.border_margin_px},
      {"max_capture_seconds", calibration.max_capture_seconds},
      {"flags", calibration.flags}};
  nlohmann::json objects_json = nlohmann::json::array();
  for (const auto& object : objects) {
    nlohmann::json item = {{"id", object.id},
                           {"label", object.label},
                           {"angle_offset", object.angle_offset},
                           {"offset", {object.offset[0], object.offset[1]}},
                           {"size", object.size}};
    if (object.box.has_value()) {
      item["box"] = {{"length_mm", object.box->length_mm},
                     {"width_mm", object.box->width_mm},
                     {"height_mm", object.box->height_mm}};
    }
    objects_json.push_back(item);
  }
  data["objects"] = objects_json;
  nlohmann::json robots_json;
  robots_json["our_color"] = robots.our_color;
  robots_json["request_timeout_s"] = robots.request_timeout_s;
  robots_json["path_window_s"] = robots.path_window_s;
  robots_json["map_image"] = robots.map_image;
  auto robot_json = [](const RobotConfig& robot) {
    nlohmann::json item = {{"name", robot.name},
                           {"host", robot.host},
                           {"port", robot.port},
                           {"enabled", robot.enabled}};
    if (robot.tag.has_value()) {
      item["tag"] = robot.tag.value();
    }
    return item;
  };
  if (robots.main.has_value()) {
    robots_json["main"] = robot_json(robots.main.value());
  }
  if (robots.hunter.has_value()) {
    robots_json["hunter"] = robot_json(robots.hunter.value());
  }
  nlohmann::json swarm = nlohmann::json::array();
  for (const auto& robot : robots.swarm) {
    swarm.push_back(robot_json(robot));
  }
  robots_json["swarm"] = swarm;
  data["robots"] = robots_json;
  data["api_host"] = api_host;
  data["api_port"] = api_port;
  data["preview_quality"] = preview_quality;
  return data;
}

Config load_config(const std::string& path) {
  Config config;
  const std::string config_path = path.empty() ? default_config_path() : path;
  nlohmann::json data;
  if (fs::exists(config_path)) {
    std::ifstream input(config_path);
    if (input) {
      try {
        input >> data;
      } catch (const std::exception& exc) {
        MV_LOGE("configuration illisible (" << config_path << ") : " << exc.what());
        data = nlohmann::json::object();
      }
    }
  }
  if (!data.is_object()) {
    data = nlohmann::json::object();
  }

  config.camera = parse_camera(data.value("camera", nlohmann::json()));
  config.table = parse_table(data.value("table", nlohmann::json()));
  config.detection.draw = true;
  if (data.contains("detection") && data["detection"].is_object()) {
    set_if(data["detection"], "draw", config.detection.draw);
  }
  config.calibration = parse_calibration(data.value("calibration", nlohmann::json()));
  config.robots = parse_robots(data.value("robots", nlohmann::json()));
  if (data.contains("aruco") && data["aruco"].is_object()) {
    set_if(data["aruco"], "dictionary", config.aruco.dictionary);
    if (data["aruco"].contains("params") && data["aruco"]["params"].is_object()) {
      config.aruco.params = data["aruco"]["params"];
    }
  }
  if (data.contains("objects") && data["objects"].is_array()) {
    config.objects.clear();
    for (const auto& item : data["objects"]) {
      config.objects.push_back(parse_object(item));
    }
  }
  set_if(data, "api_host", config.api_host);
  set_if(data, "api_port", config.api_port);
  set_if(data, "preview_quality", config.preview_quality);

  MV_LOGD("configuration effective : " << config.objects.size() << " objets, "
                                       << config.table.markers.size() << " tags de coin, roi="
                                       << config.table.roi_mode);
  return config;
}

}  // namespace matvision
