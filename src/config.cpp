#include "matvision/config.hpp"

#include <algorithm>
#include <cstdlib>
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

std::string join_path(const std::string& base, const std::string& name) {
  return (fs::path(base) / name).string();
}

template <typename T>
nlohmann::json to_json_value(const T& value) {
  nlohmann::json data;
  to_json(data, value);
  return data;
}

template <typename T>
T from_json_value(const nlohmann::json& data) {
  T value{};
  from_json(data, value);
  return value;
}

/// Charge une section JSON ; en cas de type invalide, conserve les défauts.
template <typename T>
T parse_section(const nlohmann::json& data, const char* key) {
  if (!data.is_object() || !data.contains(key) || data[key].is_null()) {
    return T{};
  }
  try {
    return from_json_value<T>(data[key]);
  } catch (const std::exception& exc) {
    MV_LOGW("section '" << key << "' invalide, valeurs par défaut : " << exc.what());
    return T{};
  }
}

template <typename T>
T value_or(const nlohmann::json& data, const char* key, T fallback) {
  if (!data.is_object() || !data.contains(key) || data[key].is_null()) {
    return fallback;
  }
  try {
    return data[key].get<T>();
  } catch (const std::exception& exc) {
    MV_LOGW("valeur '" << key << "' invalide, défaut conservé : " << exc.what());
    return fallback;
  }
}

}  // namespace

void from_json(const nlohmann::json& data, CameraConfig& camera) {
  camera = CameraConfig{};
  if (!data.is_object()) {
    return;
  }
  if (data.contains("device") && !data["device"].is_null()) {
    if (data["device"].is_string()) {
      camera.device = data["device"].get<std::string>();
    } else {
      camera.device = std::to_string(data["device"].get<long long>());
    }
  }
  camera.backend = data.value("backend", camera.backend);
  camera.width = data.value("width", camera.width);
  camera.height = data.value("height", camera.height);
  camera.fps = data.value("fps", camera.fps);
  camera.fourcc = data.value("fourcc", camera.fourcc);
  camera.buffer_size = data.value("buffer_size", camera.buffer_size);
  camera.drain_frames = data.value("drain_frames", camera.drain_frames);
  camera.warmup_frames = data.value("warmup_frames", camera.warmup_frames);
  camera.auto_exposure = data.value("auto_exposure", camera.auto_exposure);
  camera.auto_white_balance = data.value("auto_white_balance", camera.auto_white_balance);
}

bool CameraConfig::device_is_index() const { return is_all_digits(device); }

int CameraConfig::device_index() const {
  return device_is_index() ? std::stoi(device) : 0;
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

void to_json(nlohmann::json& item, const ObjectConfig& object) {
  item = {{"id", object.id},
          {"label", object.label},
          {"angle_offset", object.angle_offset},
          {"offset", {object.offset[0], object.offset[1]}},
          {"size", object.size}};
  if (object.box.has_value()) {
    item["box"] = *object.box;
  }
}

void from_json(const nlohmann::json& data, ObjectConfig& object) {
  object = ObjectConfig{};
  if (!data.is_object()) {
    return;
  }
  object.id = data.value("id", object.id);
  object.label = data.value("label", object.label);
  object.angle_offset = data.value("angle_offset", object.angle_offset);
  if (data.contains("offset") && data["offset"].is_array() && data["offset"].size() >= 2) {
    object.offset[0] = data["offset"][0].get<double>();
    object.offset[1] = data["offset"][1].get<double>();
  }
  object.size = data.value("size", object.size);
  if (data.contains("box") && data["box"].is_object()) {
    BoxConfig box;
    from_json(data["box"], box);
    object.box = box;
  }
}

void to_json(nlohmann::json& item, const RobotConfig& robot) {
  item = {{"name", robot.name},
          {"host", robot.host},
          {"port", robot.port},
          {"enabled", robot.enabled}};
  if (robot.tag.has_value()) {
    item["tag"] = robot.tag.value();
  }
}

void from_json(const nlohmann::json& data, RobotConfig& robot) {
  robot = RobotConfig{};
  if (!data.is_object()) {
    return;
  }
  robot.name = data.value("name", robot.name);
  robot.host = data.value("host", robot.host);
  robot.port = data.value("port", robot.port);
  robot.enabled = data.value("enabled", robot.enabled);
  if (data.contains("tag") && !data["tag"].is_null()) {
    robot.tag = data["tag"].get<int>();
  }
}

void to_json(nlohmann::json& data, const RobotsConfig& robots) {
  data = {{"our_color", robots.our_color},
          {"request_timeout_s", robots.request_timeout_s},
          {"path_window_s", robots.path_window_s},
          {"map_image", robots.map_image}};
  if (robots.main.has_value()) {
    data["main"] = to_json_value(*robots.main);
  }
  if (robots.hunter.has_value()) {
    data["hunter"] = to_json_value(*robots.hunter);
  }
  nlohmann::json swarm = nlohmann::json::array();
  for (const auto& robot : robots.swarm) {
    swarm.push_back(to_json_value(robot));
  }
  data["swarm"] = swarm;
}

void from_json(const nlohmann::json& data, RobotsConfig& robots) {
  robots = RobotsConfig{};
  if (!data.is_object()) {
    return;
  }
  robots.our_color = data.value("our_color", robots.our_color);
  robots.request_timeout_s = data.value("request_timeout_s", robots.request_timeout_s);
  robots.path_window_s = data.value("path_window_s", robots.path_window_s);
  robots.map_image = data.value("map_image", robots.map_image);
  if (data.contains("main") && data["main"].is_object()) {
    robots.main = from_json_value<RobotConfig>(data["main"]);
  }
  if (data.contains("hunter") && data["hunter"].is_object()) {
    robots.hunter = from_json_value<RobotConfig>(data["hunter"]);
  }
  if (data.contains("swarm") && data["swarm"].is_array()) {
    robots.swarm.clear();
    for (const auto& item : data["swarm"]) {
      try {
        robots.swarm.push_back(from_json_value<RobotConfig>(item));
      } catch (const std::exception& exc) {
        MV_LOGW("robot d'essaim ignoré : " << exc.what());
      }
    }
  }
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
  static const std::string root = [] {
    if (const char* env = std::getenv("MATVISION_ROOT")) {
      return std::string(env);
    }
    std::error_code error;
    fs::path directory = fs::current_path(error);
    for (int depth = 0; depth < 5 && !directory.empty(); ++depth) {
      if (fs::exists(directory / "config/default.json", error)) {
        return directory.string();
      }
      const fs::path parent = directory.parent_path();
      if (parent == directory) {
        break;
      }
      directory = parent;
    }
#ifdef MATVISION_PROJECT_ROOT
    return std::string(MATVISION_PROJECT_ROOT);
#else
    return std::string(".");
#endif
  }();
  return root;
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
  data["camera"] = to_json_value(camera);
  data["aruco"] = to_json_value(aruco);
  data["table"] = to_json_value(table);
  data["detection"] = to_json_value(detection);
  data["calibration"] = to_json_value(calibration);
  nlohmann::json objects_json = nlohmann::json::array();
  for (const auto& object : objects) {
    objects_json.push_back(to_json_value(object));
  }
  data["objects"] = objects_json;
  data["robots"] = to_json_value(robots);
  data["api_host"] = api_host;
  data["api_port"] = api_port;
  data["preview_quality"] = preview_quality;
  data["preview_max_width"] = preview_max_width;
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

  config.camera = parse_section<CameraConfig>(data, "camera");
  config.aruco = parse_section<ArucoConfig>(data, "aruco");
  config.table = parse_section<TableConfig>(data, "table");
  config.detection = parse_section<DetectionConfig>(data, "detection");
  config.calibration = parse_section<CalibrationConfig>(data, "calibration");
  config.robots = parse_section<RobotsConfig>(data, "robots");
  if (data.contains("objects") && data["objects"].is_array()) {
    config.objects.clear();
    for (const auto& item : data["objects"]) {
      try {
        config.objects.push_back(from_json_value<ObjectConfig>(item));
      } catch (const std::exception& exc) {
        MV_LOGW("objet ignoré : " << exc.what());
      }
    }
  }
  config.api_host = value_or(data, "api_host", config.api_host);
  config.api_port = value_or(data, "api_port", config.api_port);
  config.preview_quality = value_or(data, "preview_quality", config.preview_quality);
  config.preview_max_width = value_or(data, "preview_max_width", config.preview_max_width);

  MV_LOGD("configuration effective : " << config.objects.size() << " objets, "
                                       << config.table.markers.size() << " tags de coin, roi="
                                       << config.table.roi_mode);
  return config;
}

}  // namespace matvision
