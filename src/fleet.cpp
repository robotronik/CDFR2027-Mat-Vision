#include "matvision/fleet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

#include <httplib.h>

#include "matvision/logger.hpp"

namespace matvision {

namespace {

/// Correspondance entre l'entier renvoyé par le robot et le nom de couleur.
const std::map<int, std::string> kColorNames = {{0, ""}, {1, "blue"}, {2, "yellow"}};
/// Correspondance inverse (nom de couleur -> entier attendu par `/set_color`).
const std::map<std::string, int> kColorIds = {{"blue", 1}, {"yellow", 2}};
/// Couleur opposée, utilisée pour désigner le robot adverse.
const std::map<std::string, std::string> kOppositeColor = {
    {"blue", "yellow"}, {"yellow", "blue"}, {"", ""}};

/// Durée de validité du cache d'état d'un robot (s).
constexpr double kStatusCacheS = 1.0;
/// Garde-fou sur le nombre de points conservés par trajectoire.
constexpr std::size_t kMaxPathPoints = 600;

double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string lowered(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

}  // namespace

std::string RobotTarget::name() const {
  const std::string value = config != nullptr ? config->name : std::string();
  return value.empty() ? key : value;
}

bool RobotTarget::configured() const {
  return config != nullptr && !config->host.empty() && config->enabled;
}

std::string RobotTarget::base_url() const {
  return config != nullptr ? config->base_url() : std::string();
}

Fleet::Fleet(const RobotsConfig& config, VisionEngine& engine)
    : config_(config), engine_(engine) {}

std::vector<RobotTarget> Fleet::targets() const {
  std::vector<RobotTarget> result;
  for (const auto& [key, robot] : config_.targets()) {
    RobotTarget target;
    target.key = key;
    const std::size_t slash = key.find('/');
    target.role = slash == std::string::npos ? key : key.substr(0, slash);
    if (slash != std::string::npos) {
      target.index = std::stoi(key.substr(slash + 1));
    }
    target.config = robot;
    result.push_back(target);
  }
  return result;
}

std::vector<RobotTarget> Fleet::resolve(const std::string& raw_key) const {
  const std::string key = lowered(raw_key);
  std::vector<RobotTarget> matches;
  const auto all = targets();
  if (key == "all") {
    matches = all;
  } else if (key == "swarm") {
    for (const auto& target : all) {
      if (target.role == "swarm") {
        matches.push_back(target);
      }
    }
  } else if (key == "main" || key == "hunter" || key.rfind("swarm/", 0) == 0) {
    for (const auto& target : all) {
      if (target.key == key) {
        matches.push_back(target);
      }
    }
  } else {
    for (const auto& target : all) {
      if (lowered(target.name()) == key) {
        matches.push_back(target);
      }
    }
  }
  if (matches.empty()) {
    throw FleetError("robot inconnu : '" + raw_key + "'", 404);
  }
  return matches;
}

nlohmann::json Fleet::request(const std::string& base_url, const std::string& path,
                              const std::string& method, const nlohmann::json& payload) {
  httplib::Client client(base_url);
  const auto timeout = std::chrono::milliseconds(
      static_cast<long long>(std::max(0.05, config_.request_timeout_s) * 1000.0));
  client.set_connection_timeout(timeout);
  client.set_read_timeout(timeout);
  client.set_write_timeout(timeout);

  httplib::Result result = httplib::Result(nullptr, httplib::Error::Unknown);
  if (method == "POST") {
    const std::string body = payload.is_null() ? std::string("{}") : payload.dump();
    result = client.Post(path, body, "application/json");
  } else {
    result = client.Get(path);
  }

  if (!result) {
    throw FleetError("robot injoignable (" + httplib::to_string(result.error()) + ")", 502);
  }
  if (result->status >= 400) {
    std::string detail;
    if (!result->body.empty()) {
      try {
        const nlohmann::json parsed = nlohmann::json::parse(result->body);
        if (parsed.is_object() && parsed.contains("message")) {
          detail = parsed["message"].get<std::string>();
        }
      } catch (const std::exception&) {
        detail.clear();
      }
    }
    throw FleetError(detail.empty() ? ("HTTP " + std::to_string(result->status)) : detail,
                     result->status);
  }
  if (result->body.empty()) {
    return nlohmann::json::object();
  }
  try {
    const nlohmann::json parsed = nlohmann::json::parse(result->body);
    return parsed.is_object() ? parsed : nlohmann::json{{"data", parsed}};
  } catch (const std::exception&) {
    return nlohmann::json::object();
  }
}

nlohmann::json Fleet::fetch_state(const RobotTarget& target) {
  if (!target.configured()) {
    return {{"key", target.key},
            {"name", target.name()},
            {"role", target.role},
            {"index", target.index.has_value() ? nlohmann::json(target.index.value())
                                               : nlohmann::json(nullptr)},
            {"host", target.config->host},
            {"port", target.config->port},
            {"configured", false},
            {"online", false},
            {"error", "adresse non renseignée"}};
  }

  const double now = steady_seconds();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cache_.find(target.key);
    if (it != cache_.end() && now - it->second.first < kStatusCacheS) {
      return it->second.second;
    }
  }

  nlohmann::json state = {{"key", target.key},
                          {"name", target.name()},
                          {"role", target.role},
                          {"index", target.index.has_value()
                                        ? nlohmann::json(target.index.value())
                                        : nlohmann::json(nullptr)},
                          {"host", target.config->host},
                          {"port", target.config->port},
                          {"configured", true}};
  try {
    const nlohmann::json robot = request(target.base_url(), "/get_robot");
    const int color_id = robot.value("team", 0);
    const auto color_it = kColorNames.find(color_id);
    state["online"] = true;
    state["color_id"] = color_id;
    state["color"] = color_it != kColorNames.end() ? color_it->second : "";
    state["strategy"] = robot.value("strategy", std::string());
    state["status"] = robot.contains("status") ? robot["status"] : nlohmann::json(nullptr);
    state["score"] = robot.contains("score") ? robot["score"] : nlohmann::json(nullptr);
    state["time"] = robot.contains("time") ? robot["time"] : nlohmann::json(nullptr);
    state["runid"] = robot.contains("runid") ? robot["runid"] : nlohmann::json(nullptr);
    state["error"] = "";
  } catch (const std::exception& exc) {
    state["online"] = false;
    state["error"] = exc.what();
  }

  try {
    const nlohmann::json listed = request(target.base_url(), "/get_strategies");
    nlohmann::json strategies = nlohmann::json::array();
    if (listed.contains("strategies") && listed["strategies"].is_array()) {
      for (const auto& name : listed["strategies"]) {
        if (name.is_string() && !name.get<std::string>().empty()) {
          strategies.push_back(name);
        }
      }
    }
    state["strategies"] = strategies;
  } catch (const std::exception& exc) {
    state["strategies"] = nlohmann::json::array();
    MV_LOGD("stratégies indisponibles pour " << target.key << " : " << exc.what());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_[target.key] = {now, state};
  }
  return state;
}

void Fleet::invalidate(const std::string& key) {
  // On marque l'entrée comme périmée (au lieu de l'effacer) : `live()` garde
  // ainsi la dernière couleur connue sans déclencher d'appel réseau, tandis que
  // `fetch_state` la rafraîchit dès que possible.
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = cache_.find(key);
  if (it != cache_.end()) {
    it->second.first = 0.0;
  }
}

nlohmann::json Fleet::cached_state(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = cache_.find(key);
  return it != cache_.end() ? it->second.second : nlohmann::json::object();
}

std::string Fleet::our_color(bool allow_fetch) {
  const auto all = targets();
  for (const auto& target : all) {
    if (target.key != "main" || !target.configured()) {
      continue;
    }
    std::string color;
    if (allow_fetch) {
      color = fetch_state(target).value("color", std::string());
    } else {
      color = cached_state(target.key).value("color", std::string());
    }
    if (!color.empty()) {
      return color;
    }
  }
  const std::string color = lowered(config_.our_color);
  return kColorIds.count(color) != 0 ? color : "";
}

std::string Fleet::opponent_color(bool allow_fetch) {
  const auto it = kOppositeColor.find(our_color(allow_fetch));
  return it != kOppositeColor.end() ? it->second : "";
}

nlohmann::json Fleet::status() {
  nlohmann::json robots = nlohmann::json::array();
  for (const auto& target : targets()) {
    robots.push_back(fetch_state(target));
  }
  return {{"robots", robots},
          {"our_color", our_color()},
          {"opponent_color", opponent_color()},
          {"table", table_info()},
          {"map_image", config_.map_image.empty() ? nlohmann::json(nullptr)
                                                  : nlohmann::json(config_.map_image)}};
}

nlohmann::json Fleet::strategies() {
  nlohmann::json robots = nlohmann::json::array();
  for (const auto& target : targets()) {
    const nlohmann::json state = fetch_state(target);
    robots.push_back({{"key", state.value("key", std::string())},
                      {"name", state.value("name", std::string())},
                      {"role", state.value("role", std::string())},
                      {"index", state.contains("index") ? state["index"]
                                                        : nlohmann::json(nullptr)},
                      {"online", state.value("online", false)},
                      {"strategy", state.value("strategy", std::string())},
                      {"strategies", state.value("strategies", nlohmann::json::array())}});
  }
  return {{"robots", robots}};
}

nlohmann::json Fleet::set_strategy(const std::string& key, const std::string& strategy) {
  return apply(key, "/set_strat", {{"strat", strategy}});
}

nlohmann::json Fleet::set_color(const std::string& key, const nlohmann::json& color) {
  int color_id = 0;
  if (color.is_string()) {
    const auto it = kColorIds.find(lowered(color.get<std::string>()));
    color_id = it != kColorIds.end() ? it->second : 0;
  } else if (color.is_number_integer()) {
    color_id = color.get<int>();
  }
  if (color_id != 1 && color_id != 2) {
    throw FleetError("couleur invalide (1 = bleu, 2 = jaune)", 400);
  }
  return apply(key, "/set_color", {{"color", color_id}});
}

nlohmann::json Fleet::apply(const std::string& key, const std::string& path,
                            const nlohmann::json& payload) {
  const auto targets_list = resolve(key);
  nlohmann::json results = nlohmann::json::array();
  bool failed = false;
  for (const auto& target : targets_list) {
    if (!target.configured()) {
      results.push_back(
          {{"key", target.key}, {"ok", false}, {"message", "adresse non renseignée"}});
      failed = true;
      continue;
    }
    try {
      request(target.base_url(), path, "POST", payload);
      invalidate(target.key);
      results.push_back({{"key", target.key}, {"ok", true}, {"message", "ok"}});
    } catch (const FleetError& exc) {
      failed = true;
      results.push_back({{"key", target.key},
                         {"ok", false},
                         {"status", exc.status()},
                         {"message", exc.what()}});
    } catch (const std::exception& exc) {
      failed = true;
      results.push_back(
          {{"key", target.key}, {"ok", false}, {"status", 502}, {"message", exc.what()}});
    }
  }

  bool all_failed = !results.empty();
  for (const auto& item : results) {
    if (item.value("ok", false)) {
      all_failed = false;
      break;
    }
  }
  if (failed && all_failed) {
    const nlohmann::json& first = results[0];
    throw FleetError(first.value("message", std::string("échec")),
                     first.value("status", 502));
  }
  return {{"ok", !failed}, {"results", results}};
}

nlohmann::json Fleet::table_info() {
  const nlohmann::json table = engine_.table_info();
  return {{"width_mm", table.value("width_mm", 0.0)},
          {"height_mm", table.value("height_mm", 0.0)}};
}

nlohmann::json Fleet::main_position(const std::vector<nlohmann::json>& objects,
                                    const std::string& our_color) {
  if (our_color.empty()) {
    return nullptr;
  }
  std::vector<nlohmann::json> candidates;
  for (const auto& object : objects) {
    if (object.value("label", std::string()) == our_color) {
      candidates.push_back(object);
    }
  }
  for (const auto& target : targets()) {
    if (target.key != "main" || target.config == nullptr ||
        !target.config->tag.has_value()) {
      continue;
    }
    const int tag = target.config->tag.value();
    for (const auto& candidate : candidates) {
      if (candidate.value("id", -1) == tag) {
        return candidate;
      }
    }
  }
  return candidates.empty() ? nlohmann::json(nullptr) : candidates.front();
}

nlohmann::json Fleet::live() {
  const double now = steady_seconds();
  const nlohmann::json payload = engine_.objects();
  std::vector<nlohmann::json> objects;
  for (const auto& item : payload.value("objects", nlohmann::json::array())) {
    if (!item.value("label", std::string()).empty()) {
      objects.push_back(item);
    }
  }
  const std::string our = our_color(false);
  const std::string opponent = opponent_color(false);

  const nlohmann::json main_value = main_position(objects, our);
  std::vector<nlohmann::json> opponents;
  std::vector<nlohmann::json> game_objects;
  for (const auto& object : objects) {
    const std::string label = object.value("label", std::string());
    if (!opponent.empty() && label == opponent) {
      opponents.push_back(object);
    } else if (kColorIds.count(label) == 0) {
      game_objects.push_back(object);
    }
  }

  if (!main_value.is_null()) {
    push_path("main", main_value.value("x", 0.0), main_value.value("y", 0.0),
              main_value.value("a", 0.0), now);
  }
  for (const auto& opponent_obj : opponents) {
    push_path("opponent/" + std::to_string(opponent_obj.value("id", -1)),
              opponent_obj.value("x", 0.0), opponent_obj.value("y", 0.0),
              opponent_obj.value("a", 0.0), now);
  }

  nlohmann::json robots = live_robots(our, main_value);
  nlohmann::json opponents_live = nlohmann::json::array();
  for (const auto& opponent_obj : opponents) {
    nlohmann::json item = {{"id", opponent_obj.value("id", -1)},
                           {"label", opponent_obj.value("label", std::string())},
                           {"x", opponent_obj.value("x", 0.0)},
                           {"y", opponent_obj.value("y", 0.0)},
                           {"a", opponent_obj.value("a", 0.0)},
                           {"path", path("opponent/" +
                                         std::to_string(opponent_obj.value("id", -1)))}};
    opponents_live.push_back(item);
  }

  nlohmann::json game = nlohmann::json::array();
  for (const auto& object : game_objects) {
    game.push_back({{"id", object.value("id", -1)},
                    {"label", object.value("label", std::string())},
                    {"x", object.value("x", 0.0)},
                    {"y", object.value("y", 0.0)},
                    {"a", object.value("a", 0.0)}});
  }

  return {{"our_color", our},
          {"opponent_color", opponent},
          {"table", table_info()},
          {"map_image", config_.map_image.empty() ? nlohmann::json(nullptr)
                                                  : nlohmann::json(config_.map_image)},
          {"robots", robots},
          {"opponents", opponents_live},
          {"objects", game},
          {"timestamp", wall_seconds()}};
}

nlohmann::json Fleet::live_robots(const std::string& our_color,
                                  const nlohmann::json& main_position_value) {
  nlohmann::json entries = nlohmann::json::array();
  for (const auto& target : targets()) {
    nlohmann::json position = nullptr;
    std::string source = "report";
    if (target.role == "main") {
      position = main_position_value;
      source = "camera";
    } else {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = reports_.find(target.key);
      if (it != reports_.end()) {
        position = it->second;
      }
    }
    std::string color = our_color;
    if (target.configured()) {
      const std::string robot_color = cached_state(target.key).value("color", std::string());
      if (!robot_color.empty()) {
        color = robot_color;
      }
    }
    nlohmann::json entry = {{"key", target.key},
                            {"role", target.role},
                            {"index", target.index.has_value()
                                          ? nlohmann::json(target.index.value())
                                          : nlohmann::json(nullptr)},
                            {"name", target.name()},
                            {"color", color},
                            {"source", source},
                            {"online", target.configured()},
                            {"x", position.is_null() ? nlohmann::json(nullptr)
                                                     : position["x"]},
                            {"y", position.is_null() ? nlohmann::json(nullptr)
                                                     : position["y"]},
                            {"a", position.is_null()
                                      ? nlohmann::json(nullptr)
                                      : nlohmann::json(position.value("a", 0.0))},
                            {"path", path(target.key)}};
    entries.push_back(entry);
  }
  return entries;
}

nlohmann::json Fleet::report(const std::string& key, double x, double y, double a) {
  const auto targets_list = resolve(key);
  if (targets_list.size() != 1) {
    throw FleetError("précisez un robot (clé ou nom)", 400);
  }
  const RobotTarget& target = targets_list.front();
  if (target.role == "main") {
    throw FleetError("le robot principal est vu par la caméra", 400);
  }

  const double now = steady_seconds();
  const nlohmann::json position = {
      {"x", x}, {"y", y}, {"a", a}, {"ts", wall_seconds()}};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    reports_[target.key] = position;
  }
  push_path(target.key, x, y, a, now);

  const nlohmann::json live_payload = live();
  nlohmann::json opponents = nlohmann::json::array();
  for (const auto& opponent : live_payload.value("opponents", nlohmann::json::array())) {
    opponents.push_back({{"id", opponent.value("id", -1)},
                         {"x", opponent.value("x", 0.0)},
                         {"y", opponent.value("y", 0.0)},
                         {"a", opponent.value("a", 0.0)}});
  }
  return {{"robot", target.key},
          {"name", target.name()},
          {"position", position},
          {"our_color", live_payload.value("our_color", std::string())},
          {"opponent_color", live_payload.value("opponent_color", std::string())},
          {"objects", live_payload.value("objects", nlohmann::json::array())},
          {"opponents", opponents}};
}

void Fleet::push_path(const std::string& key, double x, double y, double a, double now) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& path_points = paths_[key];
  path_points.push_back(PathPoint{now, x, y, a});
  while (path_points.size() > kMaxPathPoints) {
    path_points.pop_front();
  }
  const double window = std::max(1.0, config_.path_window_s);
  while (!path_points.empty() && now - path_points.front().t > window) {
    path_points.pop_front();
  }
}

nlohmann::json Fleet::path(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = paths_.find(key);
  if (it == paths_.end()) {
    return nlohmann::json::array();
  }
  nlohmann::json points = nlohmann::json::array();
  for (const auto& point : it->second) {
    points.push_back({std::round(point.x * 10.0) / 10.0, std::round(point.y * 10.0) / 10.0});
  }
  return points;
}

void Fleet::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  paths_.clear();
  reports_.clear();
  cache_.clear();
}

}  // namespace matvision
