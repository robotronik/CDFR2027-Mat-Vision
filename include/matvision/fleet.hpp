#pragma once

/// Flotte de robots pilotée par le mat.
///
/// Le mat pilote le robot principal, un robot « chasseur » et un essaim de
/// petits robots à travers leur API REST embarquée (`/get_robot`,
/// `/get_strategies`, `/set_strat`, `/set_color`). Le robot principal est vu par
/// la caméra grâce à son tag ArUco ; le chasseur et les petits robots n'ont pas
/// de tag : ils déclarent leur position (`POST /fleet/report`) et reçoivent en
/// retour les éléments de jeu détectés par le mat.

#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "matvision/config.hpp"
#include "matvision/json.hpp"
#include "matvision/vision.hpp"

namespace matvision {

/// Erreur de pilotage d'un robot, convertie en réponse HTTP par l'API.
class FleetError : public std::runtime_error {
 public:
  explicit FleetError(const std::string& message, int status = 502)
      : std::runtime_error(message), status_(status) {}
  int status() const { return status_; }

 private:
  int status_ = 502;
};

/// Un robot de la flotte : clé stable, rôle et configuration.
struct RobotTarget {
  std::string key;  ///< "main", "hunter" ou "swarm/<index>"
  std::string role;
  std::optional<int> index;
  const RobotConfig* config = nullptr;

  std::string name() const;
  /// Un robot sans adresse n'est jamais contacté (aucun appel réseau).
  bool configured() const;
  std::string base_url() const;
};

/// Client REST de la flotte + suivi des trajectoires.
class Fleet {
 public:
  Fleet(const RobotsConfig& config, VisionEngine& engine);

  std::vector<RobotTarget> targets() const;
  nlohmann::json status();
  nlohmann::json strategies();
  nlohmann::json set_strategy(const std::string& key, const std::string& strategy);
  nlohmann::json set_color(const std::string& key, const nlohmann::json& color);
  nlohmann::json live();
  nlohmann::json report(const std::string& key, double x, double y, double a = 0.0);
  void clear();

 private:
  struct PathPoint {
    double t;
    double x;
    double y;
    double a;
  };

  std::vector<RobotTarget> resolve(const std::string& key) const;
  nlohmann::json request(const std::string& base_url, const std::string& path,
                         const std::string& method = "GET",
                         const nlohmann::json& payload = nlohmann::json());
  nlohmann::json fetch_state(const RobotTarget& target);
  void invalidate(const std::string& key);
  nlohmann::json cached_state(const std::string& key) const;
  std::string our_color(bool allow_fetch = true);
  std::string opponent_color(bool allow_fetch = true);
  nlohmann::json apply(const std::string& key, const std::string& path,
                       const nlohmann::json& payload);
  nlohmann::json table_info();
  nlohmann::json main_position(const std::vector<nlohmann::json>& objects,
                               const std::string& our_color);
  void push_path(const std::string& key, double x, double y, double a, double now);
  nlohmann::json path(const std::string& key) const;
  nlohmann::json live_robots(const std::string& our_color,
                             const nlohmann::json& main_position_value);

  RobotsConfig config_;
  VisionEngine& engine_;
  mutable std::mutex mutex_;
  std::map<std::string, std::pair<double, nlohmann::json>> cache_;
  std::map<std::string, std::deque<PathPoint>> paths_;
  std::map<std::string, nlohmann::json> reports_;
};

}  // namespace matvision
