#pragma once

#include <sstream>
#include <string>

namespace matvision {

enum class LogLevel { Debug = 0, Info, Warning, Error };

/// Niveau minimal des messages écrits sur la console (stderr).
void set_log_level(LogLevel level);
LogLevel log_level();
bool log_enabled(LogLevel level);
void log_write(LogLevel level, const std::string& message);

/// Horodatage local au format `YYYY-MM-DD HH:MM:SS`.
std::string timestamp_now();

}  // namespace matvision

/// Journalise un message construit par flux (`MV_LOG(level, "x=" << x)`).
#define MV_LOG(level, ...)                                                    \
  do {                                                                        \
    if (::matvision::log_enabled(level)) {                                    \
      std::ostringstream mv_log_oss;                                          \
      mv_log_oss << __VA_ARGS__;                                              \
      ::matvision::log_write(level, mv_log_oss.str());                        \
    }                                                                         \
  } while (0)

#define MV_LOGD(...) MV_LOG(::matvision::LogLevel::Debug, __VA_ARGS__)
#define MV_LOGI(...) MV_LOG(::matvision::LogLevel::Info, __VA_ARGS__)
#define MV_LOGW(...) MV_LOG(::matvision::LogLevel::Warning, __VA_ARGS__)
#define MV_LOGE(...) MV_LOG(::matvision::LogLevel::Error, __VA_ARGS__)
