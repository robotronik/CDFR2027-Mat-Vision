#include "matvision/logger.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace matvision {

namespace {
std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
std::mutex g_mutex;

const char* level_name(LogLevel level) {
  switch (level) {
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warning:
      return "WARN";
    case LogLevel::Error:
      return "ERROR";
  }
  return "?";
}
}  // namespace

void set_log_level(LogLevel level) { g_level.store(static_cast<int>(level)); }

LogLevel log_level() { return static_cast<LogLevel>(g_level.load()); }

bool log_enabled(LogLevel level) {
  return static_cast<int>(level) >= g_level.load();
}

void log_write(LogLevel level, const std::string& message) {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::cerr << timestamp_now() << ' ' << level_name(level) << " matvision: " << message
            << std::endl;
}

std::string timestamp_now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  std::tm parts{};
#if defined(_WIN32)
  localtime_s(&parts, &seconds);
#else
  localtime_r(&seconds, &parts);
#endif
  std::ostringstream oss;
  oss << std::put_time(&parts, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

}  // namespace matvision
