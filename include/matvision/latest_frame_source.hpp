#pragma once

// One camera owner, one waiting decoded frame, one processing consumer.
// Driver buffers and this decoded-image slot are separate storage stages.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "matvision/camera.hpp"
#include "matvision/util.hpp"

namespace matvision {

class LatestFrameSource final : public IFrameSource {
 public:
  explicit LatestFrameSource(std::unique_ptr<IFrameSource> source, int drain)
      : source_(std::move(source)), drain_(drain) {}
  ~LatestFrameSource() override { close(); }

  bool open() override {
    if (running_.load()) return true;
    if (worker_.joinable()) worker_.join();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      waiting_ = {};
      ready_ = false;
      read_error_.clear();
      last_error_.clear();
      cached_info_ = {{"opened", false}};
      captured_ = delivered_ = overwritten_ = delivered_sequence_ = failures_ = 0;
      read_sum_ms_ = wait_sum_ms_ = queue_sum_ms_ = queue_max_ms_ = 0.0;
    }
    stop_.store(false);
    running_.store(true);
    try { worker_ = std::thread([this] { capture_loop(); }); }
    catch (...) { running_.store(false); throw; }
    return true;
  }

  // Signals only: camera operations and release remain owned by capture_loop().
  void request_stop() override {
    stop_.store(true);
    cv_.notify_all();
  }

  void close() override {
    request_stop();
    if (worker_.joinable()) worker_.join();
    running_.store(false);
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_ = {};
    ready_ = false;
  }

  // Indicates the acquisition service is running (including reconnect attempts).
  // info()["opened"] reports the underlying camera's cached hardware state.
  bool is_open() const override { return running_.load() && !stop_.load(); }

  bool read(cv::Mat& frame) override {
    const double start = (steady_seconds() * 1000.0);
    Packet packet;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return ready_ || stop_.load() || !read_error_.empty(); });
      if (stop_.load()) { frame.release(); return false; }
      if (!ready_) {
        const std::string error = std::exchange(read_error_, "");
        throw CameraError(error);
      }
      packet = std::move(waiting_);
      ready_ = false;
      ++delivered_;
      delivered_sequence_ = packet.sequence;
      const double now = (steady_seconds() * 1000.0);
      wait_sum_ms_ += now - start;
      const double queued = now - packet.completed_ms;
      queue_sum_ms_ += queued;
      queue_max_ms_ = std::max(queue_max_ms_, queued);
    }
    // Move ownership: capture never writes to this image again.
    frame = std::move(packet.frame);
    return !frame.empty();
  }

  // Draining belongs to the producer and is configured once at construction.
  bool read_latest(cv::Mat& frame, int /*drain*/) override { return read(frame); }

  nlohmann::json info() const override {
    nlohmann::json result;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      result = cached_info_;
      result["threaded_capture"] = {
          {"enabled", true}, {"waiting_capacity", 1}, {"waiting_frame", ready_},
          {"captured", captured_}, {"delivered", delivered_},
          {"overwritten_waiting_frames", overwritten_},
          {"last_delivered_sequence", delivered_sequence_},
          {"capture_failures", failures_}, {"last_error", last_error_},
          {"producer_read_mean_ms", captured_ ? read_sum_ms_ / captured_ : 0.0},
          {"consumer_wait_mean_ms", delivered_ ? wait_sum_ms_ / delivered_ : 0.0},
          {"decoded_queue_mean_ms", delivered_ ? queue_sum_ms_ / delivered_ : 0.0},
          {"decoded_queue_max_ms", queue_max_ms_}};
    }
    return result;
  }

 private:
  struct Packet {
    cv::Mat frame;
    double completed_ms = 0.0;
    std::uint64_t sequence = 0;
  };

  void capture_loop() {
    double last_info_ms = 0.0;
    while (!stop_.load()) {
      try {
        if (!source_->is_open()) {
          if (!source_->open()) throw CameraError("capture source failed to open");
          // Read source information only on the owner thread.
          auto info = source_->info();
          { std::lock_guard<std::mutex> lock(mutex_); cached_info_ = std::move(info); }
          last_info_ms = (steady_seconds() * 1000.0);
        }
        if (stop_.load()) break;
        // Fresh output allocation: no buffer currently used by processing is reused.
        Packet packet;
        const double started = (steady_seconds() * 1000.0);
        if (!source_->read_latest(packet.frame, drain_) || packet.frame.empty())
          throw CameraError("capture source returned no image");
        packet.completed_ms = (steady_seconds() * 1000.0);
        if (stop_.load()) break;
        nlohmann::json refreshed;
        if (packet.completed_ms - last_info_ms >= 1000.0) {
          refreshed = source_->info();
          last_info_ms = packet.completed_ms;
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stop_.load()) break;
          if (ready_) ++overwritten_;
          packet.sequence = ++captured_;
          read_sum_ms_ += packet.completed_ms - started;
          waiting_ = std::move(packet); // Older WAITING image is discarded here.
          ready_ = true;
          read_error_.clear();
          last_error_.clear();
          if (!refreshed.is_null()) cached_info_ = std::move(refreshed);
        }
        cv_.notify_all();
      } catch (const std::exception& error) {
        try { source_->close(); } catch (...) {}
        {
          std::lock_guard<std::mutex> lock(mutex_);
          ++failures_;
          read_error_ = last_error_ = error.what();
          cached_info_["opened"] = false;
          // Discard a pending pre-disconnection image, never deliver it as new.
          waiting_ = {};
          ready_ = false;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::seconds(2), [this] { return stop_.load(); });
      }
    }
    try { source_->close(); } catch (...) {}
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cached_info_["opened"] = false;
    }
    running_.store(false);
    cv_.notify_all();
  }

  std::unique_ptr<IFrameSource> source_;
  int drain_;
  std::atomic<bool> stop_{true}, running_{false};
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  Packet waiting_;
  bool ready_ = false;
  nlohmann::json cached_info_ = {{"opened", false}};
  std::string read_error_, last_error_;
  std::uint64_t captured_ = 0, delivered_ = 0, overwritten_ = 0;
  std::uint64_t delivered_sequence_ = 0, failures_ = 0;
  double read_sum_ms_ = 0, wait_sum_ms_ = 0, queue_sum_ms_ = 0, queue_max_ms_ = 0;
};
} // namespace matvision
