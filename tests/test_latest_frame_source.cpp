#include "matvision/latest_frame_source.hpp"
#include "matvision/vision.hpp"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error("failed: " #x); } while (0)

struct Counters {
  std::atomic<int> reads{0}, closes{0}, wrong_owner{0};
};
class Source final : public matvision::IFrameSource {
 public:
  Source(std::shared_ptr<Counters> counters, bool never_open = false, bool fail_once = false)
      : counters_(std::move(counters)), never_open_(never_open), fail_once_(fail_once) {}
  bool open() override {
    owner();
    if (never_open_) throw matvision::CameraError("synthetic unavailable camera");
    opened_ = true; return true;
  }
  bool read(cv::Mat& frame) override {
    owner();
    if (fail_once_) { fail_once_ = false; return false; }
    std::this_thread::sleep_for(3ms);
    const int id = ++counters_->reads;
    frame = cv::Mat(8, 8, CV_32SC1, cv::Scalar(id));
    return true;
  }
  bool read_latest(cv::Mat& frame, int) override { return read(frame); }
  void close() override { owner(); opened_ = false; ++counters_->closes; owner_ = {}; }
  bool is_open() const override { owner(); return opened_; }
  nlohmann::json info() const override { owner(); return {{"opened", opened_}, {"device", "synthetic"}}; }
 private:
  void owner() const {
    if (owner_ == std::thread::id{}) owner_ = std::this_thread::get_id();
    else if (owner_ != std::this_thread::get_id()) ++counters_->wrong_owner;
  }
  std::shared_ptr<Counters> counters_;
  bool opened_ = false, never_open_, fail_once_;
  mutable std::thread::id owner_;
};

int main() {
  {
    auto counters = std::make_shared<Counters>();
    matvision::LatestFrameSource camera(std::make_unique<Source>(counters), 0);
    REQUIRE(camera.open());
    int previous = 0;
    for (int i = 0; i < 8; ++i) {
      cv::Mat image;
      REQUIRE(camera.read(image));
      const int id = image.at<int>(0,0);
      REQUIRE(id > previous);
      REQUIRE(camera.info()["threaded_capture"]["last_delivered_sequence"] == id);
      std::this_thread::sleep_for(20ms); // Producer outruns simulated processing.
      REQUIRE(image.at<int>(0,0) == id); // Image in use was not overwritten.
      previous = id;
    }
    const auto info = camera.info()["threaded_capture"];
    REQUIRE(info["waiting_capacity"] == 1);
    REQUIRE(info["overwritten_waiting_frames"].get<int>() > 0);
    REQUIRE(info["captured"].get<int>() == info["delivered"].get<int>() +
        info["overwritten_waiting_frames"].get<int>() + (info["waiting_frame"].get<bool>() ? 1 : 0));
    camera.close();
    REQUIRE(counters->wrong_owner == 0);
    REQUIRE(counters->closes >= 1);
    // A restarted service must not deliver a queued frame from before close().
    REQUIRE(camera.info()["threaded_capture"]["waiting_frame"] == false);
    const int before_restart = counters->reads.load();
    REQUIRE(camera.open());
    cv::Mat fresh;
    REQUIRE(camera.read(fresh));
    REQUIRE(fresh.at<int>(0,0) > before_restart);
    camera.close();
    REQUIRE(counters->wrong_owner == 0);
  }
  {
    auto counters = std::make_shared<Counters>();
    matvision::LatestFrameSource camera(std::make_unique<Source>(counters, true), 0);
    camera.open();
    cv::Mat frame;
    bool failed = false;
    try { camera.read(frame); } catch (const matvision::CameraError&) { failed = true; }
    REQUIRE(failed);
    auto consumer = std::async(std::launch::async, [&] { cv::Mat f; return camera.read(f); });
    std::this_thread::sleep_for(20ms);
    camera.request_stop();
    REQUIRE(consumer.wait_for(500ms) == std::future_status::ready);
    REQUIRE(consumer.get() == false);
    const auto started = std::chrono::steady_clock::now();
    camera.close();
    REQUIRE(std::chrono::steady_clock::now() - started < 500ms);
    REQUIRE(counters->wrong_owner == 0);
  }
  {
    auto counters = std::make_shared<Counters>();
    matvision::LatestFrameSource camera(std::make_unique<Source>(counters, false, true), 0);
    camera.open();
    cv::Mat frame;
    bool failed = false;
    try { camera.read(frame); } catch (const matvision::CameraError&) { failed = true; }
    REQUIRE(failed);
    REQUIRE(camera.read(frame)); // Reopen/recovery after a synthetic read failure.
    REQUIRE(camera.info()["threaded_capture"]["capture_failures"] == 1);
    camera.close();
    REQUIRE(counters->wrong_owner == 0);
  }
  {
    auto counters = std::make_shared<Counters>();
    matvision::Config config;
    config.detection.draw = false;
    matvision::VisionEngine engine(config, false, std::make_unique<Source>(counters, true));
    engine.start();
    for (int i = 0; i < 100 && counters->closes == 0; ++i)
      std::this_thread::sleep_for(10ms);
    REQUIRE(counters->closes > 0);
    std::this_thread::sleep_for(20ms);
    const auto started = std::chrono::steady_clock::now();
    engine.shutdown(); // Must signal cancellation BEFORE joining the consumer.
    REQUIRE(std::chrono::steady_clock::now() - started < 1s);
    REQUIRE(counters->wrong_owner == 0);
  }
  {
    auto counters = std::make_shared<Counters>();
    matvision::Config config;
    config.detection.draw = false;
    matvision::VisionEngine engine(config, false, std::make_unique<Source>(counters, false, true));
    engine.start();
    bool saw_failure = false, recovered = false;
    for (int i = 0; i < 400; ++i) {
      const auto status = engine.status();
      if (!status["error"].get<std::string>().empty()) saw_failure = true;
      if (saw_failure && status["stats"]["frames"].get<int>() > 0 &&
          status["error"].get<std::string>().empty()) { recovered = true; break; }
      std::this_thread::sleep_for(10ms);
    }
    engine.shutdown();
    REQUIRE(saw_failure);
    REQUIRE(recovered); // Successful reads clear a preceding capture error.
    REQUIRE(counters->wrong_owner == 0);
  }
  std::cout << "Ownership, latest-frame replacement, image stability, ordering, restart, cancellation, engine shutdown and recovery passed\n";
}
