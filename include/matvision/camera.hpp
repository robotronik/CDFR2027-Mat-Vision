#pragma once

/// Accès à la webcam USB (Logitech 4K Stream Edition sur LattePanda Delta).

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "matvision/config.hpp"
#include "matvision/json.hpp"

namespace matvision {

/// Erreur d'accès à la caméra.
class CameraError : public std::runtime_error {
 public:
  explicit CameraError(const std::string& message) : std::runtime_error(message) {}
};

/// Source d'images ; permet de substituer une caméra factice en test.
class IFrameSource {
 public:
  virtual ~IFrameSource() = default;
  /// Ouvre la source et applique la configuration demandée.
  virtual bool open() = 0;
  /// Lit une image ; renvoie `false` en cas d'échec.
  virtual bool read(cv::Mat& frame) = 0;
  /// Lit l'image la plus récente (vide le tampon avant lecture).
  virtual bool read_latest(cv::Mat& frame, int drain = 3) = 0;
  virtual void close() = 0;
  virtual bool is_open() const = 0;
  virtual cv::Size size() const = 0;
  virtual nlohmann::json info() const = 0;
};

/// Fine enveloppe autour de `cv::VideoCapture` avec configuration UVC.
class Camera : public IFrameSource {
 public:
  explicit Camera(const CameraConfig& config = CameraConfig());

  bool open() override;
  bool read(cv::Mat& frame) override;
  bool read_latest(cv::Mat& frame, int drain = 3) override;
  void close() override;

  bool is_open() const override;
  cv::Size size() const override { return size_; }
  nlohmann::json info() const override;

 private:
  void read_actual_settings();

  CameraConfig config_;
  cv::VideoCapture capture_;
  cv::Size size_{0, 0};
  std::string fourcc_;
  double fps_ = 0.0;
};

/// Liste les périphériques vidéo disponibles (Linux : `/dev/video*`).
std::vector<std::string> list_video_devices();

/// Ouvre brièvement la caméra pour vérifier qu'elle fonctionne.
nlohmann::json probe(const std::string& device = "0", double timeout_s = 5.0);

}  // namespace matvision
