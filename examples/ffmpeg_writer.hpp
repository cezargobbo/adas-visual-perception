#pragma once
// Video output through an ffmpeg child process fed with raw BGR frames on its stdin (H.264, audio copied
// from the source segment), like open_writer of the Python reference.
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace adasvp::example {

struct WriterSpec {
  std::string path;
  int width = 0, height = 0;
  double fps = 30.0;
  std::string encoder = "x264";  // nvenc | x264 (already resolved, not "auto")
  std::string audio_from;        // source video for the audio track; empty = no audio
  double start_s = 0, duration_s = 0;  // audio segment
};

// Arguments after the "ffmpeg" program name.
std::vector<std::string> ffmpeg_args(const WriterSpec& s);

// Resolves "auto" (nvenc when `ffmpeg -encoders` lists h264_nvenc, else x264); nvenc and x264 pass through.
std::string resolve_encoder(const std::string& requested);

class FfmpegWriter {
 public:
  explicit FfmpegWriter(const WriterSpec& s);  // throws std::runtime_error if ffmpeg cannot be started
  ~FfmpegWriter();
  FfmpegWriter(const FfmpegWriter&) = delete;
  FfmpegWriter& operator=(const FfmpegWriter&) = delete;
  void write(const cv::Mat& bgr);  // continuous CV_8UC3 frame of the size given in the spec
  void close();                    // closes the pipe and waits for ffmpeg; returns normally if already closed

 private:
  int fd_ = -1;
  int pid_ = -1;
  size_t frame_bytes_ = 0;
  std::string path_;
};

}  // namespace adasvp::example
