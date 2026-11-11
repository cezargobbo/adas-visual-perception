#pragma once
// Public API of the adasvp library. Only OpenCV and the STL may be included here.
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace adasvp {

struct Box {
  float x1, y1, x2, y2;
};

struct Object {
  Box box;
  int cls;
  float score;
  int track_id;  // -1 if untracked
};

using Polyline = std::vector<cv::Point2f>;

struct Timings {
  double det_ms = 0, seg_ms = 0, lanes_ms = 0, track_ms = 0, total_ms = 0;
};

struct FrameResult {
  double t = 0;
  std::vector<Object> objects;     // tracked objects (clipped to the frame)
  std::vector<Object> detections;  // detector output before the tracker (conf + class filtered)
  std::vector<Polyline> lanes;
  std::vector<float> lane_scores;
  cv::Mat seg_classes;
  Timings timings;
};

enum class Precision { FP16, FP32 };

// ByteTrack parameters (Ultralytics bytetrack.yaml defaults).
struct TrackerParams {
  float high = 0.25f, low = 0.1f, new_track = 0.25f;
  int buffer = 30;
  float match = 0.8f;
  bool fuse_score = true;
};

struct Config {
  std::string models_dir = "models";
  std::string cache_dir;  // empty = ${XDG_CACHE_HOME:-$HOME/.cache}/adasvp/engines

  std::string det_onnx = "yolo26s.onnx";
  Precision det_precision = Precision::FP16;
  int det_imgsz = 960;
  float conf = 0.30f;

  std::string seg_onnx = "segformer-b1-cityscapes-1024x576.onnx";
  Precision seg_precision = Precision::FP16;
  int seg_every = 1;

  std::string lane_onnx = "polarrcnn_curvelanes_dla34.onnx";
  float lanes_thr = 0.44f;

  bool enable_det = true;
  bool enable_seg = true;
  bool enable_lanes = true;

  TrackerParams tracker;
  double frame_rate = 30.0;  // passed to the tracker (ignored by it, like Ultralytics 8.4.208)

  static Config fast();
  static Config accurate();
};

class Pipeline {
 public:
  explicit Pipeline(const Config& cfg);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Detection + tracking, segmentation (every cfg.seg_every frames, the last map is reused in between)
  // and lanes. Frames may change resolution between calls (the tracker is reset when they do).
  FrameResult process(const cv::Mat& bgr, double t_seconds);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace adasvp
