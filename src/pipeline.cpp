#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>

#include "adasvp/adasvp.hpp"
#include "cuda/cuda_check.hpp"
#include "models/lane_detector.hpp"
#include "models/segmenter.hpp"
#include "models/yolo_detector.hpp"
#include "track/byte_tracker.hpp"

namespace adasvp {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

trt::Precision to_trt(Precision p) { return p == Precision::FP16 ? trt::Precision::FP16 : trt::Precision::FP32; }

std::string model_path(const Config& c, const std::string& name) {
  std::filesystem::path p(name);
  return p.is_absolute() ? p.string() : (std::filesystem::path(c.models_dir) / p).string();
}

}  // namespace

struct Pipeline::Impl {
  Config cfg;
  cudaStream_t stream = nullptr;
  std::unique_ptr<YoloDetector> det;
  std::unique_ptr<Segmenter> seg;
  std::unique_ptr<LaneDetector> lanes;
  std::unique_ptr<ByteTracker> tracker;
  cv::Mat seg_map;
  long frame_index = 0;
  cv::Size last_size;
  std::vector<int> classes{0, 1, 2, 3, 5, 7, 9};

  explicit Impl(const Config& c) : cfg(c) {
    ADASVP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      if (cfg.enable_det) {
        // opt shape: 16:9 letterbox of the profile size (960x544 / 1280x736)
        const int opt_w = cfg.det_imgsz;
        const int opt_h = ((cfg.det_imgsz * 9 / 16 + 31) / 32) * 32;
        det = std::make_unique<YoloDetector>(model_path(cfg, cfg.det_onnx), to_trt(cfg.det_precision),
                                             cfg.cache_dir, opt_w, opt_h);
        tracker = std::make_unique<ByteTracker>(cfg.tracker, cfg.frame_rate);
      }
      if (cfg.enable_seg)
        seg = std::make_unique<Segmenter>(model_path(cfg, cfg.seg_onnx), to_trt(cfg.seg_precision), cfg.cache_dir);
      if (cfg.enable_lanes)
        lanes = std::make_unique<LaneDetector>(model_path(cfg, cfg.lane_onnx), to_trt(Precision::FP32),
                                               cfg.cache_dir);
    } catch (...) {
      cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Impl() {
    det.reset();
    seg.reset();
    lanes.reset();
    if (stream) cudaStreamDestroy(stream);
  }
};

Pipeline::Pipeline(const Config& cfg) : impl_(std::make_unique<Impl>(cfg)) {}
Pipeline::~Pipeline() = default;

FrameResult Pipeline::process(const cv::Mat& bgr, double t_seconds) {
  Impl& s = *impl_;
  const Config& cfg = s.cfg;
  const auto t_all = Clock::now();
  FrameResult r;
  r.t = t_seconds;

  if (bgr.size() != s.last_size) {  // new video: stale tracks / segmentation map no longer apply
    if (s.tracker) s.tracker = std::make_unique<ByteTracker>(cfg.tracker, cfg.frame_rate);
    s.seg_map.release();
    s.last_size = bgr.size();
  }

  if (s.det) {
    auto t0 = Clock::now();
    std::vector<Object> dets = s.det->detect(bgr, cfg.det_imgsz, cfg.conf, s.classes, s.stream);
    r.timings.det_ms = ms_since(t0);
    t0 = Clock::now();
    r.objects = s.tracker->update(dets);
    r.detections = std::move(dets);
    // Ultralytics clips the tracked boxes to the frame (Results.update)
    const float W = static_cast<float>(bgr.cols), H = static_cast<float>(bgr.rows);
    for (Object& o : r.objects) {
      o.box.x1 = std::clamp(o.box.x1, 0.f, W);
      o.box.x2 = std::clamp(o.box.x2, 0.f, W);
      o.box.y1 = std::clamp(o.box.y1, 0.f, H);
      o.box.y2 = std::clamp(o.box.y2, 0.f, H);
    }
    r.timings.track_ms = ms_since(t0);
  }

  if (s.lanes) {
    const auto t0 = Clock::now();
    RawLanes rl = s.lanes->detect(bgr, cfg.lanes_thr, s.stream);
    r.timings.lanes_ms = ms_since(t0);
    r.lanes = std::move(rl.lanes);
    r.lane_scores = std::move(rl.scores);
  }

  if (s.seg) {
    if (s.seg_map.empty() || s.frame_index % std::max(1, cfg.seg_every) == 0) {
      const auto t0 = Clock::now();
      s.seg_map = s.seg->segment(bgr, s.stream);
      r.timings.seg_ms = ms_since(t0);
    }
    r.seg_classes = s.seg_map;
  }

  ++s.frame_index;
  r.timings.total_ms = ms_since(t_all);
  return r;
}

}  // namespace adasvp
