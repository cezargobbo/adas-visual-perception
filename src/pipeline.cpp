#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>

#include "adas/ego_lane.hpp"
#include "adas/fcw.hpp"
#include "adas/ldw.hpp"
#include "adas/lead_vehicle.hpp"
#include "adas/rlr.hpp"
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

// The ADAS modules, which depend on the frame size W x H and on the nominal fps. `ego` is declared first so
// that it outlives the LDW monitor, which holds a reference to it (members are destroyed in reverse order).
struct AdasState {
  std::unique_ptr<EgoLaneTracker> ego;  // null when lanes are off
  std::unique_ptr<LeadVehicleEstimator> lead;
  std::unique_ptr<FCWMonitor> fcw;
  std::unique_ptr<LDWMonitor> ldw;
  std::unique_ptr<RLRMonitor> rlr;
  bool lane_corridor = false;

  // Same enabling rules as the reference CLI: FCW and RLR need the detector, LDW needs the lanes.
  AdasState(const Config& c, int W, int H, bool has_det, bool has_lanes) {
    const double fps = c.frame_rate;
    if (has_lanes) {
      ego = std::make_unique<EgoLaneTracker>(W, H, fps, c.cam_height, c.horizon, c.lanes_hold);
      lane_corridor = c.lane_corridor;
    }
    if (has_det) {
      lead = std::make_unique<LeadVehicleEstimator>(W, H, c.hfov, c.cam_height, c.horizon, c.corridor_bottom,
                                                    c.corridor_top, static_cast<int>(fps * 0.3));
      if (c.enable_fcw) fcw = std::make_unique<FCWMonitor>(W, H, fps, c.fcw_ttc, c.fcw_ttc_critical);
      if (c.enable_rlr) rlr = std::make_unique<RLRMonitor>(W, H, fps, c.rlr_zone);
    }
    if (has_lanes && c.enable_ldw)
      ldw = std::make_unique<LDWMonitor>(W, H, fps, *ego, c.ldw_car_width, c.ldw_margin, 3.5, c.horizon);
  }
};

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
  AdasStats adas_stats;  // filled by the ADAS modules
  std::unique_ptr<AdasState> adas;  // built on the first frame (needs the frame size)
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
  void reset() {
    if (det) tracker = std::make_unique<ByteTracker>(cfg.tracker, cfg.frame_rate);
    seg_map.release();
    frame_index = 0;
    adas_stats = AdasStats{};
    adas.reset();  // the monitors are recreated with the next frame
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
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

const AdasStats& Pipeline::stats() const { return impl_->adas_stats; }

void Pipeline::reset() {
  impl_->reset();
  impl_->last_size = cv::Size();
}

FrameResult Pipeline::process(const cv::Mat& bgr, double t_seconds) {
  Impl& s = *impl_;
  const Config& cfg = s.cfg;
  const auto t_all = Clock::now();
  FrameResult r;
  r.t = t_seconds;

  if (bgr.size() != s.last_size) {  // new video: stale tracks / segmentation map no longer apply
    s.reset();
    s.last_size = bgr.size();
  }

  if (!s.adas) s.adas = std::make_unique<AdasState>(cfg, bgr.cols, bgr.rows, s.det != nullptr, s.lanes != nullptr);
  AdasState& ad = *s.adas;

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
    r.lanes.reserve(rl.lanes.size());
    for (size_t i = 0; i < rl.lanes.size(); ++i)
      r.lanes.push_back(Lane{std::move(rl.lanes[i]), i < rl.scores.size() ? rl.scores[i] : 0.f});
    ad.ego->update(r.lanes);
    r.ego_lane = ad.ego->snapshot();
  }

  // ADAS: lead vehicle, then FCW, LDW, RLR (order of the reference loop)
  if (ad.lead) {
    std::vector<Object> tracked;
    for (const Object& o : r.objects)
      if (o.track_id >= 0) tracked.push_back(o);
    r.lead = ad.lead->update(tracked, ad.lane_corridor ? ad.ego.get() : nullptr);
  }
  // log_event of the reference: FCW and LDW rows carry the lead vehicle, RLR rows do not
  auto add_alert = [&](const char* type, const std::string& side, bool with_lead = true) -> Alert& {
    Alert& a = r.alerts.emplace_back();
    a.type = type;
    a.level_side = side;
    a.t = t_seconds;
    if (with_lead && r.lead) {
      a.lead_id = r.lead->track_id;
      a.lead_dist_m = r.lead->distance_m;
    }
    return a;
  };
  if (ad.fcw) {
    std::optional<LeadVehicle> lead_f = r.lead;  // the lead must be inside the ego lane (no margin)
    if (r.lead && ad.ego && ad.lane_corridor && ad.ego->available() && !ad.ego->contains(r.lead->box, 0.0))
      lead_f = std::nullopt;
    r.fcw = ad.fcw->update(t_seconds, lead_f);
    if (!r.fcw.event.empty()) {
      ++s.adas_stats.fcw_events;
      add_alert("FCW", r.fcw.event).ttc_s = r.fcw.ttc;
    }
  }
  if (ad.ldw) {
    r.ldw = ad.ldw->update(t_seconds);
    r.debug.ldw_fail = ad.ldw->fail();
    r.debug.ldw_last_v = ad.ldw->last_v();
    r.debug.ldw_m = ad.ldw->metrics();
    if (!r.ldw.event.empty()) {
      ++s.adas_stats.ldw_events;
      Alert& a = add_alert("LDW", r.ldw.event);
      a.line_dist_m = r.ldw.d;
      a.lat_vel_mps = r.ldw.v;
      a.tlc_s = r.ldw.tlc;
    }
    s.adas_stats.ldw_frames = ad.ldw->stats();
    s.adas_stats.ldw_crossings_suppressed = ad.ldw->n_cross();
  }
  if (ad.rlr) {
    std::vector<std::pair<Box, float>> tl;
    for (const Object& o : r.objects)
      if (o.cls == 9) tl.emplace_back(o.box, o.score);
    r.rlr = ad.rlr->update(t_seconds, bgr, tl);
    r.lights = ad.rlr->lights();
    if (r.rlr.event) {
      ++s.adas_stats.rlr_events;
      add_alert("RLR", "red", false).red_for_s = r.rlr.event->red_for;
    }
  }

  if (s.seg) {
    if (s.seg_map.empty() || s.frame_index % std::max(1, cfg.seg_every) == 0) {
      const auto t0 = Clock::now();
      s.seg_map = s.seg->segment(bgr, s.stream);
      r.timings.seg_ms = ms_since(t0);
    }
    r.seg_classes = s.seg_map.clone();  // the cached map is reused between seg_every frames
  }

  ++s.frame_index;
  r.timings.total_ms = ms_since(t_all);
  return r;
}

}  // namespace adasvp
