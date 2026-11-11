#pragma once
// Public API of the adasvp library. Only OpenCV and the STL may be included here.
#include <opencv2/core.hpp>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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

struct Lane {
  Polyline points;  // pixel coordinates in the input frame
  float score = 0.f;
};

struct Timings {
  double det_ms = 0, seg_ms = 0, lanes_ms = 0, track_ms = 0, total_ms = 0;
};

// ---------------------------------------------------------------------------------------------------------
// ADAS result types. Field names are the snake_case names of the Python reference (dashcam_vision.py).
// "No value" of the Python (None) is std::optional, or an empty string for the enum-like string fields.

enum class LaneState { None, Partial, Estimated, Detected };

// Lead vehicle = the tuple (track_id, cls, box, dist_m) returned by LeadVehicleEstimator.update.
struct LeadVehicle {
  int track_id = -1;
  int cls = -1;
  Box box{};
  double distance_m = 0;  // smoothed distance
};

// Ego lane selected and tracked from the detected lanes (EgoLaneTracker). xl / xr hold the x position of
// the left / right boundary at the rows `rows_y` (empty while state == None).
struct EgoLane {
  LaneState state = LaneState::None;
  bool available = false;  // state != None
  double fade = 0.0;       // 1 while observed, decreasing to 0 while the last lane is only held
  std::vector<double> rows_y, xl, xr;
  bool obs_l = false, obs_r = false;  // boundary observed (not inferred / held) in this frame
  bool glitch = false;  // this frame: jump-gate rejection / re-acquisition / implausible lane
};

struct FcwStatus {
  std::optional<double> ttc;  // seconds
  bool active = false;
  std::string level;  // "", "warning" or "critical" (only while active)
  std::string event;  // "" or the newly raised level
};

struct LdwStatus {
  bool armed = false;
  std::string side;   // active side: "", "left" or "right"
  std::string event;  // "" or the newly triggered side
  std::optional<double> d, v, tlc;  // wheel-to-line distance (m), approach speed (m/s), time to crossing (s)
};

struct RlrEvent {
  int id = -1;  // traffic-light track id
  double cx = 0;
  double red_for = 0;  // seconds the light had been red
  double growth = 0;
  Box box{};
};

struct RlrStatus {
  std::optional<RlrEvent> event;
};

enum class LightColor { Unknown, Red, Yellow, Green };

struct TrafficLight {
  Box box;
  float score = 0.f;
  int track_id = -1;
  LightColor color = LightColor::Unknown;
};

// One row of the Python alerts CSV (time_s,frame,type,level_side,ttc_s,line_dist_m,lat_vel_mps,tlc_s,
// lead_id,lead_dist_m,red_for_s). `frame` is filled by the caller; the library writes -1.
struct Alert {
  std::string type;        // "FCW", "LDW" or "RLR"
  std::string level_side;  // "warning" / "critical", "left" / "right", "red"
  double t = 0;
  int frame = -1;
  std::optional<double> ttc_s, line_dist_m, lat_vel_mps, tlc_s, lead_dist_m, red_for_s;
  int lead_id = -1;  // -1 = no lead vehicle
};

// Counters behind the "[info] alerts:" summary line of the Python (events per kind, LDW frame statistics).
struct AdasStats {
  int fcw_events = 0, ldw_events = 0, rlr_events = 0;
  // frames per LDW disarm reason (key "armed" = frames armed), like LDWMonitor.stats
  std::map<std::string, long long> ldw_frames;
  int ldw_crossings_suppressed = 0;  // LDWMonitor.n_cross
};

struct FrameResult {
  double t = 0;
  std::vector<Object> objects;     // tracked objects (clipped to the frame)
  std::vector<Object> detections;  // detector output before the tracker (conf + class filtered)
  std::vector<Lane> lanes;  // all detected lanes
  cv::Mat seg_classes;      // CV_8UC1 Cityscapes class map (frame size); always a fresh copy owned by the caller
  Timings timings;

  // ADAS (filled when the corresponding modules are enabled; idle values otherwise)
  std::optional<LeadVehicle> lead;
  EgoLane ego_lane;
  std::vector<TrafficLight> lights;  // traffic-light detections (class 9) with their colour
  std::vector<Alert> alerts;         // events that are new in this frame
  FcwStatus fcw;
  LdwStatus ldw;
  RlrStatus rlr;
  // Values behind the Python --adas-debug trace.
  struct AdasDebug {
    std::string ldw_fail;                  // disarm reason of the frame ("" when armed)
    std::optional<double> ldw_last_v;      // d(DL)/dt, m/s
    std::map<std::string, double> ldw_m;   // y0, wres, resl, resr, sll, slr of the last full check
  } debug;
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
  double frame_rate = 30.0;  // passed to the tracker; also sizes every ADAS window (frames = seconds * fps)

  // Camera / lead vehicle (--hfov, --cam-height, --horizon, --corridor-bottom, --corridor-top)
  double hfov = 75.0;            // degrees
  double cam_height = 1.5;       // metres
  double horizon = 0.59;         // fraction of the frame height
  double corridor_bottom = 0.12;  // fractions of the frame width
  double corridor_top = 0.02;

  // Ego lane
  double lanes_hold = 2.5;    // seconds the last good lane is held when nothing is detected
  bool lane_corridor = true;  // use the ego lane (when available) as the lead-vehicle corridor

  // ADAS modules and thresholds
  bool enable_fcw = true, enable_ldw = true, enable_rlr = true;
  double fcw_ttc = 2.7, fcw_ttc_critical = 1.5;  // seconds
  double ldw_margin = 0.25, ldw_car_width = 1.8;  // metres
  std::pair<double, double> rlr_zone{0.25, 0.75};  // central zone, fractions of the width

  // Display parameters of the Python CLI (used by the example / CLI, not by the library)
  double alert_display = 2.0;  // seconds an alert stays on screen
  double seg_alpha = 0.45;
  int show_width = 1280;

  static Config fast();
  static Config accurate();
};

class Pipeline {
 public:
  explicit Pipeline(const Config& cfg);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  // Movable. A moved-from Pipeline may only be destroyed or assigned to.
  Pipeline(Pipeline&&) noexcept;
  Pipeline& operator=(Pipeline&&) noexcept;

  // Detection + tracking, segmentation (every cfg.seg_every frames, the last map is reused in between)
  // and lanes. Frames may change resolution between calls (the tracker, the segmentation map and the
  // frame counter are reset when they do).
  //
  // Threading: process() is not thread-safe; never call it concurrently on one Pipeline (use one
  // Pipeline per thread). Errors: a CUDA error thrown by process() leaves the Pipeline unusable;
  // destroy it and construct a new one.
  // FrameResult::seg_classes is a fresh copy on every call (about 2 MB for 1080p), so the caller may
  // keep or modify it freely.
  FrameResult process(const cv::Mat& bgr, double t_seconds);

  // Clears the tracker (track ids restart at 1), the cached segmentation map and the frame counter
  // (the next frame runs segmentation). Call it between unrelated videos of the same resolution.
  void reset();

  // Event counters and LDW statistics accumulated since construction or the last reset().
  const AdasStats& stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace adasvp
