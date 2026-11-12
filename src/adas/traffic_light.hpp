#pragma once
// Traffic-light lamp colour and a small dedicated tracker for traffic lights.
// Line-by-line ports of traffic_light_state and TrafficLightTracker of the Python prototype (dashcam_vision.py).
// Internal math is double (the colour features are float32 like the Python prototype).

#include <array>
#include <map>
#include <vector>

#include <opencv2/core.hpp>

#include "adasvp/adasvp.hpp"

namespace adasvp {

// Colour of the lit lamp (Unknown = Python None). `frame` is BGR uint8.
LightColor traffic_light_state(const cv::Mat& frame, const Box& box, double thr = 14.0);

struct TlDet {
  std::array<double, 4> box;
  double conf = 0;
  LightColor state = LightColor::Unknown;
};

struct TlEntry {
  double t = 0;
  std::array<double, 4> box{};
  double conf = 0;
  LightColor state = LightColor::Unknown;
};

struct TlTrack {
  int id = 0;
  std::vector<TlEntry> hist;
  int lost = 0;
  bool done = false;
};

// Greedy matching against a constant-velocity prediction of the box centre and log-size.
class TrafficLightTracker {
 public:
  TrafficLightTracker(double fps, double max_lost_s = 0.25);

  // Returns the tracks closed (lost for too long) by this update.
  std::vector<TlTrack> update(double t, const std::vector<TlDet>& dets);

  std::map<int, TlTrack>& tracks() { return tracks_; }  // ascending id = insertion order
  const std::map<int, TlTrack>& tracks() const { return tracks_; }
  int max_lost() const { return max_lost_; }

 private:
  static std::array<double, 4> predict(const TlTrack& tr, double t);

  int max_lost_;
  std::map<int, TlTrack> tracks_;
  int next_id_ = 1;
};

}  // namespace adasvp
