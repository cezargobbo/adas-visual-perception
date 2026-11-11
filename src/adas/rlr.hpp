#pragma once
// Red-light running: the car drives under / past a RED traffic light that belongs to its path.
// Line-by-line port of RLRMonitor of the reference (dashcam_vision.py). Internal math is double.

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "adas/traffic_light.hpp"
#include "adasvp/adasvp.hpp"

namespace adasvp {

class RLRMonitor {
 public:
  RLRMonitor(int W, int H, double fps, std::pair<double, double> zone = {0.25, 0.75}, double top_cut = 0.01,
             double top_band = 0.12, double min_track_s = 1.0, double min_growth = 0.15, double red_min_s = 1.5,
             double cooldown_s = 4.0);

  // Classifies every class-9 box (not cut by the top edge) and calls update_with_states.
  RlrStatus update(double t, const cv::Mat& frame, const std::vector<std::pair<Box, float>>& boxes);
  // Same with the colours given. The colour of a box cut by the top edge (y1 <= top_cut*H) is ignored, as
  // update() never classifies those.
  RlrStatus update_with_states(double t, const std::vector<std::tuple<Box, float, LightColor>>& boxes);

  // Traffic lights matched by the last update, with their track id.
  const std::vector<TrafficLight>& lights() const { return lights_; }
  const TrafficLightTracker& tracker() const { return trk_; }

 private:
  std::optional<RlrEvent> evaluate(const TlTrack& tr) const;
  bool below_top_cut(float y1) const;

  int W_, H_;
  std::pair<double, double> zone_;
  double top_cut_, top_band_, min_track_s_, min_growth_, red_min_s_, cooldown_s_;
  TrafficLightTracker trk_;
  double last_alert_ = -1e9;
  std::vector<TrafficLight> lights_;
};

}  // namespace adasvp
