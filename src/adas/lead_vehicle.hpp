#pragma once
// Lead-vehicle selection (ego-lane or trapezoid corridor, hysteresis by track id) and monocular distance
// (per-track median of 9 + EMA). Line-by-line port of LeadVehicleEstimator of the reference
// (dashcam_vision.py). Internal math is double.

#include <deque>
#include <map>
#include <optional>
#include <vector>

#include "adas/ego_lane.hpp"
#include "adasvp/adasvp.hpp"

namespace adasvp {

class LeadVehicleEstimator {
 public:
  LeadVehicleEstimator(int W, int H, double hfov_deg, double cam_height, double horizon_frac,
                       double corridor_bottom, double corridor_top, int hold_frames);

  // `tracked`: tracks with an id (track_id >= 0); `lane`: used as the corridor when it holds a lane.
  std::optional<LeadVehicle> update(const std::vector<Object>& tracked, const EgoLaneTracker* lane);
  double distance(const Box& box, int cls);
  bool in_corridor(const Box& box, const EgoLaneTracker* lane) const;

 private:
  double corridor_halfwidth(double y) const;

  int W_, H_;
  double f_, cam_h_, y_hor_, cb_, ct_;
  int hold_frames_;
  std::optional<int> lead_id_;
  int missing_ = 0;
  std::map<int, std::deque<double>> hist_;  // per track id, maxlen 9
  std::map<int, double> ema_;
};

}  // namespace adasvp
