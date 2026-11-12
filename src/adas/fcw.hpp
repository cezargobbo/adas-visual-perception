#pragma once
// Forward Collision Warning from the scale expansion of the lead-vehicle box (calibration-free TTC).
// Line-by-line port of FCWMonitor of the Python prototype (dashcam_vision.py). Internal math is double.

#include <deque>
#include <optional>

#include "adasvp/adasvp.hpp"

namespace adasvp {

class FCWMonitor {
 public:
  FCWMonitor(int W, int H, double fps, double ttc_alert, double ttc_crit, double win = 0.8);

  void reset();
  FcwStatus update(double t, const std::optional<LeadVehicle>& lead);

 private:
  struct Sample {
    double t, lw, z, lh;  // t, ln w, distance, ln h
  };

  int W_, H_;
  double fps_, ta_, tc_, win_;
  int on_n_, off_n_;
  std::deque<Sample> hist_;
  std::optional<int> lead_id_;
  double since_ = 0;
  int miss_ = 0;
  std::deque<double> ttcs_;  // maxlen 5
  bool active_ = false;
  std::string level_;  // "" = None
  int cnt_on_ = 0, cnt_off_ = 0;
};

}  // namespace adasvp
