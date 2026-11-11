#pragma once
// Lane Departure Warning with the "robustness first" arming logic. Line-by-line port of LDWMonitor of the
// reference (dashcam_vision.py). Internal math is double.

#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <string>

#include "adas/ego_lane.hpp"
#include "adasvp/adasvp.hpp"

namespace adasvp {

class LDWMonitor {
 public:
  LDWMonitor(int W, int H, double fps, const EgoLaneTracker& trk, double car_width = 1.8, double margin = 0.25,
             double lane_w_m = 3.5, double horizon_frac = 0.59, double arm_s = 1.0, double fit_s = 0.5);

  LdwStatus update(double t, bool lane_ok = true);

  int row() const { return row_; }
  bool armed() const { return armed_; }
  // Disarm reason of the last update ("" when armed; "start" before the first update).
  const std::string& fail() const { return fail_; }
  // d(DL)/dt of the last update; nullopt when it was not computed (Python None).
  const std::optional<double>& last_v() const { return last_v_; }
  // y0, wres, resl, resr, sll, slr as set by the last full check (stale between full checks, like the Python).
  const std::map<std::string, double>& metrics() const { return m_; }
  // Frames per disarm reason; key "armed" counts the armed frames.
  const std::map<std::string, long long>& stats() const { return stats_; }
  int n_cross() const { return n_cross_; }

 private:
  struct Sample {
    double t, DL, w;
  };
  // Returns the disarm reason, or "" when the frame is clean.
  std::string check();

  int W_, H_;
  double fps_;
  const EgoLaneTracker& trk_;
  int row_;
  double car_w_, margin_, lane_w_, hor_, arm_s_, fit_s_;
  int on_n_, off_n_;
  double cool_s_ = 2.5, cross_s_ = 2.0, vmin_ = 0.15, vmax_ = 1.5;
  std::deque<Sample> hist_;
  int cnt_[2] = {0, 0}, off_[2] = {0, 0};  // [0] = left, [1] = right
  double cool_[2] = {-1e9, -1e9};
  int active_ = -1;  // -1 none, 0 left, 1 right
  double last_active_t_ = -1e9, suppress_until_ = -1e9;
  bool armed_ = false;
  std::string fail_ = "start";
  std::map<std::string, double> m_;
  std::map<std::string, long long> stats_;
  std::optional<double> last_v_;
  int n_cross_ = 0;
};

}  // namespace adasvp
