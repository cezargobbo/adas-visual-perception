#pragma once
// Ego-lane selection and temporal tracking. Line-by-line port of EgoLaneTracker / lane_x_at of the
// Python prototype (dashcam_vision.py). Internal math is double.

#include <array>
#include <optional>
#include <vector>

#include "adasvp/adasvp.hpp"

namespace adasvp {

// x(y) of a detected lane sampled at rows `ys`: polynomial (degree <= 2) inside the observed y-range,
// straight tangent continuation outside it. nullopt when the lane is too short / does not reach the lower
// part of the image.
std::optional<std::vector<double>> lane_x_at(const Polyline& lane, const std::vector<double>& ys, int H);

// np.polyfit(y, x, deg) (scaled Vandermonde least squares via SVD, highest power first) and np.polyval.
std::vector<double> polyfit(const std::vector<double>& y, const std::vector<double>& x, int deg);
double polyval(const std::vector<double>& c, double x);

class EgoLaneTracker {
 public:
  EgoLaneTracker(int W, int H, double fps, double cam_height, double horizon_frac, double hold_s, int n_rows = 40);

  void reset();
  // Returns the new state (as the Python does).
  LaneState update(const std::vector<Lane>& lanes);

  bool available() const { return state_ != LaneState::None; }
  // Is the bottom-centre of `box` between the ego-lane boundaries at its y_bottom (+ margin)? False when
  // the tracker holds no lane.
  bool contains(const Box& box, double margin = 0.10) const;
  EgoLane snapshot() const;

  // State read by the LDW and the drawing code. xl / xr are empty while state == None.
  LaneState state() const { return state_; }
  const std::vector<double>& xl() const { return xl_; }
  const std::vector<double>& xr() const { return xr_; }
  const std::vector<double>& rows_y() const { return ys_; }  // Python `ys`
  const std::vector<double>& t_row() const { return t_row_; }
  const std::vector<double>& wprof() const { return wprof_; }
  int since() const { return since_; }
  double fade() const { return fade_; }
  bool obs_l() const { return obs_l_; }
  bool obs_r() const { return obs_r_; }
  bool glitch() const { return glitch_; }
  int free_l() const { return free_[0]; }
  int free_r() const { return free_[1]; }
  const std::optional<std::vector<double>>& raw_l() const { return raw_l_; }
  const std::optional<std::vector<double>>& raw_r() const { return raw_r_; }
  // Geometry / windows.
  int width() const { return W_; }
  int height() const { return H_; }
  double y_hor() const { return y_hor_; }
  double wmin() const { return wmin_; }
  double wmax() const { return wmax_; }
  double gate() const { return gate_; }
  int persist() const { return persist_; }
  int hold_frames() const { return hold_frames_; }

 private:
  using Vec = std::vector<double>;
  struct Pend {
    std::optional<Vec> obs;
    int count = 0;
  };

  bool gate_accept(int side, const Vec& obs);
  std::pair<std::optional<Vec>, std::optional<Vec>> pick(const std::vector<Vec>& cands) const;
  bool plausible(const Vec& xl, const Vec& xr) const;

  int W_, H_;
  double y_hor_;
  Vec ys_, t_row_, wdef_;
  double wmin_, wmax_, gate_;
  int persist_, hold_frames_;
  double alpha_;

  Vec xl_, xr_;  // empty = None
  Vec wprof_;
  LaneState state_;
  int since_;
  double fade_;
  bool obs_l_, obs_r_;
  std::array<Pend, 2> pend_;  // [0] = left, [1] = right
  std::array<int, 2> free_;
  bool glitch_;
  std::optional<Vec> raw_l_, raw_r_;
};

}  // namespace adasvp
