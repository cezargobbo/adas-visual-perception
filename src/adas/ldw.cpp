#include "adas/ldw.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "adas/theil_sen.hpp"

namespace adasvp {

namespace {

using Vec = std::vector<double>;

double max_abs_resid(const Vec& c, const Vec& xs, const Vec& ys) {
  double m = 0;
  for (size_t i = 0; i < xs.size(); ++i) m = std::max(m, std::fabs(polyval(c, xs[i]) - ys[i]));
  return m;
}

}  // namespace

LDWMonitor::LDWMonitor(int W, int H, double fps, const EgoLaneTracker& trk, double car_width, double margin,
                       double lane_w_m, double horizon_frac, double arm_s, double fit_s)
    : W_(W), H_(H), fps_(fps), trk_(trk), car_w_(car_width), margin_(margin), lane_w_(lane_w_m),
      hor_(horizon_frac * H), arm_s_(arm_s), fit_s_(fit_s) {
  const Vec& ys = trk.rows_y();
  row_ = 0;  // np.argmin(np.abs(ys - 0.93 * H)): first minimum
  for (size_t i = 1; i < ys.size(); ++i)
    if (std::fabs(ys[i] - 0.93 * H) < std::fabs(ys[row_] - 0.93 * H)) row_ = static_cast<int>(i);
  on_n_ = std::max(1, static_cast<int>(0.3 * fps));
  off_n_ = std::max(1, static_cast<int>(0.5 * fps));
}

std::string LDWMonitor::check() {
  const EgoLaneTracker& trk = trk_;
  const int r = row_;
  if (trk.state() != LaneState::Detected || !(trk.obs_l() && trk.obs_r())) return "state";
  if (trk.glitch() || trk.free_l() > 0 || trk.free_r() > 0) return "glitch";
  const Vec& ys = trk.rows_y();
  const Vec &xl = trk.xl(), &xr = trk.xr();
  const size_t n = ys.size() / 2;
  const Vec &rl = *trk.raw_l(), &rr = *trk.raw_r();
  double sl = 0, sr = 0;
  for (size_t i = n; i < ys.size(); ++i) {
    sl += std::fabs(rl[i] - xl[i]);
    sr += std::fabs(rr[i] - xr[i]);
  }
  const double cnt = static_cast<double>(ys.size() - n);
  if (sl / cnt > 40 || sr / cnt > 40) return "obs_vs_trk";
  Vec w(ys.size());
  for (size_t i = 0; i < ys.size(); ++i) w[i] = xr[i] - xl[i];
  const double wn = w[r] / trk.t_row()[r];
  if (!(trk.wmin() <= wn && wn <= trk.wmax())) return "width";  // bottom-row widths; normalise row r
  const Vec p = polyfit(ys, w, 1);
  if (p[0] <= 0) return "convergence";
  const double y0 = -p[1] / p[0];  // row where the lane width vanishes
  if (!(hor_ - 0.07 * H_ <= y0 && y0 <= hor_ + 0.07 * H_)) return "convergence";
  if (max_abs_resid(p, ys, w) > 0.04 * w[r]) return "width_profile";
  double xv[2];
  m_.clear();  // the Python rebinds self.m: a failure below keeps the entries already set
  m_["y0"] = y0;
  m_["wres"] = max_abs_resid(p, ys, w) / w[r];
  int k = 0;
  for (const char* side : {"l", "r"}) {
    const Vec& x = k == 0 ? xl : xr;
    const Vec q = polyfit(ys, x, 1);
    const double res = max_abs_resid(q, ys, x);
    m_[std::string("res") + side] = res;
    m_[std::string("sl") + side] = q[0];
    if (res > 0.0065 * W_) return "curve";  // ~12 px
    xv[k++] = polyval(q, y0);
  }
  if (std::fabs(0.5 * (xv[0] + xv[1]) - W_ / 2.0) > 0.2 * W_ || std::fabs(xv[0] - xv[1]) > 0.08 * W_)
    return "vanishing_point";
  return "";
}

LdwStatus LDWMonitor::update(double t, bool lane_ok) {
  const EgoLaneTracker& trk = trk_;
  const int r = row_;
  std::string fail = lane_ok ? check() : "state";
  double DL = 0, vL = 0;
  last_v_.reset();
  if (fail.empty()) {
    const double w = trk.xr()[r] - trk.xl()[r];
    DL = (W_ / 2.0 - trk.xl()[r]) / w * lane_w_;
    hist_.push_back({t, DL, w});
    while (!hist_.empty() && t - hist_.front().t > arm_s_ + 1.5 / fps_) hist_.pop_front();
    // ws.std() / ws.mean()
    double mean = 0;
    for (const auto& h : hist_) mean += h.w;
    mean /= static_cast<double>(hist_.size());
    double var = 0;
    for (const auto& h : hist_) var += (h.w - mean) * (h.w - mean);
    var /= static_cast<double>(hist_.size());
    if (t - hist_.front().t < arm_s_ - 0.5 / fps_ || std::sqrt(var) / mean >= 0.05) {
      fail = "arming";
    } else {
      Vec ft, fd;
      for (const auto& h : hist_)
        if (t - h.t <= fit_s_) {
          ft.push_back(h.t);
          fd.push_back(h.DL);
        }
      vL = ft.size() >= 8 ? theil_sen(ft, fd) : 0.0;  // d(DL)/dt
      last_v_ = vL;
      if (std::fabs(vL) > vmax_) fail = "lat_speed";
    }
  } else {
    hist_.clear();
  }
  if (!fail.empty() && fail != "state" && fail != "arming") {
    if (active_ >= 0 || t - last_active_t_ < 1.5) {
      if (t > suppress_until_) ++n_cross_;
      suppress_until_ = t + cross_s_;
    }
  }
  if (fail == "lat_speed") hist_.clear();
  fail_ = fail;
  armed_ = fail.empty();
  ++stats_[armed_ ? "armed" : fail];
  LdwStatus out;
  if (!armed_) {
    cnt_[0] = cnt_[1] = 0;
    if (active_ >= 0) last_active_t_ = t;  // disarming cancels an active alert at once
    active_ = -1;
    return out;
  }
  out.armed = true;
  const double d_now[2] = {DL - car_w_ / 2, (lane_w_ - DL) - car_w_ / 2};
  const double speed[2] = {-vL, vL};  // approach speed toward each line (m/s)
  const double DL0 = hist_.front().DL;  // sustained drift: net approach over the last ~1 s
  const double dnet[2] = {DL0 - DL, DL - DL0};
  static const char* const names[2] = {"left", "right"};
  for (int s = 0; s < 2; ++s) {
    const double d = d_now[s], v = speed[s];
    const double tlc = v > 1e-3 ? d / v : std::numeric_limits<double>::infinity();
    const bool cond = d < margin_ + 0.3 * v && d > -0.5 && v >= vmin_ && tlc < 1.0 && dnet[s] >= 0.25 &&
                      t >= cool_[s] && t >= suppress_until_;
    if (cond) {
      ++cnt_[s];
      off_[s] = 0;
      if (cnt_[s] >= on_n_) {
        if (active_ != s) out.event = names[s];
        active_ = s;
      }
    } else {
      cnt_[s] = 0;
      if (active_ == s) {
        ++off_[s];
        if (off_[s] >= off_n_) {
          active_ = -1;
          cool_[s] = t + cool_s_;
          last_active_t_ = t;
        }
      }
    }
    if (active_ == s) {
      out.d = d;
      out.v = v;
      out.tlc = tlc;
    }
  }
  out.side = active_ >= 0 ? names[active_] : "";
  if (active_ >= 0) last_active_t_ = t;
  return out;
}

}  // namespace adasvp
