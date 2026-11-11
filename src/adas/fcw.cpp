#include "adas/fcw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "adas/theil_sen.hpp"

namespace adasvp {

FCWMonitor::FCWMonitor(int W, int H, double fps, double ttc_alert, double ttc_crit, double win)
    : W_(W), H_(H), fps_(fps), ta_(ttc_alert), tc_(ttc_crit), win_(win) {
  on_n_ = std::max(1, static_cast<int>(0.3 * fps));
  off_n_ = std::max(1, static_cast<int>(0.5 * fps));
  reset();
}

void FCWMonitor::reset() {
  hist_.clear();
  lead_id_.reset();
  since_ = 0;
  miss_ = 0;
  ttcs_.clear();
}

FcwStatus FCWMonitor::update(double t, const std::optional<LeadVehicle>& lead) {
  std::optional<double> ttc;
  if (!lead) {
    ++miss_;
    if (miss_ > 0.3 * fps_) reset();
  } else {
    const int tid = lead->track_id;
    const double dist = lead->distance_m;
    miss_ = 0;
    if (!lead_id_ || tid != *lead_id_) {
      reset();
      lead_id_ = tid;
      since_ = t;
    }
    const double x1 = lead->box.x1, y1 = lead->box.y1, x2 = lead->box.x2, y2 = lead->box.y2;
    const double w = x2 - x1, h = y2 - y1;
    const bool cut = x1 <= 2 || x2 >= W_ - 2 || y2 >= H_ - 2 || y1 <= 2;
    // off-path (parked car beside the lane that we pass): box centre too far from the image centre in box widths
    const bool off_path = std::fabs(0.5 * (x1 + x2) - W_ / 2.0) > 1.0 * w;
    if (cut || w < 40 || h < 10 || off_path) {
      hist_.clear();  // unreliable scale -> restart the window
    } else {
      hist_.push_back({t, std::log(w), dist, std::log(h)});
      while (!hist_.empty() && t - hist_.front().t > win_) hist_.pop_front();
      if (static_cast<int>(hist_.size()) >= std::max(8, static_cast<int>(0.55 * win_ * fps_)) &&
          t - since_ >= 0.5 && dist < 40.0 && hist_.back().t - hist_.front().t >= 0.6 * win_) {
        std::vector<double> ts, lw, zs, lh;
        for (const auto& s : hist_) {
          ts.push_back(s.t);
          lw.push_back(s.lw);
          zs.push_back(s.z);
          lh.push_back(s.lh);
        }
        const double r = theil_sen(ts, lw);
        const double rh = theil_sen(ts, lh);  // the height must expand consistently (turning vehicles only widen)
        const double vz = -theil_sen(ts, zs);  // closing speed from the distance track (m/s)
        if (r > 0.03 && vz >= 0.5 && dist / vz < 2.5 * ta_ && 0.5 * r <= rh && rh <= 2.0 * r + 0.05 &&
            dist * r <= 14.0) {  // implied closing speed <= 50 km/h (bbox noise otherwise)
          ttc = 1.0 / r;
        }
      }
    }
  }
  if (ttc) {
    ttcs_.push_back(*ttc);
    if (ttcs_.size() > 5) ttcs_.pop_front();
    ttc = median_of(std::vector<double>(ttcs_.begin(), ttcs_.end()));
  } else {
    ttcs_.clear();
  }
  const bool cond = ttc.has_value() && *ttc < ta_;
  const std::string lvl = !cond ? "" : (*ttc < tc_ ? "critical" : "warning");
  FcwStatus out;
  if (cond) {
    ++cnt_on_;
    cnt_off_ = 0;
    if (cnt_on_ >= on_n_) {
      if (!active_ || (lvl == "critical" && level_ != "critical")) out.event = lvl;
      active_ = true;
      if (lvl == "critical" || level_.empty()) level_ = lvl;
    }
  } else {
    cnt_on_ = 0;
    ++cnt_off_;
    if (active_ && cnt_off_ >= off_n_) {
      active_ = false;
      level_.clear();
    }
  }
  out.ttc = ttc;
  out.active = active_;
  out.level = active_ ? level_ : "";
  return out;
}

}  // namespace adasvp
