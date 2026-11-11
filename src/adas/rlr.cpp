#include "adas/rlr.hpp"

#include <algorithm>
#include <cmath>

#include "adas/theil_sen.hpp"

namespace adasvp {

RLRMonitor::RLRMonitor(int W, int H, double fps, std::pair<double, double> zone, double top_cut, double top_band,
                       double min_track_s, double min_growth, double red_min_s, double cooldown_s)
    : W_(W),
      H_(H),
      zone_(zone),
      top_cut_(top_cut),
      top_band_(top_band),
      min_track_s_(min_track_s),
      min_growth_(min_growth),
      red_min_s_(red_min_s),
      cooldown_s_(cooldown_s),
      trk_(fps) {}

// Python: `b[1] > self.top_cut * H` with b a float32 numpy box compares in float32 (the Python float is
// weak under NumPy 2 promotion), so the threshold is rounded to float32 here too.
bool RLRMonitor::below_top_cut(float y1) const { return y1 > static_cast<float>(top_cut_ * H_); }

std::optional<RlrEvent> RLRMonitor::evaluate(const TlTrack& tr) const {
  const double H = H_, W = W_;
  std::vector<const TlEntry*> full;  // not cut by the top edge
  for (const auto& e : tr.hist)
    if (e.box[1] > top_cut_ * H) full.push_back(&e);
  if (full.size() < 5) return std::nullopt;
  const TlEntry& last = *full.back();
  const auto& b = last.box;
  const double cx = (b[0] + b[2]) / 2 / W;
  if (!(zone_.first <= cx && cx <= zone_.second)) return std::nullopt;
  if (last.t - full.front()->t < min_track_s_ || b[3] - b[1] < 10) return std::nullopt;
  std::vector<double> tt, ly, cyv;
  for (const auto* e : full) {
    if (e->t >= last.t - 1.0) {
      tt.push_back(e->t);
      ly.push_back(std::log(e->box[3] - e->box[1]));
      cyv.push_back((e->box[1] + e->box[3]) / 2 / H);
    }
  }
  const double g = theil_sen(tt, ly);
  const double vy = theil_sen(tt, cyv);
  if (g < min_growth_ || vy > -0.03) return std::nullopt;
  // colour: ignore the last 0.2 s before the light touches the top edge
  std::vector<const TlEntry*> col;
  for (const auto* e : full)
    if (e->t <= last.t - 0.2) col.push_back(e);
  if (col.empty()) return std::nullopt;
  int n = 0, n_red = 0, n_other = 0;
  for (const auto* e : col) {
    if (e->t >= last.t - 2.2) {
      ++n;
      n_red += e->state == LightColor::Red;
      n_other += e->state == LightColor::Green || e->state == LightColor::Yellow;
    }
  }
  if (n_red < std::max(6.0, 0.5 * n) || n_other > 0.15 * n) return std::nullopt;
  const TlEntry* recent_last = nullptr;
  for (const auto* e : col)
    if (e->t >= last.t - 0.7 && e->state != LightColor::Unknown) recent_last = e;
  if (!recent_last || recent_last->state != LightColor::Red) return std::nullopt;
  double red_since = last.t;
  for (auto it = col.rbegin(); it != col.rend(); ++it) {
    const LightColor s = (*it)->state;
    if (s == LightColor::Green || s == LightColor::Yellow) break;
    if (s == LightColor::Red) red_since = (*it)->t;
  }
  const double red_for = last.t - red_since;
  if (red_for < red_min_s_) return std::nullopt;
  RlrEvent ev;
  ev.id = tr.id;
  ev.cx = cx;
  ev.red_for = red_for;
  ev.growth = g;
  ev.box = Box{static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2]), static_cast<float>(b[3])};
  return ev;
}

RlrStatus RLRMonitor::update(double t, const cv::Mat& frame, const std::vector<std::pair<Box, float>>& boxes) {
  std::vector<std::tuple<Box, float, LightColor>> in;
  in.reserve(boxes.size());
  for (const auto& [b, c] : boxes)
    in.emplace_back(b, c, below_top_cut(b.y1) ? traffic_light_state(frame, b) : LightColor::Unknown);
  return update_with_states(t, in);
}

RlrStatus RLRMonitor::update_with_states(double t, const std::vector<std::tuple<Box, float, LightColor>>& boxes) {
  std::vector<TlDet> dets;
  for (const auto& [b, c, st] : boxes) {
    TlDet d;
    d.box = {static_cast<double>(b.x1), static_cast<double>(b.y1), static_cast<double>(b.x2),
             static_cast<double>(b.y2)};
    d.conf = static_cast<double>(c);
    d.state = below_top_cut(b.y1) ? st : LightColor::Unknown;
    dets.push_back(d);
  }
  std::vector<TlTrack> closed = trk_.update(t, dets);
  const double H = H_;
  std::vector<TlTrack*> cands;
  lights_.clear();
  for (auto& [k, tr] : trk_.tracks()) {
    if (tr.lost == 0) {
      const auto& e = tr.hist.back();
      lights_.push_back({Box{static_cast<float>(e.box[0]), static_cast<float>(e.box[1]),
                             static_cast<float>(e.box[2]), static_cast<float>(e.box[3])},
                         static_cast<float>(e.conf), tr.id, e.state});
    }
    if (!tr.done && tr.lost == 0 && tr.hist.back().box[1] <= top_cut_ * H) cands.push_back(&tr);
  }
  for (auto& tr : closed)
    if (!tr.done && tr.hist.back().box[1] < top_band_ * H) cands.push_back(&tr);
  RlrStatus status;
  for (TlTrack* tr : cands) {
    tr->done = true;
    const auto info = evaluate(*tr);
    if (info && t - last_alert_ >= cooldown_s_) {
      last_alert_ = t;
      status.event = info;
    }
  }
  return status;
}

}  // namespace adasvp
