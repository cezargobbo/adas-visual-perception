#include "adas/lead_vehicle.hpp"

#include <algorithm>
#include <cmath>

namespace adasvp {

namespace {

bool is_vehicle(int cls) { return cls == 2 || cls == 3 || cls == 5 || cls == 7; }

double real_width_m(int cls) {
  switch (cls) {
    case 2: return 1.80;
    case 3: return 0.80;
    case 5: return 2.55;
    case 7: return 2.40;
    default: return 1.8;  // REAL_WIDTH_M.get(cls, 1.8)
  }
}

constexpr double kPi = 3.14159265358979323846;

// np.median of a non-empty sequence.
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

}  // namespace

LeadVehicleEstimator::LeadVehicleEstimator(int W, int H, double hfov_deg, double cam_height, double horizon_frac,
                                           double corridor_bottom, double corridor_top, int hold_frames)
    : W_(W),
      H_(H),
      f_((W / 2.0) / std::tan(hfov_deg * kPi / 180.0 / 2.0)),
      cam_h_(cam_height),
      y_hor_(horizon_frac * H),
      cb_(corridor_bottom * W),
      ct_(corridor_top * W),
      hold_frames_(hold_frames) {}

double LeadVehicleEstimator::corridor_halfwidth(double y) const {
  const double t = std::clamp((y - y_hor_) / (H_ - y_hor_), 0.0, 1.0);
  return ct_ + t * (cb_ - ct_);
}

bool LeadVehicleEstimator::in_corridor(const Box& box, const EgoLaneTracker* lane) const {
  if (lane != nullptr && lane->available()) return lane->contains(box);
  const double x1 = box.x1, x2 = box.x2, y2 = box.y2;
  const double cx = 0.5 * (x1 + x2);
  if (y2 <= y_hor_ + 5) return false;
  return std::fabs(cx - W_ / 2.0) <= corridor_halfwidth(y2);
}

double LeadVehicleEstimator::distance(const Box& box, int cls) {
  const double x1 = box.x1, y2 = box.y2, x2 = box.x2;
  const double w_px = std::max(x2 - x1, 1.0);
  const double z_width = f_ * real_width_m(cls) / w_px;
  const double dy = y2 - y_hor_;
  const bool has_ground = dy > 8;
  const double z_ground = has_ground ? f_ * cam_h_ / dy : 0.0;
  // bbox cut by the image border -> width cue unreliable
  const bool cut = x1 <= 2 || x2 >= W_ - 2;
  if (!has_ground) return z_width;
  if (cut) return z_ground;
  return 0.6 * z_width + 0.4 * z_ground;
}

std::optional<LeadVehicle> LeadVehicleEstimator::update(const std::vector<Object>& tracked,
                                                        const EgoLaneTracker* lane) {
  std::vector<const Object*> cands;
  for (const auto& d : tracked)
    if (d.track_id >= 0 && is_vehicle(d.cls) && in_corridor(d.box, lane)) cands.push_back(&d);
  // max(key=y2): first maximum wins on ties
  const Object* best = nullptr;
  for (const Object* c : cands)
    if (best == nullptr || c->box.y2 > best->box.y2) best = c;
  // hysteresis: keep the current lead unless another candidate is clearly closer
  if (lead_id_) {
    const Object* cur = nullptr;
    for (const Object* c : cands)
      if (c->track_id == *lead_id_) {
        cur = c;
        break;
      }
    if (cur != nullptr && best != nullptr && (double)best->box.y2 - (double)cur->box.y2 < 0.03 * H_) best = cur;
  }
  if (best == nullptr) {
    ++missing_;
    if (missing_ > hold_frames_) lead_id_.reset();
    return std::nullopt;
  }
  const int tid = best->track_id;
  if (!lead_id_ || tid != *lead_id_) {
    if (lead_id_) ema_.erase(*lead_id_);
  }
  lead_id_ = tid;
  missing_ = 0;
  const double z = distance(best->box, best->cls);
  auto& h = hist_[tid];
  h.push_back(z);
  if (h.size() > 9) h.pop_front();
  const double zm = median(std::vector<double>(h.begin(), h.end()));
  auto it = ema_.find(tid);
  ema_[tid] = it == ema_.end() ? zm : 0.8 * it->second + 0.2 * zm;
  return LeadVehicle{tid, best->cls, best->box, ema_[tid]};
}

}  // namespace adasvp
