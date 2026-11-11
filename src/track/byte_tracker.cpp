#include "track/byte_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_set>

#include "track/assignment.hpp"

namespace adasvp {
namespace {

enum class State { New, Tracked, Lost, Removed };

struct STrack {
  float tlwh0[4];  // _tlwh (float32), used while mean is not set
  bool has_mean = false;
  Vec8 mean{};
  Mat8 cov{};
  bool is_activated = false;
  State state = State::New;
  float score = 0;
  int cls = 0;
  int track_id = 0;
  int frame_id = 0, start_frame = 0;
  int tracklet_len = 0;

  // tlwh as double (float32 values widen exactly)
  void tlwh(double out[4]) const {
    if (!has_mean) {
      for (int i = 0; i < 4; ++i) out[i] = tlwh0[i];
      return;
    }
    double r[4] = {mean[0], mean[1], mean[2], mean[3]};
    r[2] *= r[3];
    r[0] -= r[2] / 2;
    r[1] -= r[3] / 2;
    for (int i = 0; i < 4; ++i) out[i] = r[i];
  }
  // xyxy rounded to float32 like np.ascontiguousarray(..., dtype=np.float32)
  void xyxy_f32(float out[4]) const {
    if (!has_mean) {  // float32 arithmetic
      out[0] = tlwh0[0];
      out[1] = tlwh0[1];
      out[2] = tlwh0[2] + tlwh0[0];
      out[3] = tlwh0[3] + tlwh0[1];
      return;
    }
    double t[4];
    tlwh(t);
    out[0] = static_cast<float>(t[0]);
    out[1] = static_cast<float>(t[1]);
    out[2] = static_cast<float>(t[2] + t[0]);
    out[3] = static_cast<float>(t[3] + t[1]);
  }
  // convert_coords on the float32 _tlwh of a detection (float32 arithmetic) -> (cx, cy, a, h) widened
  static Vec4 tlwh_to_xyah_f32(const float t[4]) {
    float cx = t[0] + t[2] / 2, cy = t[1] + t[3] / 2, a = t[2] / t[3];
    return {cx, cy, a, t[3]};
  }
  static Vec4 tlwh_to_xyah_f64(const double t[4]) {
    return {t[0] + t[2] / 2, t[1] + t[3] / 2, t[2] / t[3], t[3]};
  }
};
using SP = std::shared_ptr<STrack>;

// iou_distance: 1 - bbox_ioa(iou=True) in float32
std::vector<std::vector<float>> iou_distance(const std::vector<SP>& a, const std::vector<SP>& b) {
  std::vector<std::vector<float>> out(a.size(), std::vector<float>(b.size(), 0.0f));
  if (a.empty() || b.empty()) return out;  // 0xN / Nx0: nothing to compare
  std::vector<std::array<float, 4>> A(a.size()), B(b.size());
  for (size_t i = 0; i < a.size(); ++i) a[i]->xyxy_f32(A[i].data());
  for (size_t j = 0; j < b.size(); ++j) b[j]->xyxy_f32(B[j].data());
  const float eps = 1e-7f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float area1 = (A[i][2] - A[i][0]) * (A[i][3] - A[i][1]);
    for (size_t j = 0; j < b.size(); ++j) {
      const float w = std::min(A[i][2], B[j][2]) - std::max(A[i][0], B[j][0]);
      const float h = std::min(A[i][3], B[j][3]) - std::max(A[i][1], B[j][1]);
      const float inter = std::max(w, 0.0f) * std::max(h, 0.0f);
      float area = (B[j][2] - B[j][0]) * (B[j][3] - B[j][1]);
      area = area + area1 - inter;
      out[i][j] = 1.0f - inter / (area + eps);
    }
  }
  return out;
}

}  // namespace

struct ByteTracker::Impl {
  TrackerParams p;
  int max_time_lost;
  KalmanXYAH kf;
  int frame_id = 0;
  int next_id = 0;
  std::vector<SP> tracked, lost, removed;

  // python float for thresholds that Ultralytics passes to lap (float32 params print back as 0.8, 0.7...)
  static double snap(float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.7g", static_cast<double>(v));
    return std::strtod(buf, nullptr);
  }

  static std::vector<std::vector<double>> to_double(const std::vector<std::vector<float>>& m) {
    std::vector<std::vector<double>> o(m.size());
    for (size_t i = 0; i < m.size(); ++i) o[i].assign(m[i].begin(), m[i].end());
    return o;
  }

  std::vector<std::vector<float>> get_dists(const std::vector<SP>& tr, const std::vector<SP>& det) const {
    auto d = iou_distance(tr, det);
    if (p.fuse_score && !tr.empty() && !det.empty()) {
      for (size_t i = 0; i < d.size(); ++i)
        for (size_t j = 0; j < d[i].size(); ++j) {
          const float iou_sim = 1.0f - d[i][j];
          d[i][j] = 1.0f - iou_sim * det[j]->score;
        }
    }
    return d;
  }

  static std::vector<SP> joint(const std::vector<SP>& a, const std::vector<SP>& b) {
    std::unordered_set<int> ids;
    for (auto& t : a) ids.insert(t->track_id);
    std::vector<SP> r = a;
    for (auto& t : b)
      if (!ids.count(t->track_id)) r.push_back(t);
    return r;
  }
  static std::vector<SP> sub(const std::vector<SP>& a, const std::vector<SP>& b) {
    std::unordered_set<int> ids;
    for (auto& t : b) ids.insert(t->track_id);
    std::vector<SP> r;
    for (auto& t : a)
      if (!ids.count(t->track_id)) r.push_back(t);
    return r;
  }

  void kupdate(STrack& t, const STrack& det, bool reactivate) {
    const Vec4 z = STrack::tlwh_to_xyah_f32(det.tlwh0);
    auto r = kf.update(t.mean, t.cov, z);
    t.mean = r.first;
    t.cov = r.second;
    if (!reactivate) t.tracklet_len += 1;
    else t.tracklet_len = 0;
    t.state = State::Tracked;
    t.is_activated = true;
    t.frame_id = frame_id;
    t.score = det.score;
    t.cls = det.cls;
  }

  void apply_match(const SP& t, const SP& det, std::vector<SP>& activated, std::vector<SP>& refind) {
    if (t->state == State::Tracked) {
      kupdate(*t, *det, false);
      activated.push_back(t);
    } else {
      kupdate(*t, *det, true);
      refind.push_back(t);
    }
  }

  std::vector<Object> update(const std::vector<Object>& dets) {
    ++frame_id;
    std::vector<SP> activated, refind, lost_now, removed_now;

    // _split_detections: float32 xywh, float32 comparisons (numpy 2 weak python scalars)
    std::vector<SP> high, low;
    for (const Object& o : dets) {
      const float x1 = o.box.x1, y1 = o.box.y1, x2 = o.box.x2, y2 = o.box.y2;
      const float cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, w = x2 - x1, h = y2 - y1;
      const bool valid = w > 0 && h > 0;
      const bool is_high = valid && o.score >= p.high;
      const bool is_low = valid && o.score > p.low && o.score < p.high;
      if (!is_high && !is_low) continue;
      auto t = std::make_shared<STrack>();
      t->tlwh0[0] = cx - w / 2;
      t->tlwh0[1] = cy - h / 2;
      t->tlwh0[2] = w;
      t->tlwh0[3] = h;
      t->score = o.score;
      t->cls = o.cls;
      (is_high ? high : low).push_back(t);
    }

    std::vector<SP> unconfirmed, tracked_conf;
    for (auto& t : tracked) (t->is_activated ? tracked_conf : unconfirmed).push_back(t);
    std::vector<SP> pool = joint(tracked_conf, lost);

    // multi_predict (non-Tracked tracks get zero height velocity)
    if (!pool.empty()) {
      std::vector<Vec8> m;
      std::vector<Mat8> c;
      for (auto& t : pool) {
        Vec8 mm = t->mean;
        if (t->state != State::Tracked) mm[7] = 0;
        m.push_back(mm);
        c.push_back(t->cov);
      }
      kf.multi_predict(m, c);
      for (size_t i = 0; i < pool.size(); ++i) {
        pool[i]->mean = m[i];
        pool[i]->cov = c[i];
      }
    }

    // first association
    std::vector<int> u_track, u_det;
    {
      auto d = to_double(get_dists(pool, high));
      auto matches = linear_assignment(d, snap(p.match), &u_track, &u_det, static_cast<int>(high.size()));
      for (auto& m : matches) apply_match(pool[m.first], high[m.second], activated, refind);
    }

    // second association (IoU only, thresh 0.5)
    std::vector<SP> r_tracked;
    for (int i : u_track)
      if (pool[i]->state == State::Tracked) r_tracked.push_back(pool[i]);
    std::vector<int> u2;
    if (!r_tracked.empty() && !low.empty()) {
      auto d = to_double(iou_distance(r_tracked, low));
      auto matches = linear_assignment(d, 0.5, &u2, nullptr, static_cast<int>(low.size()));
      for (auto& m : matches) apply_match(r_tracked[m.first], low[m.second], activated, refind);
    } else {
      for (size_t i = 0; i < r_tracked.size(); ++i) u2.push_back(static_cast<int>(i));
    }
    for (int it : u2) {
      r_tracked[it]->state = State::Lost;
      lost_now.push_back(r_tracked[it]);
    }

    // unconfirmed association (thresh 0.7)
    std::vector<SP> rest;
    for (int i : u_det) rest.push_back(high[i]);
    std::vector<int> u_det2;
    if (unconfirmed.empty()) {
      for (size_t i = 0; i < rest.size(); ++i) u_det2.push_back(static_cast<int>(i));
    } else {
      auto d = to_double(get_dists(unconfirmed, rest));
      std::vector<int> u_unc;
      auto matches = linear_assignment(d, 0.7, &u_unc, &u_det2, static_cast<int>(rest.size()));
      for (auto& m : matches) {
        kupdate(*unconfirmed[m.first], *rest[m.second], false);
        activated.push_back(unconfirmed[m.first]);
      }
      for (int it : u_unc) {
        unconfirmed[it]->state = State::Removed;
        removed_now.push_back(unconfirmed[it]);
      }
    }

    // new tracks
    for (int i : u_det2) {
      const SP& t = rest[i];
      if (t->score < p.new_track) continue;
      t->track_id = ++next_id;
      auto r = kf.initiate(STrack::tlwh_to_xyah_f32(t->tlwh0));
      t->mean = r.first;
      t->cov = r.second;
      t->has_mean = true;
      t->tracklet_len = 0;
      t->state = State::Tracked;
      if (frame_id == 1) t->is_activated = true;
      t->frame_id = frame_id;
      t->start_frame = frame_id;
      activated.push_back(t);
    }

    // remove stale lost
    for (auto& t : lost)
      if (frame_id - t->frame_id > max_time_lost) {
        t->state = State::Removed;
        removed_now.push_back(t);
      }

    // merge_track_pools
    {
      std::vector<SP> keep;
      for (auto& t : tracked)
        if (t->state == State::Tracked) keep.push_back(t);
      tracked = joint(joint(keep, activated), refind);
      lost = sub(lost, tracked);
      lost.insert(lost.end(), lost_now.begin(), lost_now.end());
      lost = sub(lost, removed);
      remove_duplicates();
      removed.insert(removed.end(), removed_now.begin(), removed_now.end());
      if (removed.size() > 1000) removed.erase(removed.begin(), removed.end() - 1000);
    }

    std::vector<Object> out;
    for (auto& t : tracked) {
      if (!t->is_activated) continue;
      double b[4];
      t->tlwh(b);
      Object o;
      o.box = {static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2] + b[0]),
               static_cast<float>(b[3] + b[1])};
      o.cls = t->cls;
      o.score = t->score;
      o.track_id = t->track_id;
      out.push_back(o);
    }
    return out;
  }

  // remove_duplicate_stracks(tracked, lost), dup_thresh 0.15 (float32 comparison)
  void remove_duplicates() {
    auto pd = iou_distance(tracked, lost);
    std::unordered_set<size_t> da, db;
    for (size_t i = 0; i < tracked.size(); ++i)
      for (size_t j = 0; j < lost.size(); ++j)
        if (pd[i][j] < 0.15f) {
          const int tp = tracked[i]->frame_id - tracked[i]->start_frame;
          const int tq = lost[j]->frame_id - lost[j]->start_frame;
          if (tp > tq) db.insert(j);
          else da.insert(i);
        }
    std::vector<SP> ra, rb;
    for (size_t i = 0; i < tracked.size(); ++i)
      if (!da.count(i)) ra.push_back(tracked[i]);
    for (size_t j = 0; j < lost.size(); ++j)
      if (!db.count(j)) rb.push_back(lost[j]);
    tracked = std::move(ra);
    lost = std::move(rb);
  }
};

ByteTracker::ByteTracker(TrackerParams params, double frame_rate) : impl_(std::make_unique<Impl>()) {
  impl_->p = params;
  (void)frame_rate;  // Ultralytics 8.4.208 ignores the frame rate: max_frames_lost = args.track_buffer
  impl_->max_time_lost = params.buffer;
}
ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

std::vector<Object> ByteTracker::update(const std::vector<Object>& dets) { return impl_->update(dets); }

}  // namespace adasvp
