#pragma once
// Port of ultralytics.trackers.byte_tracker.BYTETracker (8.4.208).
#include <memory>
#include <vector>

#include "adasvp/adasvp.hpp"
#include "track/kalman_xyah.hpp"

namespace adasvp {

struct TrackerParams {
  float high = 0.25f, low = 0.1f, new_track = 0.25f;
  int buffer = 30;
  float match = 0.8f;
  bool fuse_score = true;
};

class ByteTracker {
 public:
  // Lost tracks are dropped after max_time_lost = int(frame_rate / 30 * buffer) frames.
  ByteTracker(TrackerParams params, double frame_rate);
  ~ByteTracker();
  ByteTracker(ByteTracker&&) noexcept;
  ByteTracker& operator=(ByteTracker&&) noexcept;

  // Detections in frame pixels (already filtered by the caller). Returns the activated tracks of this frame,
  // UNCLIPPED, in Ultralytics' output order. Object::track_id is the track id (starting at 1).
  std::vector<Object> update(const std::vector<Object>& dets);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace adasvp
