#pragma once
// Port of ultralytics.trackers.byte_tracker.BYTETracker (8.4.208).
#include <memory>
#include <vector>

#include "adasvp/adasvp.hpp"
#include "track/kalman_xyah.hpp"

namespace adasvp {

class ByteTracker {
 public:
  // frame_rate is unused (kept for API stability): like Ultralytics 8.4.208 (max_frames_lost = args.track_buffer),
  // a lost track is dropped when frame_id - last_tracked_frame > params.buffer, whatever the video fps.
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
