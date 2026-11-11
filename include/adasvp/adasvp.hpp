#pragma once
// Public API of the adasvp library. Only OpenCV and the STL may be included here.
#include <opencv2/core.hpp>

#include <vector>

namespace adasvp {

struct Box {
  float x1, y1, x2, y2;
};

struct Object {
  Box box;
  int cls;
  float score;
  int track_id;  // -1 if untracked
};

using Polyline = std::vector<cv::Point2f>;

struct Timings {
  double det_ms = 0, seg_ms = 0, lanes_ms = 0, track_ms = 0, total_ms = 0;
};

struct FrameResult {
  double t = 0;
  std::vector<Object> objects;
  std::vector<Polyline> lanes;
  std::vector<float> lane_scores;
  cv::Mat seg_classes;
  Timings timings;
};

}  // namespace adasvp
