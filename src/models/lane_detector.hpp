#pragma once
// Polar R-CNN (CurveLanes, DLA34) TensorRT lane detector. Not part of the public API.
// Spec: prototype/python/lanes_polar.py (GlobalPolarHead keep filter, decode, PolarLaneDetector).
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "adasvp/adasvp.hpp"
#include "trt/trt_engine.hpp"

namespace adasvp {

struct RawLanes {
  std::vector<Polyline> lanes;  // frame pixels, ordered top -> bottom
  std::vector<float> scores;    // o2o confidence of each returned lane
};

// pts: (n,72,2) float32 (x,y in the 800x320 crop, bottom -> top), endp: (n,2), o2o: (n).
// Keeps priors with o2o >= thr (order preserved), decodes them like lanes_polar.decode and scales
// to pixels of `frame` (x*W, y*H). Lanes with <= 1 valid point are dropped.
RawLanes decode_polar(const float* pts, const float* endp, const float* o2o, int n, float thr,
                      cv::Size frame);

class LaneDetector {
 public:
  LaneDetector(const std::string& onnx, trt::Precision p, const std::string& cache_dir);
  ~LaneDetector();
  LaneDetector(const LaneDetector&) = delete;
  LaneDetector& operator=(const LaneDetector&) = delete;

  // Synchronises `stream`.
  RawLanes detect(const cv::Mat& bgr, float thr, cudaStream_t stream);

 private:
  std::unique_ptr<trt::TrtEngine> engine_;
  int n_ = 0;
  std::vector<float> h_pts_, h_endp_, h_o2o_;
};

}  // namespace adasvp
