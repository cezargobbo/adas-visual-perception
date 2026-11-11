#pragma once
// SegFormer (Cityscapes) TensorRT segmenter. Not part of the public API.
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "trt/trt_engine.hpp"

namespace adasvp {

class Segmenter {
 public:
  Segmenter(const std::string& onnx, trt::Precision p, const std::string& cache_dir);
  ~Segmenter();
  Segmenter(const Segmenter&) = delete;
  Segmenter& operator=(const Segmenter&) = delete;

  // Returns a CV_8UC1 class map (Cityscapes train ids) of the frame size. Synchronises `stream`.
  cv::Mat segment(const cv::Mat& bgr, cudaStream_t stream);

 private:
  struct Buf;  // device scratch for the class map (grows with the frame)
  std::unique_ptr<trt::TrtEngine> engine_;
  std::unique_ptr<Buf> buf_;
  int in_w_ = 0, in_h_ = 0, classes_ = 0, lh_ = 0, lw_ = 0;
  bool in_half_ = false, out_half_ = false;
};

}  // namespace adasvp
