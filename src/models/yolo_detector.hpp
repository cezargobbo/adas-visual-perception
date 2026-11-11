#pragma once
// YOLO26 (end-to-end, NMS baked in) TensorRT detector. Not part of the public API.
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "adasvp/adasvp.hpp"
#include "preprocess/letterbox.hpp"
#include "trt/trt_engine.hpp"

namespace adasvp {

// Decodes a (300, 6) network output (x1,y1,x2,y2,score,cls in letterbox coordinates).
// Keeps score > conf and cls in `classes`, undoes the letterbox and clips to [0,W]x[0,H]
// (Ultralytics scale_boxes / clip_boxes). Rows with score 0 (padding) are never kept.
// Objects come out with track_id = -1.
std::vector<Object> decode_yolo(const float* out300x6, const LetterboxInfo& info, cv::Size frame,
                                float conf, const std::vector<int>& classes);

class YoloDetector {
 public:
  // opt_imgsz_w/h: optimal input size of the shape profile (min 320x320, max 1280x1280).
  YoloDetector(const std::string& onnx, trt::Precision p, const std::string& cache_dir,
               int opt_imgsz_w, int opt_imgsz_h);
  ~YoloDetector();
  YoloDetector(const YoloDetector&) = delete;
  YoloDetector& operator=(const YoloDetector&) = delete;

  std::vector<Object> detect(const cv::Mat& bgr, int imgsz, float conf,
                             const std::vector<int>& classes, cudaStream_t stream);

 private:
  std::unique_ptr<trt::TrtEngine> engine_;
};

}  // namespace adasvp
