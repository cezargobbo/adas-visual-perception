#include "models/yolo_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cuda/cuda_check.hpp"

namespace adasvp {

std::vector<Object> decode_yolo(const float* out, const LetterboxInfo& info, cv::Size frame,
                                float conf, const std::vector<int>& classes) {
  constexpr int kRows = 300;
  const float W = static_cast<float>(frame.width), H = static_cast<float>(frame.height);
  // Ultralytics scale_boxes (ratio_pad=None): per-axis gains from the rounded resized size and
  // pad = round((out - new)/2 - 0.1) (half-to-even, like Python's round()).
  const double gain_x = static_cast<double>(info.new_w) / frame.width;
  const double gain_y = static_cast<double>(info.new_h) / frame.height;
  const double pad_x = std::nearbyint((info.out_w - info.new_w) / 2.0 - 0.1);
  const double pad_y = std::nearbyint((info.out_h - info.new_h) / 2.0 - 0.1);
  auto clampf = [](double v, float hi) { return static_cast<float>(std::clamp(v, 0.0, static_cast<double>(hi))); };
  std::vector<Object> res;
  for (int i = 0; i < kRows; ++i) {
    const float* r = out + i * 6;
    const float score = r[4];
    if (!(score > conf) || score <= 0.f) continue;
    const int cls = static_cast<int>(r[5]);
    if (std::find(classes.begin(), classes.end(), cls) == classes.end()) continue;
    auto cx = [&](float v) { return clampf((static_cast<double>(v) - pad_x) / gain_x, W); };
    auto cy = [&](float v) { return clampf((static_cast<double>(v) - pad_y) / gain_y, H); };
    Object o;
    o.box = {cx(r[0]), cy(r[1]), cx(r[2]), cy(r[3])};
    o.cls = cls;
    o.score = score;
    o.track_id = -1;
    res.push_back(o);
  }
  return res;
}

YoloDetector::YoloDetector(const std::string& onnx, trt::Precision p, const std::string& cache_dir,
                           int opt_w, int opt_h) {
  trt::ShapeProfile prof{"images", nvinfer1::Dims4{1, 3, 320, 320}, nvinfer1::Dims4{1, 3, opt_h, opt_w},
                         nvinfer1::Dims4{1, 3, 1280, 1280}};
  engine_ = std::make_unique<trt::TrtEngine>(onnx, p, cache_dir, prof);
}

YoloDetector::~YoloDetector() = default;

std::vector<Object> YoloDetector::detect(const cv::Mat& bgr, int imgsz, float conf,
                                         const std::vector<int>& classes, cudaStream_t stream) {
  const LetterboxInfo li = letterbox_shape(bgr.size(), imgsz);
  // Contract: setInputShape may reallocate, so query buffers afterwards and fill the input.
  engine_->setInputShape("images", nvinfer1::Dims4{1, 3, li.out_h, li.out_w});
  LetterboxInfo info{};
  letterbox_to_device(bgr, imgsz, static_cast<float*>(engine_->deviceBuffer("images")), stream, &info);
  engine_->enqueue(stream);

  const auto sh = engine_->shape("output0");
  if (sh.size() != 3 || sh[1] != 300 || sh[2] != 6)
    throw std::runtime_error("unexpected YOLO output shape");
  std::vector<float> host(300 * 6);
  ADASVP_CUDA_CHECK(cudaMemcpyAsync(host.data(), engine_->deviceBuffer("output0"),
                                    host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  ADASVP_CUDA_CHECK(cudaStreamSynchronize(stream));
  return decode_yolo(host.data(), info, bgr.size(), conf, classes);
}

}  // namespace adasvp
