#include "preprocess/letterbox.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cuda/cuda_check.hpp"
#include "preprocess/kernels.hpp"

namespace adasvp {

namespace {

struct Geometry {
  double r;
  int new_w, new_h, top, bottom, left, right;
};

// std::nearbyint rounds half to even in the default FP environment, like Python's round().
Geometry compute_geometry(cv::Size frame, int imgsz, int stride) {
  if (frame.width <= 0 || frame.height <= 0) throw std::invalid_argument("letterbox: empty frame");
  if (imgsz <= 0 || stride <= 0) throw std::invalid_argument("letterbox: bad imgsz/stride");
  Geometry g{};
  g.r = std::min(static_cast<double>(imgsz) / frame.height, static_cast<double>(imgsz) / frame.width);
  g.new_w = std::max(static_cast<int>(std::nearbyint(frame.width * g.r)), 1);
  g.new_h = std::max(static_cast<int>(std::nearbyint(frame.height * g.r)), 1);
  const double dw = ((imgsz - g.new_w) % stride) / 2.0;
  const double dh = ((imgsz - g.new_h) % stride) / 2.0;
  g.top = static_cast<int>(std::nearbyint(dh - 0.1));
  g.bottom = static_cast<int>(std::nearbyint(dh + 0.1));
  g.left = static_cast<int>(std::nearbyint(dw - 0.1));
  g.right = static_cast<int>(std::nearbyint(dw + 0.1));
  return g;
}

// Stages a (possibly non-continuous) uint8 BGR image on the device for the duration of the call
// on `stream` (stream-ordered allocation, freed after the kernels already queued).
class DeviceImage {
 public:
  DeviceImage(const cv::Mat& m, cudaStream_t stream) : stream_(stream) {
    if (m.type() != CV_8UC3) throw std::invalid_argument("expected an 8-bit 3-channel BGR image");
    w_ = m.cols;
    h_ = m.rows;
    pitch_ = w_ * 3;
    ADASVP_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_), static_cast<size_t>(pitch_) * h_, stream_));
    // The constructor does not complete if the copy throws, so ~DeviceImage would not run: free here.
    try {
      ADASVP_CUDA_CHECK(cudaMemcpy2DAsync(d_, pitch_, m.data, m.step, pitch_, h_, cudaMemcpyHostToDevice, stream_));
    } catch (...) {
      cudaFreeAsync(d_, stream_);
      d_ = nullptr;
      throw;
    }
  }
  ~DeviceImage() {
    if (d_) cudaFreeAsync(d_, stream_);
  }
  DeviceImage(const DeviceImage&) = delete;
  DeviceImage& operator=(const DeviceImage&) = delete;
  const uint8_t* ptr() const { return d_; }
  int w() const { return w_; }
  int h() const { return h_; }
  int pitch() const { return pitch_; }

 private:
  uint8_t* d_ = nullptr;
  int w_ = 0, h_ = 0, pitch_ = 0;
  cudaStream_t stream_;
};

}  // namespace

LetterboxInfo letterbox_shape(cv::Size frame, int imgsz, int stride) {
  const Geometry g = compute_geometry(frame, imgsz, stride);
  LetterboxInfo li;
  li.scale = static_cast<float>(g.r);
  li.pad_left = static_cast<float>(g.left);
  li.pad_top = static_cast<float>(g.top);
  li.new_w = g.new_w;
  li.new_h = g.new_h;
  li.out_w = g.new_w + g.left + g.right;
  li.out_h = g.new_h + g.top + g.bottom;
  return li;
}

void letterbox_to_device(const cv::Mat& bgr, int imgsz, float* d_nchw, cudaStream_t stream,
                         LetterboxInfo* info) {
  const Geometry g = compute_geometry(bgr.size(), imgsz, 32);
  cv::Mat resized;
  if (bgr.cols != g.new_w || bgr.rows != g.new_h)
    cv::resize(bgr, resized, cv::Size(g.new_w, g.new_h), 0, 0, cv::INTER_LINEAR);
  else
    resized = bgr;
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, g.top, g.bottom, g.left, g.right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  {
    DeviceImage dev(padded, stream);
    // Pageable host memory is staged before cudaMemcpy2DAsync returns, so `padded` may die here.
    preprocess::launch_bgr_to_chw_f32(dev.ptr(), dev.w(), dev.h(), dev.pitch(), d_nchw, stream);
  }
  if (info) *info = letterbox_shape(bgr.size(), imgsz);
}

void polar_to_device(const cv::Mat& bgr, float* d_nchw, cudaStream_t stream) {
  constexpr int kW = 800, kH = 320;
  if (bgr.empty()) throw std::invalid_argument("polar: empty frame");
  const int y_off = static_cast<int>(std::nearbyint(bgr.rows * (640.0 / 1440.0)));
  if (y_off >= bgr.rows) throw std::invalid_argument("polar: frame too small");
  cv::Mat resized;
  cv::resize(bgr.rowRange(y_off, bgr.rows), resized, cv::Size(kW, kH), 0, 0, cv::INTER_LINEAR);
  DeviceImage dev(resized, stream);
  preprocess::launch_bgr_to_chw_f32(dev.ptr(), kW, kH, dev.pitch(), d_nchw, stream);
}

void segformer_to_device(const cv::Mat& bgr, int w, int h, void* d_nchw_or_half, bool fp16_input,
                         cudaStream_t stream) {
  if (bgr.empty() || w <= 0 || h <= 0) throw std::invalid_argument("segformer: bad size");
  DeviceImage dev(bgr, stream);
  preprocess::launch_segformer_pre(dev.ptr(), dev.w(), dev.h(), dev.pitch(), d_nchw_or_half, w, h,
                                   fp16_input, stream);
}

}  // namespace adasvp
