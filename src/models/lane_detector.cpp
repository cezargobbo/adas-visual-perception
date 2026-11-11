#include "models/lane_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cuda/cuda_check.hpp"
#include "preprocess/letterbox.hpp"

namespace adasvp {

namespace {
constexpr int kOriH = 1440, kCutH = 640, kImgH = 320, kImgW = 800, kOffsets = 72;
}

// All arithmetic is float32 where numpy's is (pts are float32 arrays; python scalars do not
// promote them); start/end use float64 (float(e) * int) and round half to even like round().
RawLanes decode_polar(const float* pts, const float* endp, const float* o2o, int n, float thr,
                      cv::Size frame) {
  RawLanes out;
  constexpr int ns = kOffsets - 1;
  const float y_scale = static_cast<float>(static_cast<double>(kOriH - kCutH) / kImgH);
  const float W = static_cast<float>(frame.width), H = static_cast<float>(frame.height);
  for (int k = 0; k < n; ++k) {
    if (!(o2o[k] >= thr)) continue;
    const float* p = pts + static_cast<size_t>(k) * kOffsets * 2;
    const float* e = endp + static_cast<size_t>(k) * 2;
    float xs[kOffsets], ys[kOffsets];
    for (int i = 0; i < kOffsets; ++i) { xs[i] = p[2 * i]; ys[i] = p[2 * i + 1]; }

    const int start = std::min(std::max(0, static_cast<int>(std::nearbyint(static_cast<double>(e[0]) * ns))), ns);
    const int end = std::min(static_cast<int>(std::nearbyint(static_cast<double>(e[1]) * ns)) - 1, kOffsets - 1);
    if (start > 0) {
      // reversed cumprod of (0 <= x <= 1) over xs[:start]
      bool valid = true;
      for (int i = start - 1; i >= 0; --i) {
        valid = valid && (xs[i] >= 0.f && xs[i] <= 1.f);
        if (!valid) xs[i] = -2.f;
      }
    }
    // xs[end+1:] = -2 with Python slice semantics (negative index counts from the end)
    int from = end + 1;
    if (from < 0) from = std::max(0, kOffsets + from);
    for (int i = from; i < kOffsets; ++i) xs[i] = -2.f;

    Polyline lane;
    for (int i = kOffsets - 1; i >= 0; --i) {
      if (!(xs[i] >= 0.f)) continue;
      const float yn = (ys[i] * y_scale + static_cast<float>(kCutH)) / static_cast<float>(kOriH);
      lane.emplace_back((xs[i] / static_cast<float>(kImgW)) * W, yn * H);
    }
    if (lane.size() <= 1) continue;
    out.lanes.push_back(std::move(lane));
    out.scores.push_back(o2o[k]);
  }
  return out;
}

LaneDetector::LaneDetector(const std::string& onnx, trt::Precision p, const std::string& cache_dir)
    : engine_(std::make_unique<trt::TrtEngine>(onnx, p, cache_dir, std::nullopt)) {
  const auto in = engine_->shape("image"), pts = engine_->shape("pts");
  if (in != std::vector<int64_t>{1, 3, kImgH, kImgW}) throw std::runtime_error("unexpected Polar input shape");
  if (pts.size() != 4 || pts[0] != 1 || pts[2] != kOffsets || pts[3] != 2)
    throw std::runtime_error("unexpected Polar pts shape");
  for (const char* name : {"image", "pts", "endp", "o2o"})
    if (engine_->dtype(name) != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error(std::string("Polar tensor is not float32: ") + name);
  n_ = static_cast<int>(pts[1]);
  h_pts_.resize(static_cast<size_t>(n_) * kOffsets * 2);
  h_endp_.resize(static_cast<size_t>(n_) * 2);
  h_o2o_.resize(n_);
}

LaneDetector::~LaneDetector() = default;

RawLanes LaneDetector::detect(const cv::Mat& bgr, float thr, cudaStream_t stream) {
  if (bgr.empty()) throw std::invalid_argument("detect: empty frame");
  polar_to_device(bgr, static_cast<float*>(engine_->deviceBuffer("image")), stream);
  engine_->enqueue(stream);
  ADASVP_CUDA_CHECK(cudaMemcpyAsync(h_pts_.data(), engine_->deviceBuffer("pts"), h_pts_.size() * 4,
                                    cudaMemcpyDeviceToHost, stream));
  ADASVP_CUDA_CHECK(cudaMemcpyAsync(h_endp_.data(), engine_->deviceBuffer("endp"), h_endp_.size() * 4,
                                    cudaMemcpyDeviceToHost, stream));
  ADASVP_CUDA_CHECK(cudaMemcpyAsync(h_o2o_.data(), engine_->deviceBuffer("o2o"), h_o2o_.size() * 4,
                                    cudaMemcpyDeviceToHost, stream));
  ADASVP_CUDA_CHECK(cudaStreamSynchronize(stream));
  return decode_polar(h_pts_.data(), h_endp_.data(), h_o2o_.data(), n_, thr, bgr.size());
}

}  // namespace adasvp
