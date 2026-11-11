#include "models/segmenter.hpp"

#include <stdexcept>
#include <vector>

#include "cuda/cuda_check.hpp"
#include "postprocess/seg_kernels.hpp"
#include "preprocess/letterbox.hpp"

namespace adasvp {

struct Segmenter::Buf {
  uint8_t* d_out = nullptr;
  size_t out_bytes = 0;
  float* d_logits_f32 = nullptr;  // only when the engine output is half
  size_t logits_bytes = 0;
  ~Buf() {
    cudaFree(d_out);
    cudaFree(d_logits_f32);
  }
};

Segmenter::Segmenter(const std::string& onnx, trt::Precision p, const std::string& cache_dir)
    : engine_(std::make_unique<trt::TrtEngine>(onnx, p, cache_dir, std::nullopt)),
      buf_(std::make_unique<Buf>()) {
  const auto in = engine_->shape("pixel_values");
  const auto out = engine_->shape("logits");
  if (in.size() != 4 || in[0] != 1 || in[1] != 3) throw std::runtime_error("unexpected SegFormer input shape");
  if (out.size() != 4 || out[0] != 1) throw std::runtime_error("unexpected SegFormer output shape");
  in_h_ = static_cast<int>(in[2]);
  in_w_ = static_cast<int>(in[3]);
  classes_ = static_cast<int>(out[1]);
  lh_ = static_cast<int>(out[2]);
  lw_ = static_cast<int>(out[3]);
  const auto ti = engine_->dtype("pixel_values"), to = engine_->dtype("logits");
  if (ti != nvinfer1::DataType::kFLOAT && ti != nvinfer1::DataType::kHALF)
    throw std::runtime_error("unsupported SegFormer input dtype");
  if (to != nvinfer1::DataType::kFLOAT && to != nvinfer1::DataType::kHALF)
    throw std::runtime_error("unsupported SegFormer output dtype");
  in_half_ = ti == nvinfer1::DataType::kHALF;
  out_half_ = to == nvinfer1::DataType::kHALF;
}

Segmenter::~Segmenter() = default;

cv::Mat Segmenter::segment(const cv::Mat& bgr, cudaStream_t stream) {
  if (bgr.empty()) throw std::invalid_argument("segment: empty frame");
  const size_t npix = static_cast<size_t>(bgr.cols) * bgr.rows;
  if (npix > buf_->out_bytes) {
    cudaFree(buf_->d_out);
    buf_->d_out = nullptr;
    buf_->out_bytes = 0;
    ADASVP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf_->d_out), npix));
    buf_->out_bytes = npix;
  }
  const size_t nlog = static_cast<size_t>(classes_) * lh_ * lw_;
  if (out_half_ && nlog * sizeof(float) > buf_->logits_bytes) {
    cudaFree(buf_->d_logits_f32);
    buf_->d_logits_f32 = nullptr;
    buf_->logits_bytes = 0;
    ADASVP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf_->d_logits_f32), nlog * sizeof(float)));
    buf_->logits_bytes = nlog * sizeof(float);
  }

  segformer_to_device(bgr, in_w_, in_h_, engine_->deviceBuffer("pixel_values"),
                      in_half_, stream);
  engine_->enqueue(stream);

  const float* logits = nullptr;
  if (out_half_) {
    seg_half_to_float(engine_->deviceBuffer("logits"), buf_->d_logits_f32, nlog, stream);
    logits = buf_->d_logits_f32;
  } else {
    logits = static_cast<const float*>(engine_->deviceBuffer("logits"));
  }
  seg_upsample_argmax(logits, classes_, lh_, lw_, buf_->d_out, bgr.rows, bgr.cols, stream);

  cv::Mat out(bgr.rows, bgr.cols, CV_8UC1);
  ADASVP_CUDA_CHECK(cudaMemcpyAsync(out.data, buf_->d_out, npix, cudaMemcpyDeviceToHost, stream));
  ADASVP_CUDA_CHECK(cudaStreamSynchronize(stream));
  return out;
}

}  // namespace adasvp
