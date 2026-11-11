#pragma once
// Preprocessing of camera frames into model input tensors on the GPU.
// Spec: tools/export/preproc.py::letterbox (YOLO), reference/python/lanes_polar.py (Polar R-CNN),
// reference/python/dashcam_vision.py::Segmenter (SegFormer).
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>

namespace adasvp {

struct LetterboxInfo {
  float scale;               // r = min(imgsz/h, imgsz/w)
  float pad_left, pad_top;   // padding added on the left / top, in letterboxed pixels
  int out_w, out_h;          // letterboxed (network input) size
  int new_w, new_h;          // resized (unpadded) image size inside the letterbox
};

// Rectangular letterbox geometry (Ultralytics, auto=True, centred). Throws on empty frame.
LetterboxInfo letterbox_shape(cv::Size frame, int imgsz, int stride = 32);

// BGR uint8 frame -> (3, out_h, out_w) float32 RGB in [0,1] at d_nchw (device, capacity of at
// least out_h*out_w*3 floats; use letterbox_shape for the size). Resize and padding run on the
// CPU (cv::resize INTER_LINEAR, border 114); channel swap, /255 and HWC->CHW on the GPU.
// The call returns once the host frame has been staged; the kernel runs asynchronously on `stream`.
void letterbox_to_device(const cv::Mat& bgr, int imgsz, float* d_nchw, cudaStream_t stream,
                         LetterboxInfo* info);

// Polar R-CNN input: crop rows [round(H*640/1440):], resize to 800x320 (linear), RGB, /255.
// Writes 3*320*800 floats to d_nchw.
void polar_to_device(const cv::Mat& bgr, float* d_nchw, cudaStream_t stream);

// SegFormer input: RGB /255, bilinear (align_corners=False) to w x h, ImageNet mean/std.
// Writes 3*h*w values to the device buffer d_nchw_or_half: float32 (3*h*w*4 bytes) when fp16_input is
// false, __half (3*h*w*2 bytes) when it is true. Pick fp16_input from the engine's input dtype.
void segformer_to_device(const cv::Mat& bgr, int w, int h, void* d_nchw_or_half, bool fp16_input,
                         cudaStream_t stream);

}  // namespace adasvp
