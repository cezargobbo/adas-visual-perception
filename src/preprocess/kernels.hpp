#pragma once
// CUDA kernel launchers for the preprocessing stages (device pointers, asynchronous on `stream`).
#include <cuda_runtime_api.h>

#include <cstdint>

namespace adasvp::preprocess {

// uint8 HWC BGR -> float32 CHW RGB, value / 255 (IEEE division, bit-exact with numpy).
void launch_bgr_to_chw_f32(const uint8_t* d_bgr, int w, int h, int pitch_bytes, float* d_out,
                           cudaStream_t stream);

// uint8 HWC BGR (in_w x in_h) -> RGB, /255, bilinear resize (align_corners=False, PyTorch
// formula) to out_w x out_h, ImageNet mean/std normalisation. Writes fp32 or fp16 NCHW.
void launch_segformer_pre(const uint8_t* d_bgr, int in_w, int in_h, int pitch_bytes, void* d_out,
                          int out_w, int out_h, bool fp16_out, cudaStream_t stream);

}  // namespace adasvp::preprocess
