#pragma once
// CUDA kernels for SegFormer post-processing (device pointers, asynchronous on `stream`).
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace adasvp {

// d_logits: (C, h, w) float32, C-order. Bilinear upsample per channel (align_corners=False,
// PyTorch formula, float32) to out_h x out_w, then argmax over channels into d_out (out_h*out_w
// bytes). Ties resolve to the lowest channel index (torch.argmax). C must be <= 256.
void seg_upsample_argmax(const float* d_logits, int C, int h, int w, uint8_t* d_out, int out_h,
                         int out_w, cudaStream_t stream);

// Converts n __half values (d_half, device) to float32 (for engines with fp16 logits).
void seg_half_to_float(const void* d_half, float* d_out, size_t n, cudaStream_t stream);

}  // namespace adasvp
