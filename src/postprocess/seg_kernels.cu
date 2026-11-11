#include <cuda_fp16.h>

#include <stdexcept>

#include "cuda/cuda_check.hpp"
#include "postprocess/seg_kernels.hpp"

namespace adasvp {

namespace {

__global__ void upsample_argmax_kernel(const float* __restrict__ logits, int C, int h, int w,
                                       uint8_t* __restrict__ out, int out_h, int out_w) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;

  // PyTorch upsample_bilinear2d, align_corners=False: scale = in/out (float), clamp src at 0.
  const float sx = static_cast<float>(w) / static_cast<float>(out_w);
  const float sy = static_cast<float>(h) / static_cast<float>(out_h);
  const float fx = fmaxf(__fsub_rn(__fmul_rn(__fadd_rn(static_cast<float>(x), 0.5f), sx), 0.5f), 0.0f);
  const float fy = fmaxf(__fsub_rn(__fmul_rn(__fadd_rn(static_cast<float>(y), 0.5f), sy), 0.5f), 0.0f);
  const int x0 = min(static_cast<int>(fx), w - 1);
  const int y0 = min(static_cast<int>(fy), h - 1);
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  const float lx = __fsub_rn(fx, static_cast<float>(x0));
  const float ly = __fsub_rn(fy, static_cast<float>(y0));
  const float hx = __fsub_rn(1.0f, lx), hy = __fsub_rn(1.0f, ly);

  const size_t plane = static_cast<size_t>(h) * w;
  const size_t o00 = static_cast<size_t>(y0) * w + x0, o01 = static_cast<size_t>(y0) * w + x1;
  const size_t o10 = static_cast<size_t>(y1) * w + x0, o11 = static_cast<size_t>(y1) * w + x1;

  float best = 0.f;
  int best_c = 0;
  for (int c = 0; c < C; ++c) {
    const float* p = logits + c * plane;
    const float top = __fadd_rn(__fmul_rn(hx, p[o00]), __fmul_rn(lx, p[o01]));
    const float bot = __fadd_rn(__fmul_rn(hx, p[o10]), __fmul_rn(lx, p[o11]));
    const float v = __fadd_rn(__fmul_rn(hy, top), __fmul_rn(ly, bot));
    if (c == 0 || v > best) {  // strict: ties keep the lowest index
      best = v;
      best_c = c;
    }
  }
  out[static_cast<size_t>(y) * out_w + x] = static_cast<uint8_t>(best_c);
}

__global__ void half_to_float_kernel(const __half* in, float* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __half2float(in[i]);
}

}  // namespace

void seg_half_to_float(const void* d_half, float* d_out, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  half_to_float_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      static_cast<const __half*>(d_half), d_out, n);
  ADASVP_CUDA_CHECK(cudaGetLastError());
}

void seg_upsample_argmax(const float* d_logits, int C, int h, int w, uint8_t* d_out, int out_h,
                         int out_w, cudaStream_t stream) {
  if (C <= 0 || C > 256 || h <= 0 || w <= 0 || out_h <= 0 || out_w <= 0)
    throw std::invalid_argument("seg_upsample_argmax: bad dimensions");
  const dim3 block(32, 8);
  const dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
  upsample_argmax_kernel<<<grid, block, 0, stream>>>(d_logits, C, h, w, d_out, out_h, out_w);
  ADASVP_CUDA_CHECK(cudaGetLastError());
}

}  // namespace adasvp
