#include <cuda_fp16.h>

#include "cuda/cuda_check.hpp"
#include "preprocess/kernels.hpp"

namespace adasvp::preprocess {

namespace {

__global__ void bgr_to_chw_f32_kernel(const uint8_t* __restrict__ in, int w, int h, int pitch,
                                      float* __restrict__ out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const uint8_t* p = in + static_cast<size_t>(y) * pitch + x * 3;
  const size_t plane = static_cast<size_t>(w) * h;
  const size_t o = static_cast<size_t>(y) * w + x;
  out[o] = __fdiv_rn(static_cast<float>(p[2]), 255.0f);              // R
  out[plane + o] = __fdiv_rn(static_cast<float>(p[1]), 255.0f);      // G
  out[2 * plane + o] = __fdiv_rn(static_cast<float>(p[0]), 255.0f);  // B
}

__global__ void segformer_pre_kernel(const uint8_t* __restrict__ in, int in_w, int in_h, int pitch,
                                     void* out, int out_w, int out_h, bool fp16_out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;

  // PyTorch upsample_bilinear2d, align_corners=False: scale = in/out (float), clamp src at 0.
  const float sx = static_cast<float>(in_w) / static_cast<float>(out_w);
  const float sy = static_cast<float>(in_h) / static_cast<float>(out_h);
  const float fx = fmaxf((static_cast<float>(x) + 0.5f) * sx - 0.5f, 0.0f);
  const float fy = fmaxf((static_cast<float>(y) + 0.5f) * sy - 0.5f, 0.0f);
  const int x0 = min(static_cast<int>(fx), in_w - 1);
  const int y0 = min(static_cast<int>(fy), in_h - 1);
  const int x1 = min(x0 + 1, in_w - 1);
  const int y1 = min(y0 + 1, in_h - 1);
  const float lx = fx - static_cast<float>(x0), ly = fy - static_cast<float>(y0);
  const float hx = 1.0f - lx, hy = 1.0f - ly;

  const uint8_t* r0 = in + static_cast<size_t>(y0) * pitch;
  const uint8_t* r1 = in + static_cast<size_t>(y1) * pitch;
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stdv[3] = {0.229f, 0.224f, 0.225f};
  const size_t plane = static_cast<size_t>(out_w) * out_h;
  const size_t o = static_cast<size_t>(y) * out_w + x;

  for (int c = 0; c < 3; ++c) {  // output channel c = R,G,B -> source byte 2-c
    const int sc = 2 - c;
    const float v00 = __fdiv_rn(static_cast<float>(r0[x0 * 3 + sc]), 255.0f);
    const float v01 = __fdiv_rn(static_cast<float>(r0[x1 * 3 + sc]), 255.0f);
    const float v10 = __fdiv_rn(static_cast<float>(r1[x0 * 3 + sc]), 255.0f);
    const float v11 = __fdiv_rn(static_cast<float>(r1[x1 * 3 + sc]), 255.0f);
    const float top = __fadd_rn(__fmul_rn(hx, v00), __fmul_rn(lx, v01));
    const float bot = __fadd_rn(__fmul_rn(hx, v10), __fmul_rn(lx, v11));
    const float v = __fadd_rn(__fmul_rn(hy, top), __fmul_rn(ly, bot));
    const float n = __fdiv_rn(__fsub_rn(v, mean[c]), stdv[c]);
    if (fp16_out)
      static_cast<__half*>(out)[c * plane + o] = __float2half_rn(n);
    else
      static_cast<float*>(out)[c * plane + o] = n;
  }
}

dim3 grid_for(int w, int h, dim3 block) {
  return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

}  // namespace

void launch_bgr_to_chw_f32(const uint8_t* d_bgr, int w, int h, int pitch_bytes, float* d_out,
                           cudaStream_t stream) {
  const dim3 block(32, 8);
  bgr_to_chw_f32_kernel<<<grid_for(w, h, block), block, 0, stream>>>(d_bgr, w, h, pitch_bytes, d_out);
  ADASVP_CUDA_CHECK(cudaGetLastError());
}

void launch_segformer_pre(const uint8_t* d_bgr, int in_w, int in_h, int pitch_bytes, void* d_out,
                          int out_w, int out_h, bool fp16_out, cudaStream_t stream) {
  const dim3 block(32, 8);
  segformer_pre_kernel<<<grid_for(out_w, out_h, block), block, 0, stream>>>(
      d_bgr, in_w, in_h, pitch_bytes, d_out, out_w, out_h, fp16_out);
  ADASVP_CUDA_CHECK(cudaGetLastError());
}

}  // namespace adasvp::preprocess
