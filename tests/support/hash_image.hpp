#pragma once
// Regenerates the synthetic images described in tests/fixtures/README.md (lowbias32 hash)
// and provides the order-sensitive checksums over float32 bit patterns.
#include <cstdint>
#include <cstring>
#include <opencv2/core.hpp>

namespace adasvp::test {

inline cv::Mat hash_image(int h, int w, uint32_t seed) {
  cv::Mat img(h, w, CV_8UC3);
  const uint32_t add = seed * 0x9E3779B9u;
  const size_t n = static_cast<size_t>(h) * w * 3;
  uint8_t* p = img.data;
  for (size_t i = 0; i < n; ++i) {
    uint32_t v = static_cast<uint32_t>(i) + add;
    v ^= v >> 16;
    v *= 0x7FEB352Du;
    v ^= v >> 15;
    v *= 0x846CA68Bu;
    v ^= v >> 16;
    p[i] = static_cast<uint8_t>(v >> 24);
  }
  return img;
}

struct BitsChecksum {
  uint64_t sum = 0, wsum = 0;
};

inline BitsChecksum bits_checksum(const float* data, size_t n) {
  BitsChecksum c;
  for (size_t i = 0; i < n; ++i) {
    uint32_t w;
    std::memcpy(&w, data + i, 4);
    c.sum += w;
    c.wsum += static_cast<uint64_t>(i + 1) * w;
  }
  return c;
}

}  // namespace adasvp::test
