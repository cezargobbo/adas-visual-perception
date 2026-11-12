#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "cuda/cuda_check.hpp"
#include "preprocess/letterbox.hpp"
#include "support/hash_image.hpp"
#include "support/mini_json.hpp"
#include "support/npy.hpp"

using namespace adasvp;
using namespace adasvp::test;

namespace {
const std::string kFix = std::string(ADASVP_SOURCE_DIR) + "/tests/fixtures/";

struct Dev {
  float* p = nullptr;
  explicit Dev(size_t n) { ADASVP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(float))); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  std::vector<float> download(size_t n) const {
    std::vector<float> h(n);
    ADASVP_CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
  }
};

size_t numel(const std::vector<double>& s) {
  size_t n = 1;
  for (double d : s) n *= static_cast<size_t>(d);
  return n;
}

cv::Mat image_from_json(const MiniJson& j) {
  return hash_image(static_cast<int>(j.number("height")), static_cast<int>(j.number("width")),
                    static_cast<uint32_t>(j.number("seed")));
}

std::string hex16(uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

cv::Mat npy_to_mat(const NpyArray& a) {
  cv::Mat m(static_cast<int>(a.shape[0]), static_cast<int>(a.shape[1]), CV_8UC3);
  std::memcpy(m.data, a.data.data(), a.data.size());
  return m;
}

// Compares sample tensor[0, :, ::stride, ::stride] against the npy samples. exact -> bit equality.
void check_samples(const std::vector<float>& t, const std::vector<double>& shape, int stride,
                   const NpyArray& samples, double tol) {
  const int C = static_cast<int>(shape[1]), H = static_cast<int>(shape[2]),
            W = static_cast<int>(shape[3]);
  const int sh = (H + stride - 1) / stride, sw = (W + stride - 1) / stride;
  REQUIRE(samples.shape == std::vector<int64_t>{C, sh, sw});
  const float* s = samples.as<float>();
  double worst = 0;
  for (int c = 0; c < C; ++c)
    for (int y = 0; y < sh; ++y)
      for (int x = 0; x < sw; ++x) {
        const float got = t[(static_cast<size_t>(c) * H + y * stride) * W + x * stride];
        worst = std::max(worst, std::abs(static_cast<double>(got) - s[(c * sh + y) * sw + x]));
      }
  CHECK(worst <= tol);
}
}  // namespace

TEST_CASE("letterbox matches Ultralytics") {
  SECTION("tiny cases (full tensors, bit-exact)") {
    for (int k = 0; k < 5; ++k) {
      const std::string name = kFix + "letterbox_tiny" + std::to_string(k);
      INFO(name);
      MiniJson j(name + ".json");
      const cv::Mat img = npy_to_mat(load_npy(name + "_input.npy"));
      const NpyArray exp = load_npy(name + "_expected.npy");
      const int imgsz = static_cast<int>(j.number("imgsz"));
      const auto shape = j.array("shape");
      const auto pad = j.array("pad");

      const LetterboxInfo li = letterbox_shape(img.size(), imgsz);
      CHECK(li.scale == static_cast<float>(j.number("ratio")));
      CHECK(li.pad_left == static_cast<float>(pad[0]));
      CHECK(li.pad_top == static_cast<float>(pad[1]));
      CHECK(li.out_h == static_cast<int>(shape[2]));
      CHECK(li.out_w == static_cast<int>(shape[3]));

      const size_t n = numel(shape);
      Dev d(n);
      LetterboxInfo info{};
      letterbox_to_device(img, imgsz, d.p, nullptr, &info);
      ADASVP_CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(info.scale == li.scale);
      CHECK(info.pad_left == li.pad_left);
      CHECK(info.pad_top == li.pad_top);
      CHECK(info.out_w == li.out_w);
      CHECK(info.out_h == li.out_h);
      REQUIRE(exp.count() == n);
      CHECK(std::memcmp(d.download(n).data(), exp.data.data(), n * sizeof(float)) == 0);
    }
  }

  SECTION("hash images (checksums and samples)") {
    for (auto [w, h] : {std::pair{1920, 1080}, {1280, 720}, {854, 474}}) {
      for (int imgsz : {960, 1280}) {
        const std::string name =
            kFix + "letterbox_" + std::to_string(w) + "x" + std::to_string(h) + "_s" + std::to_string(imgsz);
        INFO(name);
        MiniJson j(name + ".json");
        const cv::Mat img = image_from_json(j);
        const auto shape = j.array("shape");
        const auto pad = j.array("pad");
        const size_t n = numel(shape);
        Dev d(n);
        LetterboxInfo info{};
        letterbox_to_device(img, imgsz, d.p, nullptr, &info);
        ADASVP_CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(info.scale == static_cast<float>(j.number("ratio")));
        CHECK(info.pad_left == static_cast<float>(pad[0]));
        CHECK(info.pad_top == static_cast<float>(pad[1]));
        CHECK(info.out_h == static_cast<int>(shape[2]));
        CHECK(info.out_w == static_cast<int>(shape[3]));
        const auto t = d.download(n);
        const auto cs = bits_checksum(t.data(), n);
        CHECK(hex16(cs.sum) == j.string("bits_sum"));
        CHECK(hex16(cs.wsum) == j.string("bits_wsum"));
        check_samples(t, shape, 32, load_npy(name + "_samples.npy"), 0.0);
      }
    }
  }

  SECTION("letterbox_shape guards") {
    const LetterboxInfo li = letterbox_shape(cv::Size(1920, 1080), 1280);
    CHECK(li.out_w == 1280);
    CHECK(li.out_h == 736);
    CHECK_THROWS(letterbox_shape(cv::Size(0, 10), 640));
  }
}

TEST_CASE("polar preprocess matches the Python prototype") {
  for (auto [w, h] : {std::pair{1920, 1080}, {854, 474}, {1281, 721}}) {
    const std::string name = kFix + "polarpre_" + std::to_string(w) + "x" + std::to_string(h);
    INFO(name);
    MiniJson j(name + ".json");
    const cv::Mat img = image_from_json(j);
    const auto shape = j.array("shape");
    const size_t n = numel(shape);
    Dev d(n);
    polar_to_device(img, d.p, nullptr);
    ADASVP_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t = d.download(n);
    const auto cs = bits_checksum(t.data(), n);
    CHECK(hex16(cs.sum) == j.string("bits_sum"));
    CHECK(hex16(cs.wsum) == j.string("bits_wsum"));
    check_samples(t, shape, 32, load_npy(name + "_samples.npy"), 0.0);
  }
}

TEST_CASE("segformer preprocess matches the Python prototype") {
  SECTION("tiny cases (full tensors)") {
    for (int k = 0; k < 2; ++k) {
      const std::string name = kFix + "segpre_tiny" + std::to_string(k);
      INFO(name);
      MiniJson j(name + ".json");
      const cv::Mat img = npy_to_mat(load_npy(name + "_input.npy"));
      const NpyArray exp = load_npy(name + "_expected.npy");
      const auto tgt = j.array("target");
      const size_t n = exp.count();
      Dev d(n);
      segformer_to_device(img, static_cast<int>(tgt[0]), static_cast<int>(tgt[1]), d.p, false, nullptr);
      ADASVP_CUDA_CHECK(cudaDeviceSynchronize());
      const auto t = d.download(n);
      double worst = 0;
      for (size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(static_cast<double>(t[i]) - exp.as<float>()[i]));
      CHECK(worst <= 1e-5);
    }
  }

  SECTION("hash images (samples)") {
    for (const char* nm : {"1920x1080_to_1024x576", "1920x1080_to_1280x720", "854x474_to_1024x576"}) {
      const std::string name = kFix + "segpre_" + nm;
      INFO(name);
      MiniJson j(name + ".json");
      const cv::Mat img = image_from_json(j);
      const auto shape = j.array("shape");
      const auto tgt = j.array("target");
      const size_t n = numel(shape);
      Dev d(n);
      segformer_to_device(img, static_cast<int>(tgt[0]), static_cast<int>(tgt[1]), d.p, false, nullptr);
      ADASVP_CUDA_CHECK(cudaDeviceSynchronize());
      check_samples(d.download(n), shape, 16, load_npy(name + "_samples.npy"), 1e-5);
    }
  }

  SECTION("fp16 output is the fp32 result rounded to half") {
    const std::string name = kFix + "segpre_tiny0";
    const cv::Mat img = npy_to_mat(load_npy(name + "_input.npy"));
    const NpyArray exp = load_npy(name + "_expected.npy");
    const size_t n = exp.count();
    Dev d(n);  // large enough for n halves
    segformer_to_device(img, 64, 36, d.p, true, nullptr);
    ADASVP_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> h(n);
    ADASVP_CUDA_CHECK(cudaMemcpy(h.data(), d.p, n * 2, cudaMemcpyDeviceToHost));
    cv::Mat src(1, static_cast<int>(n), CV_32F, const_cast<uint8_t*>(exp.data.data()));
    cv::Mat half16(1, static_cast<int>(n), CV_16F, h.data());
    cv::Mat back;
    half16.convertTo(back, CV_32F);
    double worst = 0;
    for (size_t i = 0; i < n; ++i)
      worst = std::max(worst, std::abs(static_cast<double>(back.at<float>(0, static_cast<int>(i))) -
                                       src.at<float>(0, static_cast<int>(i))));
    CHECK(worst <= 2e-3);  // fp16 has ~1e-3 relative precision at |x| <= 2.6
  }
}
