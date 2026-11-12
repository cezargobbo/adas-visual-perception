#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

#include "cuda/cuda_check.hpp"
#include "models/segmenter.hpp"
#include "postprocess/seg_kernels.hpp"
#include "support/npy.hpp"

namespace fs = std::filesystem;
using namespace adasvp;
using namespace adasvp::test;

namespace {
const std::string kFix = std::string(ADASVP_SOURCE_DIR) + "/tests/fixtures/";

// Pixels whose two largest upsampled channels are equal (double-precision CPU check; forced ties
// are exactly equal logits, so any arithmetic agrees).
std::vector<bool> tied_pixels(const float* lg, int C, int h, int w, int oh, int ow) {
  std::vector<bool> tied(static_cast<size_t>(oh) * ow, false);
  const double sx = static_cast<double>(w) / ow, sy = static_cast<double>(h) / oh;
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x) {
      const double fx = std::max((x + 0.5) * sx - 0.5, 0.0), fy = std::max((y + 0.5) * sy - 0.5, 0.0);
      const int x0 = std::min(static_cast<int>(fx), w - 1), y0 = std::min(static_cast<int>(fy), h - 1);
      const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
      const double lx = fx - x0, ly = fy - y0;
      double b1 = -1e30, b2 = -1e30;
      for (int c = 0; c < C; ++c) {
        const float* p = lg + static_cast<size_t>(c) * h * w;
        const double v = (1 - ly) * ((1 - lx) * p[y0 * w + x0] + lx * p[y0 * w + x1]) +
                         ly * ((1 - lx) * p[y1 * w + x0] + lx * p[y1 * w + x1]);
        if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
      }
      tied[static_cast<size_t>(y) * ow + x] = std::fabs(b1 - b2) < 1e-9;
    }
  return tied;
}
}  // namespace

TEST_CASE("seg upsample+argmax matches torch") {
  const NpyArray lg = load_npy(kFix + "segpost_logits.npy");
  REQUIRE(lg.shape == std::vector<int64_t>{1, 19, 36, 64});
  float* d_lg = nullptr;
  ADASVP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lg), lg.data.size()));
  ADASVP_CUDA_CHECK(cudaMemcpy(d_lg, lg.data.data(), lg.data.size(), cudaMemcpyHostToDevice));

  struct Case { int w, h; const char* file; };
  for (const Case& c : {Case{271, 153, "segpost_argmax_271x153.npy"}, Case{640, 360, "segpost_argmax_640x360.npy"}}) {
    const NpyArray ex = load_npy(kFix + c.file);
    REQUIRE(ex.count() == static_cast<size_t>(c.w) * c.h);
    uint8_t* d_out = nullptr;
    ADASVP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), ex.count()));
    seg_upsample_argmax(d_lg, 19, 36, 64, d_out, c.h, c.w, nullptr);
    std::vector<uint8_t> got(ex.count());
    ADASVP_CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    const auto tied = tied_pixels(lg.as<float>(), 19, 36, 64, c.h, c.w);
    size_t equal = 0, ntied = 0, tied_ok = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool eq = got[i] == ex.as<uint8_t>()[i];
      equal += eq;
      if (tied[i]) { ++ntied; tied_ok += eq; }
    }
    const double frac = static_cast<double>(equal) / got.size();
    std::cout << "[seg] " << c.w << "x" << c.h << " equal=" << frac << " tied=" << tied_ok << "/" << ntied << std::endl;
    CHECK(frac >= 0.999);
    CHECK(ntied > 0);
    CHECK(tied_ok == ntied);
  }
  cudaFree(d_lg);
}

TEST_CASE("Segmenter matches Python prototype dump") {
  const char* vdir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!vdir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path video = fs::path(vdir) / "sp1_crop_30fps.mp4";
  const fs::path dumpdir = fs::path(ADASVP_SOURCE_DIR) / "outputs/dumps/fast/sp1_crop_30fps/seg";
  const fs::path onnx = fs::path(ADASVP_SOURCE_DIR) / "models/segformer-b1-cityscapes-1024x576.onnx";
  if (!fs::exists(video)) SKIP("video not found");
  if (!fs::exists(dumpdir)) SKIP("Python prototype dump not found");
  if (!fs::exists(onnx)) SKIP("segformer ONNX not found");

  const fs::path cache = fs::temp_directory_path() / "adasvp_test_engine_cache_seg";
  Segmenter seg(onnx.string(), trt::Precision::FP16, cache.string());
  cudaStream_t stream;
  ADASVP_CUDA_CHECK(cudaStreamCreate(&stream));

  cv::VideoCapture cap(video.string());
  REQUIRE(cap.isOpened());
  int checked = 0;
  double min_agree = 1.0;
  for (int f = 0;; ++f) {
    cv::Mat frame;
    if (!cap.read(frame)) break;
    char name[32];
    std::snprintf(name, sizeof name, "%06d.png", f);
    const fs::path png = dumpdir / name;
    if (!fs::exists(png)) continue;
    const cv::Mat ref = cv::imread(png.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(ref.type() == CV_8UC1);
    const cv::Mat got = seg.segment(frame, stream);
    REQUIRE(got.size() == ref.size());
    const double agree = 1.0 - static_cast<double>(cv::countNonZero(got != ref)) / ref.total();
    std::cout << "[seg] frame " << f << " agreement=" << agree << std::endl;
    min_agree = std::min(min_agree, agree);
    CHECK(agree >= 0.99);
    ++checked;
  }
  std::cout << "[seg] checked=" << checked << " min=" << min_agree << std::endl;
  REQUIRE(checked > 0);
  cudaStreamDestroy(stream);
}
