#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "cuda/cuda_check.hpp"
#include "support/npy.hpp"
#include "trt/trt_engine.hpp"

namespace fs = std::filesystem;
using adasvp::trt::Precision;
using adasvp::trt::TrtEngine;

namespace {
std::vector<fs::path> engineFiles(const fs::path& dir) {
  std::vector<fs::path> out;
  for (auto& e : fs::directory_iterator(dir)) out.push_back(e.path());
  return out;
}
}  // namespace

TEST_CASE("TrtEngine builds, caches and runs polar") {
  const fs::path onnx = fs::path(ADASVP_SOURCE_DIR) / "models/polarrcnn_curvelanes_dla34.onnx";
  if (!fs::exists(onnx)) SKIP("polar ONNX not found: " << onnx);

  const fs::path cache = fs::temp_directory_path() / "adasvp_test_engine_cache";
  fs::remove_all(cache);

  {
    TrtEngine eng(onnx.string(), Precision::FP32, cache.string(), std::nullopt);
    REQUIRE(eng.shape("pts") == std::vector<int64_t>{1, 50, 72, 2});

    const auto in = eng.shape("image");
    size_t in_n = 1;
    for (auto d : in) in_n *= static_cast<size_t>(d);
    ADASVP_CUDA_CHECK(cudaMemset(eng.deviceBuffer("image"), 0, in_n * sizeof(float)));

    cudaStream_t stream;
    ADASVP_CUDA_CHECK(cudaStreamCreate(&stream));
    eng.enqueue(stream);
    ADASVP_CUDA_CHECK(cudaStreamSynchronize(stream));
    ADASVP_CUDA_CHECK(cudaStreamDestroy(stream));

    for (const char* name : {"pts", "endp", "o2o"}) {
      const auto sh = eng.shape(name);
      size_t n = 1;
      for (auto d : sh) n *= static_cast<size_t>(d);
      std::vector<float> host(n);
      ADASVP_CUDA_CHECK(cudaMemcpy(host.data(), eng.deviceBuffer(name), n * sizeof(float),
                                   cudaMemcpyDeviceToHost));
      for (float v : host) REQUIRE(std::isfinite(v));
    }
  }

  auto files = engineFiles(cache);
  REQUIRE(files.size() == 1);
  const auto mtime = fs::last_write_time(files[0]);

  {  // second construction reuses the cached engine
    TrtEngine eng(onnx.string(), Precision::FP32, cache.string(), std::nullopt);
    REQUIRE(eng.shape("pts") == std::vector<int64_t>{1, 50, 72, 2});
  }
  REQUIRE(engineFiles(cache).size() == 1);
  REQUIRE(fs::last_write_time(files[0]) == mtime);

  // corrupt the cache file (truncate) and expect a silent rebuild
  fs::resize_file(files[0], fs::file_size(files[0]) / 2);
  const auto corrupt_size = fs::file_size(files[0]);
  REQUIRE_NOTHROW(TrtEngine(onnx.string(), Precision::FP32, cache.string(), std::nullopt));
  REQUIRE(engineFiles(cache).size() == 1);
  REQUIRE(fs::file_size(files[0]) > corrupt_size);

  fs::remove_all(cache);
}
