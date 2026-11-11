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

TEST_CASE("sha256Hex matches known vectors") {
  using adasvp::trt::sha256Hex;
  REQUIRE(sha256Hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  REQUIRE(sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // multi-block message (56 bytes forces a second padding block)
  REQUIRE(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("TrtEngine grows buffers when the dynamic input shape grows") {
  const fs::path onnx = fs::path(ADASVP_SOURCE_DIR) / "models/yolo26s.onnx";
  if (!fs::exists(onnx)) SKIP("yolo26s ONNX not found: " << onnx);

  const fs::path cache = fs::temp_directory_path() / "adasvp_test_engine_cache_yolo";  // reused across runs
  adasvp::trt::ShapeProfile prof{"images", nvinfer1::Dims4{1, 3, 320, 320}, nvinfer1::Dims4{1, 3, 544, 960},
                                 nvinfer1::Dims4{1, 3, 1280, 1280}};
  TrtEngine eng(onnx.string(), Precision::FP16, cache.string(), prof);

  REQUIRE(eng.shape("images") == std::vector<int64_t>{1, 3, 544, 960});
  REQUIRE(eng.shape("output0").size() == 3);

  eng.setInputShape("images", nvinfer1::Dims4{1, 3, 320, 320});
  REQUIRE(eng.shape("images") == std::vector<int64_t>{1, 3, 320, 320});
  void* small = eng.deviceBuffer("images");
  REQUIRE(small != nullptr);

  eng.setInputShape("images", nvinfer1::Dims4{1, 3, 1280, 1280});  // grows: buffer is reallocated
  REQUIRE(eng.shape("images") == std::vector<int64_t>{1, 3, 1280, 1280});
  void* big = eng.deviceBuffer("images");
  REQUIRE(big != nullptr);
  // The enlarged buffer must be writable over its whole extent.
  const size_t bytes = size_t(3) * 1280 * 1280 * sizeof(float);
  REQUIRE(cudaMemset(big, 0, bytes) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  cudaStream_t stream;
  ADASVP_CUDA_CHECK(cudaStreamCreate(&stream));
  REQUIRE_NOTHROW(eng.enqueue(stream));
  ADASVP_CUDA_CHECK(cudaStreamSynchronize(stream));
  ADASVP_CUDA_CHECK(cudaStreamDestroy(stream));

  // Shrinking again keeps the larger allocation (no realloc) and the shape tracks the request.
  eng.setInputShape("images", nvinfer1::Dims4{1, 3, 320, 320});
  REQUIRE(eng.shape("images") == std::vector<int64_t>{1, 3, 320, 320});
  REQUIRE(eng.deviceBuffer("images") == big);
}
