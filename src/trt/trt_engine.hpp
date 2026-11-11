#pragma once
// Internal TensorRT engine wrapper. Not part of the public API.
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace adasvp::trt {

enum class Precision { FP16, FP32 };

struct ShapeProfile {
  std::string input;
  nvinfer1::Dims4 min, opt, max;
};

// Default engine cache directory: ${XDG_CACHE_HOME:-$HOME/.cache}/adasvp/engines
std::string defaultCacheDir();

// Hex SHA-256 of a byte string (used for cache keys; exposed for testing).
std::string sha256Hex(const std::string& data);

class TrtEngine {
 public:
  // Builds (or loads from cache_dir) an engine for the ONNX model. An empty cache_dir selects
  // defaultCacheDir(). Dynamic-shape models require a profile.
  TrtEngine(const std::string& onnx_path, Precision p, const std::string& cache_dir,
            std::optional<ShapeProfile> profile);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // Sets the shape of a dynamic input; (re)allocates buffers only when they must grow.
  // Contract: when a buffer grows, its previous contents are lost and its device pointer changes.
  // Callers must re-query deviceBuffer() and (re)fill all inputs after every setInputShape().
  void setInputShape(const std::string& name, const nvinfer1::Dims4& dims);
  // Device buffer of an input or output tensor. The pointer is only valid until the next
  // setInputShape() call (which may reallocate).
  void* deviceBuffer(const std::string& name);
  // Current shape of a tensor.
  std::vector<int64_t> shape(const std::string& name) const;
  void enqueue(cudaStream_t stream);

 private:
  // RAII device allocation: freed on destruction, so a throwing constructor cannot leak.
  struct Buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
    Buffer& operator=(Buffer&& o) noexcept {
      if (this != &o) { release(); ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr; o.bytes = 0; }
      return *this;
    }
    ~Buffer() { release(); }
    void release() noexcept {
      if (ptr) cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
  };
  void ensureBuffers();

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::map<std::string, Buffer> buffers_;
};

}  // namespace adasvp::trt
