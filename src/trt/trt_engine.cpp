#include "trt/trt_engine.hpp"

#include <NvOnnxParser.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "cuda/cuda_check.hpp"

namespace fs = std::filesystem;

namespace adasvp::trt {
namespace {

// ---- SHA-256 (used only to key the engine cache) ----
class Sha256 {
 public:
  Sha256() { reset(); }
  void update(const uint8_t* data, size_t len) {
    total_ += len;
    while (len > 0) {
      size_t n = std::min(len, size_t(64) - fill_);
      std::copy(data, data + n, buf_.begin() + fill_);
      fill_ += n; data += n; len -= n;
      if (fill_ == 64) { block(buf_.data()); fill_ = 0; }
    }
  }
  void update(const std::string& s) { update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
  std::string hexdigest() {
    uint64_t bits = total_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0;
    while (fill_ != 56) update(&zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    update(len, 8);
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h_)
      for (int i = 28; i >= 0; i -= 4) out += hex[(v >> i) & 0xF];
    return out;
  }

 private:
  void reset() {
    h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  }
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
             uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, hh] = h_;
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }
  std::array<uint32_t, 8> h_;
  std::array<uint8_t, 64> buf_{};
  size_t fill_ = 0;
  uint64_t total_ = 0;
};

std::string sha256File(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open ONNX file: " + path);
  Sha256 h;
  std::vector<char> chunk(1 << 20);
  while (f) {
    f.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    h.update(reinterpret_cast<const uint8_t*>(chunk.data()), static_cast<size_t>(f.gcount()));
  }
  return h.hexdigest();
}

std::string dimsStr(const nvinfer1::Dims& d) {
  std::ostringstream s;
  for (int i = 0; i < d.nbDims; ++i) s << (i ? "x" : "") << d.d[i];
  return s.str();
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << std::endl;
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

std::string gpuName() {
  cudaDeviceProp prop{};
  int dev = 0;
  ADASVP_CUDA_CHECK(cudaGetDevice(&dev));
  ADASVP_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return prop.name;
}

std::string cacheKey(const std::string& onnx_path, Precision p, const std::optional<ShapeProfile>& prof) {
  std::string profile_desc = "none";
  if (prof) {
    profile_desc = prof->input + ":" + dimsStr(prof->min) + ":" + dimsStr(prof->opt) + ":" + dimsStr(prof->max);
  }
  std::string desc = sha256File(onnx_path).substr(0, 12) + "_" + (p == Precision::FP16 ? "fp16" : "fp32") +
                     "_trt" + std::to_string(getInferLibVersion()) + "_" + gpuName() + "_" +
                     sha256Hex(profile_desc).substr(0, 8);
  for (char& c : desc)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) c = '-';
  return desc;
}

size_t elemSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kINT64: return 8;
    default: throw std::runtime_error("unsupported TensorRT data type");
  }
}

std::vector<char> readFile(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::vector<char> buildSerialized(const std::string& onnx_path, Precision p,
                                  const std::optional<ShapeProfile>& profile) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
  if (!builder) throw std::runtime_error("failed to create TensorRT builder");
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::string msg = "failed to parse ONNX " + onnx_path;
    for (int i = 0; i < parser->getNbErrors(); ++i) msg += std::string("\n  ") + parser->getError(i)->desc();
    throw std::runtime_error(msg);
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (p == Precision::FP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
  if (profile) {
    auto* op = builder->createOptimizationProfile();
    op->setDimensions(profile->input.c_str(), nvinfer1::OptProfileSelector::kMIN, profile->min);
    op->setDimensions(profile->input.c_str(), nvinfer1::OptProfileSelector::kOPT, profile->opt);
    op->setDimensions(profile->input.c_str(), nvinfer1::OptProfileSelector::kMAX, profile->max);
    if (config->addOptimizationProfile(op) < 0) throw std::runtime_error("invalid optimization profile");
  }
  std::unique_ptr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*network, *config));
  if (!blob) throw std::runtime_error("TensorRT engine build failed for " + onnx_path);
  const char* d = static_cast<const char*>(blob->data());
  return std::vector<char>(d, d + blob->size());
}

// Unique per writer (pid + thread id + random), so concurrent builders never share a temp file.
std::string uniqueSuffix() {
  static std::atomic<unsigned long> counter{0};
  std::random_device rd;
  std::ostringstream o;
  o << ".tmp." << ::getpid() << "." << std::hash<std::thread::id>{}(std::this_thread::get_id()) << "."
    << std::hex << rd() << rd() << "." << counter++;
  return o.str();
}

// Best effort: the cache is an optimization. Returns false (after a warning) when the directory is not
// writable or the write fails; the caller keeps using the in-memory engine.
bool writeAtomic(const fs::path& dst, const std::vector<char>& data) {
  fs::path tmp = dst;
  tmp += uniqueSuffix();
  try {
    {
      std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      if (!f) throw std::runtime_error("cannot open " + tmp.string());
      f.write(data.data(), static_cast<std::streamsize>(data.size()));
      f.flush();
      if (!f) throw std::runtime_error("failed to write " + tmp.string());
    }
    fs::rename(tmp, dst);  // atomic on the same filesystem
    return true;
  } catch (const std::exception& e) {
    std::error_code ec;
    fs::remove(tmp, ec);
    std::cerr << "[adasvp] warning: engine cache not written (" << e.what() << "); continuing without cache"
              << std::endl;
    return false;
  }
}

}  // namespace

std::string sha256Hex(const std::string& data) {
  Sha256 h;
  h.update(data);
  return h.hexdigest();
}

std::string defaultCacheDir() {
  const char* xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg && *xdg) return (fs::path(xdg) / "adasvp" / "engines").string();
  const char* home = std::getenv("HOME");
  return ((home && *home ? fs::path(home) : fs::path(".")) / ".cache" / "adasvp" / "engines").string();
}

struct TrtEngine::Impl {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
};

TrtEngine::TrtEngine(const std::string& onnx_path, Precision p, const std::string& cache_dir,
                     std::optional<ShapeProfile> profile)
    : impl_(std::make_unique<Impl>()) {
  const fs::path dir = cache_dir.empty() ? fs::path(defaultCacheDir()) : fs::path(cache_dir);
  {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) std::cerr << "[adasvp] warning: cannot create engine cache dir " << dir << ": " << ec.message() << std::endl;
  }
  const fs::path file = dir / (fs::path(onnx_path).stem().string() + "_" + cacheKey(onnx_path, p, profile) + ".engine");

  impl_->runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (!impl_->runtime) throw std::runtime_error("failed to create TensorRT runtime");

  std::error_code exists_ec;
  if (fs::exists(file, exists_ec)) {
    auto blob = readFile(file);
    if (!blob.empty()) impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!impl_->engine) {
      std::cerr << "[adasvp] cached engine unusable, rebuilding: " << file << std::endl;
      std::error_code ec;
      fs::remove(file, ec);
    }
  }
  if (!impl_->engine) {
    auto blob = buildSerialized(onnx_path, p, profile);
    writeAtomic(file, blob);
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!impl_->engine) throw std::runtime_error("failed to deserialize freshly built engine");
  }
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw std::runtime_error("failed to create execution context");

  if (profile) impl_->context->setInputShape(profile->input.c_str(), profile->opt);
  ensureBuffers();
}

TrtEngine::~TrtEngine() = default;

void TrtEngine::ensureBuffers() {
  auto& eng = *impl_->engine;
  for (int i = 0; i < eng.getNbIOTensors(); ++i) {
    const char* name = eng.getIOTensorName(i);
    nvinfer1::Dims d = impl_->context->getTensorShape(name);
    size_t n = 1;
    for (int k = 0; k < d.nbDims; ++k) {
      if (d.d[k] < 0) throw std::runtime_error(std::string("tensor shape not resolved: ") + name);
      n *= static_cast<size_t>(d.d[k]);
    }
    const size_t bytes = std::max<size_t>(n * elemSize(eng.getTensorDataType(name)), 1);
    Buffer& b = buffers_[name];
    if (bytes > b.bytes) {
      b.release();
      ADASVP_CUDA_CHECK(cudaMalloc(&b.ptr, bytes));
      b.bytes = bytes;
    }
    if (!impl_->context->setTensorAddress(name, b.ptr))
      throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
  }
}

void TrtEngine::setInputShape(const std::string& name, const nvinfer1::Dims4& dims) {
  if (!impl_->context->setInputShape(name.c_str(), dims))
    throw std::runtime_error("setInputShape rejected " + name + " " + dimsStr(dims));
  ensureBuffers();
}

void* TrtEngine::deviceBuffer(const std::string& name) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) throw std::invalid_argument("unknown tensor: " + name);
  return it->second.ptr;
}

std::vector<int64_t> TrtEngine::shape(const std::string& name) const {
  if (!buffers_.count(name)) throw std::invalid_argument("unknown tensor: " + name);
  nvinfer1::Dims d = impl_->context->getTensorShape(name.c_str());
  return std::vector<int64_t>(d.d, d.d + d.nbDims);
}

nvinfer1::DataType TrtEngine::dtype(const std::string& name) const {
  if (!buffers_.count(name)) throw std::invalid_argument("unknown tensor: " + name);
  return impl_->engine->getTensorDataType(name.c_str());
}

void TrtEngine::enqueue(cudaStream_t stream) {
  if (!impl_->context->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueueV3 failed");
}

}  // namespace adasvp::trt
