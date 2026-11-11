#pragma once
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace adasvp {

inline void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error '") + cudaGetErrorString(err) + "' in " +
                             expr + " at " + file + ":" + std::to_string(line));
  }
}

}  // namespace adasvp

#define ADASVP_CUDA_CHECK(expr) ::adasvp::cudaCheck((expr), #expr, __FILE__, __LINE__)
