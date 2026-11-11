#pragma once
// Minimal .npy reader (format 1.0/2.0/3.0) for float32, int32, int64 and uint8 in C order.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace adasvp::test {

struct NpyArray {
  std::vector<int64_t> shape;
  std::string dtype;  // "float32", "int32", "int64" or "uint8"
  std::vector<uint8_t> data;

  size_t count() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(data.data());
  }
};

inline NpyArray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not an npy file: " + path);
  uint8_t ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint8_t b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    hlen = b[0] | (b[1] << 8);
  } else {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
  }
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);

  std::smatch m;
  if (!std::regex_search(header, m, std::regex("'descr':\\s*'([^']+)'")))
    throw std::runtime_error("npy: no descr");
  std::string descr = m[1];
  if (!std::regex_search(header, m, std::regex("'fortran_order':\\s*(True|False)")) || m[1] == "True")
    throw std::runtime_error("npy: fortran order not supported");

  NpyArray a;
  if (descr == "<f4") a.dtype = "float32";
  else if (descr == "<i4") a.dtype = "int32";
  else if (descr == "<i8") a.dtype = "int64";
  else if (descr == "|u1" || descr == "<u1") a.dtype = "uint8";
  else throw std::runtime_error("npy: unsupported dtype " + descr);
  size_t esize = a.dtype == "float32" || a.dtype == "int32" ? 4 : a.dtype == "int64" ? 8 : 1;

  if (!std::regex_search(header, m, std::regex("'shape':\\s*\\(([^)]*)\\)")))
    throw std::runtime_error("npy: no shape");
  std::string s = m[1];
  std::regex num("[0-9]+");
  for (auto it = std::sregex_iterator(s.begin(), s.end(), num); it != std::sregex_iterator(); ++it)
    a.shape.push_back(std::stoll(it->str()));

  a.data.resize(a.count() * esize);
  f.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(a.data.size()));
  if (!f) throw std::runtime_error("npy: truncated data in " + path);
  return a;
}

}  // namespace adasvp::test
