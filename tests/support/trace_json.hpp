#pragma once
// Helpers shared by the tests that replay the reference trace (adas.jsonl / meta.json).

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace adasvp_test {

// Number, or one of the strings "inf" / "-inf" / "nan" the dump uses for non-finite floats.
inline double num(const nlohmann::json& v) {
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    if (s == "inf") return std::numeric_limits<double>::infinity();
    if (s == "-inf") return -std::numeric_limits<double>::infinity();
    return std::numeric_limits<double>::quiet_NaN();
  }
  return v.get<double>();
}

inline std::vector<double> vec(const nlohmann::json& a) {
  std::vector<double> r;
  for (const auto& v : a) r.push_back(num(v));
  return r;
}

// Max |a-b| of two vectors that must have the same length (infinity when they do not).
inline double vdiff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace adasvp_test
