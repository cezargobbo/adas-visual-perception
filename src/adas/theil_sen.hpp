#pragma once
// Robust slope dy/dt (median of pairwise slopes). Port of theil_sen of the Python prototype (dashcam_vision.py).

#include <algorithm>
#include <vector>

namespace adasvp {

// np.median: for an even count the mean of the two middle values.
inline double median_of(std::vector<double> v) {
  const size_t n = v.size();
  const size_t k = n / 2;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  const double hi = v[k];
  if (n % 2) return hi;
  const double lo = *std::max_element(v.begin(), v.begin() + k);
  return (lo + hi) / 2.0;
}

inline double theil_sen(const std::vector<double>& t, const std::vector<double>& y) {
  std::vector<double> s;
  const size_t n = t.size();
  for (size_t i = 0; i < n; ++i) {      // np.triu_indices(n, 1): i < j in row-major order
    for (size_t j = i + 1; j < n; ++j) {
      const double dt = t[j] - t[i];
      if (dt > 1e-6) s.push_back((y[j] - y[i]) / dt);
    }
  }
  return s.empty() ? 0.0 : median_of(std::move(s));
}

}  // namespace adasvp
