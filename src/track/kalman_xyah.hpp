#pragma once
// Port of ultralytics.trackers.utils.kalman_filter.KalmanFilterXYAH (8.4.208), float64 math.
#include <array>
#include <utility>
#include <vector>

namespace adasvp {

using Vec4 = std::array<double, 4>;
using Vec8 = std::array<double, 8>;
using Mat8 = std::array<double, 64>;  // row-major 8x8

class KalmanXYAH {
 public:
  KalmanXYAH();
  // measurement = (cx, cy, aspect, h)
  std::pair<Vec8, Mat8> initiate(const Vec4& measurement) const;
  std::pair<Vec8, Mat8> predict(const Vec8& mean, const Mat8& cov) const;
  std::pair<Vec8, Mat8> update(const Vec8& mean, const Mat8& cov, const Vec4& measurement) const;
  void multi_predict(std::vector<Vec8>& means, std::vector<Mat8>& covs) const;

 private:
  double std_pos_, std_vel_;
};

}  // namespace adasvp
