#include "track/kalman_xyah.hpp"

#include <cmath>
#include <utility>

namespace adasvp {
namespace {

// Dense helpers on small fixed sizes (no FMA contraction: see CMakeLists).
void motion_predict(Vec8& mean, Mat8& cov, double std_pos, double std_vel) {
  const double h = mean[3];
  const double sp = std_pos * h, sv = std_vel * h;
  const double q[8] = {sp * sp, sp * sp, 1e-2 * 1e-2, sp * sp, sv * sv, sv * sv, 1e-5 * 1e-5, sv * sv};
  // mean = mean @ F.T  (F = I with F[i][4+i] = 1)
  for (int i = 0; i < 4; ++i) mean[i] = mean[i] + mean[4 + i];
  // cov = F P F.T + Q
  Mat8 fp;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) fp[i * 8 + j] = i < 4 ? cov[i * 8 + j] + cov[(i + 4) * 8 + j] : cov[i * 8 + j];
  Mat8 out;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) out[i * 8 + j] = j < 4 ? fp[i * 8 + j] + fp[i * 8 + j + 4] : fp[i * 8 + j];
  for (int i = 0; i < 8; ++i) out[i * 8 + i] += q[i];
  cov = out;
}

}  // namespace

KalmanXYAH::KalmanXYAH() : std_pos_(1.0 / 20), std_vel_(1.0 / 160) {}

std::pair<Vec8, Mat8> KalmanXYAH::initiate(const Vec4& m) const {
  Vec8 mean{};
  for (int i = 0; i < 4; ++i) mean[i] = m[i];
  const double h = m[3];
  const double s[8] = {2 * std_pos_ * h, 2 * std_pos_ * h, 1e-2, 2 * std_pos_ * h,
                       10 * std_vel_ * h, 10 * std_vel_ * h, 1e-5, 10 * std_vel_ * h};
  Mat8 cov{};
  for (int i = 0; i < 8; ++i) cov[i * 8 + i] = s[i] * s[i];
  return {mean, cov};
}

std::pair<Vec8, Mat8> KalmanXYAH::predict(const Vec8& mean, const Mat8& cov) const {
  Vec8 m = mean;
  Mat8 c = cov;
  motion_predict(m, c, std_pos_, std_vel_);
  return {m, c};
}

void KalmanXYAH::multi_predict(std::vector<Vec8>& means, std::vector<Mat8>& covs) const {
  for (size_t k = 0; k < means.size(); ++k) motion_predict(means[k], covs[k], std_pos_, std_vel_);
}

std::pair<Vec8, Mat8> KalmanXYAH::update(const Vec8& mean, const Mat8& cov, const Vec4& z) const {
  // project: S = P[:4,:4] + diag((w*h, w*h, 1e-1, w*h)^2)
  const double sp = std_pos_ * mean[3];
  const double inn[4] = {sp * sp, sp * sp, 1e-1 * 1e-1, sp * sp};
  double S[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) S[i][j] = cov[i * 8 + j] + (i == j ? inn[i] : 0.0);

  // Kalman gain K (8x4) = P[:, :4] S^-1  <=> solve S X = P[:, :4].T, K = X.T.  LU with partial pivoting.
  double A[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) A[i][j] = S[i][j];
  double B[4][8];  // B[i][r] = P[r][i]
  for (int i = 0; i < 4; ++i)
    for (int r = 0; r < 8; ++r) B[i][r] = cov[r * 8 + i];
  for (int c = 0; c < 4; ++c) {
    int p = c;
    for (int r = c + 1; r < 4; ++r)
      if (std::fabs(A[r][c]) > std::fabs(A[p][c])) p = r;
    if (p != c) {
      for (int j = 0; j < 4; ++j) std::swap(A[p][j], A[c][j]);
      for (int j = 0; j < 8; ++j) std::swap(B[p][j], B[c][j]);
    }
    for (int r = c + 1; r < 4; ++r) {
      const double f = A[r][c] / A[c][c];
      for (int j = c; j < 4; ++j) A[r][j] -= f * A[c][j];
      for (int j = 0; j < 8; ++j) B[r][j] -= f * B[c][j];
    }
  }
  for (int j = 0; j < 8; ++j)
    for (int i = 3; i >= 0; --i) {
      double s = B[i][j];
      for (int k = i + 1; k < 4; ++k) s -= A[i][k] * B[k][j];
      B[i][j] = s / A[i][i];
    }
  double K[8][4];
  for (int r = 0; r < 8; ++r)
    for (int i = 0; i < 4; ++i) K[r][i] = B[i][r];

  double innov[4];
  for (int i = 0; i < 4; ++i) innov[i] = z[i] - mean[i];
  Vec8 nm;
  for (int r = 0; r < 8; ++r) {
    double s = 0;
    for (int i = 0; i < 4; ++i) s += innov[i] * K[r][i];
    nm[r] = mean[r] + s;
  }
  // new_cov = P - K S K^T
  double KS[8][4];
  for (int r = 0; r < 8; ++r)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += K[r][k] * S[k][j];
      KS[r][j] = s;
    }
  Mat8 nc;
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += KS[r][k] * K[c][k];
      nc[r * 8 + c] = cov[r * 8 + c] - s;
    }
  return {nm, nc};
}

}  // namespace adasvp
