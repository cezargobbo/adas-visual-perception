#include "adas/ego_lane.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace adasvp {

namespace {

using Vec = std::vector<double>;

// np.mean(np.abs(a[n:] - b[n:]))
double mean_abs_diff_tail(const Vec& a, const Vec& b, size_t n) {
  double s = 0;
  for (size_t i = n; i < a.size(); ++i) s += std::fabs(a[i] - b[i]);
  return s / static_cast<double>(a.size() - n);
}

// np.polyder
Vec polyder(const Vec& c) {
  const int n = static_cast<int>(c.size()) - 1;
  Vec d(n);
  for (int i = 0; i < n; ++i) d[i] = c[i] * (n - i);
  return d;
}

double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

// np.interp for increasing xp (clamped outside the range)
double interp(double x, const Vec& xp, const Vec& fp) {
  if (x <= xp.front()) return fp.front();
  if (x >= xp.back()) return fp.back();
  const size_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();
  const double slope = (fp[j] - fp[j - 1]) / (xp[j] - xp[j - 1]);
  return slope * (x - xp[j - 1]) + fp[j - 1];
}

}  // namespace

// np.polyfit(y, x, deg): scaled Vandermonde least squares (SVD), coefficients highest power first.
Vec polyfit(const Vec& y, const Vec& x, int deg) {
  const int m = static_cast<int>(y.size()), n = deg + 1;
  cv::Mat A(m, n, CV_64F), b(m, 1, CV_64F);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) A.at<double>(i, j) = std::pow(y[i], deg - j);
    b.at<double>(i) = x[i];
  }
  Vec scale(n);
  for (int j = 0; j < n; ++j) {
    double s = 0;
    for (int i = 0; i < m; ++i) s += A.at<double>(i, j) * A.at<double>(i, j);
    scale[j] = std::sqrt(s);
    for (int i = 0; i < m; ++i) A.at<double>(i, j) /= scale[j];
  }
  cv::Mat c;
  cv::solve(A, b, c, cv::DECOMP_SVD);
  Vec out(n);
  for (int j = 0; j < n; ++j) out[j] = c.at<double>(j) / scale[j];
  return out;
}

double polyval(const Vec& c, double x) {
  double r = 0;
  for (double v : c) r = r * x + v;
  return r;
}

std::optional<Vec> lane_x_at(const Polyline& lane, const Vec& ys, int H) {
  const size_t n = lane.size();
  Vec x(n), y(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = lane[i].x;
    y[i] = lane[i].y;
  }
  if (n < 5) return std::nullopt;
  const double ymin = *std::min_element(y.begin(), y.end());
  const double ymax = *std::max_element(y.begin(), y.end());
  const double ptp = ymax - ymin;
  if (ptp < 0.15 * H || ymax < 0.75 * H || ymin > 0.85 * H) return std::nullopt;
  const Vec c = polyfit(y, x, ptp > 0.25 * H ? 2 : 1);
  const Vec dc = polyder(c);
  Vec out(ys.size());
  for (size_t i = 0; i < ys.size(); ++i) {
    const double yc = clip(ys[i], ymin, ymax);
    const double slope = clip(polyval(dc, yc), -2.0, 2.0);
    out[i] = polyval(c, yc) + slope * (ys[i] - yc);
  }
  return out;
}

EgoLaneTracker::EgoLaneTracker(int W, int H, double fps, double cam_height, double horizon_frac, double hold_s,
                               int n_rows)
    : W_(W), H_(H), alpha_(0.35) {
  y_hor_ = horizon_frac * H;
  // np.linspace(start, H - 1, n_rows)
  const double start = std::max(horizon_frac + 0.04, 0.63) * H, stop = H - 1;
  ys_.resize(n_rows);
  const double step = n_rows > 1 ? (stop - start) / (n_rows - 1) : 0.0;
  for (int i = 0; i < n_rows; ++i) ys_[i] = start + i * step;
  if (n_rows > 1) ys_.back() = stop;
  t_row_.resize(n_rows);
  for (int i = 0; i < n_rows; ++i) t_row_[i] = (ys_[i] - y_hor_) / (H - 1 - y_hor_);
  const double scale = (H - 1 - y_hor_) / cam_height;
  wmin_ = 2.7 * scale * 0.75;
  wmax_ = 4.0 * scale * 1.10;
  wdef_.resize(n_rows);
  for (int i = 0; i < n_rows; ++i) wdef_[i] = 3.4 * scale * t_row_[i];
  gate_ = 0.07 * W;
  persist_ = std::max(3, static_cast<int>(0.3 * fps));
  hold_frames_ = std::max(1, static_cast<int>(hold_s * fps));
  reset();
}

void EgoLaneTracker::reset() {
  xl_.clear();
  xr_.clear();
  wprof_ = wdef_;
  state_ = LaneState::None;
  since_ = 0;
  fade_ = 0.0;
  obs_l_ = obs_r_ = false;
  pend_ = {Pend{}, Pend{}};
  free_ = {0, 0};
  glitch_ = false;
  raw_l_.reset();
  raw_r_.reset();
}

bool EgoLaneTracker::gate_accept(int side, const Vec& obs) {
  const Vec& trk = side == 0 ? xl_ : xr_;
  if (trk.empty() || free_[side] > 0) return true;
  const size_t n = ys_.size() / 2;
  if (mean_abs_diff_tail(obs, trk, n) <= gate_) {
    pend_[side] = Pend{};
    return true;
  }
  glitch_ = true;
  Pend& pd = pend_[side];
  int k = (pd.obs && mean_abs_diff_tail(obs, *pd.obs, n) <= gate_) ? pd.count + 1 : 1;
  pd = Pend{obs, k};
  if (k >= persist_) {  // persistent new position -> re-acquire (blended by the EMA)
    pd = Pend{};
    free_[side] = persist_;
    return true;
  }
  return false;
}

std::pair<std::optional<Vec>, std::optional<Vec>> EgoLaneTracker::pick(const std::vector<Vec>& cands) const {
  std::vector<const Vec*> left, right;
  for (const auto& c : cands) (c.back() < W_ / 2.0 ? left : right).push_back(&c);
  const Vec *bl = nullptr, *br = nullptr;
  double best_cost = 1e18;
  const size_t n = ys_.size();
  const bool tracked = !xl_.empty();
  for (const Vec* l : left) {
    for (const Vec* r : right) {
      Vec w(n);
      for (size_t i = 0; i < n; ++i) w[i] = (*r)[i] - (*l)[i];
      if (!(wmin_ <= w[n - 1] && w[n - 1] <= wmax_)) continue;
      bool bad = false;
      for (size_t i = 0; i < n && !bad; ++i) {
        const double rel = w[i] / w[n - 1];  // must follow the perspective profile
        if (rel < 0.4 * t_row_[i] - 0.02 || rel > 1.8 * t_row_[i] + 0.04) bad = true;
      }
      if (bad) continue;
      const double cost = tracked ? (std::fabs((*l)[n - 1] - xl_[n - 1]) + std::fabs((*r)[n - 1] - xr_[n - 1]))
                                  : w[n - 1];
      if (cost < best_cost) {
        bl = l;
        br = r;
        best_cost = cost;
      }
    }
  }
  if (bl != nullptr) return {*bl, *br};

  // no valid pair: one boundary only
  auto single = [&](const std::vector<const Vec*>& lst, const Vec& trk) -> std::optional<Vec> {
    if (lst.empty()) return std::nullopt;
    const bool has = !trk.empty();
    const double ref = has ? trk[n - 1] : W_ / 2.0;
    const Vec* c = lst[0];
    for (const Vec* a : lst)
      if (std::fabs(a->back() - ref) < std::fabs(c->back() - ref)) c = a;
    const double lim = has ? 2.0 * gate_ : 0.85 * wmax_;
    return std::fabs(c->back() - ref) < lim ? std::optional<Vec>(*c) : std::nullopt;
  };
  return {single(left, xl_), single(right, xr_)};
}

bool EgoLaneTracker::plausible(const Vec& xl, const Vec& xr) const {
  const double w = xr.back() - xl.back(), c = 0.5 * (xr.back() + xl.back());
  return wmin_ <= w && w <= wmax_ && std::fabs(c - W_ / 2.0) <= 0.18 * W_;
}

LaneState EgoLaneTracker::update(const std::vector<Lane>& lanes) {
  glitch_ = false;
  std::vector<Vec> cands;
  for (const auto& p : lanes) {
    auto xs = lane_x_at(p.points, ys_, H_);
    if (xs) cands.push_back(std::move(*xs));
  }
  auto [ol, orr] = pick(cands);
  for (auto& f : free_) f = std::max(0, f - 1);
  if (ol && !gate_accept(0, *ol)) ol.reset();
  if (orr && !gate_accept(1, *orr)) orr.reset();
  const double a = alpha_;
  const size_t n = ys_.size();
  const bool first = xl_.empty();
  Vec nxl, nxr, nw;  // empty = None
  auto ema = [&](const Vec& trk, const Vec& obs) {
    Vec r(n);
    for (size_t i = 0; i < n; ++i) r[i] = (1 - a) * trk[i] + a * obs[i];
    return r;
  };
  if (ol && orr) {
    nxl = first ? *ol : ema(xl_, *ol);
    nxr = first ? *orr : ema(xr_, *orr);
    nw.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const double d = (*orr)[i] - (*ol)[i];
      nw[i] = first ? d : 0.9 * wprof_[i] + 0.1 * d;
    }
  } else if (ol || orr) {
    if (ol) {
      nxl = first ? *ol : ema(xl_, *ol);
      nxr.resize(n);
      for (size_t i = 0; i < n; ++i) nxr[i] = nxl[i] + wprof_[i];
    } else {
      nxr = first ? *orr : ema(xr_, *orr);
      nxl.resize(n);
      for (size_t i = 0; i < n; ++i) nxl[i] = nxr[i] - wprof_[i];
    }
  }
  if (!nxl.empty() && !plausible(nxl, nxr)) {  // drifting / leaving the image -> treat as a miss
    nxl.clear();
    ol.reset();
    orr.reset();
    glitch_ = true;
  }
  raw_l_ = ol;
  raw_r_ = orr;
  obs_l_ = ol.has_value();
  obs_r_ = orr.has_value();
  if (!nxl.empty()) {
    xl_ = std::move(nxl);
    xr_ = std::move(nxr);
    if (!nw.empty()) wprof_ = std::move(nw);
    state_ = (ol && orr) ? LaneState::Detected : LaneState::Partial;
    since_ = 0;
    fade_ = 1.0;
  } else {
    ++since_;
    if (xl_.empty() || since_ > hold_frames_) {
      reset();
    } else {
      state_ = LaneState::Estimated;
      fade_ = std::max(0.0, 1.0 - static_cast<double>(since_) / hold_frames_);
    }
  }
  return state_;
}

bool EgoLaneTracker::contains(const Box& box, double margin) const {
  if (xl_.empty()) return false;
  const double y2 = box.y2;
  if (y2 <= y_hor_ + 5) return false;
  const double cx = 0.5 * (static_cast<double>(box.x1) + box.x2);
  const double y = clip(y2, ys_.front(), ys_.back());
  const double xl = interp(y, ys_, xl_), xr = interp(y, ys_, xr_);
  const double m = std::max(margin * (xr - xl), 0.01 * W_);
  return xl - m <= cx && cx <= xr + m;
}

EgoLane EgoLaneTracker::snapshot() const {
  EgoLane e;
  e.state = state_;
  e.available = available();
  e.fade = fade_;
  e.rows_y = ys_;
  e.xl = xl_;
  e.xr = xr_;
  e.obs_l = obs_l_;
  e.obs_r = obs_r_;
  e.glitch = glitch_;
  return e;
}

}  // namespace adasvp
