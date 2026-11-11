#include "adas/traffic_light.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>

#include "adas/theil_sen.hpp"

namespace adasvp {

namespace {

// _topk_mean: mean of the largest round(frac*n) values, float32 like numpy (the values are integers, so the
// float32 sum is exact and only the final division rounds).
double topk_mean(std::vector<float>& a, double frac = 0.25) {
  std::sort(a.begin(), a.end());
  const int k = std::max(1, static_cast<int>(std::nearbyint(frac * static_cast<double>(a.size()))));
  float s = 0.f;
  for (size_t i = a.size() - static_cast<size_t>(k); i < a.size(); ++i) s += a[i];
  return static_cast<double>(s / static_cast<float>(k));
}

}  // namespace

LightColor traffic_light_state(const cv::Mat& frame, const Box& box, double thr) {
  const int x1 = static_cast<int>(std::nearbyint(static_cast<double>(box.x1)));
  const int y1 = static_cast<int>(std::nearbyint(static_cast<double>(box.y1)));
  const int x2 = static_cast<int>(std::nearbyint(static_cast<double>(box.x2)));
  const int y2 = static_cast<int>(std::nearbyint(static_cast<double>(box.y2)));
  // frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)] (numpy slicing clips to the image)
  const int cx1 = std::min(std::max(x1, 0), frame.cols), cx2 = std::min(std::max(x2, 0), frame.cols);
  const int cy1 = std::min(std::max(y1, 0), frame.rows), cy2 = std::min(std::max(y2, 0), frame.rows);
  if (cx2 <= cx1 || cy2 <= cy1) return LightColor::Unknown;
  const cv::Mat crop = frame(cv::Range(cy1, cy2), cv::Range(cx1, cx2));
  const int h = crop.rows, w = crop.cols;
  if (std::min(h, w) < 4) return LightColor::Unknown;
  const bool vertical = h >= 1.3 * w;
  if (!vertical && w < 1.3 * h) return LightColor::Unknown;  // square box: side view / partial housing

  // Region used, with the long axis first (L rows along the long axis, S samples across).
  int L, S, off = 0;
  if (vertical) {
    int m = static_cast<int>(std::nearbyint(0.15 * w));
    if (!(w - 2 * m >= 2)) m = 0;
    off = m;
    L = h;
    S = w - 2 * m;
  } else {
    int m = static_cast<int>(std::nearbyint(0.15 * h));
    if (!(h - 2 * m >= 2)) m = 0;
    off = m;
    L = w;
    S = h - 2 * m;
  }
  auto px = [&](int l, int s) -> const uchar* {
    return vertical ? crop.ptr<uchar>(l) + 3 * (off + s) : crop.ptr<uchar>(off + s) + 3 * l;
  };

  // cells = [slice(int(k*L/3), max(int((k+1)*L/3), int(k*L/3)+1)) for k in range(3)]
  std::array<std::pair<int, int>, 3> cells;
  for (int k = 0; k < 3; ++k) {
    const int a = static_cast<int>(static_cast<double>(k * L) / 3.0);
    const int b = std::max(static_cast<int>(static_cast<double>((k + 1) * L) / 3.0), a + 1);
    cells[k] = {a, b};
  }
  std::array<double, 3> bright{};
  std::array<std::array<double, 3>, 3> f{};  // [red, yellow, green][cell]
  std::vector<float> v, fr, fy, fg;
  for (int k = 0; k < 3; ++k) {
    v.clear();
    fr.clear();
    fy.clear();
    fg.clear();
    for (int l = cells[k].first; l < std::min(cells[k].second, L); ++l) {
      for (int s = 0; s < S; ++s) {
        const uchar* p = px(l, s);
        const float B = p[0], G = p[1], R = p[2];
        v.push_back(std::max(std::max(B, G), R));
        fr.push_back(R - G);
        fy.push_back(std::min(R, G) - B);
        fg.push_back(G - R);
      }
    }
    bright[k] = topk_mean(v);
    f[0][k] = topk_mean(fr);
    f[1][k] = topk_mean(fy);
    f[2][k] = topk_mean(fg);
  }
  // expected = {"red": (0,) | (0, 2), "yellow": (1,), "green": (2,) | (2, 0)}
  const std::array<std::vector<int>, 3> expected = {
      vertical ? std::vector<int>{0} : std::vector<int>{0, 2}, std::vector<int>{1},
      vertical ? std::vector<int>{2} : std::vector<int>{2, 0}};
  std::array<double, 3> scores{};
  for (int col = 0; col < 3; ++col) {
    double best = -1e9;
    for (int k : expected[col]) {
      int o[2], n = 0;
      for (int j = 0; j < 3; ++j)
        if (j != k) o[n++] = j;
      const double colour = f[col][k] - std::max(f[col][o[0]], f[col][o[1]]);  // colour excess vs. the other cells
      const double light = bright[k] - (bright[o[0]] + bright[o[1]]) / 2.0;     // the lit cell is also brighter
      best = std::max(best, colour + 0.25 * light);
    }
    scores[col] = best;
  }
  int st = 0;  // max(scores, key=scores.get): first maximum
  for (int c = 1; c < 3; ++c)
    if (scores[c] > scores[st]) st = c;
  if (!(scores[st] >= thr)) return LightColor::Unknown;
  return st == 0 ? LightColor::Red : st == 1 ? LightColor::Yellow : LightColor::Green;
}

// ----------------------------------------------------------------------------- TrafficLightTracker
namespace {

double box_iou(const std::array<double, 4>& a, const std::array<double, 4>& b) {
  const double ix = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  const double iy = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const double inter = ix * iy;
  const double ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0.0;
}

}  // namespace

TrafficLightTracker::TrafficLightTracker(double fps, double max_lost_s)
    : max_lost_(std::max(1, static_cast<int>(std::nearbyint(max_lost_s * fps)))) {}

std::array<double, 4> TrafficLightTracker::predict(const TlTrack& tr, double t) {
  const auto& h = tr.hist;
  if (h.size() < 3) return h.back().box;
  const size_t n0 = h.size() > 6 ? h.size() - 6 : 0;
  std::vector<double> tt, cx, cy, lw, lh;
  for (size_t i = n0; i < h.size(); ++i) {
    const auto& b = h[i].box;
    tt.push_back(h[i].t);
    cx.push_back((b[0] + b[2]) / 2);
    cy.push_back((b[1] + b[3]) / 2);
    lw.push_back(std::log(b[2] - b[0]));
    lh.push_back(std::log(b[3] - b[1]));
  }
  double p[4];
  int k = 0;
  for (const auto* v : {&cx, &cy, &lw, &lh}) {
    p[k++] = v->back() + theil_sen(tt, *v) * (t - tt.back());
  }
  const double w = std::exp(p[2]), hh = std::exp(p[3]);
  return {p[0] - w / 2, p[1] - hh / 2, p[0] + w / 2, p[1] + hh / 2};
}

std::vector<TlTrack> TrafficLightTracker::update(double t, const std::vector<TlDet>& dets) {
  std::vector<std::tuple<double, int, int>> pairs;
  for (const auto& [k, tr] : tracks_) {
    const auto pb = predict(tr, t);
    const double pw = pb[2] - pb[0], ph = pb[3] - pb[1];
    const double pcx = (pb[0] + pb[2]) / 2, pcy = (pb[1] + pb[3]) / 2;
    for (size_t d = 0; d < dets.size(); ++d) {
      const auto& b = dets[d].box;
      const double iou = box_iou(pb, b);
      const double dc = std::hypot((b[0] + b[2]) / 2 - pcx, (b[1] + b[3]) / 2 - pcy);
      const double ratio = (b[3] - b[1]) / std::max(ph, 1e-3);
      if ((iou > 0.1 || dc < std::max(pw, ph)) && 0.5 < ratio && ratio < 2.0)
        pairs.emplace_back(iou - 0.01 * dc / std::max(std::max(pw, ph), 1.0), k, static_cast<int>(d));
    }
  }
  std::sort(pairs.begin(), pairs.end(), std::greater<>());  // pairs.sort(reverse=True)
  std::vector<int> used_t;
  std::vector<bool> used_d(dets.size(), false);
  for (const auto& [score, k, d] : pairs) {
    (void)score;
    if (std::find(used_t.begin(), used_t.end(), k) != used_t.end() || used_d[d]) continue;
    used_t.push_back(k);
    used_d[d] = true;
    auto& tr = tracks_[k];
    tr.hist.push_back({t, dets[d].box, dets[d].conf, dets[d].state});
    tr.lost = 0;
  }
  for (size_t d = 0; d < dets.size(); ++d) {
    if (used_d[d]) continue;
    TlTrack tr;
    tr.id = next_id_;
    tr.hist.push_back({t, dets[d].box, dets[d].conf, dets[d].state});
    tracks_[next_id_] = std::move(tr);
    ++next_id_;
  }
  std::vector<TlTrack> closed;
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    auto& tr = it->second;
    if (tr.hist.size() > 120) tr.hist.erase(tr.hist.begin(), tr.hist.end() - 90);  // keep ~3 s of history
    bool pop = false;
    if (tr.hist.back().t != t) {  // not updated this frame (new tracks were)
      ++tr.lost;
      if (tr.lost > max_lost_) pop = true;
    }
    if (pop) {
      closed.push_back(std::move(tr));
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }
  return closed;
}

}  // namespace adasvp
