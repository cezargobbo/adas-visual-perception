#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>
#include <set>
#include <string>
#include <vector>

#include "cuda/cuda_check.hpp"
#include "models/lane_detector.hpp"
#include "support/npy.hpp"

namespace fs = std::filesystem;
using namespace adasvp;
using namespace adasvp::test;

namespace {
const std::string kFix = std::string(ADASVP_SOURCE_DIR) + "/tests/fixtures/";

// np.interp for sorted xp (clamped outside the range).
double interp(double x, const std::vector<cv::Point2f>& a) {
  if (x <= a.front().y) return a.front().x;
  if (x >= a.back().y) return a.back().x;
  size_t j = std::upper_bound(a.begin(), a.end(), x, [](double v, const cv::Point2f& p) { return v < p.y; }) - a.begin();
  const auto &p0 = a[j - 1], &p1 = a[j];
  return p0.x + (x - p0.y) * (static_cast<double>(p1.x) - p0.x) / (static_cast<double>(p1.y) - p0.y);
}

// tools/export/validate_polar.py::lane_dx. Returns false when a polyline is empty or the y-ranges do not overlap.
bool lane_dx(Polyline a, Polyline b, std::vector<double>& out) {
  if (a.empty() || b.empty()) return false;
  auto byy = [](const cv::Point2f& p, const cv::Point2f& q) { return p.y < q.y; };
  std::sort(a.begin(), a.end(), byy);
  std::sort(b.begin(), b.end(), byy);
  const double lo = std::max(a.front().y, b.front().y), hi = std::min(a.back().y, b.back().y);
  if (!std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo)) return false;  // "no overlap", never NaN
  std::set<float> ys;
  for (auto& p : a) ys.insert(p.y);
  for (auto& p : b) ys.insert(p.y);
  out.clear();
  for (float y : ys)
    if (y >= lo && y <= hi) out.push_back(std::fabs(interp(y, a) - interp(y, b)));
  return true;
}

struct Match { size_t n_ref, n_cand; double mean; };

// validate_polar.py::match_lanes (greedy minimum mean |dx|).
std::vector<Match> match_lanes(const std::vector<Polyline>& ref, const std::vector<Polyline>& cand) {
  std::vector<Match> out;
  if (ref.empty() || cand.empty()) return out;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> cost(ref.size(), std::vector<double>(cand.size(), inf));
  for (size_t i = 0; i < ref.size(); ++i)
    for (size_t j = 0; j < cand.size(); ++j) {
      std::vector<double> d;
      if (lane_dx(ref[i], cand[j], d) && !d.empty()) {
        double s = 0;
        for (double v : d) s += v;
        cost[i][j] = s / d.size();
      }
    }
  for (size_t it = 0; it < std::min(ref.size(), cand.size()); ++it) {
    size_t bi = 0, bj = 0;
    double best = inf;
    for (size_t i = 0; i < ref.size(); ++i)
      for (size_t j = 0; j < cand.size(); ++j)
        if (cost[i][j] < best) { best = cost[i][j]; bi = i; bj = j; }
    if (!std::isfinite(best)) break;
    out.push_back({ref[bi].size(), cand[bj].size(), best});
    for (size_t j = 0; j < cand.size(); ++j) cost[bi][j] = inf;
    for (size_t i = 0; i < ref.size(); ++i) cost[i][bj] = inf;
  }
  return out;
}
}  // namespace

TEST_CASE("decode_polar matches lanes_polar.decode") {
  for (const char* thr_s : {"0.44", "0.05"}) {
    const float thr = std::stof(thr_s);
    for (int n = 0; n < 4; ++n) {
      const std::string base = kFix + "polar_decode_" + std::to_string(n);
      const std::string pre = base + "_thr" + thr_s;
      const NpyArray pts = load_npy(base + "_pts.npy"), endp = load_npy(base + "_endp.npy"),
                     o2o = load_npy(base + "_o2o.npy");
      const NpyArray px = load_npy(pre + "_lanes_px1920x1080.npy"), off = load_npy(pre + "_offsets.npy"),
                     sc = load_npy(pre + "_scores.npy");
      INFO("case " << n << " thr " << thr_s);
      REQUIRE(pts.shape == std::vector<int64_t>{1, 50, 72, 2});

      const RawLanes r = decode_polar(pts.as<float>(), endp.as<float>(), o2o.as<float>(), 50, thr, {1920, 1080});
      const size_t nl = off.count() ? off.count() - 1 : 0;
      REQUIRE(r.lanes.size() == nl);
      REQUIRE(r.scores.size() == nl);
      for (size_t l = 0; l < nl; ++l) {
        const int a = off.as<int32_t>()[l], b = off.as<int32_t>()[l + 1];
        REQUIRE(r.lanes[l].size() == static_cast<size_t>(b - a));
        CHECK(r.scores[l] == sc.as<float>()[l]);
        float maxd = 0;
        for (int i = a; i < b; ++i) {
          maxd = std::max(maxd, std::fabs(r.lanes[l][i - a].x - px.as<float>()[2 * i]));
          maxd = std::max(maxd, std::fabs(r.lanes[l][i - a].y - px.as<float>()[2 * i + 1]));
        }
        std::cout << "[lanes] decode case " << n << " thr " << thr_s << " lane " << l << " pts " << (b - a)
                  << " max|diff| " << maxd << " px\n";
        CHECK(maxd <= 1e-4f);
      }
      if (n == 3) CHECK(r.lanes.empty());
    }
  }
}

TEST_CASE("decode_polar edge cases") {
  // Python semantics: round half to even, negative slice index for end+1, <=1 point dropped.
  std::vector<float> pts(2 * 72 * 2), endp(4), o2o{0.9f, 0.9f};
  for (int k = 0; k < 2; ++k)
    for (int i = 0; i < 72; ++i) { pts[(k * 72 + i) * 2] = 0.5f; pts[(k * 72 + i) * 2 + 1] = 0.f; }
  // lane 0: start=round(0.5/71*71)... use exact .5 products: e0*71 = 0.5 -> start 0 (half-even)
  endp[0] = 0.5f / 71.f; endp[1] = 1.f;
  // lane 1: end = round(-0.5)-1 = -1 -> end+1 = 0 -> everything invalid -> dropped
  endp[2] = 0.f; endp[3] = -0.5f / 71.f;
  const RawLanes r = decode_polar(pts.data(), endp.data(), o2o.data(), 2, 0.44f, {800, 320});
  REQUIRE(r.lanes.size() == 1);
  CHECK(r.lanes[0].size() == 71);  // end = 71 - 1 = 70 -> xs[71] invalidated
  CHECK(r.scores[0] == 0.9f);
  const RawLanes none = decode_polar(pts.data(), endp.data(), o2o.data(), 2, 0.95f, {800, 320});
  CHECK(none.lanes.empty());
}

TEST_CASE("LaneDetector matches reference dump") {
  const char* vdir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!vdir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path video = fs::path(vdir) / "sp1_crop_30fps.mp4";
  const fs::path dump = fs::path(ADASVP_SOURCE_DIR) / "outputs/dumps/fast/sp1_crop_30fps/frames.jsonl";
  const fs::path onnx = fs::path(ADASVP_SOURCE_DIR) / "models/polarrcnn_curvelanes_dla34.onnx";
  if (!fs::exists(video)) SKIP("video not found");
  if (!fs::exists(dump)) SKIP("reference dump not found");
  if (!fs::exists(onnx)) SKIP("polar ONNX not found");

  const fs::path cache = fs::temp_directory_path() / "adasvp_test_engine_cache_polar";
  LaneDetector det(onnx.string(), trt::Precision::FP32, cache.string());
  cudaStream_t stream;
  ADASVP_CUDA_CHECK(cudaStreamCreate(&stream));
  cv::VideoCapture cap(video.string());
  REQUIRE(cap.isOpened());
  std::ifstream js(dump);
  REQUIRE(js);

  const int kFrames = 300;
  int frames = 0, count_eq = 0, matched = 0, unmatched_ref = 0, unmatched_cand = 0, pt_bad = 0, pt_max = 0;
  double dx_sum = 0, dx_worst = 0;
  size_t dx_n = 0;
  std::string line;
  for (int f = 0; f < kFrames && std::getline(js, line); ++f) {
    cv::Mat frame;
    REQUIRE(cap.read(frame));
    const auto j = nlohmann::json::parse(line);
    std::vector<Polyline> ref;
    for (const auto& l : j["lanes"]) {
      Polyline p;
      for (const auto& q : l) p.emplace_back(q[0].get<float>(), q[1].get<float>());
      ref.push_back(std::move(p));
    }
    const RawLanes got = det.detect(frame, 0.44f, stream);
    ++frames;
    if (got.lanes.size() == ref.size()) ++count_eq;
    const auto m = match_lanes(ref, got.lanes);
    matched += static_cast<int>(m.size());
    unmatched_ref += static_cast<int>(ref.size() - m.size());
    unmatched_cand += static_cast<int>(got.lanes.size() - m.size());
    for (const auto& mm : m) {
      dx_sum += mm.mean;
      dx_worst = std::max(dx_worst, mm.mean);
      ++dx_n;
      const int d = std::abs(static_cast<int>(mm.n_ref) - static_cast<int>(mm.n_cand));
      pt_max = std::max(pt_max, d);
      if (d > 1) ++pt_bad;
    }
  }
  ADASVP_CUDA_CHECK(cudaStreamDestroy(stream));
  REQUIRE(frames == kFrames);
  const double eq = double(count_eq) / frames, mean_dx = dx_n ? dx_sum / dx_n : 0.0;
  std::cout << "[lanes] frames " << frames << " count_equal " << count_eq << " (" << eq * 100 << "%) matched " << matched
            << " unmatched_ref " << unmatched_ref << " unmatched_cand " << unmatched_cand << " mean|dx| " << mean_dx
            << " px worst lane mean|dx| " << dx_worst << " px max point-count diff " << pt_max
            << " lanes with diff>1: " << pt_bad << "\n";
  CHECK(eq >= 0.98);
  CHECK(mean_dx <= 2.0);
  CHECK(pt_bad == 0);
}
