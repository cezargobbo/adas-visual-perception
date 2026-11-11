#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "support/npy.hpp"
#include "track/assignment.hpp"
#include "track/byte_tracker.hpp"
#include "track/kalman_xyah.hpp"

using namespace adasvp;
using namespace adasvp::test;
using nlohmann::json;

namespace {
const std::string kFix = std::string(ADASVP_SOURCE_DIR) + "/tests/fixtures/";

json load_json(const std::string& name) {
  std::ifstream f(kFix + name);
  REQUIRE(f.good());
  return json::parse(f);
}

Vec8 vec8(const json& j) {
  Vec8 v;
  for (int i = 0; i < 8; ++i) v[i] = j[i].get<double>();
  return v;
}
Mat8 mat8(const json& j) {
  Mat8 m;
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 8; ++k) m[i * 8 + k] = j[i][k].get<double>();
  return m;
}

// relative to the larger of |expected| and a small absolute floor (covariance has entries near 0)
void expect_close(double got, double want, double scale) {
  REQUIRE(std::fabs(got - want) <= 1e-6 * std::max(std::fabs(want), scale));
}
void check_state(const Vec8& m, const Mat8& c, const json& jm, const json& jc) {
  const Vec8 em = vec8(jm);
  const Mat8 ec = mat8(jc);
  double ms = 0, cs = 0;
  for (double v : em) ms = std::max(ms, std::fabs(v));
  for (double v : ec) cs = std::max(cs, std::fabs(v));
  for (int i = 0; i < 8; ++i) expect_close(m[i], em[i], 1e-9 * ms);
  for (int i = 0; i < 64; ++i) expect_close(c[i], ec[i], 1e-9 * cs);
}
}  // namespace

TEST_CASE("Kalman matches") {
  const json d = load_json("kalman_xyah.json");
  KalmanXYAH kf;
  for (const json& c : d["cases"]) {
    Vec8 m{};
    Mat8 cov{};
    const auto& ops = c["ops"];
    for (size_t k = 0; k < ops.size(); ++k) {
      const std::string op = ops[k];
      if (op == "initiate") {
        Vec4 z;
        for (int i = 0; i < 4; ++i) z[i] = c["measurements"][k][i].get<double>();
        std::tie(m, cov) = kf.initiate(z);
      } else if (op == "predict") {
        std::tie(m, cov) = kf.predict(m, cov);
      } else {
        REQUIRE(op == "update");
        Vec4 z;
        for (int i = 0; i < 4; ++i) z[i] = c["measurements"][k][i].get<double>();
        std::tie(m, cov) = kf.update(m, cov, z);
      }
      check_state(m, cov, c["means"][k], c["covariances"][k]);
    }
  }
  const json& mp = d["multi_predict"];
  std::vector<Vec8> ms;
  std::vector<Mat8> cs;
  for (size_t i = 0; i < mp["means_in"].size(); ++i) {
    ms.push_back(vec8(mp["means_in"][i]));
    cs.push_back(mat8(mp["covariances_in"][i]));
  }
  kf.multi_predict(ms, cs);
  for (size_t i = 0; i < ms.size(); ++i) check_state(ms[i], cs[i], mp["means_out"][i], mp["covariances_out"][i]);
}

TEST_CASE("linear_assignment matches lap.lapjv") {
  const NpyArray costs = load_npy(kFix + "lap_costs.npy");
  const NpyArray shapes = load_npy(kFix + "lap_shapes.npy");
  const NpyArray limits = load_npy(kFix + "lap_cost_limits.npy");
  const NpyArray xs = load_npy(kFix + "lap_x.npy");
  const NpyArray ys = load_npy(kFix + "lap_y.npy");
  const size_t n = shapes.shape[0];
  REQUIRE(n == 20);
  size_t co = 0, xo = 0, yo = 0;
  for (size_t k = 0; k < n; ++k) {
    const int r = shapes.as<int32_t>()[2 * k], c = shapes.as<int32_t>()[2 * k + 1];
    INFO("matrix " << k << " " << r << "x" << c);
    std::vector<std::vector<double>> cost(r, std::vector<double>(c));
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) cost[i][j] = costs.as<float>()[co + static_cast<size_t>(i) * c + j];
    const double limit = limits.as<float>()[k];
    std::vector<int> ua, ub;
    const auto m = linear_assignment(cost, limit, &ua, &ub, c);
    std::vector<int> x(r, -1), y(c, -1);
    for (auto& p : m) {
      x[p.first] = p.second;
      y[p.second] = p.first;
    }
    for (int i = 0; i < r; ++i) REQUIRE(x[i] == xs.as<int32_t>()[xo + i]);
    for (int j = 0; j < c; ++j) REQUIRE(y[j] == ys.as<int32_t>()[yo + j]);
    size_t na = 0, nb = 0;
    for (int i = 0; i < r; ++i) na += x[i] < 0;
    for (int j = 0; j < c; ++j) nb += y[j] < 0;
    REQUIRE(ua.size() == na);
    REQUIRE(ub.size() == nb);
    co += static_cast<size_t>(r) * c;
    xo += r;
    yo += c;
  }
}

namespace {
struct SeqResult {
  int frames = 0, mismatches = 0;
};

void run_sequence(const std::string& name, std::vector<std::vector<Object>>* all_out = nullptr) {
  const NpyArray dets = load_npy(kFix + "bytetrack_" + name + "_dets.npy");
  const NpyArray dc = load_npy(kFix + "bytetrack_" + name + "_det_counts.npy");
  const NpyArray tracks = load_npy(kFix + "bytetrack_" + name + "_tracks.npy");
  const NpyArray tc = load_npy(kFix + "bytetrack_" + name + "_track_counts.npy");
  const size_t frames = dc.shape[0];
  REQUIRE(tc.shape[0] == frames);
  auto cnt = [](const NpyArray& a, size_t i) -> size_t {
    return a.dtype == "int64" ? static_cast<size_t>(a.as<int64_t>()[i]) : static_cast<size_t>(a.as<int32_t>()[i]);
  };
  ByteTracker tracker(TrackerParams{}, 30.0);
  size_t doff = 0, toff = 0;
  for (size_t f = 0; f < frames; ++f) {
    std::vector<Object> in;
    for (size_t i = 0; i < cnt(dc, f); ++i) {
      const float* r = dets.as<float>() + (doff + i) * 6;
      in.push_back(Object{{r[0], r[1], r[2], r[3]}, static_cast<int>(r[5]), r[4], -1});
    }
    doff += cnt(dc, f);
    const auto out = tracker.update(in);
    INFO("sequence " << name << " frame " << f);
    REQUIRE(out.size() == cnt(tc, f));
    for (size_t i = 0; i < out.size(); ++i) {
      const float* r = tracks.as<float>() + (toff + i) * 8;
      REQUIRE(out[i].track_id == static_cast<int>(r[4]));
      REQUIRE(std::fabs(out[i].box.x1 - r[0]) <= 1e-3);
      REQUIRE(std::fabs(out[i].box.y1 - r[1]) <= 1e-3);
      REQUIRE(std::fabs(out[i].box.x2 - r[2]) <= 1e-3);
      REQUIRE(std::fabs(out[i].box.y2 - r[3]) <= 1e-3);
      REQUIRE(out[i].score == r[5]);
      REQUIRE(out[i].cls == static_cast<int>(r[6]));
    }
    toff += cnt(tc, f);
    if (all_out) all_out->push_back(out);
  }
}
}  // namespace

TEST_CASE("ByteTracker matches Ultralytics") {
  for (const char* n : {"A", "B", "C"}) run_sequence(n);
}

TEST_CASE("ByteTracker empty frames and lost-track expiry") {
  // sequence C: empty frames; a gap longer than the buffer returns with a new id (checked against ultralytics ids)
  std::vector<std::vector<Object>> out;
  run_sequence("C", &out);
  std::vector<int> ids;
  for (auto& f : out)
    for (auto& o : f)
      if (std::find(ids.begin(), ids.end(), o.track_id) == ids.end()) ids.push_back(o.track_id);
  REQUIRE(ids.size() == 3);

  ByteTracker t(TrackerParams{}, 30.0);
  const Object d{{100, 100, 200, 300}, 0, 0.9f, -1};
  int first = -1;
  for (int f = 0; f < 3; ++f) {
    auto o = t.update({d});
    REQUIRE(o.size() == 1);
    first = o[0].track_id;
  }
  REQUIRE(first == 1);
  for (int f = 0; f < 35; ++f) REQUIRE(t.update({}).empty());  // > max_time_lost frames without detections
  REQUIRE(t.update({d}).empty());  // new tracks are hidden on their first frame (frame_id != 1)
  auto o = t.update({d});
  REQUIRE(o.size() == 1);
  REQUIRE(o[0].track_id != first);
  REQUIRE(o[0].track_id == 2);
}
