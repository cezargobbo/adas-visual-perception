#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "adas/ego_lane.hpp"
#include "adas/lead_vehicle.hpp"
#include "support/trace_json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace adasvp;

namespace {

Box box_of(const json& b) {
  return Box{b[0].get<float>(), b[1].get<float>(), b[2].get<float>(), b[3].get<float>()};
}

LeadVehicleEstimator make_est(int W, int H, int hold = 9) {
  return LeadVehicleEstimator(W, H, 75.0, 1.5, 0.59, 0.12, 0.02, hold);
}

Object car(int id, float x1, float y1, float x2, float y2, int cls = 2) {
  return Object{Box{x1, y1, x2, y2}, cls, 0.9f, id};
}

void replay(const std::string& stem) {
  const fs::path dir = fs::path(ADASVP_SOURCE_DIR) / "outputs/dumps/fast" / stem;
  if (!fs::exists(dir / "adas.jsonl") || !fs::exists(dir / "meta.json")) {
    SKIP("Python prototype trace missing: " << (dir / "adas.jsonl").string());
  }
  json meta;
  {
    std::ifstream f(dir / "meta.json");
    meta = json::parse(f);
  }
  const auto& ad = meta["adas"];
  const auto& df = ad["defaults"];
  const int W = meta["width"].get<int>(), H = meta["height"].get<int>();
  EgoLaneTracker trk(W, H, ad["fps_used"].get<double>(), df["cam_height"].get<double>(),
                     df["horizon"].get<double>(), df["lanes_hold"].get<double>());
  LeadVehicleEstimator est(W, H, df["hfov"].get<double>(), df["cam_height"].get<double>(),
                           df["horizon"].get<double>(), df["corridor_bottom"].get<double>(),
                           df["corridor_top"].get<double>(), ad["lead_hold_frames"].get<int>());
  const bool lane_corridor = true;  // Python prototype default (--lane-corridor), not recorded in meta.json

  std::ifstream f(dir / "adas.jsonl");
  REQUIRE(f.good());
  std::string line;
  long frames = 0, id_mismatch = 0, null_mismatch = 0, cls_mismatch = 0;
  long id_switches = 0, with_lead = 0, appear = 0, vanish = 0, lane_used = 0, trapezoid_used = 0;
  double max_box = 0, max_dist = 0;
  int prev_id = -1;
  bool prev_has = false;
  while (std::getline(f, line)) {
    const json j = json::parse(line);
    std::vector<Lane> lanes;
    for (const auto& pl : j["lanes_in"]) {
      Lane l;
      for (const auto& p : pl) l.points.emplace_back(p[0].get<float>(), p[1].get<float>());
      lanes.push_back(std::move(l));
    }
    trk.update(lanes);
    std::vector<Object> tracked;
    for (const auto& d : j["lead_in"])
      tracked.push_back(Object{box_of(d[2]), d[1].get<int>(), 1.f, d[0].get<int>()});
    (trk.available() && lane_corridor ? lane_used : trapezoid_used)++;
    const auto got = est.update(tracked, lane_corridor ? &trk : nullptr);
    ++frames;

    const json& want = j["lead"];
    const bool has = !want.is_null();
    if (has != got.has_value()) {
      ++null_mismatch;
    } else if (has) {
      if (got->track_id != want[0].get<int>()) ++id_mismatch;
      if (got->cls != want[1].get<int>()) ++cls_mismatch;
      const Box b = box_of(want[2]);
      max_box = std::max({max_box, (double)std::fabs(got->box.x1 - b.x1), (double)std::fabs(got->box.y1 - b.y1),
                          (double)std::fabs(got->box.x2 - b.x2), (double)std::fabs(got->box.y2 - b.y2)});
      const double dd = std::fabs(got->distance_m - adasvp_test::num(want[3]));
      if (!(dd <= max_dist)) max_dist = dd;  // NaN-propagating max
    }
    // Statistics of the scenario exercised by the trace.
    if (has) {
      ++with_lead;
      const int id = want[0].get<int>();
      if (!prev_has) ++appear;
      if (prev_has && id != prev_id) ++id_switches;
      prev_id = id;
    } else if (prev_has) {
      ++vanish;
    }
    prev_has = has;
  }
  std::cout << "[lead " << stem << "] frames=" << frames << " with_lead=" << with_lead
            << " id_switches=" << id_switches << " appearances=" << appear << " disappearances=" << vanish
            << " lane_corridor_frames=" << lane_used << " trapezoid_frames=" << trapezoid_used
            << " max|box|=" << max_box << " max|dist|=" << max_dist << " id_mismatch=" << id_mismatch
            << " null_mismatch=" << null_mismatch << " cls_mismatch=" << cls_mismatch << std::endl;

  CHECK(frames == meta["frames"].get<long>());
  CHECK(null_mismatch == 0);
  CHECK(id_mismatch == 0);
  CHECK(cls_mismatch == 0);
  CHECK(max_box <= 1e-3);
  CHECK(max_dist <= 1e-4);
  // The trace must actually exercise hysteresis / loss / re-acquisition (Review Focus 1).
  CHECK(id_switches > 0);
  CHECK(appear > 0);
  CHECK(vanish > 0);
}

}  // namespace

TEST_CASE("LeadVehicleEstimator replays the Python prototype trace") { replay("sp1_crop_30fps"); }
TEST_CASE("LeadVehicleEstimator replays the Python prototype trace (second video)") { replay("sp1_crop_4_30fps"); }

TEST_CASE("distance blends width and ground cues, ground only when cut, width only near the horizon") {
  auto est = make_est(1920, 1080);
  const double f = 960.0 / std::tan(75.0 * M_PI / 180.0 / 2.0);
  const double y_hor = 0.59 * 1080;
  const Box b{900, 600, 1000, 700};
  const double zw = f * 1.8 / 100.0, zg = f * 1.5 / (700 - y_hor);
  CHECK(est.distance(b, 2) == Catch::Approx(0.6 * zw + 0.4 * zg).margin(1e-9));
  const Box cut{0, 600, 100, 700};
  CHECK(est.distance(cut, 2) == Catch::Approx(zg).margin(1e-9));
  const Box near_hor{900, 600, 1000, y_hor + 5.0f};  // dy <= 8 -> width cue only
  CHECK(est.distance(near_hor, 7) == Catch::Approx(f * 2.4 / 100.0).margin(1e-9));
  const Box unknown{900, 600, 1000, 700};  // class without a table entry -> 1.8 m
  CHECK(est.distance(unknown, 0) == est.distance(unknown, 2));
}

TEST_CASE("trapezoid corridor is used without a lane and gates candidates") {
  auto est = make_est(1920, 1080);
  CHECK(est.in_corridor(Box{900, 600, 1020, 700}, nullptr));
  CHECK(!est.in_corridor(Box{100, 600, 200, 700}, nullptr));   // far to the left
  CHECK(!est.in_corridor(Box{900, 600, 1020, 640}, nullptr));  // bottom above horizon + 5
  EgoLaneTracker none(1920, 1080, 30.0, 1.5, 0.59, 2.5);       // no lane held -> trapezoid fallback
  CHECK(est.in_corridor(Box{900, 600, 1020, 700}, &none));
  CHECK(!est.in_corridor(Box{100, 600, 200, 700}, &none));
}

TEST_CASE("update: non-vehicles are ignored, hysteresis, lead switch") {
  auto est = make_est(1920, 1080, 3);
  // person (cls 0) and an off-corridor car -> no lead
  CHECK(!est.update({car(1, 900, 600, 1020, 700, 0), car(2, 100, 600, 200, 700)}, nullptr));
  auto a = est.update({car(3, 900, 600, 1020, 700)}, nullptr);
  REQUIRE(a);
  CHECK(a->track_id == 3);
  CHECK(a->distance_m == Catch::Approx(est.distance(Box{900, 600, 1020, 700}, 2)).margin(1e-9));
  // Another car only slightly closer (< 3% H) does not steal the lead; a clearly closer one does.
  auto b = est.update({car(3, 900, 600, 1020, 700), car(4, 905, 600, 1025, 720)}, nullptr);
  REQUIRE(b);
  CHECK(b->track_id == 3);
  auto c = est.update({car(3, 900, 600, 1020, 700), car(4, 905, 600, 1025, 760)}, nullptr);
  REQUIRE(c);
  CHECK(c->track_id == 4);
}

TEST_CASE("update: per-track median window (9) and EMA 0.8/0.2 on a hand-computed sequence") {
  auto est = make_est(1920, 1080, 3);
  // y2 sequence -> distances; the independent model below is median of the last 9 then EMA.
  const float y2s[] = {700, 705, 690, 760, 695, 700, 702, 698, 701, 699, 703};
  std::deque<double> win;
  double ema = 0;
  bool first = true;
  for (float y2 : y2s) {
    const Box bx{900, 600, 1020, y2};
    const double z = est.distance(bx, 2);
    win.push_back(z);
    if (win.size() > 9) win.pop_front();
    std::vector<double> v(win.begin(), win.end());
    std::sort(v.begin(), v.end());
    const double zm = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    ema = first ? zm : 0.8 * ema + 0.2 * zm;
    first = false;
    auto r = est.update({car(3, 900, 600, 1020, y2)}, nullptr);
    REQUIRE(r);
    CHECK(r->distance_m == Catch::Approx(ema).margin(1e-9));
  }
}

TEST_CASE("update: the lead is held for hold_frames misses, then dropped") {
  // car 4 is slightly closer than car 3 (< 3% H), so it only wins once the lead id of car 3 is forgotten.
  const std::vector<Object> both = {car(3, 900, 600, 1020, 700), car(4, 905, 600, 1025, 720)};
  {
    auto est = make_est(1920, 1080, 3);
    REQUIRE(est.update({car(3, 900, 600, 1020, 700)}, nullptr));
    for (int k = 0; k < 3; ++k) CHECK(!est.update({}, nullptr));  // missing_ == hold_frames: still held
    auto r = est.update(both, nullptr);
    REQUIRE(r);
    CHECK(r->track_id == 3);
  }
  {
    auto est = make_est(1920, 1080, 3);
    REQUIRE(est.update({car(3, 900, 600, 1020, 700)}, nullptr));
    for (int k = 0; k < 4; ++k) CHECK(!est.update({}, nullptr));  // missing_ > hold_frames: id dropped
    auto r = est.update(both, nullptr);
    REQUIRE(r);
    CHECK(r->track_id == 4);
  }
}
