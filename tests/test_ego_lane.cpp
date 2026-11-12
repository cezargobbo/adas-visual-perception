#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "adas/ego_lane.hpp"
#include "support/trace_json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace adasvp;
using adasvp_test::num;
using adasvp_test::vdiff;
using adasvp_test::vec;

namespace {

struct Diffs {
  std::map<std::string, double> maxd;
  void add(const std::string& k, double d) {
    double& m = maxd[k];
    if (!(d <= m)) m = d;  // NaN-propagating max
  }
};

// Optional-array field: null in the trace <-> empty / nullopt here.
double odiff(const json& j, const std::optional<std::vector<double>>& o) {
  if (j.is_null()) return o ? std::numeric_limits<double>::infinity() : 0.0;
  if (!o) return std::numeric_limits<double>::infinity();
  return vdiff(vec(j), *o);
}

LaneState state_of(const std::string& s) {
  if (s == "detected") return LaneState::Detected;
  if (s == "partial") return LaneState::Partial;
  if (s == "estimated") return LaneState::Estimated;
  return LaneState::None;
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
  EgoLaneTracker trk(meta["width"].get<int>(), meta["height"].get<int>(), ad["fps_used"].get<double>(),
                     df["cam_height"].get<double>(), df["horizon"].get<double>(), df["lanes_hold"].get<double>());

  // Geometry recorded by the Python prototype.
  CHECK(vdiff(trk.rows_y(), vec(ad["ego_ys"])) <= 1e-9);
  CHECK(vdiff(trk.t_row(), vec(ad["ego_t_row"])) <= 1e-12);
  CHECK(std::fabs(trk.wmin() - ad["ego_wmin"].get<double>()) <= 1e-9);
  CHECK(std::fabs(trk.wmax() - ad["ego_wmax"].get<double>()) <= 1e-9);
  CHECK(std::fabs(trk.gate() - ad["ego_gate"].get<double>()) <= 1e-9);
  CHECK(trk.hold_frames() == ad["ego_hold_frames"].get<int>());
  CHECK(trk.persist() == ad["ego_persist"].get<int>());

  std::ifstream f(dir / "adas.jsonl");
  REQUIRE(f.good());
  std::string line;
  Diffs d;
  long frames = 0, state_mismatch = 0, avail_mismatch = 0, flag_mismatch = 0;
  while (std::getline(f, line)) {
    const json j = json::parse(line);
    std::vector<Lane> lanes;
    for (const auto& pl : j["lanes_in"]) {
      Lane l;
      for (const auto& p : pl) l.points.emplace_back(p[0].get<float>(), p[1].get<float>());
      lanes.push_back(std::move(l));
    }
    trk.update(lanes);
    const json& e = j["ego"];
    const EgoLane s = trk.snapshot();
    ++frames;
    if (s.state != state_of(e["state"].get<std::string>())) ++state_mismatch;
    if (s.available != e["available"].get<bool>() || trk.available() != s.available) ++avail_mismatch;
    if (s.obs_l != e["obs_l"].get<bool>() || s.obs_r != e["obs_r"].get<bool>() ||
        s.glitch != e["glitch"].get<bool>() || trk.free_l() != e["free"]["l"].get<int>() ||
        trk.free_r() != e["free"]["r"].get<int>() || trk.since() != e["since"].get<int>())
      ++flag_mismatch;
    d.add("xl", e["xl"].is_null() ? (s.xl.empty() ? 0.0 : INFINITY) : vdiff(vec(e["xl"]), s.xl));
    d.add("xr", e["xr"].is_null() ? (s.xr.empty() ? 0.0 : INFINITY) : vdiff(vec(e["xr"]), s.xr));
    d.add("fade", std::fabs(num(e["fade"]) - s.fade));
    d.add("raw_l", odiff(e["raw_l"], trk.raw_l()));
    d.add("raw_r", odiff(e["raw_r"], trk.raw_r()));
    d.add("wprof", vdiff(vec(e["wprof"]), trk.wprof()));
  }
  std::cout << "[ego_lane " << stem << "] frames=" << frames;
  for (const auto& [k, v] : d.maxd) std::cout << " max|" << k << "|=" << v;
  std::cout << " state_mismatch=" << state_mismatch << " avail_mismatch=" << avail_mismatch
            << " flag_mismatch=" << flag_mismatch << std::endl;

  CHECK(frames == meta["frames"].get<long>());
  CHECK(state_mismatch == 0);
  CHECK(avail_mismatch == 0);
  CHECK(flag_mismatch == 0);
  CHECK(d.maxd["xl"] <= 1e-3);
  CHECK(d.maxd["xr"] <= 1e-3);
  CHECK(d.maxd["fade"] <= 1e-6);
  CHECK(d.maxd["raw_l"] <= 1e-3);
  CHECK(d.maxd["raw_r"] <= 1e-3);
  CHECK(d.maxd["wprof"] <= 1e-3);
}

}  // namespace

TEST_CASE("EgoLaneTracker replays the Python prototype trace") { replay("sp1_crop_30fps"); }
TEST_CASE("EgoLaneTracker replays the Python prototype trace (second video)") { replay("sp1_crop_4_30fps"); }

TEST_CASE("lane_x_at rejects short or high lanes") {
  std::vector<cv::Point2f> few{{100, 900}, {100, 950}, {100, 1000}};
  CHECK(!lane_x_at(few, std::vector<double>{800, 900}, 1080).has_value());
  std::vector<cv::Point2f> high;
  for (int i = 0; i < 10; ++i) high.emplace_back(100.f, 300.f + 20.f * i);  // never reaches 0.75 H
  CHECK(!lane_x_at(high, std::vector<double>{800, 900}, 1080).has_value());
  std::vector<cv::Point2f> straight;  // x = 0.5 y, reaches the bottom: linear fit, exact
  for (int i = 0; i < 10; ++i) straight.emplace_back(0.5f * (800.f + 30.f * i), 800.f + 30.f * i);
  const auto r = lane_x_at(straight, std::vector<double>{700, 900, 1079}, 1080);
  REQUIRE(r.has_value());
  CHECK(std::fabs((*r)[0] - 350.0) < 1e-6);
  CHECK(std::fabs((*r)[2] - 539.5) < 1e-6);
}

TEST_CASE("EgoLaneTracker starts unavailable and contains() is false then") {
  EgoLaneTracker t(1920, 1080, 30.0, 1.5, 0.59, 2.5);
  CHECK(!t.available());
  CHECK(!t.contains(Box{900, 800, 1000, 900}));
}
