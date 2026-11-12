#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "adas/rlr.hpp"
#include "adas/traffic_light.hpp"
#include "support/trace_json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace adasvp;

namespace {

LightColor color_of(const json& v) {
  if (v.is_null()) return LightColor::Unknown;
  const std::string s = v.get<std::string>();
  return s == "red" ? LightColor::Red : s == "yellow" ? LightColor::Yellow : LightColor::Green;
}

const char* name_of(LightColor c) {
  switch (c) {
    case LightColor::Red: return "red";
    case LightColor::Yellow: return "yellow";
    case LightColor::Green: return "green";
    default: return "none";
  }
}

Box box_of(const json& b) {
  return Box{b[0].get<float>(), b[1].get<float>(), b[2].get<float>(), b[3].get<float>()};
}

}  // namespace

TEST_CASE("traffic_light_state matches Python prototype crops") {
  const fs::path dir = fs::path(ADASVP_SOURCE_DIR) / "tests/fixtures";
  std::ifstream f(dir / "tl_state_expected.json");
  REQUIRE(f.good());
  const json expected = json::parse(f);
  int match = 0, total = 0;
  for (const auto& e : expected) {
    const cv::Mat img = cv::imread((dir / e["file"].get<std::string>()).string(), cv::IMREAD_COLOR);
    REQUIRE(!img.empty());
    const auto& cb = e["crop_box"];
    const LightColor got = traffic_light_state(img, box_of(cb));
    const LightColor want = color_of(e["state"]);
    ++total;
    if (got == want) {
      ++match;
    } else {
      std::cout << "[tl_state] MISMATCH " << e["file"] << " got=" << name_of(got) << " want=" << name_of(want)
                << std::endl;
    }
  }
  std::cout << "[tl_state] " << match << "/" << total << " crops match" << std::endl;
  CHECK(total == 60);
  CHECK(match == total);
}

TEST_CASE("traffic_light_state rejects degenerate boxes") {
  const cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
  CHECK(traffic_light_state(img, Box{10, 10, 12, 40}) == LightColor::Unknown);    // narrower than 4 px
  CHECK(traffic_light_state(img, Box{10, 10, 40, 40}) == LightColor::Unknown);    // square
  CHECK(traffic_light_state(img, Box{-30, -30, -5, -5}) == LightColor::Unknown);  // outside the image
  CHECK(traffic_light_state(img, Box{10, 10, 30, 60}) == LightColor::Unknown);    // black: below the threshold
}

TEST_CASE("traffic_light_state reads the lit lamp of a synthetic housing") {
  auto housing = [](int lit) {  // vertical housing, 3 cells of 20 px, lamp lit = 0 red, 1 yellow, 2 green
    cv::Mat m(60, 20, CV_8UC3, cv::Scalar(30, 30, 30));
    const cv::Scalar cols[3] = {{20, 20, 240}, {20, 220, 230}, {40, 230, 20}};
    m(cv::Rect(0, lit * 20, 20, 20)).setTo(cols[lit]);
    return m;
  };
  CHECK(traffic_light_state(housing(0), Box{0, 0, 20, 60}) == LightColor::Red);
  CHECK(traffic_light_state(housing(1), Box{0, 0, 20, 60}) == LightColor::Yellow);
  CHECK(traffic_light_state(housing(2), Box{0, 0, 20, 60}) == LightColor::Green);
  cv::Mat h;  // horizontal: red at the left end, green at the right end
  cv::transpose(housing(0), h);
  CHECK(traffic_light_state(h, Box{0, 0, 60, 20}) == LightColor::Red);
  cv::transpose(housing(2), h);
  CHECK(traffic_light_state(h, Box{0, 0, 60, 20}) == LightColor::Green);
}

namespace {

void replay_rlr(const std::string& stem, const std::vector<long>& expected_frames, double expected_red_for) {
  const fs::path dir = fs::path(ADASVP_SOURCE_DIR) / "outputs/dumps/fast" / stem;
  if (!fs::exists(dir / "adas.jsonl") || !fs::exists(dir / "meta.json")) SKIP("Python prototype trace missing: " << stem);
  std::ifstream m(dir / "meta.json");
  const json meta = json::parse(m);
  std::ifstream f(dir / "adas.jsonl");
  const auto& df = meta["adas"]["defaults"];
  RLRMonitor mon(meta["width"].get<int>(), meta["height"].get<int>(), meta["adas"]["fps_used"].get<double>(),
                 {df["rlr_zone"][0].get<double>(), df["rlr_zone"][1].get<double>()});
  REQUIRE(mon.tracker().max_lost() == meta["adas"]["tl_max_lost"].get<int>());

  std::string line;
  long frames = 0, event_mismatch = 0, track_mismatch = 0;
  double max_box = 0, max_ev = 0;
  std::vector<long> got_ev, want_ev;
  std::vector<double> red_fors;
  while (std::getline(f, line)) {
    const json j = json::parse(line);
    std::vector<std::tuple<Box, float, LightColor>> in;
    for (const auto& d : j["tl_in"])
      in.emplace_back(box_of(d["box"]), d["score"].get<float>(), color_of(d["state"]));
    const RlrStatus s = mon.update_with_states(j["t"].get<double>(), in);

    // event
    const json& w = j["rlr"]["event"];
    if (s.event) got_ev.push_back(frames);
    if (!w.is_null()) want_ev.push_back(frames);
    if (s.event.has_value() != !w.is_null()) {
      ++event_mismatch;
    } else if (s.event) {
      const auto& e = *s.event;
      const auto& bb = w["box"];
      const Box wb = box_of(bb);
      max_ev = std::max({max_ev, std::fabs(e.cx - w["cx"].get<double>()),
                         std::fabs(e.red_for - w["red_for"].get<double>()),
                         std::fabs(e.growth - w["growth"].get<double>())});
      if (e.id != w["id"].get<int>() || e.box.x1 != wb.x1 || e.box.y1 != wb.y1 || e.box.x2 != wb.x2 ||
          e.box.y2 != wb.y2)
        ++event_mismatch;
      red_fors.push_back(e.red_for);
    }

    // tracks (the dump keeps only the last history entry and n_hist, hist is capped like the Python prototype)
    const auto& tracks = mon.tracker().tracks();
    const json& wt = j["tl_tracks"];
    if (tracks.size() != wt.size()) {
      ++track_mismatch;
    } else {
      size_t i = 0;
      for (const auto& [id, tr] : tracks) {
        const json& r = wt[i++];
        const auto& last = tr.hist.back();
        const auto& lb = r["last"]["box"];
        if (id != r["id"].get<int>() || tr.lost != r["lost"].get<int>() || tr.done != r["done"].get<bool>() ||
            static_cast<int>(tr.hist.size()) != r["n_hist"].get<int>()) {
          ++track_mismatch;
          continue;
        }
        for (int k = 0; k < 4; ++k) max_box = std::max(max_box, std::fabs(last.box[k] - lb[k].get<double>()));
        if (last.t != r["last"]["t"].get<double>() || last.state != color_of(r["last"]["state"]) ||
            last.conf != r["last"]["conf"].get<double>())
          ++track_mismatch;
      }
    }
    ++frames;
  }
  std::cout << "[rlr " << stem << "] frames=" << frames << " event_mismatch=" << event_mismatch
            << " track_mismatch=" << track_mismatch << " max|d box|=" << max_box << " max|d event|=" << max_ev
            << "\n  events C++: " << got_ev.size() << " trace: " << want_ev.size();
  for (size_t k = 0; k < got_ev.size(); ++k) std::cout << "  frame " << got_ev[k] << " red_for=" << red_fors[k];
  std::cout << std::endl;
  CHECK(frames == meta["frames"].get<long>());
  CHECK(got_ev == want_ev);
  CHECK(got_ev == expected_frames);
  CHECK(event_mismatch == 0);
  CHECK(track_mismatch == 0);
  CHECK(max_box <= 1e-9);
  CHECK(max_ev <= 1e-9);
  for (double r : red_fors) CHECK(r == Catch::Approx(expected_red_for).margin(0.01));
}

}  // namespace

TEST_CASE("RLRMonitor replays the Python prototype trace") {
  SECTION("red_crop_11_30fps: one RLR") { replay_rlr("red_crop_11_30fps", {466}, 3.3667); }
  SECTION("sp1_crop_30fps: no RLR") { replay_rlr("sp1_crop_30fps", {}, 0); }
}

// ----------------------------------------------------------------------------- synthetic gates
namespace {

constexpr int kW = 1000, kH = 1000;
constexpr double kFps = 30;

// A light that grows exponentially and rises at a constant speed (a car driving towards it), seen from
// t_begin. It is fed until its box touches the top edge (y1 <= 10), or until `drop_at_y1` when it is set
// (then it just disappears inside the top band and the tracker closes it).
struct Light {
  double cx = 500;          // px
  double t_begin = 0;       // s, the light exists (kinematics of t = 0) from here on
  double delay = 0;         // s of real time before it appears (kinematics start at t_begin then)
  double growth = 0.6;      // d ln(h) / dt
  double rise = 156;        // px / s
  double h0 = 40, cy0 = 600;
  std::function<LightColor(double)> color = [](double) { return LightColor::Red; };
  double drop_at_y1 = -1;
};

struct Ev {
  int frame;
  RlrEvent e;
};

std::vector<Ev> simulate(std::vector<Light> lights, RLRMonitor& mon, int n_frames = 150) {
  std::vector<Ev> out;
  for (int i = 0; i < n_frames; ++i) {
    const double t = i / kFps;
    std::vector<std::tuple<Box, float, LightColor>> in;
    for (auto& L : lights) {
      if (t < L.delay) continue;
      const double tk = t - L.delay + L.t_begin;
      const double h = L.h0 * std::exp(L.growth * tk), cy = L.cy0 - L.rise * tk, w = 0.45 * h;
      const double y1 = cy - h / 2;
      if (L.drop_at_y1 >= 0 && y1 < L.drop_at_y1) continue;
      if (y1 < -50) continue;  // long gone
      in.emplace_back(Box{static_cast<float>(L.cx - w / 2), static_cast<float>(y1), static_cast<float>(L.cx + w / 2),
                          static_cast<float>(cy + h / 2)},
                      0.8f, L.color(tk));
    }
    const RlrStatus s = mon.update_with_states(t, in);
    if (s.event) out.push_back({i, *s.event});
  }
  return out;
}

std::vector<Ev> run(Light L) {
  RLRMonitor mon(kW, kH, kFps);
  return simulate({L}, mon);
}

}  // namespace

TEST_CASE("RLRMonitor synthetic: a red light leaving through the central top edge raises one event") {
  const auto ev = run(Light{});
  REQUIRE(ev.size() == 1);
  CHECK(ev[0].e.id == 1);
  CHECK(ev[0].e.cx == Catch::Approx(0.5).margin(0.01));
  CHECK(ev[0].e.growth == Catch::Approx(0.6).margin(0.01));
  CHECK(ev[0].e.red_for > 2.0);
  CHECK(ev[0].e.box.y1 > 10.8f);  // the last box not cut by the top edge
  CHECK(ev[0].e.box.y1 < 40.f);
}

TEST_CASE("RLRMonitor synthetic: gates that must suppress the event") {
  SECTION("exit outside the central zone (left and right)") {
    Light a;
    a.cx = 150;
    CHECK(run(a).empty());
    Light b;
    b.cx = 850;
    CHECK(run(b).empty());
  }
  SECTION("zone is configurable") {
    Light a;
    a.cx = 150;
    RLRMonitor mon(kW, kH, kFps, {0.1, 0.9});
    CHECK(simulate({a}, mon).size() == 1);
  }
  SECTION("track shorter than 1 s") {
    Light a;
    a.t_begin = 2.4;  // leaves at ~3.0 s: 0.6 s of track
    CHECK(run(a).empty());
    // red_for <= track length, so with the default 1.5 s red_min_s this gate is hidden behind it:
    // relax red_min_s to isolate min_track_s.
    RLRMonitor shorter(kW, kH, kFps, {0.25, 0.75}, 0.01, 0.12, 1.0, 0.15, 0.3);
    CHECK(simulate({a}, shorter).empty());
    RLRMonitor shorter_ok(kW, kH, kFps, {0.25, 0.75}, 0.01, 0.12, 0.5, 0.15, 0.3);
    CHECK(simulate({a}, shorter_ok).size() == 1);
  }
  SECTION("insufficient growth") {
    Light a;
    a.growth = 0.05;
    CHECK(run(a).empty());
  }
  SECTION("not moving up") {
    Light a;
    a.cy0 = 600;
    a.rise = 0;
    a.growth = 1.2;  // grows until its top reaches the edge without the centre rising
    a.h0 = 40;
    CHECK(run(a).empty());
  }
  SECTION("red share below 50%") {
    Light a;
    // 40% red, the rest undetermined, ends on a red sample
    a.color = [](double t) { return static_cast<int>(std::lround(t * kFps)) % 5 < 2 ? LightColor::Red : LightColor::Unknown; };
    CHECK(run(a).empty());
  }
  SECTION("more than 15% green/yellow in the 2 s window") {
    // green long before the end (red for 1.7 s: red_min_s is met), so only the share gate can reject it
    auto sched = [](double green_until) {
      return [green_until](double t) {
        return t < 0.8 ? LightColor::Unknown : t < green_until ? LightColor::Green : LightColor::Red;
      };
    };
    Light a;
    a.color = sched(1.3);  // 25% of the window
    CHECK(run(a).empty());
    Light ok;
    ok.color = sched(1.0);  // 10%
    CHECK(run(ok).size() == 1);
  }
  SECTION("red for less than 1.5 s (entered on amber)") {
    Light a;
    a.color = [](double t) { return t < 1.7 ? LightColor::Unknown : LightColor::Red; };
    CHECK(run(a).empty());
    Light b;
    b.color = [](double t) { return t < 1.2 ? LightColor::Unknown : LightColor::Red; };
    const auto ev = run(b);
    REQUIRE(ev.size() == 1);
    CHECK(ev[0].e.red_for >= 1.5);
  }
  SECTION("no determined colour in the 0.7 s before the last sample") {
    Light a;
    a.color = [](double t) { return t > 2.2 ? LightColor::Unknown : LightColor::Red; };
    CHECK(run(a).empty());
  }
}

TEST_CASE("RLRMonitor synthetic: cooldown keeps one alert per intersection") {
  Light a;               // leaves the top at ~3.0 s
  Light b;               // a second light of the same intersection, a bit slower
  b.cx = 600;
  b.rise = 140;
  RLRMonitor mon(kW, kH, kFps);
  const auto ev = simulate({a, b}, mon);
  REQUIRE(ev.size() == 1);
  CHECK(ev[0].e.id == 1);
  // Alone, the second light would have raised its own event.
  CHECK(run(b).size() == 1);
  // After the cooldown a new event is allowed again: the second light leaves 4.5 s after the first.
  Light c = a;
  c.delay = 4.5;  // same kinematics, 4.5 s later
  RLRMonitor mon2(kW, kH, kFps);
  const auto ev2 = simulate({a, c}, mon2, 300);
  REQUIRE(ev2.size() == 2);
  CHECK(ev2[1].frame - ev2[0].frame >= static_cast<int>(4.0 * kFps));
  // ... but not within the cooldown (1.5 s later)
  Light d = a;
  d.delay = 1.5;
  RLRMonitor mon3(kW, kH, kFps);
  CHECK(simulate({a, d}, mon3, 300).size() == 1);
}

TEST_CASE("RLRMonitor synthetic: a light lost inside the top band is evaluated when its track closes") {
  Light a;
  a.drop_at_y1 = 80;  // vanishes at y1 ~ 80 (< 0.12*H), never touches the top edge
  const auto ev = run(a);
  REQUIRE(ev.size() == 1);
  CHECK(ev[0].e.box.y1 >= 80.f);
  Light low;
  low.drop_at_y1 = 200;  // vanishes below the top band: no candidate
  CHECK(run(low).empty());
}

TEST_CASE("RLRMonitor::update classifies boxes below the top cut and reports the lights") {
  cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(30, 30, 30));
  frame(cv::Rect(60, 40, 20, 20)).setTo(cv::Scalar(20, 20, 240));
  RLRMonitor mon(200, 200, 30);
  mon.update(0.0, frame, {{Box{60, 40, 80, 100}, 0.9f}, {Box{100, 0, 120, 60}, 0.5f}});
  const auto& l = mon.lights();
  REQUIRE(l.size() == 2);
  CHECK(l[0].track_id == 1);
  CHECK(l[0].color == LightColor::Red);
  CHECK(l[1].track_id == 2);
  CHECK(l[1].color == LightColor::Unknown);  // y1 = 0 <= 0.01*H: not classified
}
