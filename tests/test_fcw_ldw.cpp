#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "adas/ego_lane.hpp"
#include "adas/fcw.hpp"
#include "adas/ldw.hpp"
#include "adas/theil_sen.hpp"
#include "support/trace_json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace adasvp;

namespace {

constexpr double kTiny = 1e-9;

// Relative difference; 0 when both are the same infinity / both NaN, infinity when only one is non-finite.
double rel(double a, double b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::infinity();
  if (a == b) return 0;
  if (std::isinf(a) || std::isinf(b)) return std::numeric_limits<double>::infinity();
  return std::fabs(a - b) / std::max(std::fabs(b), kTiny);
}

// Relative difference of an optional C++ value against a trace value (null = None).
double rel_opt(const std::optional<double>& a, const json& b) {
  if (b.is_null()) return a ? std::numeric_limits<double>::infinity() : 0;
  if (!a) return std::numeric_limits<double>::infinity();
  return rel(*a, adasvp_test::num(b));
}

std::string str_of(const json& v) { return v.is_null() ? std::string() : v.get<std::string>(); }

struct Trace {
  json meta;
  std::ifstream f;
  bool ok = false;
};

bool open_trace(const std::string& stem, Trace& tr) {
  const fs::path dir = fs::path(ADASVP_SOURCE_DIR) / "outputs/dumps/fast" / stem;
  if (!fs::exists(dir / "adas.jsonl") || !fs::exists(dir / "meta.json")) return false;
  std::ifstream m(dir / "meta.json");
  tr.meta = json::parse(m);
  tr.f.open(dir / "adas.jsonl");
  return tr.f.good();
}

std::optional<LeadVehicle> lead_of(const json& v) {
  if (v.is_null()) return std::nullopt;
  const auto& b = v[2];
  return LeadVehicle{v[0].get<int>(), v[1].get<int>(),
                     Box{b[0].get<float>(), b[1].get<float>(), b[2].get<float>(), b[3].get<float>()},
                     adasvp_test::num(v[3])};
}

std::string events_str(const std::vector<std::pair<long, std::string>>& e) {
  std::string s;
  for (const auto& [i, k] : e) s += k + "@" + std::to_string(i) + " ";
  return s;
}

void replay_fcw(const std::string& stem, const std::vector<std::pair<long, std::string>>& expected) {
  Trace tr;
  if (!open_trace(stem, tr)) SKIP("reference trace missing: " << stem);
  const auto& ad = tr.meta["adas"];
  const auto& df = ad["defaults"];
  FCWMonitor mon(tr.meta["width"].get<int>(), tr.meta["height"].get<int>(), ad["fps_used"].get<double>(),
                 df["fcw_ttc"].get<double>(), df["fcw_ttc_critical"].get<double>());
  std::string line;
  long frames = 0, flag_mismatch = 0, ttc_frames = 0, active_frames = 0;
  double max_ttc = 0;
  std::vector<std::pair<long, std::string>> got_ev, want_ev;
  while (std::getline(tr.f, line)) {
    const json j = json::parse(line);
    const FcwStatus s = mon.update(j["t"].get<double>(), lead_of(j["fcw_lead_in"]));
    const json& w = j["fcw"];
    if (s.active != w["active"].get<bool>() || s.level != str_of(w["level"]) || s.event != str_of(w["event"]))
      ++flag_mismatch;
    const double d = rel_opt(s.ttc, w["ttc"]);
    if (!(d <= max_ttc)) max_ttc = d;  // NaN / inf propagating max
    ttc_frames += !w["ttc"].is_null();
    active_frames += w["active"].get<bool>();
    if (!s.event.empty()) got_ev.emplace_back(frames, s.event);
    if (!w["event"].is_null()) want_ev.emplace_back(frames, w["event"].get<std::string>());
    ++frames;
  }
  std::cout << "[fcw " << stem << "] frames=" << frames << " ttc_frames=" << ttc_frames
            << " active_frames=" << active_frames << " max_rel(ttc)=" << max_ttc
            << " flag_mismatch=" << flag_mismatch << "\n  events C++: " << events_str(got_ev)
            << "\n  events trace: " << events_str(want_ev) << std::endl;
  CHECK(frames == tr.meta["frames"].get<long>());
  CHECK(flag_mismatch == 0);
  CHECK(max_ttc <= 1e-6);
  CHECK(got_ev == want_ev);
  CHECK(got_ev == expected);
}

void replay_ldw(const std::string& stem, const std::vector<std::pair<long, std::string>>& expected) {
  Trace tr;
  if (!open_trace(stem, tr)) SKIP("reference trace missing: " << stem);
  const auto& ad = tr.meta["adas"];
  const auto& df = ad["defaults"];
  const int W = tr.meta["width"].get<int>(), H = tr.meta["height"].get<int>();
  const double fps = ad["fps_used"].get<double>();
  EgoLaneTracker trk(W, H, fps, df["cam_height"].get<double>(), df["horizon"].get<double>(),
                     df["lanes_hold"].get<double>());
  LDWMonitor mon(W, H, fps, trk, df["ldw_car_width"].get<double>(), df["ldw_margin"].get<double>(), 3.5,
                 df["horizon"].get<double>());
  CHECK(mon.row() == ad["ldw_row"].get<int>());
  std::string line;
  long frames = 0, flag_mismatch = 0, fail_mismatch = 0, m_mismatch = 0, armed_frames = 0, cross_mismatch = 0;
  double max_d = 0, max_v = 0, max_tlc = 0, max_lastv = 0, max_m = 0;
  std::vector<std::pair<long, std::string>> got_ev, want_ev;
  json last;
  long m_viol = 0;
  std::map<std::string, double> m_rel, m_abs, m_mag;
  while (std::getline(tr.f, line)) {
    const json j = json::parse(line);
    std::vector<Lane> lanes;
    for (const auto& pl : j["lanes_in"]) {
      Lane l;
      for (const auto& p : pl) l.points.emplace_back(p[0].get<float>(), p[1].get<float>());
      lanes.push_back(std::move(l));
    }
    trk.update(lanes);
    const LdwStatus s = mon.update(j["t"].get<double>());
    const json& w = j["ldw"];
    const json& m = w["mon"];
    if (s.armed != w["armed"].get<bool>() || s.side != str_of(w["side"]) || s.event != str_of(w["event"]))
      ++flag_mismatch;
    if (mon.fail() != str_of(m["fail"])) ++fail_mismatch;
    if (mon.n_cross() != m["n_cross"].get<int>()) ++cross_mismatch;
    auto upd = [](double& mx, double d) {
      if (!(d <= mx)) mx = d;
    };
    upd(max_d, rel_opt(s.d, w["d"]));
    upd(max_v, rel_opt(s.v, w["v"]));
    upd(max_tlc, rel_opt(s.tlc, w["tlc"]));
    upd(max_lastv, rel_opt(mon.last_v(), m["last_v"]));
    const auto& wm = m["m"];
    if (wm.size() != mon.metrics().size()) {
      ++m_mismatch;
    } else {
      for (const auto& [k, v] : mon.metrics()) {
        if (!wm.contains(k)) {
          ++m_mismatch;
          break;
        }
        const double want = adasvp_test::num(wm[k]);
        upd(max_m, rel(v, want));
        // allclose(rtol=1e-6, atol=1e-9): the fit residuals resl/resr/wres are ~1e-8 when a boundary is an
        // exact line, and then only the ~1e-12 double-precision noise of the least squares is left.
        if (!(std::fabs(v - want) <= 1e-6 * std::fabs(want) + 1e-9)) ++m_viol;
        upd(m_rel[k], rel(v, want));
        m_abs[k] = std::max(m_abs[k], std::fabs(v - want));
        m_mag[k] = std::max(m_mag[k], std::fabs(want));
      }
    }
    armed_frames += s.armed;
    if (!s.event.empty()) got_ev.emplace_back(frames, s.event);
    if (!w["event"].is_null()) want_ev.emplace_back(frames, w["event"].get<std::string>());
    last = m["stats"];
    ++frames;
  }
  std::map<std::string, long long> want_stats;
  for (auto it = last.begin(); it != last.end(); ++it) want_stats[it.key()] = it.value().get<long long>();
  std::cout << "[ldw " << stem << "] frames=" << frames << " armed_frames=" << armed_frames
            << " max_rel(d)=" << max_d << " max_rel(v)=" << max_v << " max_rel(tlc)=" << max_tlc
            << " max_rel(last_v)=" << max_lastv << " max_rel(m)=" << max_m << " flag_mismatch=" << flag_mismatch
            << " fail_mismatch=" << fail_mismatch << " m_key_mismatch=" << m_mismatch
            << " n_cross_mismatch=" << cross_mismatch << " n_cross=" << mon.n_cross()
            << "\n  events C++: " << events_str(got_ev) << "\n  events trace: " << events_str(want_ev)
            << "\n  stats C++:";
  for (const auto& [k, v] : mon.stats()) std::cout << " " << k << "=" << v;
  for (const auto& [k, v] : m_rel)
    std::cout << "\n  m[" << k << "]: max_rel=" << v << " max_abs=" << m_abs[k] << " max|value|=" << m_mag[k];
  std::cout << std::endl;
  CHECK(frames == tr.meta["frames"].get<long>());
  CHECK(flag_mismatch == 0);
  CHECK(fail_mismatch == 0);  // 100% of the frames
  CHECK(m_mismatch == 0);
  CHECK(cross_mismatch == 0);
  CHECK(max_d <= 1e-6);
  CHECK(max_v <= 1e-6);
  CHECK(max_tlc <= 1e-6);
  CHECK(max_lastv <= 1e-6);
  CHECK(m_viol == 0);  // allclose, see above; max_rel(m) is printed for information
  CHECK(mon.stats() == want_stats);
  CHECK(got_ev == want_ev);
  CHECK(got_ev == expected);
}

}  // namespace

TEST_CASE("FCWMonitor replays the reference trace") {
  replay_fcw("sp1_crop_30fps", {{3835, "warning"}, {4656, "warning"}});
}
TEST_CASE("FCWMonitor replays the reference trace (second video)") {
  replay_fcw("sp1_crop_4_30fps", {{1047, "warning"}, {1058, "critical"}, {2015, "warning"}, {4743, "warning"}});
}
TEST_CASE("LDWMonitor replays the reference trace") {
  replay_ldw("sp1_crop_30fps", {{4593, "left"}, {4749, "right"}});
}
TEST_CASE("LDWMonitor replays the reference trace (second video)") {
  replay_ldw("sp1_crop_4_30fps", {{484, "right"}, {4034, "left"}, {4438, "left"}});
}

TEST_CASE("theil_sen: median of pairwise slopes, numpy median for even counts") {
  CHECK(theil_sen({0, 1, 2, 3}, {0, 1, 2, 3}) == Catch::Approx(1.0));
  // 4 points -> 6 pairs (even count): slopes 1,2,3,1,2... compute by hand: y = {0,1,3,4}
  // pairs (0,1)=1 (0,2)=1.5 (0,3)=4/3 (1,2)=2 (1,3)=1.5 (2,3)=1 -> sorted 1,1,4/3,1.5,1.5,2 -> (4/3+1.5)/2
  CHECK(theil_sen({0, 1, 2, 3}, {0, 1, 3, 4}) == Catch::Approx((4.0 / 3.0 + 1.5) / 2.0).margin(1e-12));
  CHECK(theil_sen({0, 1, 2}, {0, 10, 2}) == Catch::Approx(1.0));  // slopes 10, 1, -8 -> median 1
  CHECK(theil_sen({1.0}, {5.0}) == 0.0);                           // no pairs
  CHECK(theil_sen({1.0, 1.0, 1.0}, {1, 2, 3}) == 0.0);              // dt <= 1e-6 pairs are dropped
}

TEST_CASE("FCWMonitor: approaching lead raises a warning, missing lead resets") {
  FCWMonitor mon(1920, 1080, 30.0, 2.7, 1.5);
  const double dt = 1.0 / 30.0;
  FcwStatus s;
  std::string ev;
  for (int k = 0; k < 90; ++k) {
    const double w = 100.0 * std::exp(0.5 * k * dt);  // d ln w / dt = 0.5 -> TTC 2 s
    const double h = 0.8 * w;
    LeadVehicle l{1, 2, Box{float(960 - w / 2), 500.f, float(960 + w / 2), float(500 + h)}, 30.0 - 6.0 * k * dt};
    s = mon.update(k * dt, l);
    if (!s.event.empty()) ev = s.event;
  }
  CHECK(ev == "warning");
  CHECK(s.active);
  CHECK(s.ttc.has_value());
  for (int k = 90; k < 140; ++k) s = mon.update(k * dt, std::nullopt);
  CHECK(!s.active);
  CHECK(!s.ttc.has_value());
}
