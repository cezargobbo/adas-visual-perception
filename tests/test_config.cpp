#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "adasvp/adasvp.hpp"

using namespace adasvp;
using Catch::Approx;
using nlohmann::json;

namespace {

json load_defaults() {
  std::ifstream f(std::string(ADASVP_SOURCE_DIR) + "/tests/fixtures/argparse_defaults.json");
  REQUIRE(f.good());
  return json::parse(f);
}

void check_common(const Config& c, const json& d) {
  CHECK(c.hfov == Approx(d["hfov"].get<double>()));
  CHECK(c.cam_height == Approx(d["cam_height"].get<double>()));
  CHECK(c.horizon == Approx(d["horizon"].get<double>()));
  CHECK(c.corridor_bottom == Approx(d["corridor_bottom"].get<double>()));
  CHECK(c.corridor_top == Approx(d["corridor_top"].get<double>()));
  CHECK(c.lanes_thr == Approx(d["lanes_thr"].get<double>()));
  CHECK(c.lanes_hold == Approx(d["lanes_hold"].get<double>()));
  CHECK(c.lane_corridor == !d["no_lane_corridor"].get<bool>());
  CHECK(c.fcw_ttc == Approx(d["fcw_ttc"].get<double>()));
  CHECK(c.fcw_ttc_critical == Approx(d["fcw_ttc_critical"].get<double>()));
  CHECK(c.ldw_margin == Approx(d["ldw_margin"].get<double>()));
  CHECK(c.ldw_car_width == Approx(d["ldw_car_width"].get<double>()));
  CHECK(c.rlr_zone.first == Approx(d["rlr_zone"][0].get<double>()));
  CHECK(c.rlr_zone.second == Approx(d["rlr_zone"][1].get<double>()));
  CHECK(c.alert_display == Approx(d["alert_display"].get<double>()));
  CHECK(c.seg_alpha == Approx(d["seg_alpha"].get<double>()));
  CHECK(c.show_width == d["show_width"].get<int>());
  CHECK(c.conf == Approx(d["conf"].get<double>()));
  CHECK(c.seg_every == d["seg_every"].get<int>());
  CHECK(c.enable_fcw == !d["no_fcw"].get<bool>());
  CHECK(c.enable_ldw == !d["no_ldw"].get<bool>());
  CHECK(c.enable_rlr == !d["no_rlr"].get<bool>());
}

}  // namespace

TEST_CASE("Config defaults match the Python argparse defaults") {
  const json j = load_defaults();
  const json& d = j["defaults"];

  SECTION("fast") {
    const Config c = Config::fast();
    check_common(c, d);
    CHECK(c.det_imgsz == j["profiles"]["fast"]["det_imgsz"].get<int>());
  }
  SECTION("accurate") {
    const Config c = Config::accurate();
    check_common(c, d);
    CHECK(c.det_imgsz == j["profiles"]["accurate"]["det_imgsz"].get<int>());
  }
  SECTION("a default-constructed Config") {
    check_common(Config{}, d);
  }
}

TEST_CASE("ADAS result types default to the idle state") {
  const FrameResult r;
  CHECK_FALSE(r.lead.has_value());
  CHECK(r.ego_lane.state == LaneState::None);
  CHECK_FALSE(r.ego_lane.available);
  CHECK(r.lights.empty());
  CHECK(r.alerts.empty());
  CHECK_FALSE(r.fcw.active);
  CHECK_FALSE(r.fcw.ttc.has_value());
  CHECK_FALSE(r.ldw.armed);
  CHECK_FALSE(r.rlr.event.has_value());
  const Alert a;
  CHECK(a.frame == -1);
  CHECK(a.lead_id == -1);
  const AdasStats s;
  CHECK(s.fcw_events == 0);
  CHECK(s.ldw_crossings_suppressed == 0);
}
