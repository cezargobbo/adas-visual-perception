#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "alerts_csv.hpp"
#include "cli.hpp"
#include "ffmpeg_writer.hpp"

using namespace adasvp;
using namespace adasvp::example;
using Catch::Approx;
using nlohmann::json;

namespace {

struct Parsed {
  std::optional<Options> opt;
  int code = 0;
  std::string out, err;
};

Parsed parse(std::vector<std::string> args) {
  args.insert(args.begin(), "video_demo");
  std::vector<const char*> argv;
  for (const auto& a : args) argv.push_back(a.c_str());
  std::ostringstream out, err;
  Parsed p;
  p.opt = parse_args(static_cast<int>(argv.size()), argv.data(), out, err, p.code);
  p.out = out.str();
  p.err = err.str();
  return p;
}

json load_defaults() {
  std::ifstream f(std::string(ADASVP_SOURCE_DIR) + "/tests/fixtures/argparse_defaults.json");
  REQUIRE(f.good());
  return json::parse(f);
}

}  // namespace

TEST_CASE("cli defaults match the argparse defaults of the reference", "[cli]") {
  const json all = load_defaults();
  const json& d = all["defaults"];
  auto p = parse({"in.mp4"});
  REQUIRE(p.opt);
  const Options& o = *p.opt;
  CHECK(o.input == "in.mp4");
  CHECK(o.alert_display == Approx(d["alert_display"].get<double>()));
  CHECK(o.cam_height == Approx(d["cam_height"].get<double>()));
  CHECK(o.conf == Approx(d["conf"].get<double>()));
  CHECK(o.corridor_bottom == Approx(d["corridor_bottom"].get<double>()));
  CHECK(o.corridor_top == Approx(d["corridor_top"].get<double>()));
  CHECK(!o.det_imgsz);
  CHECK(!o.det_model);
  CHECK(o.dry == d["dry"].get<bool>());
  CHECK(!o.duration);
  CHECK(o.encoder == d["encoder"].get<std::string>());
  CHECK(o.fcw_ttc == Approx(d["fcw_ttc"].get<double>()));
  CHECK(o.fcw_ttc_critical == Approx(d["fcw_ttc_critical"].get<double>()));
  CHECK(o.hfov == Approx(d["hfov"].get<double>()));
  CHECK(o.horizon == Approx(d["horizon"].get<double>()));
  CHECK(o.lanes_hold == Approx(d["lanes_hold"].get<double>()));
  CHECK(o.lanes_thr == Approx(d["lanes_thr"].get<double>()));
  CHECK(o.ldw_car_width == Approx(d["ldw_car_width"].get<double>()));
  CHECK(o.ldw_margin == Approx(d["ldw_margin"].get<double>()));
  CHECK(o.no_audio == d["no_audio"].get<bool>());
  CHECK(o.no_combined == d["no_combined"].get<bool>());
  CHECK(o.no_det == d["no_det"].get<bool>());
  CHECK(o.no_fcw == d["no_fcw"].get<bool>());
  CHECK(o.no_lane == d["no_lane"].get<bool>());
  CHECK(o.no_lane_corridor == d["no_lane_corridor"].get<bool>());
  CHECK(o.no_ldw == d["no_ldw"].get<bool>());
  CHECK(o.no_legend == d["no_legend"].get<bool>());
  CHECK(o.no_rlr == d["no_rlr"].get<bool>());
  CHECK(o.no_seg == d["no_seg"].get<bool>());
  CHECK(!o.out_prefix);
  CHECK(o.profile == d["profile"].get<std::string>());
  CHECK(o.rlr_zone.first == Approx(d["rlr_zone"][0].get<double>()));
  CHECK(o.rlr_zone.second == Approx(d["rlr_zone"][1].get<double>()));
  CHECK(o.seg_alpha == Approx(d["seg_alpha"].get<double>()));
  CHECK(o.seg_every == d["seg_every"].get<int>());
  CHECK(!o.seg_model);
  CHECK(!o.seg_size);
  CHECK(!o.show);
  CHECK(o.show_width == d["show_width"].get<int>());
  CHECK(o.start == Approx(d["start"].get<double>()));
  CHECK(!o.adas_debug);
}

TEST_CASE("cli profile defaults match the PROFILES of the reference", "[cli]") {
  const json all = load_defaults();
  for (const std::string name : {"accurate", "fast"}) {
    auto p = parse({"in.mp4", "--profile", name});
    REQUIRE(p.opt);
    const Config c = make_config(*p.opt, 30.0);
    CHECK(c.det_imgsz == all["profiles"][name]["det_imgsz"].get<int>());
    CHECK(c.det_onnx.find(name == "fast" ? "yolo26s" : "yolo26x") == 0);
    CHECK(c.seg_onnx.find(name == "fast" ? "1024x576" : "1280x720") != std::string::npos);
  }
}

TEST_CASE("cli --show without a value is det", "[cli]") {
  auto p = parse({"in.mp4", "--show"});
  REQUIRE(p.opt);
  CHECK(p.opt->show == "det");
  p = parse({"in.mp4", "--show", "comb"});
  REQUIRE(p.opt);
  CHECK(p.opt->show == "comb");
  p = parse({"in.mp4", "--show", "bogus"});
  CHECK(!p.opt);
  CHECK(p.code == 2);
}

TEST_CASE("cli --show falls back to what is computed", "[cli]") {
  auto p = parse({"in.mp4", "--show", "seg", "--no-seg"});
  REQUIRE(p.opt);
  CHECK(p.opt->show == "det");
  p = parse({"in.mp4", "--show", "comb", "--no-det"});
  REQUIRE(p.opt);
  CHECK(p.opt->show == "seg");
}

TEST_CASE("cli --rlr-zone takes two numbers and validates them", "[cli]") {
  auto p = parse({"in.mp4", "--rlr-zone", "0.3", "0.7"});
  REQUIRE(p.opt);
  CHECK(p.opt->rlr_zone.first == Approx(0.3));
  CHECK(p.opt->rlr_zone.second == Approx(0.7));
  p = parse({"in.mp4", "--rlr-zone", "0.7", "0.3"});
  CHECK(!p.opt);
  CHECK(p.code == 2);
  CHECK(p.err.find("--rlr-zone") != std::string::npos);
}

TEST_CASE("cli usage errors end with code 2", "[cli]") {
  auto p = parse({"in.mp4", "--no-such-flag"});
  CHECK(!p.opt);
  CHECK(p.code == 2);
  CHECK(p.err.find("no-such-flag") != std::string::npos);
  p = parse({});
  CHECK(!p.opt);
  CHECK(p.code == 2);
  p = parse({"in.mp4", "--no-seg", "--no-det"});
  CHECK(!p.opt);
  CHECK(p.code == 2);
  p = parse({"in.mp4", "--encoder", "vp9"});
  CHECK(!p.opt);
  CHECK(p.code == 2);
  p = parse({"in.mp4", "--det-precision", "fp8"});
  CHECK(!p.opt);
  CHECK(p.code == 2);
}

TEST_CASE("cli help lists groups and examples and leaves hidden flags out", "[cli]") {
  auto p = parse({"-h"});
  CHECK(!p.opt);
  CHECK(p.code == 0);
  for (const char* s : {"input and output:", "what to run:", "live view:", "ADAS alerts (FCW / LDW / RLR; on by default):",
                        "examples:", "--rlr-zone L R", "--alert-display SEC", "video_demo sp1_crop.mp4 --profile fast --show"})
    CHECK(p.out.find(s) != std::string::npos);
  CHECK(p.out.find("--dry") == std::string::npos);
  CHECK(p.out.find("--adas-debug") == std::string::npos);
  CHECK(p.out.find("--det-precision") == std::string::npos);
}

TEST_CASE("cli --no-det / --no-lane couplings", "[cli]") {
  auto p = parse({"in.mp4", "--no-det"});
  REQUIRE(p.opt);
  CHECK(!p.opt->lanes);
  CHECK(!p.opt->fcw);
  CHECK(!p.opt->ldw);
  CHECK(!p.opt->rlr);
  p = parse({"in.mp4", "--no-lane"});
  REQUIRE(p.opt);
  CHECK(!p.opt->lanes);
  CHECK(p.opt->fcw);
  CHECK(!p.opt->ldw);
  CHECK(p.opt->rlr);
  const Config c = make_config(*p.opt, 25.0);
  CHECK(!c.enable_lanes);
  CHECK(!c.enable_ldw);
  CHECK(c.enable_det);
  CHECK(c.frame_rate == Approx(25.0));
  p = parse({"in.mp4", "--no-det"});
  const Config c2 = make_config(*p.opt, 30.0);
  CHECK(!c2.enable_det);
  CHECK(!c2.enable_lanes);
  CHECK(!c2.enable_fcw);
  CHECK(!c2.enable_rlr);
}

TEST_CASE("cli make_config carries the ADAS parameters and the hidden precision", "[cli]") {
  auto p = parse({"in.mp4", "--profile", "fast", "--hfov", "80", "--ldw-margin", "0.4", "--rlr-zone", "0.2", "0.8",
                  "--no-lane-corridor", "--det-precision", "fp32", "--det-imgsz", "640", "--seg-every", "3"});
  REQUIRE(p.opt);
  const Config c = make_config(*p.opt, 30.0);
  CHECK(c.hfov == Approx(80));
  CHECK(c.ldw_margin == Approx(0.4));
  CHECK(c.rlr_zone.first == Approx(0.2));
  CHECK(!c.lane_corridor);
  CHECK(c.det_precision == Precision::FP32);
  CHECK(c.det_imgsz == 640);
  CHECK(c.seg_every == 3);
  CHECK(c.alert_display == Approx(2.0));
}

TEST_CASE("output plan", "[cli]") {
  CHECK(output_prefix(*parse({"/some/dir/sp1_crop.mp4"}).opt) == "outputs/sp1_crop");
  CHECK(output_prefix(*parse({"a.mp4", "-o", "out/x"}).opt) == "out/x");

  auto plan = plan_outputs(*parse({"/d/sp1.mp4"}).opt);
  CHECK(plan.prefix == "outputs/sp1");
  CHECK(plan.videos == std::vector<std::string>{"outputs/sp1_detections.mp4", "outputs/sp1_segmentation.mp4",
                                                "outputs/sp1_combined.mp4"});
  CHECK(plan.csv == "outputs/sp1_alerts.csv");

  plan = plan_outputs(*parse({"/d/sp1.mp4", "--show"}).opt);
  CHECK(plan.videos.empty());
  CHECK(!plan.csv);
  plan = plan_outputs(*parse({"/d/sp1.mp4", "--show", "seg", "-o", "x/y"}).opt);
  CHECK(plan.videos.empty());
  CHECK(!plan.csv);

  plan = plan_outputs(*parse({"/d/sp1.mp4", "-o", "o/s", "--no-seg"}).opt);
  CHECK(plan.videos == std::vector<std::string>{"o/s_detections.mp4"});
  CHECK(plan.csv == "o/s_alerts.csv");

  plan = plan_outputs(*parse({"/d/sp1.mp4", "-o", "o/s", "--no-det"}).opt);
  CHECK(plan.videos == std::vector<std::string>{"o/s_segmentation.mp4"});
  CHECK(!plan.csv);  // no alert monitor left

  plan = plan_outputs(*parse({"/d/sp1.mp4", "-o", "o/s", "--no-combined"}).opt);
  CHECK(plan.videos == std::vector<std::string>{"o/s_detections.mp4", "o/s_segmentation.mp4"});

  plan = plan_outputs(*parse({"/d/sp1.mp4", "-o", "o/s", "--no-fcw", "--no-ldw", "--no-rlr"}).opt);
  CHECK(plan.videos.size() == 3);
  CHECK(!plan.csv);

  plan = plan_outputs(*parse({"/d/sp1.mp4", "-o", "o/s", "--dry"}).opt);
  CHECK(plan.videos.empty());
  CHECK(!plan.csv);  // like the reference: --dry writes no file
}

TEST_CASE("alerts csv header and numeric formatting follow log_event", "[cli]") {
  CHECK(std::string(kAlertsCsvHeader) ==
        "time_s,frame,type,level_side,ttc_s,line_dist_m,lat_vel_mps,tlc_s,lead_id,lead_dist_m,red_for_s");
  Alert fcw;
  fcw.type = "FCW";
  fcw.level_side = "warning";
  fcw.t = 12.3456;
  fcw.frame = 370;
  fcw.ttc_s = 2.6549;
  fcw.lead_id = 7;
  fcw.lead_dist_m = 18.26;
  CHECK(format_alert_row(fcw) == "12.35,370,FCW,warning,2.65,,,,7,18.3,");
  CHECK(format_alert_detail(fcw) == "2.65,,,,7,18.3,");
  Alert ldw;
  ldw.type = "LDW";
  ldw.level_side = "left";
  ldw.t = 3.0;
  ldw.frame = 90;
  ldw.line_dist_m = 0.1234;
  ldw.lat_vel_mps = -0.456;
  ldw.tlc_s = 0.789;
  CHECK(format_alert_row(ldw) == "3.00,90,LDW,left,,0.12,-0.46,0.79,,,");
  Alert rlr;
  rlr.type = "RLR";
  rlr.level_side = "red";
  rlr.t = 50.0;
  rlr.frame = 1500;
  rlr.red_for_s = 3.14159;
  CHECK(format_alert_row(rlr) == "50.00,1500,RLR,red,,,,,,,3.1");
}

TEST_CASE("ffmpeg writer command line mirrors the reference", "[cli]") {
  WriterSpec s;
  s.width = 1920;
  s.height = 1080;
  s.fps = 30.0;
  s.encoder = "x264";
  s.audio_from = "in.mp4";
  s.start_s = 4.0;
  s.duration_s = 5.0;
  s.path = "o.mp4";
  const auto a = ffmpeg_args(s);
  const std::string j = [&] {
    std::string r;
    for (const auto& x : a) r += x + " ";
    return r;
  }();
  CHECK(j.find("-f rawvideo -pix_fmt bgr24 -s 1920x1080 -r 30") != std::string::npos);
  CHECK(j.find("-ss 4") != std::string::npos);
  CHECK(j.find("-t 5") != std::string::npos);
  CHECK(j.find("-map 0:v -map 1:a? -c:a aac -shortest") != std::string::npos);
  CHECK(j.find("-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p o.mp4") != std::string::npos);
  s.encoder = "nvenc";
  s.audio_from.clear();
  const auto b = ffmpeg_args(s);
  std::string jb;
  for (const auto& x : b) jb += x + " ";
  CHECK(jb.find("-c:v h264_nvenc -preset medium -rc vbr -cq 21 -b:v 0 -pix_fmt yuv420p o.mp4") != std::string::npos);
  CHECK(jb.find("-ss") == std::string::npos);
}
