#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "adasvp/render.hpp"
#include "support/trace_json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace adasvp;

namespace {

Box box_of(const json& b) {
  return Box{b[0].get<float>(), b[1].get<float>(), b[2].get<float>(), b[3].get<float>()};
}

LaneState lane_state(const std::string& s) {
  return s == "detected" ? LaneState::Detected
         : s == "partial" ? LaneState::Partial
         : s == "estimated" ? LaneState::Estimated
                            : LaneState::None;
}

std::string str_or_empty(const json& v) { return v.is_null() ? std::string() : v.get<std::string>(); }

std::optional<double> opt_num(const json& v) {
  if (v.is_null()) return std::nullopt;
  return adasvp_test::num(v);
}

// FrameResult rebuilt from the recorded trace (what the Python drawing received).
FrameResult build_result(const json& fr, const json& ad, const cv::Mat& seg, const std::vector<double>& ego_ys) {
  FrameResult r;
  r.t = ad["t"].get<double>();
  for (const auto& t : fr["tracks"])
    r.objects.push_back(Object{box_of(t), t[6].get<int>(), t[5].get<float>(), t[4].get<int>()});
  for (const auto& l : fr["lanes"]) {
    Lane lane;
    for (const auto& p : l) lane.points.emplace_back(p[0].get<float>(), p[1].get<float>());
    r.lanes.push_back(std::move(lane));
  }
  r.seg_classes = seg;
  const json& e = ad["ego"];
  r.ego_lane.state = lane_state(e["state"].get<std::string>());
  r.ego_lane.available = e["available"].get<bool>();
  r.ego_lane.fade = e["fade"].get<double>();
  r.ego_lane.obs_l = e["obs_l"].get<bool>();
  r.ego_lane.obs_r = e["obs_r"].get<bool>();
  if (!e["xl"].is_null()) {
    r.ego_lane.rows_y = ego_ys;
    r.ego_lane.xl = adasvp_test::vec(e["xl"]);
    r.ego_lane.xr = adasvp_test::vec(e["xr"]);
  }
  if (!ad["lead"].is_null()) {
    const json& l = ad["lead"];
    r.lead = LeadVehicle{l[0].get<int>(), l[1].get<int>(), box_of(l[2]), l[3].get<double>()};
  }
  r.fcw.ttc = opt_num(ad["fcw"]["ttc"]);
  r.fcw.active = ad["fcw"]["active"].get<bool>();
  r.fcw.level = str_or_empty(ad["fcw"]["level"]);
  r.ldw.enabled = true;  // the LDW monitor existed in every recorded trace
  r.ldw.armed = ad["ldw"]["armed"].get<bool>();
  r.ldw.side = str_or_empty(ad["ldw"]["side"]);
  if (!ad["rlr"]["event"].is_null()) r.rlr.event = RlrEvent{};
  return r;
}

// identical pixels (all three channels equal) and mean absolute difference per channel
void accumulate(const cv::Mat& a, const cv::Mat& b, long long& same, long long& pixels, double& sum_abs) {
  REQUIRE(a.size() == b.size());
  REQUIRE(a.type() == b.type());
  for (int y = 0; y < a.rows; ++y) {
    const uchar* pa = a.ptr<uchar>(y);
    const uchar* pb = b.ptr<uchar>(y);
    for (int x = 0; x < a.cols; ++x) {
      const int d0 = std::abs(pa[3 * x] - pb[3 * x]), d1 = std::abs(pa[3 * x + 1] - pb[3 * x + 1]),
                d2 = std::abs(pa[3 * x + 2] - pb[3 * x + 2]);
      same += (d0 | d1 | d2) == 0;
      sum_abs += d0 + d1 + d2;
    }
  }
  pixels += static_cast<long long>(a.rows) * a.cols;
}

struct Window {
  const char* trace;
  int a, b;
  const char* what;
};

}  // namespace

TEST_CASE("Renderer matches the Python drawing") {
  const char* vdir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!vdir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path src(ADASVP_SOURCE_DIR);
  const std::vector<Window> windows = {
      {"sp1_crop_30fps", 3820, 3880, "FCW"},     {"sp1_crop_30fps", 4580, 4640, "LDW"},
      {"sp1_crop_4_30fps", 470, 530, "LDW"},     {"sp1_crop_4_30fps", 1040, 1100, "FCW critical"},
      {"red_crop_11_30fps", 450, 510, "RLR"},
      // windows ending at the instants of the README GIFs (30.63 s, 148.08 s and 15.53 s)
      {"sp1_crop_30fps", 860, 920, "GIF 30.63 s"},
      {"sp1_crop_4_30fps", 4381, 4441, "GIF 148.08 s"},
      {"red_crop_11_30fps", 410, 470, "GIF 15.53 s"},
  };
  const char* dump_dir = std::getenv("ADASVP_RENDER_OUT");
  int checked = 0, missing = 0;
  for (const Window& w : windows) {
    const std::string name = std::string(w.trace) + "_" + std::to_string(w.a) + "_" + std::to_string(w.b);
    const fs::path trace = src / "outputs/dumps/fast" / w.trace;
    const fs::path ref = src / "outputs/render_ref" / name;
    if (!fs::exists(ref / "manifest.json") || !fs::exists(trace / "adas.jsonl")) {
      WARN("no reference render for " << name << " (tools/regression/render_reference.py)");
      ++missing;
      continue;
    }
    fs::path video = fs::path(vdir) / json::parse(std::ifstream(trace / "meta.json"))["video"].get<std::string>();
    if (!fs::exists(video)) video = fs::path(vdir) / "redlights" / video.filename();
    if (!fs::exists(video)) {
      WARN("test video not found for " << name);
      ++missing;
      continue;
    }
    const json manifest = json::parse(std::ifstream(ref / "manifest.json"));
    const int start = manifest["start_frame"].get<int>();
    const std::vector<double> ego_ys =
        adasvp_test::vec(json::parse(std::ifstream(trace / "meta.json"))["adas"]["ego_ys"]);

    std::vector<json> frames, adas;
    {
      std::ifstream ff(trace / "frames.jsonl"), fa(trace / "adas.jsonl");
      std::string l1, l2;
      for (int k = 0; k < w.b && std::getline(ff, l1) && std::getline(fa, l2); ++k)
        if (k >= w.a) {
          frames.push_back(json::parse(l1));
          adas.push_back(json::parse(l2));
        }
    }
    REQUIRE(static_cast<int>(frames.size()) == w.b - w.a);

    cv::VideoCapture cap(video.string());
    REQUIRE(cap.isOpened());
    cv::Mat frame;
    for (int k = 0; k < start + w.a; ++k) REQUIRE(cap.read(frame));

    RenderOptions opt;
    opt.alert_display_s = manifest["alert_display"].get<double>();
    opt.seg_alpha = manifest["seg_alpha"].get<double>();
    Renderer rend(opt);
    const struct {
      View v;
      const char* tag;
    } views[] = {{View::Detections, "det"}, {View::Segmentation, "seg"}, {View::Combined, "comb"}};
    long long same[3] = {0, 0, 0}, pixels[3] = {0, 0, 0};
    double sum_abs[3] = {0, 0, 0};
    for (int k = w.a; k < w.b; ++k) {
      REQUIRE(cap.read(frame));
      const cv::Mat seg = cv::imread(
          (trace / "seg" / manifest["seg"][std::to_string(k)].get<std::string>()).string(), cv::IMREAD_GRAYSCALE);
      REQUIRE(!seg.empty());
      const FrameResult r = build_result(frames[k - w.a], adas[k - w.a], seg, ego_ys);
      const auto out = rend.render(frame, r, {View::Detections, View::Segmentation, View::Combined});
      REQUIRE(out.size() == 3);
      for (int v = 0; v < 3; ++v) {
        char fn[64];
        std::snprintf(fn, sizeof fn, "%s_%06d.png", views[v].tag, k);
        const cv::Mat want = cv::imread((ref / fn).string(), cv::IMREAD_COLOR);
        REQUIRE(!want.empty());
        accumulate(out.at(views[v].v), want, same[v], pixels[v], sum_abs[v]);
        if (dump_dir) {  // ADASVP_RENDER_OUT: keep the C++ renders for inspection
          fs::create_directories(fs::path(dump_dir) / name);
          cv::imwrite((fs::path(dump_dir) / name / fn).string(), out.at(views[v].v));
        }
      }
    }
    for (int v = 0; v < 3; ++v) {
      const double pct = 100.0 * static_cast<double>(same[v]) / static_cast<double>(pixels[v]);
      const double mad = sum_abs[v] / (3.0 * static_cast<double>(pixels[v]));
      std::cout << "[render] " << name << " (" << w.what << ") " << views[v].tag << ": identical " << pct
                << " %, mean abs diff " << mad << "\n";
      CHECK(pct >= 99.5);
      CHECK(mad <= 0.5);
    }
    ++checked;
  }
  std::cout << "[render] windows checked: " << checked << " of " << windows.size() << "\n";
  if (checked == 0) SKIP("no reference render available");
  // some windows have their inputs and others do not: a partial check must not pass silently
  CHECK(missing == 0);
}

TEST_CASE("Renderer omits the LDW status label when LDW is disabled") {
  const int W = 1280, H = 720;
  cv::Mat frame(H, W, CV_8UC3, cv::Scalar(90, 100, 110));
  FrameResult r;
  r.ego_lane.state = LaneState::Detected;
  r.ego_lane.available = true;
  r.ego_lane.obs_l = r.ego_lane.obs_r = true;
  for (double y = 400; y < H; y += 20) {
    r.ego_lane.rows_y.push_back(y);
    r.ego_lane.xl.push_back(400);
    r.ego_lane.xr.push_back(880);
  }
  Renderer on, off;
  r.ldw.enabled = true;
  const cv::Mat with_label = on.render(frame, r, {View::Detections}).at(View::Detections);
  r.ldw.enabled = false;
  const cv::Mat without = off.render(frame, r, {View::Detections}).at(View::Detections);

  // the label sits at (12, H - 46); everything outside its box is identical, inside it the labels differ
  const cv::Rect label(12, H - 46 - 30, 170, 32);
  cv::Mat diff;
  cv::absdiff(with_label, without, diff);
  cv::Mat outside = diff.clone();
  outside(label).setTo(cv::Scalar::all(0));
  CHECK(cv::countNonZero(outside.reshape(1)) == 0);
  CHECK(cv::countNonZero(diff(label).clone().reshape(1)) > 0);
  // with the label off, the label box holds only the lane fill: same as the frame region blended by the lane
  // polygon, which is absent at x < 400, so those pixels are the untouched frame
  const cv::Rect left(12, H - 46 - 30, 170, 32);
  cv::Mat untouched = frame(left).clone(), got = without(left).clone();
  cv::absdiff(untouched, got, diff);
  CHECK(cv::countNonZero(diff.reshape(1)) == 0);
}
