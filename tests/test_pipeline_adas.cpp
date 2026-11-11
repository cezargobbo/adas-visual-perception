#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <opencv2/videoio.hpp>
#include <string>

#include "adasvp/adasvp.hpp"

namespace fs = std::filesystem;
using namespace adasvp;

TEST_CASE("Pipeline ADAS fields are consistent") {
  const char* dir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!dir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path v = fs::path(dir) / "sp1_crop_30fps.mp4";
  if (!fs::exists(v)) SKIP("test video not found");

  Config cfg = Config::fast();
  cfg.models_dir = std::string(ADASVP_SOURCE_DIR) + "/models";
  if (const char* m = std::getenv("ADASVP_MODELS_DIR")) cfg.models_dir = m;
  cfg.seg_every = 30;
  cv::VideoCapture cap(v.string());
  REQUIRE(cap.isOpened());
  cfg.frame_rate = cap.get(cv::CAP_PROP_FPS);
  Pipeline pipe(cfg);

  cv::Mat frame;
  int n_ldw_enabled = 0;
  int n_alerts = 0, n_events = 0, n_lead = 0, n_available = 0;
  LaneState first_state = LaneState::None;
  bool first_lead = false;
  for (int i = 0; i < 150; ++i) {
    REQUIRE(cap.read(frame));
    const FrameResult r = pipe.process(frame, i / cfg.frame_rate);

    // alerts only in transition frames: one per event raised in this frame
    const int events = !r.fcw.event.empty() + !r.ldw.event.empty() + r.rlr.event.has_value();
    CHECK(static_cast<int>(r.alerts.size()) == events);
    n_alerts += static_cast<int>(r.alerts.size());
    n_events += events;
    for (const Alert& a : r.alerts) {
      CHECK((a.type == "FCW" || a.type == "LDW" || a.type == "RLR"));
      CHECK(a.t == r.t);
    }

    // the lead is one of the tracked objects
    if (r.lead) {
      ++n_lead;
      bool found = false;
      for (const Object& o : r.objects)
        found |= o.track_id == r.lead->track_id && o.cls == r.lead->cls && o.box.x1 == r.lead->box.x1 &&
                 o.box.y2 == r.lead->box.y2;
      CHECK(found);
      CHECK(r.lead->distance_m > 0);
    }

    // ego lane state is valid and consistent
    n_ldw_enabled += r.ldw.enabled;
    const EgoLane& e = r.ego_lane;
    const int st = static_cast<int>(e.state);
    REQUIRE((st >= static_cast<int>(LaneState::None) && st <= static_cast<int>(LaneState::Detected)));
    CHECK(e.available == (e.state != LaneState::None));
    if (e.available) {
      ++n_available;
      CHECK(e.xl.size() == e.rows_y.size());
      CHECK(e.xr.size() == e.rows_y.size());
    } else {
      CHECK(e.xl.empty());
    }
    if (i == 0) {
      first_state = e.state;
      first_lead = r.lead.has_value();
    }
  }
  CHECK(n_available > 0);
  CHECK(n_ldw_enabled == 150);  // the LDW monitor runs on every frame by default
  CHECK(n_lead > 0);
  const AdasStats st = pipe.stats();
  CHECK(st.fcw_events + st.ldw_events + st.rlr_events == n_alerts);
  CHECK(!st.ldw_frames.empty());

  // reset() clears the counters and the monitors: the first frame behaves as on a fresh Pipeline
  pipe.reset();
  const AdasStats z = pipe.stats();
  CHECK(z.fcw_events + z.ldw_events + z.rlr_events == 0);
  CHECK(z.ldw_frames.empty());
  CHECK(z.ldw_crossings_suppressed == 0);
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);
  REQUIRE(cap.read(frame));
  const FrameResult r0 = pipe.process(frame, 0.0);
  CHECK(r0.ego_lane.state == first_state);
  CHECK(r0.lead.has_value() == first_lead);
  CHECK(r0.alerts.empty());
}

TEST_CASE("Pipeline reports ldw.enabled only when the LDW monitor exists") {
  const char* dir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!dir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path v = fs::path(dir) / "sp1_crop_30fps.mp4";
  if (!fs::exists(v)) SKIP("test video not found");

  Config cfg = Config::fast();
  cfg.models_dir = std::string(ADASVP_SOURCE_DIR) + "/models";
  if (const char* m = std::getenv("ADASVP_MODELS_DIR")) cfg.models_dir = m;
  cfg.seg_every = 30;
  cfg.enable_ldw = false;
  cv::VideoCapture cap(v.string());
  REQUIRE(cap.isOpened());
  cfg.frame_rate = cap.get(cv::CAP_PROP_FPS);
  Pipeline pipe(cfg);
  cv::Mat frame;
  for (int i = 0; i < 5; ++i) {
    REQUIRE(cap.read(frame));
    CHECK_FALSE(pipe.process(frame, i / cfg.frame_rate).ldw.enabled);
  }
}
