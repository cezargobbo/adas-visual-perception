#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "adasvp/adasvp.hpp"

namespace fs = std::filesystem;
using namespace adasvp;

TEST_CASE("Pipeline handles resolution change") {
  const char* dir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!dir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path v = fs::path(dir) / "sp1_crop_30fps.mp4";
  if (!fs::exists(v)) SKIP("test video not found");

  Config cfg = Config::fast();
  cfg.models_dir = std::string(ADASVP_SOURCE_DIR) + "/models";
  if (const char* m = std::getenv("ADASVP_MODELS_DIR")) cfg.models_dir = m;
  cfg.seg_every = 10;
  Pipeline pipe(cfg);

  int total_objects = 0;
  // Native 1920x1080, then the same frames resized to 854x474 (an odd, non-16:9 size) on the same Pipeline.
  for (const cv::Size size : {cv::Size(), cv::Size(854, 474)}) {
    cv::VideoCapture cap(v.string());
    REQUIRE(cap.isOpened());
    cv::Mat frame;
    for (int i = 0; i < 30; ++i) {
      REQUIRE(cap.read(frame));
      if (!size.empty()) cv::resize(frame, frame, size, 0, 0, cv::INTER_AREA);
      FrameResult r;
      REQUIRE_NOTHROW(r = pipe.process(frame, i / 30.0));
      REQUIRE(r.seg_classes.size() == frame.size());
      REQUIRE(r.seg_classes.type() == CV_8UC1);
      for (const Object& o : r.objects) {
        ++total_objects;
        REQUIRE(o.box.x1 >= 0);
        REQUIRE(o.box.y1 >= 0);
        REQUIRE(o.box.x2 <= frame.cols);
        REQUIRE(o.box.y2 <= frame.rows);
        REQUIRE(o.box.x2 >= o.box.x1);
        REQUIRE(o.box.y2 >= o.box.y1);
      }
    }
  }
  CHECK(total_objects > 0);
}

TEST_CASE("Pipeline::reset restarts track ids at 1") {
  const char* dir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!dir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path v = fs::path(dir) / "sp1_crop_30fps.mp4";
  if (!fs::exists(v)) SKIP("test video not found");

  Config cfg = Config::fast();
  cfg.models_dir = std::string(ADASVP_SOURCE_DIR) + "/models";
  if (const char* m = std::getenv("ADASVP_MODELS_DIR")) cfg.models_dir = m;
  if (!fs::exists(fs::path(cfg.models_dir) / cfg.det_onnx)) SKIP("detector model not found");
  cfg.enable_seg = false;
  cfg.enable_lanes = false;
  Pipeline pipe(cfg);

  // Runs the first 60 frames and returns the smallest track id of the first frame that has tracks
  // (new tracks are hidden on their first frame, so the first visible ids are the oldest ones).
  auto first_ids = [&](Pipeline& p) {
    cv::VideoCapture cap(v.string());
    REQUIRE(cap.isOpened());
    cv::Mat frame;
    int min_id = -1, max_id = -1;
    for (int i = 0; i < 60; ++i) {
      REQUIRE(cap.read(frame));
      for (const Object& o : p.process(frame, i / 30.0).objects) {
        if (min_id < 0 || o.track_id < min_id) min_id = o.track_id;
        max_id = std::max(max_id, o.track_id);
      }
    }
    return std::make_pair(min_id, max_id);
  };

  const auto first = first_ids(pipe);
  REQUIRE(first.first >= 1);
  CHECK(first.first == 1);
  pipe.reset();
  const auto second = first_ids(pipe);
  CHECK(second.first == 1);  // ids restart at 1 after reset()
  CHECK(second.second == first.second);  // same frames, same tracker state: same ids
}
