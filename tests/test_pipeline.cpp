#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <opencv2/videoio.hpp>
#include <string>

#include "adasvp/adasvp.hpp"

namespace fs = std::filesystem;
using namespace adasvp;

TEST_CASE("Pipeline handles resolution change") {
  const char* dir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!dir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path a = fs::path(dir) / "sp1_crop_30fps.mp4";
  const fs::path b = fs::path(dir) / "redlights" / "red_crop_9_30fps.mp4";
  if (!fs::exists(a) || !fs::exists(b)) SKIP("test videos not found");

  Config cfg = Config::fast();
  cfg.models_dir = std::string(ADASVP_SOURCE_DIR) + "/models";
  if (const char* m = std::getenv("ADASVP_MODELS_DIR")) cfg.models_dir = m;
  cfg.seg_every = 10;
  Pipeline pipe(cfg);

  int total_objects = 0;
  for (const fs::path& v : {a, b}) {
    cv::VideoCapture cap(v.string());
    REQUIRE(cap.isOpened());
    cv::Mat frame;
    for (int i = 0; i < 30; ++i) {
      REQUIRE(cap.read(frame));
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
