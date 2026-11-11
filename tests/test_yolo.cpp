#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

#include "cuda/cuda_check.hpp"
#include "models/yolo_detector.hpp"

namespace fs = std::filesystem;
using namespace adasvp;

namespace {
void setRow(std::vector<float>& o, int i, float x1, float y1, float x2, float y2, float s, float c) {
  float* r = &o[i * 6];
  r[0] = x1; r[1] = y1; r[2] = x2; r[3] = y2; r[4] = s; r[5] = c;
}
float iou(const Box& a, const Box& b) {
  const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = iw * ih;
  const float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return u > 0 ? inter / u : 0.f;
}
}  // namespace

TEST_CASE("decode_yolo") {
  const std::vector<int> classes{0, 1, 2, 3, 5, 7, 9};
  // 1920x1080 at imgsz 960: scale 0.5, letterbox 960x544, pad_top 2
  const cv::Size frame(1920, 1080);
  const LetterboxInfo info = letterbox_shape(frame, 960);
  std::vector<float> out(300 * 6, 0.f);
  setRow(out, 0, 100, 100, 200, 200, 0.30f, 2);   // dropped: score == conf
  setRow(out, 1, 100, 100, 200, 200, 0.31f, 2);   // kept
  setRow(out, 2, 100, 100, 200, 200, 0.90f, 4);   // dropped: class 4
  setRow(out, 3, -10, -5, -2, 1, 0.80f, 0);       // fully inside padding -> clipped, w >= 0
  setRow(out, 4, 900, 500, 1000, 600, 0.80f, 7);  // beyond right/bottom edge

  auto objs = decode_yolo(out.data(), info, frame, 0.30f, classes);
  REQUIRE(objs.size() == 3);
  CHECK(objs[0].score == 0.31f);
  CHECK(objs[0].cls == 2);
  CHECK(objs[0].track_id == -1);
  CHECK(objs[0].box.x1 == (100 - info.pad_left) / info.scale);
  CHECK(objs[0].box.y1 == (100 - info.pad_top) / info.scale);
  CHECK(objs[0].box.x2 == (200 - info.pad_left) / info.scale);
  CHECK(objs[0].box.y2 == (200 - info.pad_top) / info.scale);
  CHECK(objs[1].box.x1 == 0.f);
  CHECK(objs[1].box.y1 == 0.f);
  CHECK(objs[1].box.x2 >= objs[1].box.x1);
  CHECK(objs[1].box.y2 >= objs[1].box.y1);
  CHECK(objs[2].box.x2 == 1920.f);
  CHECK(objs[2].box.y2 <= 1080.f);
  for (auto& o : objs) {
    CHECK(o.box.x2 >= o.box.x1);
    CHECK(o.box.y2 >= o.box.y1);
  }
}

TEST_CASE("YoloDetector matches reference dump") {
  const char* vdir = std::getenv("ADASVP_VIDEOS_DIR");
  if (!vdir) SKIP("ADASVP_VIDEOS_DIR not set");
  const fs::path video = fs::path(vdir) / "sp1_crop_30fps.mp4";
  const fs::path dump = fs::path(ADASVP_SOURCE_DIR) / "outputs/dumps/fast/sp1_crop_30fps/frames.jsonl";
  const fs::path onnx = fs::path(ADASVP_SOURCE_DIR) / "models/yolo26s.onnx";
  if (!fs::exists(video)) SKIP("video not found");
  if (!fs::exists(dump)) SKIP("reference dump not found");
  if (!fs::exists(onnx)) SKIP("yolo26s ONNX not found");

  const fs::path cache = fs::temp_directory_path() / "adasvp_test_engine_cache_yolo";
  YoloDetector det(onnx.string(), trt::Precision::FP16, cache.string(), 960, 544);
  cudaStream_t stream;
  ADASVP_CUDA_CHECK(cudaStreamCreate(&stream));

  cv::VideoCapture cap(video.string());
  REQUIRE(cap.isOpened());
  std::ifstream js(dump);
  REQUIRE(js);

  const std::vector<int> classes{0, 1, 2, 3, 5, 7, 9};
  std::map<int, int> tp, nref, ndet;
  std::string line;
  for (int f = 0; f < 60 && std::getline(js, line); ++f) {
    cv::Mat frame;
    REQUIRE(cap.read(frame));
    const auto ref = nlohmann::json::parse(line)["dets"];
    auto got = det.detect(frame, 960, 0.30f, classes, stream);
    std::vector<bool> used(got.size(), false);
    for (const auto& r : ref) {
      const int c = r[5].get<int>();
      const Box rb{r[0], r[1], r[2], r[3]};
      ++nref[c];
      int best = -1;
      float bi = 0.9f;
      for (size_t k = 0; k < got.size(); ++k) {
        if (used[k] || got[k].cls != c) continue;
        const float v = iou(rb, got[k].box);
        if (v >= bi) { bi = v; best = static_cast<int>(k); }
      }
      if (best >= 0) { used[best] = true; ++tp[c]; }
    }
    for (const auto& g : got) ++ndet[g.cls];
  }
  ADASVP_CUDA_CHECK(cudaStreamDestroy(stream));

  int TP = 0, NR = 0, ND = 0;
  for (auto& [c, n] : nref) {
    std::cout << "class " << c << ": ref " << n << " det " << ndet[c] << " tp " << tp[c] << "\n";
    NR += n; TP += tp[c];
  }
  for (auto& [c, n] : ndet) ND += n;
  const double recall = NR ? double(TP) / NR : 1.0, prec = ND ? double(TP) / ND : 1.0;
  std::cout << "recall " << recall << " precision " << prec << " (ref " << NR << ", det " << ND << ")\n";
  REQUIRE(NR > 0);
  CHECK(recall >= 0.97);
  CHECK(prec >= 0.97);
  // Per-class gate only where the sample is large enough to be meaningful (a class with a single
  // reference object would turn one extra detection into 50% precision).
  for (auto& [c, n] : nref) {
    if (n < 20) continue;
    CHECK(double(tp[c]) / n >= 0.97);
    CHECK((ndet[c] == 0 || double(tp[c]) / ndet[c] >= 0.97));
  }
}
