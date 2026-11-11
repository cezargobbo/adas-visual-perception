// Dumps per-frame results in the format of tools/regression/dump_reference.py:
//   DIR/frames.jsonl, DIR/seg/NNNNNN.png (every --seg-every frames), DIR/meta.json
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "adasvp/adasvp.hpp"

namespace fs = std::filesystem;
using namespace adasvp;

namespace {
void usage() {
  std::cerr << "Usage: adasvp_dump <video> --profile fast|accurate --out DIR [--start S] [--duration D]\n"
               "                   [--seg-every N] [--models-dir DIR] [--no-write]\n";
}

void num(std::ostream& o, double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  o << b;
}
}  // namespace

int main(int argc, char** argv) {
  std::string video, profile, out, models_dir;
  double start = 0, duration = -1;
  int seg_every = 30;
  bool no_write = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--profile") profile = next();
    else if (a == "--out") out = next();
    else if (a == "--start") start = std::stod(next());
    else if (a == "--duration") duration = std::stod(next());
    else if (a == "--seg-every") seg_every = std::stoi(next());
    else if (a == "--models-dir") models_dir = next();
    else if (a == "--no-write") no_write = true;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (!a.empty() && a[0] != '-' && video.empty()) video = a;
    else { usage(); return 2; }
  }
  if (video.empty() || (profile != "fast" && profile != "accurate") || (out.empty() && !no_write)) {
    usage();
    return 2;
  }

  cv::VideoCapture cap(video);
  if (!cap.isOpened()) { std::cerr << "cannot open " << video << "\n"; return 1; }
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const int start_f = static_cast<int>(start * fps);
  const int end_f = duration < 0 ? total : std::min(total, start_f + static_cast<int>(duration * fps));
  if (start_f) cap.set(cv::CAP_PROP_POS_FRAMES, start_f);
  int n_frames = end_f - start_f;

  Config cfg = profile == "fast" ? Config::fast() : Config::accurate();
  if (!models_dir.empty()) cfg.models_dir = models_dir;
  else if (const char* e = std::getenv("ADASVP_MODELS_DIR")) cfg.models_dir = e;
  cfg.seg_every = seg_every;
  Pipeline pipe(cfg);

  std::ofstream jf;
  if (!no_write) {
    fs::create_directories(fs::path(out) / "seg");
    jf.open(fs::path(out) / "frames.jsonl");
  }

  int seg_written = 0, done = 0;
  double model_ms = 0;
  const auto t_start = std::chrono::steady_clock::now();
  cv::Mat frame;
  for (int i = 0; i < n_frames; ++i) {
    if (!cap.read(frame)) break;
    FrameResult r = pipe.process(frame, (start_f + i) / fps);
    ++done;
    model_ms += r.timings.total_ms;
    if (no_write) continue;

    std::ostringstream o;
    o << "{\"i\":" << i << ",\"t\":";
    num(o, r.t);
    o << ",\"dets\":[";
    for (size_t k = 0; k < r.detections.size(); ++k) {
      const Object& b = r.detections[k];
      o << (k ? "," : "") << "[";
      num(o, b.box.x1); o << ","; num(o, b.box.y1); o << ","; num(o, b.box.x2); o << ","; num(o, b.box.y2);
      o << ","; num(o, b.score); o << "," << b.cls << "]";
    }
    o << "],\"tracks\":[";
    for (size_t k = 0; k < r.objects.size(); ++k) {
      const Object& b = r.objects[k];
      o << (k ? "," : "") << "[";
      num(o, b.box.x1); o << ","; num(o, b.box.y1); o << ","; num(o, b.box.x2); o << ","; num(o, b.box.y2);
      o << "," << b.track_id << ","; num(o, b.score); o << "," << b.cls << "]";
    }
    o << "],\"lanes\":[";
    for (size_t k = 0; k < r.lanes.size(); ++k) {
      o << (k ? "," : "") << "[";
      for (size_t p = 0; p < r.lanes[k].size(); ++p) {
        o << (p ? "," : "") << "["; num(o, r.lanes[k][p].x); o << ","; num(o, r.lanes[k][p].y); o << "]";
      }
      o << "]";
    }
    o << "],\"lane_scores\":[";
    for (size_t k = 0; k < r.lane_scores.size(); ++k) { o << (k ? "," : ""); num(o, r.lane_scores[k]); }
    o << "]}\n";
    jf << o.str();
    if (i % seg_every == 0 && !r.seg_classes.empty()) {
      char name[32];
      std::snprintf(name, sizeof name, "%06d.png", i);
      cv::imwrite((fs::path(out) / "seg" / name).string(), r.seg_classes);
      ++seg_written;
    }
    if ((i + 1) % 100 == 0) std::cerr << "[" << i + 1 << "/" << n_frames << "]" << std::endl;
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::printf("[done] %d frames in %.1f s = %.2f fps end-to-end (model time %.2f fps)\n", done, wall,
              done / wall, done / (model_ms / 1000.0));

  if (!no_write) {
    std::ofstream mf(fs::path(out) / "meta.json");
    mf << "{\n  \"video\": \"" << fs::path(video).filename().string() << "\",\n  \"profile\": \"" << profile
       << "\",\n  \"width\": " << W << ",\n  \"height\": " << H << ",\n  \"fps\": " << fps
       << ",\n  \"start_s\": " << start << ",\n  \"start_frame\": " << start_f << ",\n  \"frames\": " << done
       << ",\n  \"seg_every\": " << seg_every << ",\n  \"seg_frames_written\": " << seg_written
       << ",\n  \"conf\": " << cfg.conf << ",\n  \"lanes_thr\": " << cfg.lanes_thr
       << ",\n  \"det_model\": \"" << cfg.det_onnx << "\",\n  \"det_imgsz\": " << cfg.det_imgsz
       << ",\n  \"seg_model\": \"" << cfg.seg_onnx << "\",\n  \"wall_s\": " << wall
       << ",\n  \"fps_end_to_end\": " << done / wall << "\n}\n";
  }
  return 0;
}
