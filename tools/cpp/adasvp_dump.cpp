// Dumps per-frame results in the format of tools/regression/dump_reference.py:
//   DIR/frames.jsonl, DIR/seg/NNNNNN.png (every --seg-every frames), DIR/meta.json
// With --adas also DIR/adas.jsonl (same schema as dump_reference.py --adas, outputs only).
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "adasvp/adasvp.hpp"

namespace fs = std::filesystem;
using namespace adasvp;

namespace {
void usage() {
  std::cerr << "Usage: adasvp_dump <video> --profile fast|accurate --out DIR [--start S] [--duration D]\n"
               "                   [--seg-every N] [--models-dir DIR] [--no-write] [--adas]\n"
               "                   [--det-precision fp16|fp32]\n";
}

void num(std::ostream& o, double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  o << b;
}

// JSON number; non-finite values are the strings "inf", "-inf", "nan" (like dump_reference.py).
void jnum(std::ostream& o, double v) {
  if (std::isnan(v)) o << "\"nan\"";
  else if (std::isinf(v)) o << (v > 0 ? "\"inf\"" : "\"-inf\"");
  else {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", v);
    o << b;
  }
}
void jopt(std::ostream& o, const std::optional<double>& v) {
  if (v) jnum(o, *v); else o << "null";
}
void jstr(std::ostream& o, const std::string& s) {
  if (s.empty()) o << "null"; else o << "\"" << s << "\"";
}
void jvec(std::ostream& o, const std::vector<double>& v) {
  o << "[";
  for (size_t k = 0; k < v.size(); ++k) { o << (k ? "," : ""); jnum(o, v[k]); }
  o << "]";
}
void jbox(std::ostream& o, const Box& b) {
  o << "["; jnum(o, b.x1); o << ","; jnum(o, b.y1); o << ","; jnum(o, b.x2); o << ","; jnum(o, b.y2); o << "]";
}
const char* lane_state_name(LaneState s) {
  switch (s) {
    case LaneState::Detected: return "detected";
    case LaneState::Partial: return "partial";
    case LaneState::Estimated: return "estimated";
    default: return "none";
  }
}

std::string adas_line(int i, const FrameResult& r) {
  std::ostringstream o;
  o << "{\"i\":" << i << ",\"t\":"; jnum(o, r.t);
  const EgoLane& e = r.ego_lane;
  o << ",\"ego\":{\"state\":\"" << lane_state_name(e.state) << "\",\"available\":" << (e.available ? "true" : "false")
    << ",\"fade\":"; jnum(o, e.fade);
  o << ",\"xl\":"; if (e.xl.empty()) o << "null"; else jvec(o, e.xl);
  o << ",\"xr\":"; if (e.xr.empty()) o << "null"; else jvec(o, e.xr);
  o << ",\"obs_l\":" << (e.obs_l ? "true" : "false") << ",\"obs_r\":" << (e.obs_r ? "true" : "false")
    << ",\"glitch\":" << (e.glitch ? "true" : "false") << "}";
  o << ",\"lead\":";
  if (r.lead) {
    o << "[" << r.lead->track_id << "," << r.lead->cls << ","; jbox(o, r.lead->box); o << ","; jnum(o, r.lead->distance_m);
    o << "]";
  } else o << "null";
  o << ",\"fcw\":{\"ttc\":"; jopt(o, r.fcw.ttc);
  o << ",\"active\":" << (r.fcw.active ? "true" : "false") << ",\"level\":"; jstr(o, r.fcw.level);
  o << ",\"event\":"; jstr(o, r.fcw.event); o << "}";
  o << ",\"ldw\":{\"armed\":" << (r.ldw.armed ? "true" : "false") << ",\"side\":"; jstr(o, r.ldw.side);
  o << ",\"event\":"; jstr(o, r.ldw.event);
  o << ",\"d\":"; jopt(o, r.ldw.d); o << ",\"v\":"; jopt(o, r.ldw.v); o << ",\"tlc\":"; jopt(o, r.ldw.tlc);
  o << ",\"mon\":{\"fail\":"; jstr(o, r.debug.ldw_fail);
  o << ",\"last_v\":"; jopt(o, r.debug.ldw_last_v);
  o << ",\"m\":{";
  bool first = true;
  for (const auto& kv : r.debug.ldw_m) {
    o << (first ? "" : ",") << "\"" << kv.first << "\":"; jnum(o, kv.second);
    first = false;
  }
  o << "}}}";
  o << ",\"rlr\":{\"event\":";
  if (r.rlr.event) {
    const RlrEvent& ev = *r.rlr.event;
    o << "{\"id\":" << ev.id << ",\"cx\":"; jnum(o, ev.cx); o << ",\"red_for\":"; jnum(o, ev.red_for);
    o << ",\"growth\":"; jnum(o, ev.growth); o << ",\"box\":"; jbox(o, ev.box); o << "}";
  } else o << "null";
  o << "}}\n";
  return o.str();
}
}  // namespace

int main(int argc, char** argv) {
  std::string video, profile, out, models_dir, det_precision;
  double start = 0, duration = -1;
  int seg_every = 30;
  bool no_write = false, adas = false;
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
    else if (a == "--det-precision") det_precision = next();
    else if (a == "--no-write") no_write = true;
    else if (a == "--adas") adas = true;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (!a.empty() && a[0] != '-' && video.empty()) video = a;
    else { usage(); return 2; }
  }
  if (seg_every < 1) {
    std::cerr << "--seg-every must be >= 1\n";
    usage();
    return 2;
  }
  if (!det_precision.empty() && det_precision != "fp16" && det_precision != "fp32") {
    std::cerr << "--det-precision must be fp16 or fp32\n";
    usage();
    return 2;
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
  if (!det_precision.empty()) cfg.det_precision = det_precision == "fp32" ? Precision::FP32 : Precision::FP16;
  cfg.seg_every = seg_every;
  cfg.frame_rate = fps;  // sizes the ADAS windows, like the fps of the reference loop
  std::unique_ptr<Pipeline> pipe_ptr;
  try {
    pipe_ptr = std::make_unique<Pipeline>(cfg);
  } catch (const std::exception& e) {
    std::cerr << "error: cannot create the pipeline: " << e.what() << "\n";
    return 1;
  }
  Pipeline& pipe = *pipe_ptr;

  std::ofstream jf, af;
  if (!no_write) {
    fs::create_directories(fs::path(out) / "seg");
    jf.open(fs::path(out) / "frames.jsonl");
    if (adas) af.open(fs::path(out) / "adas.jsonl");
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
      const Polyline& pts = r.lanes[k].points;
      for (size_t p = 0; p < pts.size(); ++p) {
        o << (p ? "," : "") << "["; num(o, pts[p].x); o << ","; num(o, pts[p].y); o << "]";
      }
      o << "]";
    }
    o << "],\"lane_scores\":[";
    for (size_t k = 0; k < r.lanes.size(); ++k) { o << (k ? "," : ""); num(o, r.lanes[k].score); }
    o << "]}\n";
    jf << o.str();
    if (adas) af << adas_line(i, r);
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
       << ",\n  \"fps_end_to_end\": " << done / wall;
    if (adas) {
      const AdasStats& st = pipe.stats();
      mf << ",\n  \"adas\": {\"fps_used\": " << fps << ", \"fcw_events\": " << st.fcw_events
         << ", \"ldw_events\": " << st.ldw_events << ", \"rlr_events\": " << st.rlr_events
         << ", \"ldw_crossings_suppressed\": " << st.ldw_crossings_suppressed << "}";
    }
    mf << "\n}\n";
  }
  return 0;
}
