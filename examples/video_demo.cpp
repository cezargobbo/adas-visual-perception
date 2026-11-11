// video_demo: processes a dashcam video with the adasvp library and writes the annotated videos + alerts CSV
// (or shows a live window). Same behaviour as main() of the Python reference (dashcam_vision.py).
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "adasvp/adasvp.hpp"
#include "adasvp/render.hpp"
#include "alerts_csv.hpp"
#include "cli.hpp"
#include "ffmpeg_writer.hpp"

namespace fs = std::filesystem;
using namespace adasvp;
using namespace adasvp::example;
using Clock = std::chrono::steady_clock;

namespace {

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char b[1024];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}

// "%g" of Python's f"{x:g}"
std::string g(double v) { return fmt("%g", v); }

// put_label of the reference: filled box with black text
void put_label(cv::Mat& img, const std::string& text, int x, int y, const cv::Scalar& color, double scale = 0.6,
               int thick = 2) {
  int bl = 0;
  const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &bl);
  y = std::max(y, ts.height + 4);
  cv::rectangle(img, cv::Point(x, y - ts.height - bl - 4), cv::Point(x + ts.width + 6, y), color, cv::FILLED);
  cv::putText(img, text, cv::Point(x + 3, y - bl - 1), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thick - 1,
              cv::LINE_AA);
}

// Per-stage timer (StageTimer of the reference): mean ms and fps per stage, in order of first use.
class StageTimer {
 public:
  void add(const std::string& k, double ms) {
    auto it = std::find_if(stages_.begin(), stages_.end(), [&](const Stage& s) { return s.name == k; });
    if (it == stages_.end()) {
      stages_.push_back({k, 0, 0});
      it = stages_.end() - 1;
    }
    it->total_ms += ms;
    ++it->n;
  }
  std::string report() const {
    std::string r;
    for (const auto& s : stages_) {
      const double ms = s.total_ms / std::max(s.n, 1);
      if (!r.empty()) r += " | ";
      r += ms > 0 ? fmt("%s %.1f ms (%.0f fps)", s.name.c_str(), ms, 1000 / ms) : s.name + " 0 ms";
    }
    return r;
  }

 private:
  struct Stage {
    std::string name;
    double total_ms;
    int n;
  };
  std::vector<Stage> stages_;
};

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// repr(round(x, nd)) of Python for the --adas-debug columns
std::string py_round(double x, int nd) {
  if (std::isnan(x)) return "nan";
  std::string s = fmt("%.*f", nd, x);
  while (s.size() > 2 && s.back() == '0' && s[s.size() - 2] != '.') s.pop_back();
  return s;
}
std::string f3(double x) { return std::isnan(x) ? "nan" : fmt("%.3f", x); }

const char* lane_state_name(LaneState s) {
  switch (s) {
    case LaneState::Detected: return "detected";
    case LaneState::Partial: return "partial";
    case LaneState::Estimated: return "estimated";
    default: return "none";
  }
}

// One line of the --adas-debug trace (columns of the reference).
std::string debug_row(int frame, int W, int H, const Options& o, const FrameResult& r) {
  const double nan = std::nan("");
  const EgoLane& e = r.ego_lane;
  int row = 0;  // ldw_mon.row (argmin |ys - 0.93 H|, first minimum) or 0 without the LDW monitor
  if (o.ldw)
    for (size_t k = 1; k < e.rows_y.size(); ++k)
      if (std::fabs(e.rows_y[k] - 0.93 * H) < std::fabs(e.rows_y[row] - 0.93 * H)) row = static_cast<int>(k);
  const bool ok = o.lanes && !e.xl.empty() && !e.xr.empty();
  const double w = ok ? e.xr[row] - e.xl[row] : nan;
  const double dl = ok ? (W / 2.0 - e.xl[row]) / w * 3.5 : nan;
  std::string s = fmt("%d,%.3f,%s,%s,%s,%s,", frame, r.t, o.lanes ? lane_state_name(e.state) : "",
                      o.ldw ? r.debug.ldw_fail.c_str() : "", std::isnan(w) ? "nan" : fmt("%.1f", w).c_str(),
                      f3(dl).c_str());
  s += (o.fcw && r.fcw.ttc ? py_round(*r.fcw.ttc, 2) : "") + ",";
  if (r.lead)
    s += std::to_string(r.lead->track_id) + "," + py_round(r.lead->distance_m, 1) + "," +
         std::to_string(std::lrint(r.lead->box.x2 - r.lead->box.x1)) + ",";
  else
    s += ",,,";
  s += (o.ldw && r.debug.ldw_last_v ? py_round(*r.debug.ldw_last_v, 3) : "");
  for (const char* k : {"y0", "wres", "resl", "resr", "sll", "slr"}) {
    auto it = r.debug.ldw_m.find(k);
    s += "," + f3(it == r.debug.ldw_m.end() ? nan : it->second);
  }
  return s + "\n";
}

View show_view(const std::string& s) { return s == "seg" ? View::Segmentation : s == "comb" ? View::Combined : View::Detections; }

int run(const Options& o) {
  std::string device_name;
  {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) device_name = p.name;
  }
  if (o.show && o.out_prefix) std::cout << "[info] --show: no video will be written (-o ignored)\n";
  if (o.seg_size) std::cout << "[warn] --seg-size is ignored: the segmentation resolution is fixed by the ONNX file\n";
  std::cout << "[info] device: cuda " << device_name << "\n";

  if (!o.show && !o.dry && !ffmpeg_available()) {  // before the (slow) model loading
    std::cerr << "[error] ffmpeg not found in PATH (needed to write the videos; use --show or --dry to run without it)\n";
    return 1;
  }
  if (const auto err = missing_model_error(make_config(o, 30.0))) {  // before opening the video / loading models
    std::cerr << *err << "\n";
    return 1;
  }
  cv::VideoCapture cap(o.input);
  if (!cap.isOpened()) {
    std::cerr << "[error] cannot open video: " << o.input << "\n";
    return 1;
  }
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  if (!(fps > 0) || W <= 0 || H <= 0) {
    std::cerr << "[error] invalid video properties: " << o.input << "\n";
    return 1;
  }
  const Segment seg = segment(o.start, o.duration, fps, total);
  const int start_f = seg.start_f, end_f = seg.end_f, n_frames = seg.n_frames;
  if (n_frames <= 0) {
    std::cerr << "[error] nothing to process: --start is at or beyond the end of the video\n";
    return 1;
  }
  if (start_f) cap.set(cv::CAP_PROP_POS_FRAMES, start_f);

  const Config cfg = make_config(o, fps);
  std::cout << "[info] profile '" << o.profile << "': " << (o.no_seg ? "" : "seg=" + cfg.seg_onnx + "  ")
            << (o.no_det ? "" : "det=" + cfg.det_onnx + " @ " + std::to_string(cfg.det_imgsz))
            << (o.lanes ? "  lanes=Polar R-CNN DLA34 (CurveLanes)" : "  lanes=off") << "\n";
  if (o.lanes)
    std::cout << "[info] lanes: drawn in the detections and combined videos (there is no lanes-only video); "
                 "lead-vehicle corridor = "
              << (o.no_lane_corridor ? "fixed trapezoid (--no-lane-corridor)" : "ego lane") << "\n";
  std::cout << "[info] alerts: FCW "
            << (o.fcw ? "on (TTC " + g(o.fcw_ttc) + " s / " + g(o.fcw_ttc_critical) + " s)" : "off") << ", LDW "
            << (o.ldw ? "on (margin " + g(o.ldw_margin) + " m, car " + g(o.ldw_car_width) + " m)" : "off") << ", RLR "
            << (o.rlr ? "on (zone " + g(o.rlr_zone.first) + "-" + g(o.rlr_zone.second) + " of the width)" : "off")
            << (o.ldw ? "  [no vehicle speed: LDW is not suppressed at low speed]" : "") << "\n";
  std::cout << fmt("[info] video %dx%d @ %.2f fps, processing frames %d..%d (%d)\n", W, H, fps, start_f, end_f, n_frames);
  std::cout.flush();

  Pipeline pipe(cfg);
  Renderer renderer({o.alert_display, o.seg_alpha, !o.no_legend});

  // outputs
  const OutputPlan plan = plan_outputs(o);
  const bool writing = !plan.videos.empty();
  if ((writing || plan.csv) && fs::path(plan.prefix).has_parent_path())
    fs::create_directories(fs::path(plan.prefix).parent_path());
  std::string encoder;
  if (writing) encoder = resolve_encoder(o.encoder);
  std::map<View, std::unique_ptr<FfmpegWriter>> writers;
  auto open_writer = [&](View v, const std::string& path) {
    WriterSpec s;
    s.path = path;
    s.width = W;
    s.height = H;
    s.fps = fps;
    s.encoder = encoder;
    if (!o.no_audio) s.audio_from = o.input;
    s.start_s = start_f / fps;
    s.duration_s = n_frames / fps;
    std::cout << "[info] writing " << path << " (" << encoder << ")\n";
    writers[v] = std::make_unique<FfmpegWriter>(s);
  };
  std::ofstream csv;
  if (plan.csv) {
    std::cout << "[info] writing " << *plan.csv << "\n";
    csv.open(*plan.csv);
    if (!csv) {
      std::cerr << "[error] cannot write " << *plan.csv << "\n";
      return 1;
    }
    csv << kAlertsCsvHeader << "\n";
  }
  std::ofstream dbg;
  if (o.adas_debug) {
    dbg.open(*o.adas_debug);
    if (!dbg) {
      std::cerr << "[error] cannot write " << *o.adas_debug << "\n";
      return 1;
    }
    dbg << "frame,t,state,fail,w,DL,ttc,lead_id,lead_dist,lead_w,vL,y0,wres,resl,resr,sll,slr\n";
  }
  if (writing) {
    for (const std::string& p : plan.videos) {
      if (p == plan.prefix + "_detections.mp4") open_writer(View::Detections, p);
      else if (p == plan.prefix + "_segmentation.mp4") open_writer(View::Segmentation, p);
      else open_writer(View::Combined, p);
    }
  }

  const std::string win = "video_demo";
  if (o.show) {
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::resizeWindow(win, o.show_width, static_cast<int>(o.show_width * static_cast<double>(H) / W));
  }
  std::vector<View> views;
  for (const auto& kv : writers) views.push_back(kv.first);
  if (o.show) views.push_back(show_view(*o.show));

  double live_fps = -1;
  auto t_prev = Clock::now();
  StageTimer timer;
  const auto t0 = Clock::now();
  cv::Mat frame;
  for (int i = 0; i < n_frames; ++i) {
    auto tm = Clock::now();
    const bool ok = cap.read(frame);
    if (!ok) break;
    timer.add("decode", ms_since(tm));

    const int frame_i = start_f + i;
    FrameResult r = pipe.process(frame, frame_i / fps);
    if (o.lanes) timer.add("lanes", r.timings.lanes_ms);
    if (!o.no_det) {
      timer.add("det", r.timings.det_ms);
      timer.add("track", r.timings.track_ms);
    }
    if (!o.no_seg && r.timings.seg_ms > 0) timer.add("seg", r.timings.seg_ms);

    for (Alert& a : r.alerts) {
      a.frame = frame_i;
      const std::string row = format_alert_row(a);
      if (csv.is_open()) csv << row << "\n";
      std::cout << fmt("[alert] t=%.2fs ", a.t) << a.type << " " << a.level_side << " " << format_alert_detail(a) << "\n";
    }
    if (dbg.is_open()) dbg << debug_row(frame_i, W, H, o, r);

    std::map<View, cv::Mat> imgs;
    if (!views.empty()) {
      tm = Clock::now();
      imgs = renderer.render(frame, r, views);
      for (auto& kv : writers) kv.second->write(imgs.at(kv.first));
      timer.add("draw+write", ms_since(tm));
    }

    const auto now = Clock::now();
    const double inst = 1.0 / std::max(std::chrono::duration<double>(now - t_prev).count(), 1e-6);
    live_fps = live_fps < 0 ? inst : 0.9 * live_fps + 0.1 * inst;
    t_prev = now;
    if (o.show) {
      tm = Clock::now();
      auto it = imgs.find(show_view(*o.show));
      if (it == imgs.end()) {
        std::cerr << "[error] the requested view is not available\n";
        return 1;
      }
      cv::Mat view;
      cv::resize(it->second, view, cv::Size(o.show_width, static_cast<int>(o.show_width * static_cast<double>(H) / W)), 0, 0,
                 cv::INTER_LINEAR);
      put_label(view, fmt("%.1f fps  [%s]", live_fps, o.profile.c_str()), view.cols - 230, 28, cv::Scalar(255, 255, 255), 0.6, 2);
      cv::imshow(win, view);
      int key = cv::waitKey(1) & 0xFF;
      if (key == ' ') key = cv::waitKey(0) & 0xFF;
      // getWindowProperty may throw when the window is already gone
      bool closed = false;
      try {
        closed = cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) < 1;
      } catch (const cv::Exception&) {
        closed = true;
      }
      timer.add("show", ms_since(tm));
      if (key == 'q' || key == 27 || closed) {
        std::cout << "[info] stopped by user\n";
        break;
      }
    }

    if ((i + 1) % 100 == 0 || i + 1 == n_frames) {
      const double el = std::chrono::duration<double>(Clock::now() - t0).count();
      const double fps_run = (i + 1) / el;
      std::cout << fmt("[%d/%d] total %.1f fps, ETA %.1f min  ||  ", i + 1, n_frames, fps_run,
                       (n_frames - i - 1) / fps_run / 60)
                << timer.report() << std::endl;
    }
  }

  cap.release();
  if (o.show) cv::destroyAllWindows();
  for (auto& kv : writers) kv.second->close();  // warns on a non-zero ffmpeg exit
  if (csv.is_open()) csv.close();
  if (dbg.is_open()) dbg.close();
  if (o.fcw || o.ldw || o.rlr) std::cout << format_alerts_summary(o.ldw, pipe.stats()) << "\n";
  std::cout << "[done]\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int code = 0;
  const auto opts = parse_args(argc, argv, std::cout, std::cerr, code);
  if (!opts) return code;
  try {
    return run(*opts);
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}
