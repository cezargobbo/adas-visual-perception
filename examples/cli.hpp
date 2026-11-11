#pragma once
// Command line of the video_demo example: same flags, defaults and help layout as the main() of the Python
// reference (dashcam_vision.py). Parsing and the output plan are free of GPU and display dependencies so they
// can be unit-tested.
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "adasvp/adasvp.hpp"

namespace adasvp::example {

struct Options {
  std::string input;
  std::optional<std::string> out_prefix;
  double start = 0.0;
  std::optional<double> duration;
  bool no_combined = false;
  std::string encoder = "auto";  // auto | nvenc | x264
  bool no_audio = false;

  std::string profile = "accurate";
  bool no_seg = false, no_det = false, no_lane = false;

  std::optional<std::string> show;  // det | seg | comb
  int show_width = 1280;

  std::optional<std::string> seg_model;  // path of an ONNX file (the Python takes a HuggingFace id)
  std::optional<std::string> seg_size;   // accepted for parity; the size is fixed by the ONNX
  int seg_every = 1;
  double seg_alpha = 0.45;
  bool no_legend = false;

  std::optional<std::string> det_model;  // path of an ONNX file (the Python takes a .pt)
  std::optional<int> det_imgsz;
  double conf = 0.30;

  double hfov = 75.0, cam_height = 1.5, horizon = 0.59;
  double corridor_bottom = 0.12, corridor_top = 0.02;

  double lanes_thr = 0.44, lanes_hold = 2.5;
  bool no_lane_corridor = false;

  bool no_fcw = false, no_ldw = false, no_rlr = false;
  double fcw_ttc = 2.7, fcw_ttc_critical = 1.5;
  double ldw_margin = 0.25, ldw_car_width = 1.8;
  std::pair<double, double> rlr_zone{0.25, 0.75};
  double alert_display = 2.0;

  // hidden flags
  std::optional<std::string> adas_debug;
  bool dry = false;
  std::optional<std::string> det_precision;  // fp16 | fp32
  std::optional<std::string> models_dir;

  // Derived by parse_args (the couplings of the reference): what actually runs.
  bool lanes = true, fcw = true, ldw = true, rlr = true;
};

// Parses argv. Returns nullopt after printing the help (exit code 0 in `exit_code`) or a usage error (exit
// code 2, message on `err`); otherwise the options, with the profile defaults not yet applied and the derived
// fields (lanes/fcw/ldw/rlr, --show fallback) filled in.
std::optional<Options> parse_args(int argc, const char* const* argv, std::ostream& out, std::ostream& err,
                                  int& exit_code);

// Library configuration for the options: profile + overrides. `fps` is the video frame rate.
Config make_config(const Options& o, double fps);

// Output prefix: -o, or outputs/<input name without extension>.
std::string output_prefix(const Options& o);

struct OutputPlan {
  std::string prefix;
  std::vector<std::string> videos;  // in writing order: detections, segmentation, combined
  std::optional<std::string> csv;
};
// Files that a run would create. With --show or --dry no video is written, and with --show no CSV either;
// the CSV also needs at least one alert monitor.
OutputPlan plan_outputs(const Options& o);

const char* help_text();

}  // namespace adasvp::example
