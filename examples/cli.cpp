#include "cli.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <filesystem>
#include <ostream>

namespace adasvp::example {
namespace fs = std::filesystem;

namespace {

const char* const kHelp = R"HELP(
usage: video_demo [-h] [-o PREFIX] [--start SEC] [--duration SEC]
                  [--no-combined] [--encoder {auto,nvenc,x264}] [--no-audio]
                  [--models-dir DIR]
                  [--profile {accurate,fast}] [--no-seg] [--no-det]
                  [--no-lane] [--show [{det,seg,comb}]] [--show-width PX]
                  [--seg-model ONNX] [--seg-size WxH] [--seg-every N]
                  [--seg-alpha A] [--no-legend] [--det-model ONNX]
                  [--det-imgsz PX] [--conf C] [--hfov DEG] [--cam-height M]
                  [--horizon FRAC] [--corridor-bottom FRAC]
                  [--corridor-top FRAC] [--lanes-thr C] [--lanes-hold SEC]
                  [--no-lane-corridor] [--no-fcw] [--no-ldw] [--no-rlr]
                  [--fcw-ttc SEC] [--fcw-ttc-critical SEC] [--ldw-margin M]
                  [--ldw-car-width M] [--rlr-zone L R] [--alert-display SEC]
                  input

Computer vision for dashcam video:
  * Cityscapes semantic segmentation (19 classes) with SegFormer (NVIDIA).
  * Detection + tracking of people, vehicles and traffic lights with YOLO26 (COCO),
    including the traffic-light colour (red/yellow/green).
  * Lead-vehicle selection (in the ego lane) and distance estimate, drawn as a
    line from the bottom of its bbox to the camera.
  * ADAS alerts: FCW (forward collision, from TTC), LDW (lane departure) and RLR (red-light running).

Outputs (PREFIX = -o PREFIX; default outputs/<input video name>):
  PREFIX_segmentation.mp4   segmentation only, with legend
  PREFIX_detections.mp4     detections only + lead vehicle and distance
  PREFIX_combined.mp4       both together
  PREFIX_alerts.csv         FCW / LDW / RLR events (time, frame, type, level/side, TTC, distance...);
                            written together with the videos (not with --show)

help:
  -h, --help            show this help and exit

input and output:
  input                 input video (e.g. sp1_crop.mp4)
  -o PREFIX, --out-prefix PREFIX
                        prefix of the output videos; writes
                        PREFIX_segmentation.mp4, PREFIX_detections.mp4 and
                        PREFIX_combined.mp4 (default: outputs/<input video
                        name without extension>; the folder is created if
                        missing)
  --start SEC           start processing at this time, in seconds (default: 0)
  --duration SEC        process only this many seconds (default: until the end
                        of the video)
  --no-combined         do not write the combined video (only separate
                        segmentation and detections)
  --encoder {auto,nvenc,x264}
                        output video encoder: nvenc (on the GPU, fast), x264
                        (on the CPU, slightly better per MB) or auto = nvenc
                        if available (default: auto)
  --no-audio            do not copy the audio of the source video to the
                        outputs
  --models-dir DIR      directory with the ONNX models (get them with
                        scripts/download_models.sh); a relative --seg-model /
                        --det-model name is looked up here (default: models)

what to run:
  --profile {accurate,fast}
                        model set: accurate (heavy, best quality) or fast
                        (light, real time); see 'profiles' below (default:
                        accurate)
  --no-seg              turn segmentation off (run detection only)
  --no-det              turn detection off (run segmentation only)
  --no-lane             turn lane detection (Polar R-CNN) off; the lead
                        vehicle goes back to being chosen by the fixed
                        trapezoid corridor

live view:
  --show [{det,seg,comb}]
                        open a window showing the result while processing: det
                        (detections, default if only --show is given), seg
                        (segmentation) or comb (both). With --show NO video is
                        written (-o, --encoder etc. are ignored)
  --show-width PX       width of the --show window, in pixels (default: 1280)

segmentation (SegFormer, Cityscapes):
  --seg-model ONNX      path of the segmentation ONNX file; a bare name is
                        looked up in the models directory (profile default:
                        segformer-b5-cityscapes-1280x720.onnx or
                        segformer-b1-cityscapes-1024x576.onnx)
  --seg-size WxH        accepted for compatibility; the network resolution is
                        fixed by the ONNX file (profile default: 1280x720 or
                        1024x576)
  --seg-every N         run segmentation every N frames and repeat the last
                        mask in between (default: 1 = every frame)
  --seg-alpha A         opacity of the segmentation colours over the video,
                        from 0 to 1 (default: 0.45)
  --no-legend           do not draw the segmentation class legend

detection (YOLO26, COCO):
  --det-model ONNX      path of the YOLO ONNX file; a bare name is looked up in
                        the models directory (profile default: yolo26x.onnx or
                        yolo26s.onnx)
  --det-imgsz PX        YOLO input image size; larger catches small/distant
                        objects but is slower (profile default: 1280 or 960)
  --conf C              minimum confidence to accept a detection, from 0 to 1
                        (default: 0.30)

lead vehicle and distance:
  --hfov DEG            horizontal field of view of the camera, in degrees;
                        defines the focal length (default: 75)
  --cam-height M        camera height above the ground, in metres (default:
                        1.5)
  --horizon FRAC        height of the horizon line in the image, as a fraction
                        of the height (0 = top, 1 = bottom) (default: 0.59)
  --corridor-bottom FRAC
                        half-width of the lane corridor at the bottom of the
                        image, as a fraction of the width; a vehicle is only a
                        'lead' candidate if the bottom-centre of its bbox is
                        inside this corridor. With lanes on (default) this
                        trapezoid corridor is only used when there is no ego
                        lane, with --no-lane or with --no-lane-corridor
                        (default: 0.12)
  --corridor-top FRAC   half-width of the lane corridor at the horizon line,
                        as a fraction of the width (default: 0.02)

lanes (Polar R-CNN, CurveLanes; on by default, --no-lane turns off):
  --lanes-thr C         minimum confidence of a detected lane, from 0 to 1
                        (default: 0.44)
  --lanes-hold SEC      for how many seconds the ego lane keeps being
                        extrapolated (held, fading) after it stops being
                        detected, e.g. at an intersection (default: 2.5)
  --no-lane-corridor    draw the lanes but keep the fixed trapezoid corridor
                        (--corridor-*) to choose the lead vehicle

ADAS alerts (FCW / LDW / RLR; on by default):
  --no-fcw              turn the forward collision warning (FCW) off
  --no-ldw              turn the lane departure warning (LDW) off; it is also
                        off with --no-lane / --no-det
  --no-rlr              turn the red-light running (RLR) alert off; it is also
                        off with --no-det
  --fcw-ttc SEC         FCW 'warning' level: TTC (time to collision) below
                        this value (default: 2.7)
  --fcw-ttc-critical SEC
                        FCW 'critical' level (flashing red banner): TTC below
                        this value (default: 1.5)
  --ldw-margin M        LDW fires when the wheel is within M metres of the
                        line (plus a small allowance that grows with the
                        approach speed) and the car is approaching it; larger
                        = more sensitive, more risk of false alarms (default:
                        0.25)
  --ldw-car-width M     car width, in metres (the wheel is half of that from
                        the centre) (default: 1.8)
  --rlr-zone L R        RLR: a red light only counts if it leaves the image
                        through the top edge between these horizontal
                        positions, as fractions of the width (default: 0.25
                        0.75)
  --alert-display SEC   minimum time the FCW / LDW / RLR banners, the
                        highlighted lane boundary and the lead-box highlight
                        stay on screen after the alert was last active; visual
                        only, events and CSV are unchanged (default: 2.0; 0 =
                        only while active)

profiles (--profile):
  accurate  SegFormer-B5 @1280x720 + YOLO26x @1280  -> best quality
            (inference only, RTX 4070 Laptop: seg ~13 fps, det ~28 fps)
  fast      SegFormer-B1 @1024x576 + YOLO26s @960   -> real time
            (inference only, RTX 4070 Laptop: seg ~70 fps, det ~80 fps)
  --seg-model, --det-model and --det-imgsz override the profile.

distance estimation (approximate; the camera is not calibrated):
  width cue  : Z = f * real_width / bbox_width_px
  ground cue : Z = f * camera_height / (bbox_bottom_y - horizon_y)
  Both are fused and smoothed over time. f comes from --hfov.
  If you know the real camera setup, tune --hfov, --cam-height and --horizon.

--show mode:
  With --show nothing is written: live window only. Without --show the videos are written
  (and the measured fps is closest to the real one, without the imshow cost).
  Keys: q / ESC quits, SPACE pauses / resumes.

examples:
  # whole video, heavy models, writes the 3 videos (no -o: outputs/sp1_crop_*.mp4)
  video_demo sp1_crop.mp4
  # light models (real time)
  video_demo sp1_crop.mp4 -o outputs/sp1_fast --profile fast
  # detection only, segmentation only
  video_demo sp1_crop.mp4 -o outputs/sp1_det --profile fast --no-seg
  video_demo sp1_crop.mp4 -o outputs/sp1_seg --profile fast --no-det
  # live window, nothing written
  video_demo sp1_crop.mp4 --profile fast --show
  # quick test: 10 s starting at 2:00
  video_demo sp1_crop.mp4 -o outputs/test --start 120 --duration 10
  # no lane detection (back to the fixed trapezoid corridor)
  video_demo sp1_crop.mp4 -o outputs/sp1_nolanes --profile fast --no-lane
  # lanes only drawn, old trapezoid corridor; hold the lane for 4 s without detection
  video_demo sp1_crop.mp4 -o outputs/sp1_lanes --profile fast --no-lane-corridor --lanes-hold 4
  # no ADAS alerts; FCW only; more sensitive LDW; earlier FCW; alerts shown for 3 s
  video_demo sp1_crop.mp4 -o outputs/sp1_noalerts --profile fast --no-fcw --no-ldw
  video_demo sp1_crop.mp4 -o outputs/sp1_fcw --profile fast --no-ldw
  video_demo sp1_crop.mp4 -o outputs/sp1_ldw --profile fast --ldw-margin 0.4 --fcw-ttc 3.5 --alert-display 3
  # no red-light-running alert; wider central exit zone for RLR
  video_demo sp1_crop.mp4 -o outputs/sp1_norlr --profile fast --no-rlr
  video_demo red_light.mp4 -o outputs/red_light --rlr-zone 0.2 0.8

lanes (on by default; --no-lane turns them off):
  Polar R-CNN finds the lanes; the tracker picks the left/right boundary of the ego lane
  (plausible width ~600-1300 px at the bottom row, = 2.7-4.0 m with the camera at 1.5 m),
  smooths it over time and rejects jumps. State shown on the video: detected / partial (one
  boundary inferred from the previous width) / estimated (nothing seen: the last lane is
  held, dashed and fading, for --lanes-hold s; assumes driving straight) / none (falls back
  to the trapezoid corridor).

ADAS alerts (on by default; --no-fcw / --no-ldw / --no-rlr turn them off):
  FCW (forward collision): TTC = 1 / (d ln(bbox_width) / dt), robust fit (Theil-Sen) over 0.8 s
    on the lead vehicle; needs no calibration. Only valid with the same vehicle for >= 0.5 s,
    bbox >= 40 px wide, distance < 40 m, bbox fully inside the image, inside the ego lane (no
    margin), bbox height growing consistently with the width, implied closing speed <= 14 m/s
    and the smoothed distance also shrinking (>= 0.5 m/s). 'warning' when TTC < --fcw-ttc,
    'critical' when TTC < --fcw-ttc-critical; needs 0.3 s in a row to trigger and 0.5 s
    without the condition to release. Requires detection.
  LDW (lane departure): the priority is NO false alarms (missing a borderline case is fine).
    Lateral position = image centre (camera assumed centred on the car) relative to the two
    lane boundaries, lane width assumed 3.5 m; wheel-to-line distance = boundary distance -
    half of --ldw-car-width. Fires when the wheel is within --ldw-margin (plus a speed-dependent
    allowance) of the line AND the car is approaching it (>= 0.15 m/s, time to cross < 1 s, net
    drift >= 0.25 m over the last 1 s) for 0.3 s; then 2.5 s without repeating on that side.
    LDW is only ARMED if, for >= 1 s in a row, BOTH boundaries were actually observed (state
    'detected', never partial/estimated), with no tracker jumps or re-acquisitions, with a
    plausible and stable width, straight boundaries converging at the horizon and consistent
    with the frame observation. Any failure disarms at once. When the line is actually crossed
    (the boundaries swap) LDW is suppressed for 2 s more. Requires lanes: it is turned off with
    --no-lane / --no-det.
    Limitation: without the car speed (no CAN/GPS) LDW cannot be suppressed at low speed
    (parking manoeuvres, stopped traffic) nor know whether the turn signal is on.
  RLR (red-light running): without the car speed, 'driving past the light' is read from the image:
    a traffic light ahead grows and moves up until it leaves the image through the TOP edge. A light
    over our own path leaves through the central part of the top edge; lights over other lanes or
    cross streets slide out through the sides, so only lights leaving inside --rlr-zone (default:
    the central half of the width) count. Fires when such a light, tracked for >= 1 s and visibly
    approaching (growing >= 15%/s, moving up), was red for >= 50% of its last 2 s, still red when
    it left and red for >= 1.5 s (a light that only just turned red = the car entered on amber ->
    no alert). The lamp colour comes from the position of the lit lamp in the housing and its colour
    relative to the other lamps. One alert per intersection (4 s). Stopping at a red light gives no
    alert (the light does not grow). Requires detection.
    Limitations: lights on a pole at the side of the road (common in Brazil) leave through the side
    and are not checked; at night YOLO often sees only the lamp glow, not the housing, so lights
    are detected much less (night RLR is mostly missed).
  Events are written to PREFIX_alerts.csv.
  --alert-display only affects how long banners / highlights stay on screen, not the events.
)HELP";

// Same behaviour as argparse's `ap.error`: usage line, message, exit code 2.
void usage_error(std::ostream& err, const std::string& msg) {
  err << "usage: video_demo [-h] [-o PREFIX] ... input\nvideo_demo: error: " << msg << "\n";
}

// A model given by the user: an existing file is used as is (made absolute so that the models directory is not
// prepended); anything else is a name looked up in the models directory.
std::string model_arg(const std::string& p) {
  std::error_code ec;
  return fs::exists(p, ec) ? fs::absolute(p).string() : p;
}

}  // namespace

const char* help_text() { return kHelp + 1; }  // skips the newline that opens the raw string

std::optional<Options> parse_args(int argc, const char* const* argv, std::ostream& out, std::ostream& err,
                                  int& exit_code) {
  Options o;
  bool help = false, old_lanes = false;
  std::vector<double> rlr_zone;

  CLI::App app{"", "video_demo"};
  // argparse parity: a repeated scalar option keeps the last value
  app.option_defaults()->multi_option_policy(CLI::MultiOptionPolicy::TakeLast);
  app.set_help_flag();  // replaced by the Python-style help below
  app.add_flag("-h,--help", help);
  app.add_option("input", o.input)->required();
  app.add_option("-o,--out-prefix", o.out_prefix);
  app.add_option("--start", o.start);
  app.add_option("--duration", o.duration);
  app.add_flag("--no-combined", o.no_combined);
  app.add_option("--encoder", o.encoder)->check(CLI::IsMember({"auto", "nvenc", "x264"}));
  app.add_flag("--no-audio", o.no_audio);

  app.add_option("--profile", o.profile)->check(CLI::IsMember({"accurate", "fast"}));
  app.add_flag("--no-seg", o.no_seg);
  app.add_flag("--no-det", o.no_det);
  app.add_flag("--no-lane", o.no_lane);

  std::string show;
  CLI::Option* show_opt = app.add_option("--show", show)->expected(0, 1)->check(CLI::IsMember({"det", "seg", "comb"}));
  app.add_option("--show-width", o.show_width);

  app.add_option("--seg-model", o.seg_model);
  app.add_option("--seg-size", o.seg_size);
  app.add_option("--seg-every", o.seg_every);
  app.add_option("--seg-alpha", o.seg_alpha);
  app.add_flag("--no-legend", o.no_legend);

  app.add_option("--det-model", o.det_model);
  app.add_option("--det-imgsz", o.det_imgsz);
  app.add_option("--conf", o.conf);

  app.add_option("--hfov", o.hfov);
  app.add_option("--cam-height", o.cam_height);
  app.add_option("--horizon", o.horizon);
  app.add_option("--corridor-bottom", o.corridor_bottom);
  app.add_option("--corridor-top", o.corridor_top);

  app.add_flag("--lanes", old_lanes);  // old opt-in of the Python prototype, now the default (hidden, no-op)
  app.add_option("--lanes-thr", o.lanes_thr);
  app.add_option("--lanes-hold", o.lanes_hold);
  app.add_flag("--no-lane-corridor", o.no_lane_corridor);

  app.add_flag("--no-fcw", o.no_fcw);
  app.add_flag("--no-ldw", o.no_ldw);
  app.add_flag("--no-rlr", o.no_rlr);
  app.add_option("--fcw-ttc", o.fcw_ttc);
  app.add_option("--fcw-ttc-critical", o.fcw_ttc_critical);
  app.add_option("--ldw-margin", o.ldw_margin);
  app.add_option("--ldw-car-width", o.ldw_car_width);
  app.add_option("--rlr-zone", rlr_zone)->expected(2);
  app.add_option("--alert-display", o.alert_display);

  app.add_option("--adas-debug", o.adas_debug);
  app.add_flag("--dry", o.dry);
  app.add_option("--det-precision", o.det_precision)->check(CLI::IsMember({"fp16", "fp32"}));
  app.add_option("--models-dir", o.models_dir);

  // CLI11 takes an option name appearing as a value after a positional as an error too, like argparse.
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    // --help is handled below: with the flag set, a missing `input` is not an error
    if (!help) {
      usage_error(err, e.what());
      exit_code = 2;
      return std::nullopt;
    }
  }
  if (help) {
    out << help_text();
    exit_code = 0;
    return std::nullopt;
  }

  auto fail = [&](const std::string& msg) -> std::optional<Options> {
    usage_error(err, msg);
    exit_code = 2;
    return std::nullopt;
  };
  if (show_opt->count() > 0) o.show = show.empty() ? "det" : show;
  if (o.no_seg && o.no_det) return fail("--no-seg and --no-det together leave nothing to do");
  if (!rlr_zone.empty()) o.rlr_zone = {rlr_zone[0], rlr_zone[1]};
  if (!(0 <= o.rlr_zone.first && o.rlr_zone.first < o.rlr_zone.second && o.rlr_zone.second <= 1))
    return fail("--rlr-zone needs 0 <= L < R <= 1");
  if (o.seg_every < 1) return fail("--seg-every must be >= 1");

  // lanes are drawn only in detections / combined, so --no-det also turns them off
  o.lanes = !(o.no_lane || o.no_det);
  o.fcw = !(o.no_fcw || o.no_det);
  o.ldw = !(o.no_ldw || !o.lanes);
  o.rlr = !(o.no_rlr || o.no_det);
  if (o.show && ((*o.show == "seg" && o.no_seg) || (*o.show == "det" && o.no_det) ||
                 (*o.show == "comb" && (o.no_seg || o.no_det))))
    o.show = o.no_det ? "seg" : "det";  // fall back to what is actually computed
  return o;
}

Segment segment(double start_s, std::optional<double> duration_s, double fps, int total) {
  Segment s;
  s.start_f = static_cast<int>(start_s * fps);  // int() truncates toward zero, like the Python
  s.end_f = duration_s ? std::min(total, s.start_f + static_cast<int>(*duration_s * fps)) : total;
  s.n_frames = s.end_f - s.start_f;
  return s;
}

Config make_config(const Options& o, double fps) {
  Config c = o.profile == "fast" ? Config::fast() : Config::accurate();
  // Default models location: Config::models_dir ("models", relative to the working directory).
  // Nothing is read from environment variables; use --models-dir to point elsewhere.
  if (o.models_dir) c.models_dir = *o.models_dir;
  if (o.det_model) c.det_onnx = model_arg(*o.det_model);
  if (o.seg_model) c.seg_onnx = model_arg(*o.seg_model);
  if (o.det_imgsz) c.det_imgsz = *o.det_imgsz;
  if (o.det_precision) c.det_precision = *o.det_precision == "fp32" ? Precision::FP32 : Precision::FP16;
  c.conf = static_cast<float>(o.conf);
  c.seg_every = o.seg_every;
  c.lanes_thr = static_cast<float>(o.lanes_thr);
  c.enable_det = !o.no_det;
  c.enable_seg = !o.no_seg;
  c.enable_lanes = o.lanes;
  c.frame_rate = fps;
  c.hfov = o.hfov;
  c.cam_height = o.cam_height;
  c.horizon = o.horizon;
  c.corridor_bottom = o.corridor_bottom;
  c.corridor_top = o.corridor_top;
  c.lanes_hold = o.lanes_hold;
  c.lane_corridor = !o.no_lane_corridor;
  c.enable_fcw = o.fcw;
  c.enable_ldw = o.ldw;
  c.enable_rlr = o.rlr;
  c.fcw_ttc = o.fcw_ttc;
  c.fcw_ttc_critical = o.fcw_ttc_critical;
  c.ldw_margin = o.ldw_margin;
  c.ldw_car_width = o.ldw_car_width;
  c.rlr_zone = o.rlr_zone;
  c.alert_display = o.alert_display;
  c.seg_alpha = o.seg_alpha;
  c.show_width = o.show_width;
  return c;
}

std::optional<std::string> missing_model_error(const Config& c) {
  struct Need {
    bool on;
    const std::string& name;
  };
  for (const Need& n : {Need{c.enable_det, c.det_onnx}, Need{c.enable_seg, c.seg_onnx}, Need{c.enable_lanes, c.lane_onnx}}) {
    if (!n.on) continue;
    const fs::path p(n.name);
    const fs::path full = p.is_absolute() ? p : fs::path(c.models_dir) / p;
    std::error_code ec;
    if (!fs::exists(full, ec))
      return "[error] model not found: " + full.string() +
             " \xE2\x80\x94 run scripts/download_models.sh or pass --models-dir DIR";
  }
  return std::nullopt;
}

std::string output_prefix(const Options& o) {
  if (o.out_prefix && !o.out_prefix->empty()) return *o.out_prefix;
  return (fs::path("outputs") / fs::path(o.input).stem()).string();
}

OutputPlan plan_outputs(const Options& o) {
  OutputPlan p;
  p.prefix = output_prefix(o);
  if (o.show) return p;  // live window only: nothing is written
  if (!o.dry) {
    if (!o.no_det) p.videos.push_back(p.prefix + "_detections.mp4");
    if (!o.no_seg) p.videos.push_back(p.prefix + "_segmentation.mp4");
    if (!o.no_det && !o.no_seg && !o.no_combined) p.videos.push_back(p.prefix + "_combined.mp4");
  }
  // Deliberate deviation from the Python tuning flag: there --dry also skips the CSV; here it writes the CSV
  // (and only the CSV), which the regression tooling relies on.
  if (o.fcw || o.ldw || o.rlr) p.csv = p.prefix + "_alerts.csv";
  return p;
}

bool show_window_closed(double visible, double aspect_ratio) {
  if (visible >= 0) return visible < 1;
  return !(aspect_ratio > 0);  // also true for NaN
}

}  // namespace adasvp::example
