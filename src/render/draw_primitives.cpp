#include "render/draw_primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "adas/traffic_light.hpp"

namespace adasvp::render {
namespace {

using cv::Scalar;

const Scalar kLaneColor(70, 220, 70);           // ego lane
const Scalar kLaneOtherColor(255, 210, 130);    // other detected lanes

const char* const kCityscapes[19] = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                                     "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                                     "truck", "bus", "train", "motorcycle", "bicycle"};
// CITYSCAPES_PALETTE_RGB of the reference
const unsigned char kPaletteRgb[19][3] = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70},   {102, 102, 156}, {190, 153, 153}, {153, 153, 153}, {250, 170, 30},
    {220, 220, 0},  {107, 142, 35}, {152, 251, 152}, {70, 130, 180},  {220, 20, 60},   {255, 0, 0},     {0, 0, 142},
    {0, 0, 70},     {0, 60, 100},   {0, 80, 100},    {0, 0, 230},     {119, 11, 32}};

Scalar palette_bgr(int cid) { return Scalar(kPaletteRgb[cid][2], kPaletteRgb[cid][1], kPaletteRgb[cid][0]); }

const char* coco_name(int cls) {
  switch (cls) {
    case 0: return "person";
    case 1: return "bicycle";
    case 2: return "car";
    case 3: return "motorcycle";
    case 5: return "bus";
    case 7: return "truck";
    case 9: return "traffic light";
    default: return "object";
  }
}

Scalar box_color(int cls) {
  switch (cls) {
    case 0: return Scalar(60, 20, 220);
    case 1: return Scalar(230, 0, 230);
    case 2: return Scalar(255, 160, 0);
    case 3: return Scalar(255, 255, 0);
    case 5: return Scalar(0, 200, 255);
    case 7: return Scalar(0, 140, 255);
    case 9: return Scalar(0, 255, 0);
    default: return Scalar(255, 255, 255);
  }
}

const char* state_name(LaneState s) {
  switch (s) {
    case LaneState::Partial: return "partial";
    case LaneState::Estimated: return "estimated";
    case LaneState::Detected: return "detected";
    default: return "none";
  }
}

std::string fmt1(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.1f", v);
  return b;
}

// Python floor division on ints
int fdiv(int a, int b) { return static_cast<int>(std::floor(static_cast<double>(a) / b)); }

// numpy .round() (half to even) then astype(int32)
int rint_i(double v) { return static_cast<int>(std::nearbyint(v)); }

cv::Point trunc_pt(const cv::Point2f& p) { return {static_cast<int>(p.x), static_cast<int>(p.y)}; }

void polyline(cv::Mat& img, const std::vector<cv::Point>& pts, const Scalar& color, int thick) {
  cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, false, color, thick, cv::LINE_AA);
}

void draw_dashed(cv::Mat& img, const std::vector<cv::Point>& pts, const Scalar& color, int thick) {
  for (size_t i = 0; i + 1 < pts.size(); ++i)
    if (i % 4 < 2) cv::line(img, pts[i], pts[i + 1], color, thick, cv::LINE_AA);
}

void draw_banner(cv::Mat& img, const std::string& text, const Scalar& color, int y, bool flash, int frame_i) {
  const int W = img.cols;
  const double scale = 1.6;
  const int thick = 4;
  int bl = 0;
  const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &bl);
  const int tw = ts.width, th = ts.height;
  const int x0 = fdiv(W - tw, 2) - 24, y0 = y;
  const Scalar fill = (!flash || (frame_i / 5) % 2 == 0) ? color : Scalar(40, 40, 160);
  cv::rectangle(img, {x0, y0}, {x0 + tw + 48, y0 + th + bl + 28}, Scalar(0, 0, 0), -1);
  cv::rectangle(img, {x0 + 4, y0 + 4}, {x0 + tw + 44, y0 + th + bl + 24}, fill, -1);
  cv::putText(img, text, {x0 + 24, y0 + th + 14}, cv::FONT_HERSHEY_SIMPLEX, scale, Scalar(255, 255, 255), thick,
              cv::LINE_AA);
}

// float rounded to half precision (round to nearest even), returned as float. Values here are in [0, 512).
float to_half(float x) {
  if (x == 0.f) return 0.f;
  const int e = std::ilogb(x);
  const float q = std::ldexp(1.f, std::max(e, -14) - 10);  // spacing of the half numbers around x
  return std::nearbyint(x / q) * q;
}

}  // namespace

void put_label(cv::Mat& img, const std::string& text, int x, int y, const Scalar& color, double scale, int thick) {
  int bl = 0;
  const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &bl);
  y = std::max(y, ts.height + 4);
  cv::rectangle(img, {x, y - ts.height - bl - 4}, {x + ts.width + 6, y}, color, -1);
  cv::putText(img, text, {x + 3, y - bl - 1}, cv::FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0), thick - 1,
              cv::LINE_AA);
}

void draw_lanes(cv::Mat& img, const std::vector<Lane>& lanes, const EgoLane& ego, const std::string& ldw_side,
                bool ldw_armed) {
  const int H = img.rows;
  for (const Lane& l : lanes) {
    std::vector<cv::Point> p;
    for (const cv::Point2f& q : l.points) p.push_back(trunc_pt(q));
    polyline(img, p, kLaneOtherColor, 2);
  }
  if (!ego.available || ego.rows_y.empty()) return;

  const double fade = ego.state == LaneState::Estimated ? ego.fade : 1.0;
  const int yt = static_cast<int>(ego.rows_y.front());
  const size_t n = ego.rows_y.size();
  std::vector<cv::Point> pl(n), pr(n);
  for (size_t k = 0; k < n; ++k) {
    pl[k] = {rint_i(ego.xl[k]), rint_i(ego.rows_y[k])};
    pr[k] = {rint_i(ego.xr[k]), rint_i(ego.rows_y[k])};
  }
  // polygon of the translucent fill, in ROI coordinates
  std::vector<cv::Point> poly_roi;
  for (size_t k = 0; k < n; ++k) poly_roi.push_back({rint_i(ego.xl[k]), rint_i(ego.rows_y[k]) - yt});
  for (size_t k = n; k-- > 0;) poly_roi.push_back({rint_i(ego.xr[k]), rint_i(ego.rows_y[k]) - yt});
  if (yt >= 0 && yt < H) {
    cv::Mat roi = img.rowRange(yt, H);
    cv::Mat ov = roi.clone();
    cv::fillPoly(ov, std::vector<std::vector<cv::Point>>{poly_roi}, kLaneColor);
    const double a = 0.28 * fade;
    cv::addWeighted(ov, a, roi, 1 - a, 0, roi);
  }
  const double k = 0.35 + 0.65 * fade;
  const Scalar line_col(static_cast<int>(kLaneColor[0] * k), static_cast<int>(kLaneColor[1] * k),
                        static_cast<int>(kLaneColor[2] * k));
  struct Side {
    const std::vector<cv::Point>& pts;
    bool seen;
    const char* name;
  };
  const Side sides[2] = {{pl, ego.obs_l, "left"}, {pr, ego.obs_r, "right"}};
  for (const Side& s : sides) {
    if (ldw_side == s.name) {  // boundary being approached (LDW)
      polyline(img, s.pts, Scalar(0, 0, 0), 12);
      polyline(img, s.pts, Scalar(0, 100, 255), 8);
    } else if (ego.state == LaneState::Detected || (ego.state == LaneState::Partial && s.seen)) {
      polyline(img, s.pts, line_col, 4);
    } else {
      draw_dashed(img, s.pts, line_col, 4);
    }
  }
  put_label(img, std::string("lane: ") + state_name(ego.state), 12, H - 14, line_col, 0.6, 2);
  put_label(img, ldw_armed ? "LDW: armed" : "LDW: disarmed", 12, H - 46,
            ldw_armed ? Scalar(90, 200, 90) : Scalar(150, 150, 150), 0.5, 1);
}

void draw_detections(cv::Mat& img, const std::vector<Object>& dets, const cv::Mat& frame_raw,
                     const std::optional<LeadVehicle>& lead) {
  for (const Object& o : dets) {
    const int x1 = static_cast<int>(o.box.x1), y1 = static_cast<int>(o.box.y1);
    const int x2 = static_cast<int>(o.box.x2), y2 = static_cast<int>(o.box.y2);
    Scalar color = box_color(o.cls);
    std::string label = coco_name(o.cls);
    if (o.cls == 9) {
      const LightColor st = traffic_light_state(frame_raw, o.box);
      if (st != LightColor::Unknown) {
        label += st == LightColor::Red ? " (red)" : st == LightColor::Yellow ? " (yellow)" : " (green)";
        color = st == LightColor::Red ? Scalar(0, 0, 255) : st == LightColor::Yellow ? Scalar(0, 220, 255)
                                                                                     : Scalar(0, 255, 0);
      }
    }
    if (o.track_id >= 0 && lead && o.track_id == lead->track_id) continue;  // drawn separately
    cv::rectangle(img, {x1, y1}, {x2, y2}, color, 2);
    char conf[32];
    std::snprintf(conf, sizeof conf, "%.2f", static_cast<double>(o.score));
    put_label(img, label + " " + conf, x1, y1 - 2, color, 0.5, 2);
  }
}

void draw_fcw_box(cv::Mat& img, const Box& box, const std::string& level, int frame_i) {
  const int x1 = static_cast<int>(box.x1), y1 = static_cast<int>(box.y1);
  const int x2 = static_cast<int>(box.x2), y2 = static_cast<int>(box.y2);
  const bool crit = level == "critical";
  const Scalar hc = (crit && (frame_i / 5) % 2 == 0) ? Scalar(0, 0, 255) : Scalar(0, 140, 255);
  cv::rectangle(img, {x1 - 6, y1 - 6}, {x2 + 6, y2 + 6}, Scalar(0, 0, 0), 12);
  cv::rectangle(img, {x1 - 6, y1 - 6}, {x2 + 6, y2 + 6}, hc, 8);
}

void draw_lead(cv::Mat& img, const LeadVehicle& lead, const FcwView* fcw, int frame_i) {
  const int H = img.rows, W = img.cols;
  const int x1 = static_cast<int>(lead.box.x1), y1 = static_cast<int>(lead.box.y1);
  const int x2 = static_cast<int>(lead.box.x2), y2 = static_cast<int>(lead.box.y2);
  const Scalar color(0, 0, 255);
  cv::rectangle(img, {x1, y1}, {x2, y2}, color, 4);
  std::string txt = std::string("LEAD ") + coco_name(lead.cls) + "  " + fmt1(lead.distance_m) + " m";
  if (fcw && fcw->ttc && *fcw->ttc < 10) txt += "  TTC " + fmt1(*fcw->ttc) + " s";
  put_label(img, txt, x1, y1 - 4, color, 0.8, 2);
  if (fcw && fcw->active) draw_fcw_box(img, lead.box, fcw->level, frame_i);  // FCW: highlight the lead box
  const cv::Point p_obj(fdiv(x1 + x2, 2), y2);
  const cv::Point p_cam(fdiv(W, 2), H - 1);
  cv::line(img, p_obj, p_cam, Scalar(0, 0, 0), 7, cv::LINE_AA);
  cv::line(img, p_obj, p_cam, Scalar(0, 255, 255), 3, cv::LINE_AA);
  cv::circle(img, p_obj, 7, Scalar(0, 255, 255), -1, cv::LINE_AA);
  const cv::Point mid(fdiv(p_obj.x + p_cam.x, 2) + 12, fdiv(p_obj.y + p_cam.y, 2));
  put_label(img, fmt1(lead.distance_m) + " m", mid.x, mid.y, Scalar(0, 255, 255), 0.9, 2);
}

void draw_alerts(cv::Mat& img, const std::optional<FcwView>& fcw, const std::string& ldw_side, int frame_i,
                 bool rlr) {
  int y = 70;
  if (rlr) {
    draw_banner(img, "RED LIGHT RUNNING", Scalar(0, 0, 230), y, true, frame_i);
    y += 90;
  }
  if (fcw && fcw->active) {
    const bool crit = fcw->level == "critical";
    std::string txt = crit ? "FCW CRITICAL" : "FCW WARNING";
    if (fcw->ttc) txt += "  TTC " + fmt1(*fcw->ttc) + " s";
    draw_banner(img, txt, crit ? Scalar(0, 0, 230) : Scalar(0, 140, 255), y, crit, frame_i);
    y += 90;
  }
  if (!ldw_side.empty())
    draw_banner(img, ldw_side == "left" ? "LDW << LEFT" : "LDW RIGHT >>", Scalar(0, 110, 255), y, false, frame_i);
}

SegOverlay::SegOverlay(double alpha) {
  // out = (f.half() * (1 - alpha) + color.half() * alpha).round(): every operation is rounded to half precision
  // (the scalars enter the kernels as float), the sum is rounded to half, then rounded to an integer.
  const float wa = static_cast<float>(1 - alpha), wb = static_cast<float>(alpha);
  for (int c = 0; c < 19; ++c)
    for (int ch = 0; ch < 3; ++ch)
      for (int f = 0; f < 256; ++f) {
        const float t1 = to_half(static_cast<float>(f) * wa);
        const float t2 = to_half(static_cast<float>(kPaletteRgb[c][2 - ch]) * wb);  // BGR channel ch
        const float s = std::nearbyint(to_half(t1 + t2));
        lut_[c][ch][f] = static_cast<unsigned char>(std::min(255.f, std::max(0.f, s)));
      }
}

cv::Mat SegOverlay::apply(const cv::Mat& frame, const cv::Mat& seg, std::set<int>& present) const {
  CV_Assert(frame.type() == CV_8UC3 && seg.type() == CV_8UC1 && seg.size() == frame.size());
  cv::Mat out = frame.clone();
  for (int y = 0; y < frame.rows; ++y) {
    const unsigned char* s = seg.ptr<unsigned char>(y);
    unsigned char* o = out.ptr<unsigned char>(y);
    for (int x = 0; x < frame.cols; ++x) {
      const int c = s[x];
      if (c >= 19) continue;  // not a Cityscapes class: leave the frame pixel
      o[3 * x] = lut_[c][0][o[3 * x]];
      o[3 * x + 1] = lut_[c][1][o[3 * x + 1]];
      o[3 * x + 2] = lut_[c][2][o[3 * x + 2]];
    }
  }
  present.clear();
  for (int y = 0; y < seg.rows; y += 8) {
    const unsigned char* s = seg.ptr<unsigned char>(y);
    for (int x = 0; x < seg.cols; x += 8) present.insert(s[x]);
  }
  return out;
}

void draw_legend(cv::Mat& img, const std::set<int>& present_ids, int x0, int y0) {
  std::vector<int> ids;
  for (int i = 0; i < 19; ++i)
    if (present_ids.count(i)) ids.push_back(i);
  if (ids.empty()) return;
  const int row_h = 22, box_w = 150;
  cv::Mat overlay = img.clone();
  cv::rectangle(overlay, {x0 - 5, y0 - 5}, {x0 + box_w, y0 + row_h * static_cast<int>(ids.size()) + 2},
                Scalar(0, 0, 0), -1);
  cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
  for (size_t k = 0; k < ids.size(); ++k) {
    const int cid = ids[k];
    const int y = y0 + static_cast<int>(k) * row_h;
    cv::rectangle(img, {x0, y + 3}, {x0 + 16, y + 17}, palette_bgr(cid), -1);
    cv::putText(img, kCityscapes[cid], {x0 + 22, y + 16}, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1,
                cv::LINE_AA);
  }
}

}  // namespace adasvp::render
