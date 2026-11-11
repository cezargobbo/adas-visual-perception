#include "adasvp/render.hpp"

#include <algorithm>
#include <set>

#include "render/draw_primitives.hpp"

namespace adasvp {
namespace {

using render::FcwView;

// Visual-only persistence of the alerts (AlertDisplay of the Python).
struct AlertDisplay {
  struct FcwSnap {
    double t = 0;
    FcwView view;
    std::optional<Box> box;
  };
  double hold = 2.0;
  std::optional<FcwSnap> fcw;
  std::optional<std::pair<double, std::string>> ldw;  // (t_last_active, side)
  std::optional<double> rlr;                          // t of the last event

  // RLR is an instantaneous event: its banner stays for max(hold, 1.5) s.
  bool update_rlr(double t, const RlrStatus& st) {
    if (st.event) rlr = t;
    if (rlr && t - *rlr >= std::max(hold, 1.5)) rlr.reset();
    return rlr.has_value();
  }

  struct Views {
    std::optional<FcwView> fcw;
    std::string ldw_side;  // "" = no LDW banner
    std::optional<Box> held_box;
  };

  Views update(double t, const FrameResult& r) {
    Views v;
    if (r.fcw.active) {
      FcwSnap s;
      s.t = t;
      s.view.active = true;
      s.view.ttc = r.fcw.ttc;
      s.view.level = r.fcw.level;
      if (r.lead) s.box = r.lead->box;
      fcw = s;
    }
    if (fcw) {
      if (r.fcw.active) {
        v.fcw = FcwView{true, false, r.fcw.ttc, r.fcw.level};
      } else if (t - fcw->t < hold) {
        v.fcw = fcw->view;
        v.fcw->held = true;
      } else {
        fcw.reset();
      }
    }
    if (!r.ldw.side.empty()) ldw = std::make_pair(t, r.ldw.side);
    if (ldw) {
      if (!r.ldw.side.empty()) v.ldw_side = r.ldw.side;
      else if (t - ldw->first < hold) v.ldw_side = ldw->second;
      else ldw.reset();
    }
    if (v.fcw && v.fcw->held && fcw) v.held_box = fcw->box;
    return v;
  }
};

}  // namespace

struct Renderer::Impl {
  RenderOptions opts;
  render::SegOverlay seg;
  AlertDisplay alerts;
  int frame_i = 0;
  explicit Impl(const RenderOptions& o) : opts(o), seg(o.seg_alpha) { alerts.hold = o.alert_display_s; }
};

Renderer::Renderer(RenderOptions opts) : impl_(std::make_unique<Impl>(opts)) {}
Renderer::~Renderer() = default;
Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

void Renderer::reset() {
  impl_->alerts = AlertDisplay{};
  impl_->alerts.hold = impl_->opts.alert_display_s;
  impl_->frame_i = 0;
}

std::map<View, cv::Mat> Renderer::render(const cv::Mat& frame, const FrameResult& r, const std::vector<View>& views) {
  CV_Assert(frame.type() == CV_8UC3);
  Impl& s = *impl_;
  const int i = s.frame_i++;

  // alerts as displayed: once per frame
  const AlertDisplay::Views av = s.alerts.update(r.t, r);
  const bool rlr = s.alerts.update_rlr(r.t, r.rlr);

  auto draw_dets_on = [&](cv::Mat& img) {
    render::draw_lanes(img, r.lanes, r.ego_lane, av.ldw_side, r.ldw.armed, r.ldw.enabled);
    render::draw_detections(img, r.objects, frame, r.lead);
    if (r.lead) {
      const bool live = av.fcw && !av.fcw->held;
      render::draw_lead(img, *r.lead, live ? &*av.fcw : nullptr, i);
    }
    if (av.held_box) render::draw_fcw_box(img, *av.held_box, av.fcw->level, i);  // held FCW highlight
    render::draw_alerts(img, av.fcw, av.ldw_side, i, rlr);
  };

  const auto wants = [&](View v) { return std::find(views.begin(), views.end(), v) != views.end(); };
  const bool have_seg = !r.seg_classes.empty();
  std::map<View, cv::Mat> out;

  if (wants(View::Detections)) {
    cv::Mat img = frame.clone();
    draw_dets_on(img);
    out[View::Detections] = std::move(img);
  }
  if (have_seg && (wants(View::Segmentation) || wants(View::Combined))) {
    std::set<int> ids;
    cv::Mat img = s.seg.apply(frame, r.seg_classes, ids);
    if (s.opts.legend) render::draw_legend(img, ids);
    if (wants(View::Combined)) {
      cv::Mat comb = img.clone();
      draw_dets_on(comb);
      out[View::Combined] = std::move(comb);
    }
    if (wants(View::Segmentation)) out[View::Segmentation] = std::move(img);
  }
  return out;
}

}  // namespace adasvp
