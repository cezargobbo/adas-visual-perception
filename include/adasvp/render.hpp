#pragma once
// HUD renderer: draws a FrameResult over its frame exactly like the drawing of the Python prototype
// (dashcam_vision.py: draw_dets_on, Segmenter.overlay + draw_legend).
//
// Design decision: the drawing API is a stateful Renderer, not a stateless draw(frame, result). The alert
// hold (an FCW / LDW / RLR banner stays on screen for `alert_display_s` after the last active frame) and the
// blinking of the banners need state that lives across frames. The Pipeline stays free of drawing state.
//
// Only OpenCV and the STL may be included here.
#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "adasvp/adasvp.hpp"

namespace adasvp {

struct RenderOptions {
  double alert_display_s = 2.0;  // seconds an alert stays on screen after it was last active
  double seg_alpha = 0.45;       // weight of the Cityscapes colours in the segmentation overlay
  bool legend = true;            // class legend on the segmentation overlay (and on the combined view)
};

enum class View {
  Detections,    // lanes, boxes, lead vehicle and alert banners on the raw frame
  Segmentation,  // Cityscapes overlay + legend
  Combined       // Detections drawn on top of the segmentation overlay
};

class Renderer {
 public:
  explicit Renderer(RenderOptions opts = {});
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) noexcept;
  Renderer& operator=(Renderer&&) noexcept;

  // Forgets the alert hold and restarts the blink counter. Call it between unrelated videos.
  void reset();

  // `frame` is the BGR uint8 frame that produced `r`. Call it once per frame, in order: it advances the
  // alert hold (using r.t) and the blink counter (counts the calls) exactly once, then draws each requested
  // view on its own copy of the frame. Segmentation and Combined need r.seg_classes (CV_8UC1, frame size)
  // and are left out of the result when it is empty. `frame` is not modified.
  std::map<View, cv::Mat> render(const cv::Mat& frame, const FrameResult& r, const std::vector<View>& views);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace adasvp
