#pragma once
// Drawing helpers mirroring the Python prototype (put_label, draw_lanes, draw_detections, draw_lead,
// draw_alerts, draw_legend and the segmentation overlay). Colours are BGR.
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "adasvp/adasvp.hpp"

namespace adasvp::render {

// Alert state as displayed (AlertDisplay views of the Python).
struct FcwView {
  bool active = false;
  bool held = false;
  std::optional<double> ttc;
  std::string level;
};

void put_label(cv::Mat& img, const std::string& text, int x, int y, const cv::Scalar& color, double scale = 0.6,
               int thick = 2);
void draw_lanes(cv::Mat& img, const std::vector<Lane>& lanes, const EgoLane& ego, const std::string& ldw_side,
                bool ldw_armed, bool ldw_enabled);
void draw_detections(cv::Mat& img, const std::vector<Object>& dets, const cv::Mat& frame_raw,
                     const std::optional<LeadVehicle>& lead);
void draw_fcw_box(cv::Mat& img, const Box& box, const std::string& level, int frame_i);
// `fcw` is nullptr when the FCW is only being held (or off): same as passing None to the Python.
void draw_lead(cv::Mat& img, const LeadVehicle& lead, const FcwView* fcw, int frame_i);
void draw_alerts(cv::Mat& img, const std::optional<FcwView>& fcw, const std::string& ldw_side, int frame_i,
                 bool rlr);

// Blend of the Cityscapes colours over the frame: the half-precision arithmetic of Segmenter.overlay,
// through a lookup table. Returns the overlay and the class ids found on seg[::8, ::8].
class SegOverlay {
 public:
  explicit SegOverlay(double alpha);
  cv::Mat apply(const cv::Mat& frame, const cv::Mat& seg_classes, std::set<int>& present_ids) const;

 private:
  unsigned char lut_[19][3][256];
};
void draw_legend(cv::Mat& img, const std::set<int>& present_ids, int x0 = 10, int y0 = 10);

}  // namespace adasvp::render
