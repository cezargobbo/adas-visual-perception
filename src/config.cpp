#include "adasvp/adasvp.hpp"

namespace adasvp {

Config Config::fast() {
  Config c;
  c.det_onnx = "yolo26s.onnx";
  c.det_precision = Precision::FP16;
  c.det_imgsz = 960;
  c.seg_onnx = "segformer-b1-cityscapes-1024x576.onnx";
  c.seg_precision = Precision::FP16;
  return c;
}

Config Config::accurate() {
  Config c;
  c.det_onnx = "yolo26x.onnx";
  c.det_precision = Precision::FP32;
  c.det_imgsz = 1280;
  c.seg_onnx = "segformer-b5-cityscapes-1280x720.onnx";
  c.seg_precision = Precision::FP16;
  return c;
}

}  // namespace adasvp
