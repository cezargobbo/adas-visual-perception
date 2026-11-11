# Model export and validation

All commands run from the repository root with the project venv. Videos are found through
`ADASVP_VIDEOS_DIR` (or `--videos-dir`). ONNX files go to `models/` and are not versioned.

## Export

    .venv/bin/python tools/export/export_yolo.py --sizes s x
    .venv/bin/python tools/export/export_segformer.py --model all
    .venv/bin/python tools/export/export_polar.py

## Validate

    # backend ort = ONNX Runtime, backend trt = TensorRT (pip wheel, engines cached in cache/trt-py/)
    .venv/bin/python tools/export/validate_yolo.py --backend trt --model s --precision fp16
    .venv/bin/python tools/export/validate_segformer.py --backend trt --model b5 --precision fp16
    .venv/bin/python tools/export/validate_polar.py --backend trt --precision fp32

`--precision fp16|fp32` selects the TensorRT engine precision. YOLO uses a dynamic profile
min 1x3x320x320, opt 1x3x544x960 (s) / 1x3x736x1280 (x), max 1x3x1280x1280.

## TensorRT 10.16.1.11 results (vs PyTorch fp32 reference)

| Model | Precision | Metric | Gate | Result |
|---|---|---|---|---|
| YOLO26s | fp16 | recall/precision, imgsz 1280, pooled / worst resolution | >=0.97 / >=0.95 | 0.995 / 0.984 pooled, worst 0.959 (PASS) |
| YOLO26s | fp32 | same | same | 0.997 / 1.000 pooled, worst 0.990 (PASS) |
| YOLO26x | fp16 | same | same | 0.996 / 0.969 pooled, worst 0.941 (FAIL) |
| YOLO26x | fp32 | same | same | 0.998 / 0.998 pooled, worst 0.998 (PASS) |
| SegFormer-B1 | fp16 | same-class pixels, worst resolution mean | >=99% | 99.84% (PASS) |
| SegFormer-B1 | fp32 | same | >=99% | 99.91% (PASS) |
| SegFormer-B5 | fp16 | same | >=99% | 99.93% (PASS) |
| SegFormer-B5 | fp32 | same | >=99% | 99.97% (PASS) |
| PolarRCNN | fp16 | lane-count match / mean dx / worst dx | >=98% / <=2 px / <=5 px | 97.5% / 0.35 px / 10.4 px (FAIL) |
| PolarRCNN | fp32 | same | same | 100% / 0.02 px / 1.6 px (PASS) |

## Chosen precision

| Model | Precision |
|---|---|
| yolo26s | fp16 |
| yolo26x | fp32 |
| segformer-b1 | fp16 |
| segformer-b5 | fp16 |
| polarrcnn | fp32 |

## ONNX sha256

    823c06d6c64177d593b1fb9e9b7a5475b9325c1671a0db900d68be6e929b6d69  polarrcnn_curvelanes_dla34.onnx
    d017eeec54b7decd9144a644458383ecd7f242fcb3796f189dc2f42892ce2a45  segformer-b1-cityscapes-1024x576.onnx
    0237d3e6073dfca91c6c53b7c4d70af761fb83248ef8128483e33cee5ce792cf  segformer-b5-cityscapes-1280x720.onnx
    7fef0d6cd24b87b7f503f8f222392e30127eff4a2abe9de48c7cd3dec0dc6e6f  yolo26s.onnx
    52909fe747ae4120ca6c6bf0285e8dcda97fca0d2176d3fbcc0e0b55cbe95399  yolo26x.onnx

## System TensorRT (C++)

Engines built with the system `trtexec` (`/usr/src/tensorrt/bin/trtexec`) from the ONNX files above,
to confirm the C++ TensorRT can consume them. Engines are written to `cache/trt-cpp/` (git-ignored).

Environment: TensorRT 10.16.1.11 (apt, built for CUDA 13.2), CUDA toolkit 12.9 (`libcudart.so.12.9.79`),
NVIDIA driver 580.178.04, RTX 4070 Laptop 8 GB. `trtexec` embeds nvinfer and cudart statically (ldd lists only libnvonnxparser); no apt CUDA 13
install was needed.

| Model | Precision | trtexec | GPU compute mean / median (ms) |
|---|---|---|---|
| yolo26s | fp16 | PASSED | 1.54 / 1.54 |
| yolo26x | fp32 | PASSED | 40.08 / 40.04 |
| segformer-b1 1024x576 | fp16 | PASSED | 5.70 / 5.68 |
| segformer-b5 1280x720 | fp16 | PASSED | 43.20 / 42.96 |
| polarrcnn dla34 | fp32 | PASSED | 3.19 / 3.18 |

YOLO profiles: min `images:1x3x320x320`, opt `1x3x544x960` (s) / `1x3x736x1280` (x),
max `1x3x1280x1280`. No other process was using the GPU during the measurements.
