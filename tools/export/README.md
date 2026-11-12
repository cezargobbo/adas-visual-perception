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

Engine cache files are keyed by ONNX hash, precision, TensorRT version, profile shapes and GPU name.
`--precision fp16|fp32` selects the TensorRT engine precision. YOLO uses a dynamic profile
min 1x3x320x320, opt 1x3x544x960 (s) / 1x3x736x1280 (x), max 1x3x1280x1280.

## Validation results (vs the PyTorch fp32 models of the Python prototype)

TensorRT 10.16.1.11 (pip wheel, GPU). YOLO rows use imgsz 1280 (the only size run
through TensorRT); the ORT rows ran on the CPU provider, because the onnxruntime-gpu wheel
is built for CUDA 13 and its CUDA libraries could not be loaded here. ORT numbers are
fp32 and pooled over the three test resolutions.

| Model | Backend | Precision | Metric | Gate | Result |
|---|---|---|---|---|---|
| YOLO26s imgsz 1280 | TensorRT | fp16 | recall/precision, pooled / worst resolution | >=0.97 / >=0.95 | 0.995 / 0.984 pooled, worst 0.959 (PASS) |
| YOLO26s imgsz 1280 | TensorRT | fp32 | same | same | 0.997 / 1.000 pooled, worst 0.990 (PASS) |
| YOLO26x imgsz 1280 | TensorRT | fp16 | same | same | 0.996 / 0.969 pooled, worst 0.941 (FAIL) |
| YOLO26x imgsz 1280 | TensorRT | fp32 | same | same | 0.998 / 0.998 pooled, worst 0.998 (PASS) |
| YOLO26s imgsz 960 (fast profile) | TensorRT | - | not run | - | no TensorRT numbers recorded |
| YOLO26s imgsz 960 | ORT (CPU) | fp32 | recall/precision, pooled | >=0.98 per resolution | 1.000 / 0.999 (PASS) |
| YOLO26s imgsz 1280 | ORT (CPU) | fp32 | same | same | 1.000 / 1.000 (PASS) |
| YOLO26x imgsz 960 | ORT (CPU) | fp32 | same | same | 1.000 / 0.999 (PASS) |
| YOLO26x imgsz 1280 | ORT (CPU) | fp32 | same | same | 0.999 / 1.000 (PASS) |
| SegFormer-B1 | TensorRT | fp16 | same-class pixels, worst resolution mean | >=99% | 99.84% (PASS) |
| SegFormer-B1 | TensorRT | fp32 | same | >=99% | 99.91% (PASS) |
| SegFormer-B5 | TensorRT | fp16 | same | >=99% | 99.93% (PASS) |
| SegFormer-B5 | TensorRT | fp32 | same | >=99% | 99.97% (PASS) |
| SegFormer-B1 | ORT (CPU) | fp32 | same-class pixels, pooled | >=99.5% | 99.98% (PASS) |
| SegFormer-B5 | ORT (CPU) | fp32 | same | >=99.5% | 99.98% (PASS) |
| PolarRCNN | TensorRT | fp16 | lane-count match / mean dx / worst dx | >=98% / <=2 px / <=5 px | 97.5% / 0.35 px / 10.4 px (FAIL) |
| PolarRCNN | TensorRT | fp32 | same | same | 100% / 0.02 px / 1.6 px (PASS) |
| PolarRCNN | ORT (CPU) | fp32 | lane-count match / mean dx | >=98% / <=2 px | 100% / 0.16 px at export time (PASS) |

The YOLO ORT gate is evaluated per resolution against the fp32 PyTorch reference; the
SegFormer ORT gate was 99.5% (TensorRT: 99%).

## Chosen precision

The detection-gate table above is the history of the export validation: yolo26x fp16 failed that per-frame gate (worst resolution
0.941 precision), so fp32 was chosen then. The accurate profile later switched to fp16 after the event-level
regression passed 17/17 strict against the Python prototype with no borderline case; fp32 remains available with the hidden
`--det-precision fp32`.

| Model | Precision |
|---|---|
| yolo26s | fp16 |
| yolo26x | fp16 (event-level parity 17/17 strict, 0 borderline; see [tools/regression/README.md](../regression/README.md)) |
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

Command lines (`--saveEngine` goes to `cache/trt-cpp/<onnx stem>.engine`):

    /usr/src/tensorrt/bin/trtexec --onnx=models/yolo26s.onnx --fp16 --saveEngine=cache/trt-cpp/yolo26s.engine \
        --minShapes=images:1x3x320x320 --optShapes=images:1x3x544x960 --maxShapes=images:1x3x1280x1280
    /usr/src/tensorrt/bin/trtexec --onnx=models/yolo26x.onnx --saveEngine=cache/trt-cpp/yolo26x.engine \
        --minShapes=images:1x3x320x320 --optShapes=images:1x3x736x1280 --maxShapes=images:1x3x1280x1280
    /usr/src/tensorrt/bin/trtexec --onnx=models/segformer-b1-cityscapes-1024x576.onnx --fp16 --saveEngine=cache/trt-cpp/segformer-b1-cityscapes-1024x576.engine
    /usr/src/tensorrt/bin/trtexec --onnx=models/segformer-b5-cityscapes-1280x720.onnx --fp16 --saveEngine=cache/trt-cpp/segformer-b5-cityscapes-1280x720.engine
    /usr/src/tensorrt/bin/trtexec --onnx=models/polarrcnn_curvelanes_dla34.onnx --saveEngine=cache/trt-cpp/polarrcnn_curvelanes_dla34.engine
