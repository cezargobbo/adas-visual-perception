# adas-visual-perception

ADAS perception for dashcam video: segmentation, detection/tracking, lead-vehicle distance, lanes, FCW, LDW and red-light running.

**Status:** the C++ library is in progress. This stage contains the Python reference pipeline (`reference/python/`) plus the model export (ONNX) and TensorRT validation tools (`tools/export/`).

## Prerequisites

- Linux
- NVIDIA GPU with a CUDA-capable driver
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

    uv venv .venv -p 3.12
    uv pip install -p .venv/bin/python -r requirements.txt

`requirements.txt` pins the PyTorch CUDA wheels through an extra index; keep that line.

## Models

Weights are not stored in the repository. Download them into `models/` with:

    .venv/bin/python tools/setup_models.py

Sources: `yolo26s.pt` / `yolo26x.pt` from the Ultralytics releases (downloaded by `YOLO("yolo26x.pt")`), SegFormer from Hugging Face `nvidia/segformer-b{1,5}-finetuned-cityscapes-1024-1024`, and Polar R-CNN from
<https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_curvelanes_dla34.pth>. The `.pt`/`.pth` files are verified by sha256. If you already have the files, pass `--from DIR` (see `--help`). Licenses and credits: [models/README.md](models/README.md).

## Test videos

The test videos are private and are not distributed. The validators and the regression tests need your own dashcam clips, passed with `--videos-dir` or the `ADASVP_VIDEOS_DIR` environment variable. The reference CSVs in `tests/regression/reference/` were produced from the author's clips, so they only apply to those videos.

## More

- Model export and validation: [tools/export/README.md](tools/export/README.md)
- Model licenses and origins: [models/README.md](models/README.md)
