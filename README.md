# adas-visual-perception

ADAS perception for dashcam video: segmentation, detection/tracking, lead-vehicle distance, lanes, FCW, LDW and red-light running.

**Status:** the Python reference pipeline (`reference/python/`), the model export and TensorRT validation tools (`tools/export/`) and the C++/TensorRT library `adasvp` (detection, tracking, segmentation and lanes; `include/adasvp/adasvp.hpp`) are available. FCW, LDW, distance and red-light logic are still Python-only.

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

## Build (C++)

Dependencies: CUDA toolkit 12.x or newer with `nvcc`, TensorRT 10, OpenCV 4 and CMake 3.24 or newer.
The tests fetch Catch2 (and nlohmann_json when it is not installed) with CMake FetchContent, so the first
configure needs network access.

    cmake --preset native-release
    cmake --build build/native-release -j
    ctest --test-dir build/native-release --output-on-failure

The preset uses `/usr/local/cuda/bin/nvcc`. For another CUDA toolkit override it on the command line:

    cmake --preset native-release -DCMAKE_CUDA_COMPILER=/path/to/nvcc

or keep it in an untracked `CMakeUserPresets.json` that inherits the preset:

    {
      "version": 3,
      "configurePresets": [{
        "name": "mine", "inherits": "native-release",
        "cacheVariables": { "CMAKE_CUDA_COMPILER": "/path/to/nvcc" }
      }]
    }

Tests that need models or videos are skipped
unless `models/` is populated and `ADASVP_VIDEOS_DIR` points to your clips. TensorRT engines are built on
first use (minutes) and cached in `${XDG_CACHE_HOME:-$HOME/.cache}/adasvp/engines`. Use `-DADASVP_BUILD_TESTS=OFF`
to skip the tests.

### What the tests cover

- Always run (no models needed): Config/CLI, the 60 traffic-light fixtures, the synthetic RLR / FCW / lead / ego
  tests and the LDW-label render test.
- Need models and videos (`models/` and `ADASVP_VIDEOS_DIR`): the pipeline and regression tests.
- Additionally need gitignored traces (skipped when missing): the trace-replay tests read
  `outputs/dumps/fast/<stem>/adas.jsonl` for `sp1_crop_30fps`, `sp1_crop_4_30fps` and `red_crop_11_30fps`, made with
  `python tools/regression/dump_reference.py VIDEO --profile fast --adas --out outputs/dumps/fast/STEM`; the
  Renderer comparison reads `outputs/render_ref/`, made with
  `python tools/regression/render_reference.py --trace DIR --frames A:B --out OUT` (see
  [tools/regression/README.md](tools/regression/README.md)).
- Last recorded full run: 70/70 executed, none skipped.
- Configuring with the examples enabled fetches CLI11 unless it is installed, so it may need network access
  (`-DADASVP_BUILD_EXAMPLES=OFF` avoids it).

## Run the comparison

`adasvp_dump` writes the same per-frame dump as the Python reference and `compare_dumps.py` gates the two against each
other; see [tools/regression/README.md](tools/regression/README.md).

## More

- Model export and validation: [tools/export/README.md](tools/export/README.md)
- Model licenses and origins: [models/README.md](models/README.md)
