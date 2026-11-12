# Installing the dependencies

`video_demo` needs an NVIDIA GPU, a driver, CUDA, TensorRT 10, OpenCV 4 and ffmpeg. There are two ways to get
them: a Docker image that carries everything except the driver (section 1), or a native build (section 2).
Models and videos are downloaded separately (section 4) in both cases.

Everything here was written on one machine: Ubuntu 24.04.5, an RTX 4070 Laptop GPU (8 GB), NVIDIA driver
580.178.04, Docker 29.4.1 with the NVIDIA Container Toolkit. Steps that were run on it are marked
**tested**. Anything else is a description, not a promise. In section 2 the package names and versions are
the ones installed on that machine; the apt commands themselves were not re-run on a clean system.

## 1. Docker

The image is built on `nvcr.io/nvidia/tensorrt:25.11-py3`, pinned by digest in the `Dockerfile`:

| Component | Version in the image |
|---|---|
| Base | NVIDIA TensorRT container 25.11 (Ubuntu 24.04) |
| TensorRT | 10.14.1.48 |
| CUDA toolkit | 13.0 (nvcc 13.0.88) |
| OpenCV | 4.6.0 (Ubuntu `libopencv-dev`) |
| ffmpeg | 6.1.1 (Ubuntu `ffmpeg`) |
| CMake | 3.28.3 |

25.11 is the newest TensorRT 10.x tag whose CUDA (13.0) runs on a 580-series driver. The next tag, 25.12,
moves to CUDA 13.1. The image is about 11 GB, and a build takes a few minutes on a fast connection once the
base image is pulled. It holds the `video_demo` binary only (in `/opt/adasvp/bin`). Models and videos are
mounted at run time.

You need the NVIDIA driver on the host and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    docker run --rm --gpus all nvcr.io/nvidia/tensorrt:25.11-py3 nvidia-smi

Build the image and check it (**tested**):

    docker build -t adas-visual-perception .
    docker run --rm --gpus all adas-visual-perception video_demo --help

The build compiles for GPU architectures 7.5, 8.0, 8.6, 8.9 and 9.0 (plus PTX for newer ones), because no GPU
is visible during `docker build`. To build for one GPU only, for example an Ada card:
`docker build --build-arg CUDA_ARCHS=89 -t adas-visual-perception .`

Run it with the wrapper, which mounts the models, the input video and the output folder for you and passes
`--models-dir` (**tested**):

    scripts/download_models.sh --profile fast
    scripts/download_videos.sh
    scripts/video_demo_docker.sh videos/redlights/red_crop_11_30fps.mp4 -o outputs/docker/red_crop_11_30fps --profile fast --no-seg --dry

(`download_videos.sh` is described in section 4. Without `-o` the wrapper writes to
`outputs/<video name>` in the current directory, like the native `video_demo`.)

- Write options as `--option value` (not `--option=value`), so the wrapper can translate paths. Put the
  video before `--` if its name starts with a dash (`... -- -clip.mp4`).
- `--show` is not supported through Docker (the container has no display): run `video_demo` natively for
  the live view. The wrapper rejects it with exit 2.
- Output paths (`-o` and `--adas-debug`) are translated and their folders are created on the host.
  `--det-model` and `--seg-model` are not translated: through the wrapper they accept only file names that
  are in `models/` (for example `--det-model yolo26x.onnx`), not host paths.
- The repository is mounted read-only at `/work/repo`, `models/` at `/work/models`, the input video's folder
  read-only, and the folder of the `-o` prefix read-write (it is created on the host first).
- TensorRT engines are cached in `${XDG_CACHE_HOME:-$HOME/.cache}/adasvp` on the host, mounted at
  `/cache/adasvp`. The first run builds them (minutes); later runs reuse them. Engine names carry the
  TensorRT version, so engines built in the container and natively do not collide.
- Environment: `ADASVP_DOCKER_IMAGE` (default `adas-visual-perception`) and `ADASVP_DOCKER_GPU_FLAGS`
  (default `--gpus all`).
- If Docker cannot start the container with a GPU, the wrapper prints the toolkit hint and exits with 2.

## 2. Native build on Ubuntu 24.04

This is the author's machine: Ubuntu 24.04.5, driver 580.178.04 (`nvidia-driver-580`), CUDA toolkit 12.9.1,
TensorRT 10.16.1.11, OpenCV 4.6.0, CMake 3.28.3. It uses the CUDA toolkit and
TensorRT packages from NVIDIA's apt repository for Ubuntu 24.04.

1. **Driver.** Install the driver from Ubuntu and reboot, then check it:

       sudo apt install nvidia-driver-580
       nvidia-smi

2. **NVIDIA apt repository.** Add the CUDA repository with the keyring package:

       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
       sudo dpkg -i cuda-keyring_1.1-1_all.deb
       sudo apt update

3. **CUDA toolkit.** The compiler is `nvcc`. On the tested machine it is CUDA 12.9:

       sudo apt install cuda-toolkit-12-9

4. **TensorRT 10.** From the same repository. On the tested machine the packages are 10.16.1.11, built for
   CUDA 13.2:

       sudo apt install libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev

   The apt repository offers the newest TensorRT 10 build. On the tested machine the CUDA toolkit is 12.9
   while these TensorRT packages are built for CUDA 13.2, and the build and the runs worked. That
   combination may not work elsewhere. If it does not, install a TensorRT build and a toolkit for the same
   CUDA version (the `+cudaXX.Y` suffix in `apt policy libnvinfer-dev` shows which).

5. **The rest.** OpenCV, CMake (3.24 or newer; Ubuntu 24.04 ships 3.28), a compiler and ffmpeg:

       sudo apt install build-essential cmake git ffmpeg libopencv-dev

   CLI11 is fetched by CMake if it is not installed; `sudo apt install libcli11-dev` avoids the download.

6. **Build** (**tested**: configure, build and `ctest` with the packages above; 74/74 tests passed, none skipped, with models and videos present and `ADASVP_VIDEOS_DIR` set):

       cmake --preset native-release
       cmake --build build/native-release -j

   The preset compiles with `/usr/local/cuda/bin/nvcc`. If `nvcc` is somewhere else, or `/usr/local/cuda`
   points at another toolkit, pass it explicitly, for example
   `cmake --preset native-release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`.
   Do not rely on the `nvcc` from the Ubuntu `nvidia-cuda-toolkit` package (CUDA 12.0 on 24.04): put the
   NVIDIA toolkit first with the override above.

   `-DADASVP_BUILD_TESTS=OFF` skips the tests and Catch2; `-DADASVP_BUILD_EXAMPLES=OFF` skips `video_demo`.

ffmpeg: Ubuntu 24.04 ships 6.1 and the Docker image uses it. The author's own machine has a locally built
ffmpeg 3.4 in `/usr/local/bin`, which has NVENC but not the `p1`-`p7` presets, so the app asks for
`-preset medium`, which every version accepts.

## 3. Python tools (optional)

Only needed to regenerate the ONNX models and the regression reference. `video_demo` itself does not use
Python. Python 3.12 and [uv](https://docs.astral.sh/uv/):

    uv venv .venv -p 3.12
    uv pip install -p .venv/bin/python -r requirements.txt

`requirements.txt` pins PyTorch CUDA 12.8 wheels (through the extra index it names), TensorRT 10.16.1.11 for
Python, ONNX Runtime and the other packages the tools use. Then:

    .venv/bin/python tools/setup_models.py     # download the PyTorch weights into models/
    # export to ONNX: see tools/export/README.md
    .venv/bin/python tools/export/export_yolo.py --sizes s x
    .venv/bin/python tools/export/export_segformer.py --model all
    .venv/bin/python tools/export/export_polar.py

The comparison scripts under `tools/regression/` use the same environment (see `regression.md` next to this
file).

## 4. Get models and videos

The release carries the five ONNX models and seven test videos, with a `SHA256SUMS` file. Each script
downloads to a `.part` file, checks the sha256 and only then renames, so an interrupted download is
repeated on the next run and a corrupt file never gets its final name.

    scripts/download_models.sh [--profile fast|accurate|all] [--dest DIR]
    scripts/download_videos.sh [--dest DIR]

| What | Size |
|---|---|
| `--profile fast` (YOLO26s, SegFormer-B1, Polar R-CNN) | about 158 MB |
| `--profile accurate` (YOLO26x, SegFormer-B5, Polar R-CNN) | about 628 MB |
| `--profile all` (the default) | about 722 MB |
| the seven videos | about 1 GB |

By default the scripts write to `models/` and `videos/` of the repository, wherever you run them from (`--dest`
changes that). Models are not versioned. Videos go to `videos/`, with the `red_crop_*` clips in
`videos/redlights/`. The `red_crop_*` clips come from
[Quinn Dickinson's YouTube channel](https://www.youtube.com/@QuinnDickinson2610). Models carry their own
licenses: see `models/README.md`. The scripts only need `curl` and `sha256sum`.

To use other models or videos, run `tools/setup_models.py` and `tools/export/` (section 3) or put your own
`.mp4` anywhere: `video_demo` takes any video file.

## 5. Check what you got

Native build:

    ctest --test-dir build/native-release --output-on-failure
    build/native-release/examples/video_demo videos/redlights/red_crop_11_30fps.mp4 \
        -o outputs/check/red_crop_11_30fps --profile fast --no-seg --dry

The tests that need models or videos are skipped when those are not there, so run section 4 first for the
full suite. `--dry` skips writing the videos and still writes `outputs/check/red_crop_11_30fps_alerts.csv`, so
a short run shows a red-light event for this clip. Docker: the wrapper line from section 1 does the same.

The first run builds the TensorRT engines for your GPU, which takes minutes (the `yolo26x` fp16 engine
about ten) and is cached afterwards.

## 6. When it does not work

| Symptom | Fix |
|---|---|
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`, or the wrapper prints the toolkit hint and exits with 2 | The NVIDIA Container Toolkit is not installed or Docker was not restarted. Install it, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. Check with `docker run --rm --gpus all nvcr.io/nvidia/tensorrt:25.11-py3 nvidia-smi`. |
| CMake picks the wrong `nvcc` (CUDA 12.0 from the Ubuntu package, or "No CMAKE_CUDA_COMPILER could be found") | The preset uses `/usr/local/cuda/bin/nvcc`. Point it at your toolkit: `cmake --preset native-release -DCMAKE_CUDA_COMPILER=/path/to/nvcc`. Delete `build/native-release` when you change the compiler. |
| `cached engine unusable, rebuilding: ...` | The cached engine does not match this TensorRT, GPU or driver. This is normal after an upgrade; the engine is rebuilt and the cache is overwritten. If it repeats on every run, check that `${XDG_CACHE_HOME:-$HOME/.cache}/adasvp/engines` is writable. |
| The first run takes many minutes | TensorRT is building the engines for your GPU (the `yolo26x` fp16 one takes about ten minutes). They are cached in `${XDG_CACHE_HOME:-$HOME/.cache}/adasvp/engines`; later runs start in seconds. |
| Models not found | Run `scripts/download_models.sh` (or `--profile fast`) from the repository root, or point to them with `--models-dir DIR`. The Docker wrapper mounts `models/` of the repository. |
| ffmpeg 3.4 rejects a `p1`-`p7` NVENC preset | That ffmpeg only knows the legacy preset names. `video_demo` already uses `-preset medium`, which it accepts. If you run your own ffmpeg command, use `medium` (or `slow`, `fast`, `hq`) instead of `p1`-`p7`, or encode with `--encoder x264`. |
| `nvidia-smi` works but TensorRT says the CUDA driver is insufficient | The driver is older than the CUDA the TensorRT packages target. Use a TensorRT build for an older CUDA (Docker: a base image tag your driver supports) or update the driver. |
| `docker run` fails with a permission error on `outputs/` | The container runs as your user (`--user`). Make sure the output folder is yours, or let the wrapper create it. |
