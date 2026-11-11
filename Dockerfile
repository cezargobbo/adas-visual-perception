# adas-visual-perception: video_demo on the NVIDIA TensorRT container.
#
# Build:  docker build -t adas-visual-perception .
# Run:    scripts/video_demo_docker.sh INPUT.mp4 -o outputs/run/clip   (needs the NVIDIA Container Toolkit)
#
# Base: TensorRT 10.14.1.48, CUDA 13.0.2, Ubuntu 24.04. 25.11 is the newest TensorRT 10.x tag whose
# CUDA (13.0) runs on a 580-series driver (the 25.12 tag moved to CUDA 13.1). Pinned by digest.
# Models and videos are not part of the image: they are mounted at run time.
FROM nvcr.io/nvidia/tensorrt:25.11-py3@sha256:fc886c8bad8fb81ada68baa70d00f2d2d166a921a039c52785e75a38a2f273db

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates ffmpeg libopencv-dev libcli11-dev \
 && rm -rf /var/lib/apt/lists/*

# No GPU is visible while building, so the CUDA architectures are listed instead of "native".
# 90 also embeds PTX, so newer GPUs still run. Override with --build-arg CUDA_ARCHS="89".
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90"

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY examples ./examples
COPY tools/cpp ./tools/cpp

RUN cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DADASVP_BUILD_TESTS=OFF \
 && cmake --build /tmp/build -j"$(nproc)" \
 && install -D /tmp/build/examples/video_demo /opt/adasvp/bin/video_demo \
 && install -D /tmp/build/libadasvp.so /opt/adasvp/lib/libadasvp.so \
 && echo /opt/adasvp/lib > /etc/ld.so.conf.d/adasvp.conf && ldconfig \
 && rm -rf /tmp/build /src

ENV PATH=/opt/adasvp/bin:${PATH}
WORKDIR /work
CMD ["video_demo", "--help"]
