#!/usr/bin/env bash
# Run video_demo inside the adas-visual-perception image.
# Usage: scripts/video_demo_docker.sh INPUT.mp4 [-o PREFIX] [video_demo options...]
#
# Mounted read-only: the repository (/work/repo), models/ (/work/models), the input video's directory.
# Mounted read-write: the directory of the -o prefix and the engine cache (/cache/adasvp).
# Environment: ADASVP_DOCKER_IMAGE (default adas-visual-perception),
#              ADASVP_DOCKER_GPU_FLAGS (default "--gpus all"; set it empty to run without a GPU).
set -euo pipefail

ROOT=$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)
IMAGE=${ADASVP_DOCKER_IMAGE:-adas-visual-perception}
GPU_FLAGS=${ADASVP_DOCKER_GPU_FLAGS---gpus all}
CACHE_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/adasvp

command -v docker >/dev/null 2>&1 || { echo "error: docker not found in PATH" >&2; exit 2; }

abspath() { # abspath PATH -> absolute path (no symlink resolution of the last component)
  case $1 in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

# Options that take one value. Needed to tell the input video (the only positional) from option values.
VALUE_OPTS=" -o --out-prefix --start --duration --encoder --profile --show-width --seg-model --seg-size
 --seg-every --seg-alpha --det-model --det-imgsz --conf --hfov --cam-height --horizon --corridor-bottom
 --corridor-top --lanes-thr --lanes-hold --fcw-ttc --fcw-ttc-critical --ldw-margin --ldw-car-width
 --alert-display --det-precision --models-dir "

args=() mounts=() input="" outprefix="" outdir=""
while [ $# -gt 0 ]; do
  case $1 in
    --models-dir|--models-dir=*)
      echo "error: --models-dir is set by this wrapper (models/ is mounted at /work/models)" >&2; exit 2 ;;
    --out-prefix=*|--*=*)
      echo "error: write '$1' as two words (OPTION VALUE); the wrapper translates paths word by word" >&2; exit 2 ;;
    -o|--out-prefix)
      [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }
      outprefix=$(abspath "$2")
      outdir=${outprefix%/*}
      [ -n "$outdir" ] || outdir=/
      mkdir -p "$outdir"
      args+=("-o" "/out/${outprefix##*/}")
      shift 2 ;;
    --rlr-zone)
      [ $# -ge 3 ] || { echo "error: --rlr-zone needs two values" >&2; exit 2; }
      args+=("$1" "$2" "$3"); shift 3 ;;
    --show)
      args+=("$1"); shift
      if [ $# -gt 0 ]; then case $1 in det|seg|comb) args+=("$1"); shift ;; esac; fi ;;
    -*)
      if [[ $VALUE_OPTS == *" $1 "* ]]; then
        [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }
        args+=("$1" "$2"); shift 2
      else
        args+=("$1"); shift
      fi ;;
    *)
      if [ -z "$input" ]; then
        input=$(abspath "$1")
        args=("/in/${input##*/}" "${args[@]}")
      else
        args+=("$1")
      fi
      shift ;;
  esac
done

mkdir -p "$CACHE_DIR"

run=(docker run --rm --init)
# shellcheck disable=SC2206  # GPU_FLAGS is meant to word-split ("--gpus all")
run+=($GPU_FLAGS)
run+=(--user "$(id -u):$(id -g)"
      -e XDG_CACHE_HOME=/cache -e HOME=/tmp
      -v "$ROOT:/work/repo:ro" -v "$ROOT/models:/work/models:ro"
      -v "$CACHE_DIR:/cache/adasvp")
[ -n "$input" ] && run+=(-v "${input%/*}:/in:ro")
[ -n "$outdir" ] && run+=(-v "$outdir:/out")
run+=("$IMAGE" video_demo --models-dir /work/models "${args[@]}")

# Keep stderr visible while also scanning it for GPU/runtime errors.
err=$(mktemp "${TMPDIR:-/tmp}/adasvp-docker.XXXXXX")
trap 'rm -f "$err"' EXIT
set +e
"${run[@]}" 2> >(tee "$err" >&2)
rc=$?
set -e
sleep 0.2  # let the tee process substitution flush
if [ $rc -ne 0 ] && grep -qiE "could not select device driver|nvidia-container|unknown or invalid runtime|nvidia.*runtime|NVIDIA Driver|no CUDA-capable|CUDA driver version is insufficient|failed to initialize NVML|libcuda" "$err"; then
  cat >&2 <<'HINT'

error: the container could not use the GPU.
Install the NVIDIA Container Toolkit and restart Docker:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
Check with: docker run --rm --gpus all adas-visual-perception nvidia-smi
See docs/install.md, section "When it does not work".
HINT
  exit 2
fi
exit $rc
