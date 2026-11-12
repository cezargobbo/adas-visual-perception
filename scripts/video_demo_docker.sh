#!/usr/bin/env bash
# Run video_demo inside the adas-visual-perception image.
# Usage: scripts/video_demo_docker.sh INPUT.mp4 [-o PREFIX] [video_demo options...]
# Without -o the outputs go to outputs/<input name> in the current directory (like the native video_demo).
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

abspath() { # abspath PATH -> absolute path (no symlink resolution of the last component)
  case $1 in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

# Options that take one value (tests/tools/test_docker_wrapper.py checks this list against examples/cli.cpp).
# Special cases handled below: -o/--out-prefix and --adas-debug (output paths, translated), --rlr-zone
# (two values), --show (rejected: no display in the container), --models-dir (rejected: set by the wrapper).
VALUE_OPTS=" --start --duration --encoder --profile --show-width --seg-model --seg-size
 --seg-every --seg-alpha --det-model --det-imgsz --conf --hfov --cam-height --horizon --corridor-bottom
 --corridor-top --lanes-thr --lanes-hold --fcw-ttc --fcw-ttc-critical --ldw-margin --ldw-car-width
 --alert-display --det-precision "

# host_outdir PATH: absolute directory of an output file path, created on the host.
host_outdir() {
  local d=${1%/*}
  [ -n "$d" ] || d=/
  mkdir -p "$d"
  printf '%s\n' "$d"
}

args=() input="" outdir="" dbgdir="" positional_only=0
while [ $# -gt 0 ]; do
  if [ $positional_only -eq 0 ]; then
    case $1 in
      --) positional_only=1; shift; continue ;;
      --models-dir|--models-dir=*)
        echo "error: --models-dir is set by this wrapper (models/ is mounted at /work/models)" >&2; exit 2 ;;
      --show|--show=*)
        echo "error: --show is not supported through Docker; run video_demo natively" >&2; exit 2 ;;
      --*=*)
        echo "error: write '$1' as two words (OPTION VALUE); the wrapper translates paths word by word" >&2; exit 2 ;;
      -o|--out-prefix)
        [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }
        p=$(abspath "$2")
        outdir=$(host_outdir "$p")
        args+=("-o" "/out/${p##*/}")
        shift 2; continue ;;
      --adas-debug)
        [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }
        p=$(abspath "$2")
        dbgdir=$(host_outdir "$p")
        args+=("--adas-debug" "/dbg/${p##*/}")
        shift 2; continue ;;
      --rlr-zone)
        [ $# -ge 3 ] || { echo "error: --rlr-zone needs two values" >&2; exit 2; }
        args+=("$1" "$2" "$3"); shift 3; continue ;;
      -*)
        if [[ $VALUE_OPTS == *" $1 "* ]]; then
          [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }
          args+=("$1" "$2"); shift 2
        else
          args+=("$1"); shift
        fi
        continue ;;
    esac
  fi
  if [ -z "$input" ]; then
    input=$(abspath "$1")
    args=("/in/${input##*/}" "${args[@]}")
  else
    args+=("$1")
  fi
  shift
done

# No -o given: mirror the native default (outputs/<input stem> under the current directory), because the
# container's /work is not writable.
if [ -n "$input" ] && [ -z "$outdir" ]; then
  base=${input##*/}
  p="$PWD/outputs/${base%.*}"
  outdir=$(host_outdir "$p")
  args+=("-o" "/out/${p##*/}")
fi

command -v docker >/dev/null 2>&1 || { echo "error: docker not found in PATH" >&2; exit 2; }
mkdir -p "$CACHE_DIR"

run=(docker run --rm --init)
# shellcheck disable=SC2206  # GPU_FLAGS is meant to word-split ("--gpus all")
run+=($GPU_FLAGS)
run+=(--user "$(id -u):$(id -g)"
      -e XDG_CACHE_HOME=/cache -e HOME=/tmp
      --mount "type=bind,src=$ROOT,dst=/work/repo,readonly"
      --mount "type=bind,src=$ROOT/models,dst=/work/models,readonly"
      --mount "type=bind,src=$CACHE_DIR,dst=/cache/adasvp")
[ -n "$input" ] && run+=(--mount "type=bind,src=${input%/*},dst=/in,readonly")
[ -n "$outdir" ] && run+=(--mount "type=bind,src=$outdir,dst=/out")
[ -n "$dbgdir" ] && run+=(--mount "type=bind,src=$dbgdir,dst=/dbg")
run+=("$IMAGE" video_demo --models-dir /work/models "${args[@]}")

# Keep stderr visible while also scanning it for GPU/runtime errors (a FIFO, so the tee can be waited for).
tmp=$(mktemp -d "${TMPDIR:-/tmp}/adasvp-docker.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
err=$tmp/stderr
mkfifo "$tmp/fifo"
tee "$err" >&2 <"$tmp/fifo" &
tee_pid=$!
set +e
"${run[@]}" 2>"$tmp/fifo"
rc=$?
wait "$tee_pid"
set -e
if [ $rc -ne 0 ] && grep -qiE "could not select device driver|nvidia-container|unknown or invalid runtime|nvidia.*runtime|NVIDIA Driver|no CUDA-capable|CUDA driver version is insufficient|failed to initialize NVML|libcuda" "$err"; then
  cat >&2 <<'HINT'

error: the container could not use the GPU.
Install the NVIDIA Container Toolkit and restart Docker:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
Check with: docker run --rm --gpus all nvcr.io/nvidia/tensorrt:25.11-py3 nvidia-smi
See docs/install.md, section "When it does not work".
HINT
  exit 2
fi
exit $rc
