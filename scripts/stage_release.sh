#!/usr/bin/env bash
# Stage the release assets (5 ONNX models + 7 videos, flat) and write SHA256SUMS.
set -euo pipefail
. "${BASH_SOURCE[0]%/*}/lib/fetch.sh"

usage() {
  cat <<USAGE
Usage: scripts/stage_release.sh --videos-dir DIR [--models-dir models] [--out outputs/release/v0.1.0]

  --videos-dir  directory holding sp1_crop*_30fps.mp4 and redlights/red_crop_*_30fps.mp4
  --models-dir  directory holding the ONNX models (default: models)
  --out         staging directory (default: outputs/release/v0.1.0)

Exit codes: 0 ok, 1 missing input file, 2 usage error or missing sha256sum.
USAGE
}

videos="" models=models out=outputs/release/v0.1.0
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --videos-dir|--models-dir|--out)
      [ $# -ge 2 ] || { usage >&2; die 2 "$1 needs a value"; }
      case $1 in
        --videos-dir) videos=$2 ;;
        --models-dir) models=$2 ;;
        --out) out=$2 ;;
      esac
      shift 2 ;;
    *) usage >&2; die 2 "unknown argument: $1" ;;
  esac
done
[ -n "$videos" ] || { usage >&2; die 2 "--videos-dir is required"; }
need_cmd sha256sum

srcs=() names=()
for n in yolo26s.onnx yolo26x.onnx segformer-b1-cityscapes-1024x576.onnx \
         segformer-b5-cityscapes-1280x720.onnx polarrcnn_curvelanes_dla34.onnx; do
  srcs+=("$models/$n") names+=("$n")
done
for n in sp1_crop_30fps.mp4 sp1_crop_2_30fps.mp4 sp1_crop_3_30fps.mp4 sp1_crop_4_30fps.mp4; do
  srcs+=("$videos/$n") names+=("$n")
done
for n in red_crop_11_30fps.mp4 red_crop_12_30fps.mp4 red_crop_13_30fps.mp4; do
  srcs+=("$videos/redlights/$n") names+=("$n")
done

missing=0
for s in "${srcs[@]}"; do
  [ -f "$s" ] || { echo "error: missing file: $s" >&2; missing=1; }
done
[ "$missing" = 0 ] || exit 1

mkdir -p "$out"
for i in "${!srcs[@]}"; do
  cp -f "${srcs[$i]}" "$out/${names[$i]}"
done
(cd "$out" && sha256sum "${names[@]}" > SHA256SUMS)
echo "staged ${#names[@]} assets + SHA256SUMS in $out"
