#!/usr/bin/env bash
# Download the ONNX models from the GitHub release and verify their sha256.
set -euo pipefail
. "${BASH_SOURCE[0]%/*}/lib/fetch.sh"
REPO_ROOT=$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)

usage() {
  cat <<USAGE
Usage: scripts/download_models.sh [--profile fast|accurate|all] [--dest DIR] [--tag TAG] [--base-url URL]

  --profile  fast: yolo26s + segformer-b1 + polar; accurate: yolo26x + segformer-b5 + polar;
             all: the five models (default)
  --dest     destination directory (default: models/ of the repository, whatever the current directory)
  --tag      release tag (default: $DEFAULT_TAG)
  --base-url download base URL (default: $REPO_URL/releases/download/<tag>)

Exit codes: 0 ok, 1 download or checksum failure, 2 usage error or missing curl/sha256sum.
USAGE
}

profile=all dest=$REPO_ROOT/models tag=$DEFAULT_TAG base=""
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --profile|--dest|--tag|--base-url)
      [ $# -ge 2 ] || { usage >&2; die 2 "$1 needs a value"; }
      case $1 in
        --profile) profile=$2 ;;
        --dest) dest=$2 ;;
        --tag) tag=$2 ;;
        --base-url) base=$2 ;;
      esac
      shift 2 ;;
    *) usage >&2; die 2 "unknown argument: $1" ;;
  esac
done

case $profile in
  fast) names=(yolo26s.onnx segformer-b1-cityscapes-1024x576.onnx polarrcnn_curvelanes_dla34.onnx) ;;
  accurate) names=(yolo26x.onnx segformer-b5-cityscapes-1280x720.onnx polarrcnn_curvelanes_dla34.onnx) ;;
  all) names=(yolo26s.onnx yolo26x.onnx segformer-b1-cityscapes-1024x576.onnx
              segformer-b5-cityscapes-1280x720.onnx polarrcnn_curvelanes_dla34.onnx) ;;
  *) usage >&2; die 2 "unknown profile: $profile (expected fast, accurate or all)" ;;
esac
base=${base:-$REPO_URL/releases/download/$tag}

need_cmd curl
need_cmd sha256sum
fetch_sums "$base"
for n in "${names[@]}"; do
  fetch_verified "$base" "$n" "$dest"
done

cat <<LICENSES
Licenses:
  YOLO26: AGPL-3.0.
  SegFormer: NVIDIA Source Code License, non-commercial use only.
  Polar R-CNN: no declared license, redistributed for non-commercial academic use with credit.
  CurveLanes and Cityscapes data: non-commercial.
LICENSES
