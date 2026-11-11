#!/usr/bin/env bash
# Download the test videos from the GitHub release and verify their sha256.
set -euo pipefail
. "${BASH_SOURCE[0]%/*}/lib/fetch.sh"

usage() {
  cat <<USAGE
Usage: scripts/download_videos.sh [--dest DIR] [--tag TAG] [--base-url URL]

  --dest     destination directory (default: videos); red_crop_* go to DIR/redlights/
  --tag      release tag (default: $DEFAULT_TAG)
  --base-url download base URL (default: $REPO_URL/releases/download/<tag>)

Exit codes: 0 ok, 1 download or checksum failure, 2 usage error or missing curl/sha256sum.
USAGE
}

dest=videos tag=$DEFAULT_TAG base=""
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --dest|--tag|--base-url)
      [ $# -ge 2 ] || { usage >&2; die 2 "$1 needs a value"; }
      case $1 in
        --dest) dest=$2 ;;
        --tag) tag=$2 ;;
        --base-url) base=$2 ;;
      esac
      shift 2 ;;
    *) usage >&2; die 2 "unknown argument: $1" ;;
  esac
done
base=${base:-$REPO_URL/releases/download/$tag}

need_cmd curl
need_cmd sha256sum
fetch_sums "$base"
for n in sp1_crop_30fps.mp4 sp1_crop_2_30fps.mp4 sp1_crop_3_30fps.mp4 sp1_crop_4_30fps.mp4; do
  fetch_verified "$base" "$n" "$dest"
done
for n in red_crop_11_30fps.mp4 red_crop_12_30fps.mp4 red_crop_13_30fps.mp4; do
  fetch_verified "$base" "$n" "$dest/redlights"
done

echo "Credits: the red_crop_* clips come from the YouTube channel https://www.youtube.com/@QuinnDickinson2610"
