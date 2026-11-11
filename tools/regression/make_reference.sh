#!/usr/bin/env bash
# Regenerate the Python reference alerts (detection only, --no-seg) for every
# video listed in videos.txt and copy the CSVs into tests/regression/reference/.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: make_reference.sh --videos-dir DIR --profile fast|accurate [--out outputs/reference]

  --videos-dir DIR   directory containing the videos listed in videos.txt
  --profile NAME     fast or accurate
  --out DIR          output root (default: outputs/reference); results go to DIR/<profile>/
  -h, --help         show this help

Writes <stem>_* outputs to DIR/<profile>/, copies the <stem>_alerts.csv files to
tests/regression/reference/<profile>/ and writes summary.txt there (one line per
video: wall time and FCW/LDW/RLR event counts).
USAGE
}

videos_dir=""; profile=""; out="outputs/reference"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-dir) videos_dir="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$videos_dir" && -d "$videos_dir" ]] || { echo "--videos-dir is required and must exist" >&2; exit 2; }
[[ "$profile" == fast || "$profile" == accurate ]] || { echo "--profile must be fast or accurate" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
py=".venv/bin/python"
dest="$out/$profile"
ref="tests/regression/reference/$profile"
mkdir -p "$dest" "$ref"
summary="$ref/summary.txt"
: > "$summary"

while IFS= read -r rel; do
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  stem="$(basename "$rel" .mp4)"
  t0=$(date +%s.%N)
  "$py" reference/python/dashcam_vision.py "$videos_dir/$rel" -o "$dest/$stem" \
      --profile "$profile" --no-seg > "$dest/$stem.log" 2>&1
  t1=$(date +%s.%N)
  csv="$dest/${stem}_alerts.csv"
  cp "$csv" "$ref/"
  fcw=$(awk -F, 'NR>1 && $3=="FCW"' "$csv" | wc -l)
  ldw=$(awk -F, 'NR>1 && $3=="LDW"' "$csv" | wc -l)
  rlr=$(awk -F, 'NR>1 && $3=="RLR"' "$csv" | wc -l)
  printf '%s time=%.0fs FCW=%d LDW=%d RLR=%d\n' "$stem" "$(echo "$t1 - $t0" | bc)" "$fcw" "$ldw" "$rlr" >> "$summary"
  echo "done $stem" >&2
done < "$here/videos.txt"
