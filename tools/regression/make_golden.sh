#!/usr/bin/env bash
# Regenerates the C++ golden alerts (tests/regression/golden/<profile>/<stem>_alerts.csv) with video_demo
# (--dry --no-seg, the flags of run_cpp_regression.sh) and copies only the CSVs.
# Changing the golden files is an intentional, reviewed change: commit it on its own, with the reason.
set -uo pipefail

usage() {
  cat <<USAGE
Usage: make_golden.sh --videos-dir DIR --profile fast|accurate [--list FILE] [--demo PATH] [--out DIR]

  --videos-dir DIR   directory containing the videos listed in videos.txt
  --profile NAME     fast or accurate
  --list FILE        video list (default: videos.txt next to this script)
  --demo PATH        video_demo binary (default: build/native-release/examples/video_demo)
  --out DIR          scratch output root (default: outputs/golden_gen); results go to DIR/<profile>/

Runs video_demo on every video, then copies <stem>_alerts.csv into tests/regression/golden/<profile>/.
Check the result with run_cpp_regression.sh (default --against golden) before committing.
USAGE
}

videos_dir=""; profile=""; list=""; demo=""; out="outputs/golden_gen"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-dir) videos_dir="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --list) list="$2"; shift 2 ;;
    --demo) demo="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$videos_dir" && -d "$videos_dir" ]] || { echo "--videos-dir is required and must exist" >&2; exit 2; }
[[ "$profile" == fast || "$profile" == accurate ]] || { echo "--profile must be fast or accurate" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$list" ]] || list="$here/videos.txt"
[[ -f "$list" ]] || { echo "--list: no such file: $list" >&2; exit 2; }
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
[[ -n "$demo" ]] || demo="build/native-release/examples/video_demo"
dest="$out/$profile"
gold="tests/regression/golden/$profile"
mkdir -p "$dest" "$gold"

n=0
while IFS= read -r rel; do
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  stem="$(basename "$rel" .mp4)"
  rm -f "$dest/${stem}_alerts.csv"
  "$demo" --dry --no-seg --profile "$profile" -o "$dest/$stem" "$videos_dir/$rel" < /dev/null > "$dest/$stem.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -f "$dest/${stem}_alerts.csv" ]]; then
    echo "video_demo failed on $rel (rc=$rc), see $dest/$stem.log" >&2
    exit 1
  fi
  cp "$dest/${stem}_alerts.csv" "$gold/"
  n=$((n + 1))
  echo "golden $profile $stem" >&2
done < "$list"
echo "$n golden CSVs written to $gold" >&2
