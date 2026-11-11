#!/usr/bin/env bash
# Runs the C++ video_demo (--no-seg) on every video of videos.txt and compares the alert CSVs with the
# Python reference in tests/regression/reference/<profile>/ (strict, see compare_alerts.py).
set -uo pipefail

usage() {
  cat <<USAGE
Usage: run_cpp_regression.sh --videos-dir DIR --profile fast|accurate [--dry] [--det-precision fp16|fp32]
                             [--out DIR] [--demo PATH]

  --videos-dir DIR       directory containing the videos listed in videos.txt
  --profile NAME         fast or accurate
  --dry                  no output videos (the alerts CSV is still written)
  --det-precision P      override the detector precision (hidden video_demo flag)
  --out DIR              output root (default: outputs/cpp_regression); results go to DIR/<profile>[_<precision>]/
  --demo PATH            video_demo binary (default: build/native-release/examples/video_demo)

Prints one line per video (counts, fps, result) and a summary; exit 0 only when every pair passes.
Pairs that fail are the input of the borderline diagnosis (README.md, compare_dumps.py --adas).
USAGE
}

videos_dir=""; profile=""; dry=""; prec=""; out="outputs/cpp_regression"; demo="build/native-release/examples/video_demo"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-dir) videos_dir="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --dry) dry="--dry"; shift ;;
    --det-precision) prec="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --demo) demo="$2"; shift 2 ;;
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
ref="tests/regression/reference/$profile"
dest="$out/$profile${prec:+_$prec}"
mkdir -p "$dest"
extra=(); [[ -n "$prec" ]] && extra=(--det-precision "$prec")

n=0; failed=0
while IFS= read -r rel; do
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  stem="$(basename "$rel" .mp4)"
  "$demo" $dry --no-seg --profile "$profile" "${extra[@]}" -o "$dest/$stem" "$videos_dir/$rel" > "$dest/$stem.log" 2>&1
  fps=$(grep -o 'total [0-9.]* fps' "$dest/$stem.log" | tail -1 | awk '{print $2}')
  "$py" "$here/compare_alerts.py" "$ref/${stem}_alerts.csv" "$dest/${stem}_alerts.csv" > "$dest/$stem.compare.txt"
  rc=$?
  n=$((n + 1)); res=PASS
  if [[ $rc -ne 0 ]]; then res=FAIL; failed=$((failed + 1)); fi
  printf '%-22s %-4s fps=%-6s %s\n' "$stem" "$res" "${fps:-?}" "$(head -1 "$dest/$stem.compare.txt")"
done < "$here/videos.txt"
echo "$profile${prec:+ $prec}: $((n - failed))/$n pairs identical, $failed failed"
[[ $failed -eq 0 ]]
