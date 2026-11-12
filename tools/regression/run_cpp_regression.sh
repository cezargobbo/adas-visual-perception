#!/usr/bin/env bash
# Runs the C++ video_demo (--no-seg) on every video of videos.txt and compares the alert CSVs with a
# baseline (strict, see compare_alerts.py): the C++ golden alerts in tests/regression/golden/<profile>/
# (default) or the alerts of the original Python prototype in tests/regression/python/<profile>/.
set -uo pipefail

usage() {
  cat <<USAGE
Usage: run_cpp_regression.sh --videos-dir DIR --profile fast|accurate [--against golden|python] [--dry] [--det-precision fp16|fp32]
                             [--out DIR] [--demo PATH] [--list FILE]

  --videos-dir DIR       directory containing the videos listed in videos.txt
  --profile NAME         fast or accurate
  --against BASELINE     golden (default): tests/regression/golden/<profile>/, the C++ baseline, expected to match
                         exactly; python: tests/regression/python/<profile>/, the alerts of the Python prototype
                         (the fast profile has one documented borderline, see tools/regression/README.md)
  --dry                  no output videos (the alerts CSV is still written)
  --det-precision P      override the detector precision (hidden video_demo flag)
  --out DIR              output root (default: outputs/cpp_regression); results go to DIR/<profile>[_<precision>]/
  --demo PATH            video_demo binary (default: build/native-release/examples/video_demo)
  --list FILE            video list (default: videos.txt next to this script)

Prints one line per video (counts, fps, result) and a summary; exit 0 only when every pair passes.
Pairs that fail are the input of the borderline diagnosis (README.md, compare_dumps.py --adas).
USAGE
}

videos_dir=""; profile=""; against="golden"; dry=""; prec=""; out="outputs/cpp_regression"; demo="build/native-release/examples/video_demo"; list=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-dir) videos_dir="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --against) against="$2"; shift 2 ;;
    --dry) dry="--dry"; shift ;;
    --det-precision) prec="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --demo) demo="$2"; shift 2 ;;
    --list) list="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$videos_dir" && -d "$videos_dir" ]] || { echo "--videos-dir is required and must exist" >&2; exit 2; }
[[ "$profile" == fast || "$profile" == accurate ]] || { echo "--profile must be fast or accurate" >&2; exit 2; }
[[ "$against" == golden || "$against" == python ]] || { echo "--against must be golden or python" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$list" ]] || list="$here/videos.txt"
[[ -f "$list" ]] || { echo "--list: no such file: $list" >&2; exit 2; }
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
py=".venv/bin/python"
[[ -x "$py" ]] || py="python3"  # compare_alerts.py is stdlib-only
ref="tests/regression/$against/$profile"
[[ -d "$ref" ]] || { echo "no baseline dir for --against $against --profile $profile: $ref" >&2; exit 2; }
dest="$out/$profile${prec:+_$prec}"
mkdir -p "$dest"
extra=(); [[ -n "$prec" ]] && extra=(--det-precision "$prec")

n=0; failed=0
while IFS= read -r rel; do
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  stem="$(basename "$rel" .mp4)"
  # a stale CSV must never make a failed run pass
  rm -f "$dest/${stem}_alerts.csv" "$dest/$stem.compare.txt"
  "$demo" $dry --no-seg --profile "$profile" ${extra[@]+"${extra[@]}"} -o "$dest/$stem" "$videos_dir/$rel" < /dev/null > "$dest/$stem.log" 2>&1
  demo_rc=$?
  fps=$(grep -o 'total [0-9.]* fps' "$dest/$stem.log" | tail -1 | awk '{print $2}')
  n=$((n + 1))
  if [[ $demo_rc -ne 0 || ! -f "$dest/${stem}_alerts.csv" ]]; then
    failed=$((failed + 1))
    printf '%-22s %-4s fps=%-6s video_demo rc=%d (no valid alerts CSV)\n' "$stem" FAIL "${fps:-?}" "$demo_rc"
    continue
  fi
  "$py" "$here/compare_alerts.py" "$ref/${stem}_alerts.csv" "$dest/${stem}_alerts.csv" > "$dest/$stem.compare.txt" 2>&1
  rc=$?
  res=PASS
  if [[ $rc -ne 0 ]]; then res=FAIL; failed=$((failed + 1)); fi
  printf '%-22s %-4s fps=%-6s %s\n' "$stem" "$res" "${fps:-?}" "$(head -1 "$dest/$stem.compare.txt")"
done < "$list"
echo "$profile${prec:+ $prec} vs $against: $((n - failed))/$n pairs identical, $failed failed"
[[ $n -gt 0 ]] || { echo "no videos in $list" >&2; exit 2; }
[[ $failed -eq 0 ]]
