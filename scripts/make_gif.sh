#!/usr/bin/env bash
# Cut a window around an event out of a video and write an optimized GIF.
# Usage: make_gif.sh INPUT.mp4 CENTER_S OUT.gif [--before 5] [--after 5]
#                    [--width 640] [--fps 12] [--colors 128]
set -euo pipefail

usage() {
  echo "usage: $0 INPUT.mp4 CENTER_S OUT.gif [--before S] [--after S] [--width PX] [--fps N] [--colors N]" >&2
  exit 2
}

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }
[ $# -ge 3 ] || usage
in=$1; center=$2; out=$3; shift 3
before=5; after=5; width=640; fps=12; colors=128
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || usage
  case $1 in
    --before) before=$2 ;;
    --after)  after=$2 ;;
    --width)  width=$2 ;;
    --fps)    fps=$2 ;;
    --colors) colors=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[ -f "$in" ] || { echo "error: input not found: $in" >&2; exit 1; }

start=$(awk -v c="$center" -v b="$before" 'BEGIN { s = c - b; if (s < 0) s = 0; printf "%.3f", s }')
dur=$(awk -v b="$before" -v a="$after" 'BEGIN { printf "%.3f", b + a }')

ffmpeg -y -v error -ss "$start" -t "$dur" -i "$in" \
  -vf "fps=${fps},scale=${width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=${colors}[p];[s1][p]paletteuse=dither=sierra2_4a" \
  -loop 0 "$out"

bytes=$(stat -c %s "$out")
echo "$out: $bytes bytes ($(awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1048576 }') MB)"
