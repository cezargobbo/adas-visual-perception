#!/usr/bin/env python3
"""Extract traffic-light crops + the reference traffic_light_state() as test fixtures.

For the published test videos (sp1_crop*_30fps, red_crop_{11,12,13}_30fps) the class-9 detections
come from the reference dumps (frames.jsonl written by dump_python.py). Each selected box is cut
out of the frame exactly as traffic_light_state() does (rounded, clipped to the image, no extra
padding; the function's own 15% side margin is applied inside it) and saved as a PNG. The expected
state is the Python traffic_light_state() result for that box on the full frame; the script checks
that running it on the crop alone gives the same answer, which is what the C++ test does.

Up to 60 crops are chosen to cover red / yellow / green / unknown, spread over videos and time.

  make_tl_fixtures.py --videos-dir DIR --dumps-dir outputs/dumps/fast --out tests/fixtures

Output: tl_state_NNN.png and tl_state_expected.json
  [{"file", "video", "frame", "box": [x1,y1,x2,y2] (floats, frame pixels), "crop_box": [0,0,w,h],
    "state": "red"|"yellow"|"green"|null}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "python"))

VIDEOS = ["sp1_crop_30fps", "sp1_crop_2_30fps", "sp1_crop_3_30fps", "sp1_crop_4_30fps",
          "red_crop_11_30fps", "red_crop_12_30fps", "red_crop_13_30fps"]
QUOTA = {"red": 18, "green": 18, "yellow": 12, None: 12}
MAX_TOTAL = 60
CANDIDATES_PER_VIDEO = 250
MAX_PNG_BYTES = 600_000


def find_video(root: Path, stem: str) -> Path | None:
    hits = sorted(root.rglob(stem + ".mp4"))
    return hits[0] if hits else None


def spread(items: list, n: int) -> list:
    """n items evenly spaced over the list."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    return [items[int(k * len(items) / n)] for k in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--dumps-dir", required=True, help="e.g. outputs/dumps/fast (one dir per video stem)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import dashcam_vision as dv

    cands = []            # dicts: video, frame, box, state, crop
    for stem in VIDEOS:
        vpath = find_video(Path(args.videos_dir), stem)
        fpath = Path(args.dumps_dir) / stem / "frames.jsonl"
        if vpath is None or not fpath.exists():
            print(f"[skip] {stem}: video or dump missing", file=sys.stderr)
            continue
        boxes = {}
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                tl = [d[:4] for d in r["dets"] if d[5] == 9 and d[2] - d[0] >= 6 and d[3] - d[1] >= 6]
                if tl:
                    boxes[r["i"]] = tl
        flat = [(i, b) for i in sorted(boxes) for b in boxes[i]]
        pick = {}
        for i, b in spread(flat, CANDIDATES_PER_VIDEO):
            pick.setdefault(i, []).append(b)
        cap = cv2.VideoCapture(str(vpath))
        for i in range(max(pick) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            for b in pick.get(i, []):
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                if crop.size == 0:
                    continue
                st = dv.traffic_light_state(frame, b)
                h, w = crop.shape[:2]
                st2 = dv.traffic_light_state(crop, [0, 0, w, h])
                assert st == st2, f"{stem} frame {i}: frame-state {st} != crop-state {st2}"
                cands.append({"video": stem, "frame": i, "box": [float(v) for v in b], "state": st,
                              "crop": crop.copy()})
        cap.release()
        print(f"[{stem}] {len(flat)} boxes, {sum(len(v) for v in pick.values())} evaluated", file=sys.stderr)

    chosen = []
    for st, q in QUOTA.items():
        pool = [c for c in cands if c["state"] == st]
        pool.sort(key=lambda c: (c["video"], c["frame"]))
        chosen += spread(pool, q)
    if len(chosen) < MAX_TOTAL:            # backfill from whatever is left, preferring rare states
        left = [c for c in cands if all(c is not d for d in chosen)]
        left.sort(key=lambda c: (c["video"], c["frame"]))
        chosen += spread(left, MAX_TOTAL - len(chosen))
    chosen = chosen[:MAX_TOTAL]
    chosen.sort(key=lambda c: (c["video"], c["frame"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("tl_state_*"):
        old.unlink()
    expected, total = [], 0
    for k, c in enumerate(chosen):
        name = f"tl_state_{k:03d}.png"
        cv2.imwrite(str(out / name), c["crop"], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        total += (out / name).stat().st_size
        h, w = c["crop"].shape[:2]
        expected.append({"file": name, "video": c["video"], "frame": c["frame"], "box": c["box"],
                         "crop_box": [0, 0, w, h], "state": c["state"]})
    (out / "tl_state_expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    total += (out / "tl_state_expected.json").stat().st_size
    dist = {str(k): sum(1 for e in expected if e["state"] == k) for k in ("red", "yellow", "green", None)}
    print(f"{len(expected)} fixtures, {total} bytes, states {dist}")
    if total >= MAX_PNG_BYTES:
        sys.exit(f"fixtures too large: {total} bytes")


if __name__ == "__main__":
    main()
