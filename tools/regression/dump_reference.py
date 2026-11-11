#!/usr/bin/env python3
"""Dump per-frame intermediate results of the Python reference pipeline.

The dump is the ground truth the C++ tests compare against. It imports
reference/python/dashcam_vision.py (never modified) and runs the models exactly as
the main loop does: same Ultralytics ``track()`` call, same ``Segmenter``, same
``PolarLaneDetector``, same ``--seg-every`` reuse rule (segmentation runs on frames
with ``i % seg_every == 0``, ``i`` relative to --start).

Output layout (DIR):
  frames.jsonl   one line per frame:
                 {"i", "t", "dets": [[x1,y1,x2,y2,score,cls],...],
                  "tracks": [[x1,y1,x2,y2,id,score,cls],...],
                  "lanes": [[[x,y],...],...], "lane_scores": [...]}
                 i is relative to --start, t = (start_frame + i) / fps (seconds).
                 dets   = exactly what entered BYTETracker.update (captured by wrapping
                          that method: post-NMS, conf >= --conf, class filter, original
                          frame pixel coordinates).
                 tracks = what the reference loop sees afterwards: result.boxes rows that
                          carry a track id (Kalman-smoothed xyxy, id, score, cls).
                 lanes  = frame-pixel polylines (top -> bottom) from PolarLaneDetector.
  seg/NNNNNN.png uint8 class-id map (frame resolution), every --seg-every frames.
  meta.json      video, profile, resolution, fps, parameters, library versions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference" / "python"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--profile", choices=["fast", "accurate"], required=True)
    ap.add_argument("--out", required=True, metavar="DIR")
    ap.add_argument("--start", type=float, default=0.0, metavar="S")
    ap.add_argument("--duration", type=float, default=None, metavar="D")
    ap.add_argument("--seg-every", type=int, default=30, metavar="N")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--lanes-thr", type=float, default=None)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker

    import dashcam_vision as dv
    from lanes_polar import PolarLaneDetector

    prof = dv.PROFILES[args.profile]
    lanes_thr = dv.LANES_THR_DEFAULT if args.lanes_thr is None else args.lanes_thr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(args.start * fps)
    end_f = total if args.duration is None else min(total, start_f + int(args.duration * fps))
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    n_frames = end_f - start_f

    sw, sh = (int(v) for v in prof["seg_size"].lower().split("x"))
    det = YOLO(prof["det_model"])
    seg = dv.Segmenter(prof["seg_model"], device, sw, sh)
    lane_det = PolarLaneDetector(thr=lanes_thr, device=device)

    # Capture the tracker input: wrap BYTETracker.update. Ultralytics' on_predict_postprocess_end
    # calls tracker.update(result.boxes.cpu().numpy(), ...) once per frame, after NMS, the conf
    # threshold and the class filter, with boxes already scaled to the original frame.
    captured = []
    orig_update = BYTETracker.update

    def update_spy(self, results, *a, **k):
        captured.append(np.asarray(results.data, dtype=np.float32).reshape(-1, 6).copy())
        return orig_update(self, results, *a, **k)

    BYTETracker.update = update_spy

    out = Path(args.out)
    (out / "seg").mkdir(parents=True, exist_ok=True)
    seg_written = 0
    with open(out / "frames.jsonl", "w") as jf:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                n_frames = i
                break
            lane_list, lane_scores = lane_det(frame)
            captured.clear()
            res = det.track(frame, persist=True, tracker=args.tracker, conf=args.conf,
                            imgsz=prof["det_imgsz"], classes=list(dv.COCO_KEEP),
                            quantize=16 if device.type == "cuda" else None, verbose=False)[0]
            assert len(captured) == 1, f"tracker.update called {len(captured)} times on frame {i}"
            dets = [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[5])]
                    for r in captured[0]]
            tracks = []
            if res.boxes is not None and len(res.boxes) and res.boxes.id is not None:
                for r in res.boxes.data.cpu().numpy():
                    tracks.append([float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                                   int(r[4]), float(r[5]), int(r[6])])
            if i % args.seg_every == 0:
                m = seg(frame).cpu().numpy()
                assert m.dtype == np.uint8 and m.shape == (H, W)
                cv2.imwrite(str(out / "seg" / f"{i:06d}.png"), m)
                seg_written += 1
            rec = {"i": i, "t": (start_f + i) / fps, "dets": dets, "tracks": tracks,
                   "lanes": [[[float(x), float(y)] for x, y in ln] for ln in lane_list],
                   "lane_scores": [float(s) for s in lane_scores]}
            jf.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{n_frames}]", flush=True)
    BYTETracker.update = orig_update
    cap.release()

    meta = {
        "video": os.path.basename(args.video), "profile": args.profile,
        "width": W, "height": H, "fps": fps, "start_s": args.start, "start_frame": start_f,
        "duration_s": args.duration, "frames": n_frames, "seg_every": args.seg_every,
        "seg_frames_written": seg_written, "conf": args.conf, "lanes_thr": lanes_thr,
        "tracker": args.tracker, "det_model": os.path.basename(prof["det_model"]),
        "det_imgsz": prof["det_imgsz"], "seg_model": os.path.basename(prof["seg_model"]),
        "seg_size": prof["seg_size"], "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__},
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {n_frames} frames -> {out}")


if __name__ == "__main__":
    main()
