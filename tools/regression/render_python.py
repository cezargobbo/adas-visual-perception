#!/usr/bin/env python3
"""Render the HUD views of the Python reference from a recorded trace.

Uses the drawing code of prototype/python/dashcam_vision.py (AlertDisplay, draw_lanes, draw_detections,
draw_lead, draw_fcw_box, draw_alerts, Segmenter.overlay, draw_legend) with the state recorded by
dump_python.py --adas (adas.jsonl + frames.jsonl + seg/*.png), driven like the main loop: one AlertDisplay
update per frame, blink counter = frame index inside the rendered window (the window starts with a fresh
AlertDisplay, like a fresh C++ Renderer).

    render_python.py --trace outputs/dumps/fast/STEM --frames A:B --out OUT [--videos-dir DIR]

Writes OUT/{det,seg,comb}_NNNNNN.png (NNNNNN = trace frame index i, A <= i < B) and OUT/manifest.json
(video, window, seg PNG used by every frame: the nearest one in time, ties to the earlier).
The raw frame of the trace frame i is frame `start_frame + i` of the video.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "python"))


def load_prototype():
    spec = importlib.util.spec_from_file_location("dashcam_vision_proto", ROOT / "prototype" / "python" / "dashcam_vision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_video(name, videos_dir):
    for cand in (Path(videos_dir) / name, Path(videos_dir) / "redlights" / name):
        if cand.exists():
            return cand
    raise SystemExit(f"video {name} not found under {videos_dir}")


def nearest_seg(i, seg_idx):
    return min(seg_idx, key=lambda s: (abs(s - i), s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--frames", required=True, help="A:B trace frame indices, B exclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos-dir", default=os.environ.get("ADASVP_VIDEOS_DIR", "."))
    ap.add_argument("--alert-display", type=float, default=2.0)
    ap.add_argument("--seg-alpha", type=float, default=0.45)
    a = ap.parse_args()
    f0, f1 = (int(v) for v in a.frames.split(":"))
    trace, out = Path(a.trace), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dv = load_prototype()
    import torch
    meta = json.load(open(trace / "meta.json"))
    ys = np.array(meta["adas"]["ego_ys"])
    start = meta["start_frame"]
    seg_idx = sorted(int(p.stem) for p in (trace / "seg").glob("*.png"))

    frames = {}
    for k, line in enumerate(open(trace / "frames.jsonl")):
        if f0 <= k < f1:
            frames[k] = json.loads(line)
    adas = {}
    for k, line in enumerate(open(trace / "adas.jsonl")):
        if f0 <= k < f1:
            adas[k] = json.loads(line)

    cap = cv2.VideoCapture(str(find_video(meta["video"], a.videos_dir)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start + f0)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start + f0:       # seek not honoured: decode sequentially
        cap.release()
        cap = cv2.VideoCapture(str(find_video(meta["video"], a.videos_dir)))
        for _ in range(start + f0):
            cap.read()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = dv.Segmenter.__new__(dv.Segmenter)                    # no model: only Segmenter.overlay is needed
    seg.device = device
    seg.palette = torch.from_numpy(dv.PALETTE_BGR).to(device)

    disp = dv.AlertDisplay(a.alert_display)
    manifest = dict(video=meta["video"], trace=trace.name, frames=[f0, f1], start_frame=start,
                    alert_display=a.alert_display, seg_alpha=a.seg_alpha, seg={})
    for n, k in enumerate(range(f0, f1)):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"video ended before frame {k}")
        fr, ad = frames[k], adas[k]
        t = ad["t"]
        dets = [(int(tr[4]), int(tr[6]), np.array(tr[:4], np.float32), np.float32(tr[5])) for tr in fr["tracks"]]
        lane_list = [np.array(p, np.float32).reshape(-1, 2) for p in fr["lanes"]]
        e = ad["ego"]
        lane_trk = SimpleNamespace(
            available=bool(e["available"]), state=e["state"], fade=e["fade"], ys=ys,
            xl=None if e["xl"] is None else np.array(e["xl"]), xr=None if e["xr"] is None else np.array(e["xr"]),
            obs_l=bool(e["obs_l"]), obs_r=bool(e["obs_r"]))
        lead = None
        if ad["lead"] is not None:
            tid, cls, box, dist = ad["lead"]
            lead = (int(tid), int(cls), np.array(box, np.float32), dist)
        fcw, ldw, rlr = ad["fcw"], ad["ldw"], ad["rlr"]
        fcw_v, ldw_v, held_box = disp.update(t, fcw, ldw, lead)
        rlr_v = disp.update_rlr(t, rlr)
        i = n                                                    # blink counter

        def draw_dets_on(img):
            dv.draw_lanes(img, lane_list, lane_trk, ldw_v["side"] if ldw_v else None, ldw["armed"] if ldw else None)
            dv.draw_detections(img, dets, frame, lead)
            if lead is not None:
                dv.draw_lead(img, lead, None if (fcw_v and fcw_v.get("held")) else fcw_v, i)
            if held_box is not None:
                dv.draw_fcw_box(img, held_box, fcw_v["level"], i)
            dv.draw_alerts(img, fcw_v, ldw_v, i, rlr_v)
            return img

        s = nearest_seg(k, seg_idx)
        manifest["seg"][str(k)] = f"{s:06d}.png"
        seg_map = torch.from_numpy(cv2.imread(str(trace / "seg" / f"{s:06d}.png"), cv2.IMREAD_GRAYSCALE)).to(device)
        det_img = draw_dets_on(frame.copy())
        seg_img, ids = seg.overlay(frame, seg_map, a.seg_alpha)
        dv.draw_legend(seg_img, ids)
        comb_img = draw_dets_on(seg_img.copy())
        for name, img in (("det", det_img), ("seg", seg_img), ("comb", comb_img)):
            cv2.imwrite(str(out / f"{name}_{k:06d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(f"[done] {f1 - f0} frames -> {out}")


if __name__ == "__main__":
    main()
