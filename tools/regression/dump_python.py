#!/usr/bin/env python3
"""Dump per-frame intermediate results of the Python prototype pipeline.

The dump is the recorded Python prototype trace that the C++ replay tests and
compare_dumps.py compare against. It imports
prototype/python/dashcam_vision.py (never modified) and runs the models exactly as
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
                 tracks = what the prototype loop sees afterwards: result.boxes rows that
                          carry a track id (Kalman-smoothed xyxy, id, score, cls).
                 lanes  = frame-pixel polylines (top -> bottom) from PolarLaneDetector.
  seg/NNNNNN.png uint8 class-id map (frame resolution), every --seg-every frames.
  adas.jsonl     only with --adas: one line per frame with the inputs and outputs of the ADAS
                 classes (EgoLaneTracker, LeadVehicleEstimator, FCWMonitor, LDWMonitor,
                 TrafficLightTracker, RLRMonitor), driven exactly like the prototype main loop with
                 the prototype defaults. Schema: tools/regression/README.md.
  meta.json      video, profile, resolution, fps, parameters, library versions (plus an "adas"
                 object with the frame windows derived from fps when --adas is given).
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
sys.path.insert(0, str(ROOT / "prototype" / "python"))

# Defaults of the prototype argparse (dashcam_vision.py main()).
ADAS_DEFAULTS = dict(hfov=75.0, cam_height=1.5, horizon=0.59, corridor_bottom=0.12, corridor_top=0.02,
                     lanes_hold=2.5, fcw_ttc=2.7, fcw_ttc_critical=1.5, ldw_margin=0.25,
                     ldw_car_width=1.8, rlr_zone=(0.25, 0.75))


def jsonable(v):
    """numpy -> lists / Python scalars, None -> null, tuples -> lists. Non-finite floats become the
    strings "inf" / "-inf" / "nan" (plain JSON has no literal for them)."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f:
            return "nan"
        return f if abs(f) != float("inf") else ("inf" if f > 0 else "-inf")
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    raise TypeError(f"cannot serialise {type(v)}")


def ego_state(trk):
    """Public + internal state of the EgoLaneTracker after update()."""
    return {"xl": trk.xl, "xr": trk.xr, "state": trk.state, "available": trk.available,
            "fade": trk.fade, "since": trk.since, "obs_l": trk.obs_l, "obs_r": trk.obs_r,
            "glitch": trk.glitch, "free": dict(trk.free), "raw_l": trk.raw_l, "raw_r": trk.raw_r,
            "wprof": trk.wprof, "pend": {k: [v[0], v[1]] for k, v in trk.pend.items()}}


def tl_tracks_state(trk):
    """TrafficLightTracker tracks: scalars + the last history entry (the history itself, up to 120
    entries, is not dumped; the tracker only reads the last 6 entries and its length)."""
    out = []
    for tr in trk.tracks.values():
        out.append({"id": tr["id"], "lost": tr["lost"], "done": tr["done"], "n_hist": len(tr["hist"]),
                    "last": tr["hist"][-1]})
    return out


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
    ap.add_argument("--adas", action="store_true",
                    help="also write adas.jsonl (inputs/outputs of the ADAS logic classes per frame)")
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

    adas_f = None
    adas_meta = None
    if args.adas:
        A = ADAS_DEFAULTS
        lane_trk = dv.EgoLaneTracker(W, H, fps, A["cam_height"], A["horizon"], A["lanes_hold"])
        fcw_mon = dv.FCWMonitor(W, H, fps, A["fcw_ttc"], A["fcw_ttc_critical"])
        ldw_mon = dv.LDWMonitor(W, H, fps, lane_trk, A["ldw_car_width"], A["ldw_margin"], horizon_frac=A["horizon"])
        rlr_mon = dv.RLRMonitor(W, H, fps, zone=tuple(A["rlr_zone"]))
        lead_est = dv.LeadVehicleEstimator(W, H, A["hfov"], A["cam_height"], A["horizon"],
                                           A["corridor_bottom"], A["corridor_top"], hold_frames=int(fps * 0.3))
        # Spy on the TrafficLightTracker to also record the tracks closed (lost too long) each frame.
        tl_closed = []
        orig_tl_update = rlr_mon.trk.update

        def tl_update_spy(t, dets):
            closed = orig_tl_update(t, dets)
            tl_closed[:] = [c["id"] for c in closed]
            return closed

        rlr_mon.trk.update = tl_update_spy
        adas_meta = {"fps_used": fps, "defaults": {k: list(v) if isinstance(v, tuple) else v for k, v in A.items()},
                     "ego_hold_frames": lane_trk.hold_frames, "ego_persist": lane_trk.persist,
                     "fcw_on_n": fcw_mon.on_n, "fcw_off_n": fcw_mon.off_n,
                     "ldw_on_n": ldw_mon.on_n, "ldw_off_n": ldw_mon.off_n, "ldw_row": int(ldw_mon.row),
                     "lead_hold_frames": lead_est.hold_frames, "tl_max_lost": rlr_mon.trk.max_lost,
                     "ego_ys": lane_trk.ys, "ego_t_row": lane_trk.t_row,
                     "ego_wmin": lane_trk.wmin, "ego_wmax": lane_trk.wmax, "ego_gate": lane_trk.gate}
        adas_f = open(out / "adas.jsonl", "w")
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
            if args.adas:
                # Same glue as the prototype main loop (dashcam_vision.py, "ADAS alerts" block).
                t_s = (start_f + i) / fps
                lane_trk.update(lane_list)
                ego = ego_state(lane_trk)
                ldets = []
                if res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    conf = res.boxes.conf.cpu().numpy()
                    ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(cls)
                    ldets = [(ids[k], cls[k], xyxy[k], conf[k]) for k in range(len(cls))]
                lead_in = [(t, c, b) for t, c, b, _ in ldets if t is not None]
                lead = lead_est.update(lead_in, lane_trk)
                lead_f = lead
                if lead is not None and lane_trk.available and not lane_trk.contains(lead[2], margin=0.0):
                    lead_f = None
                fcw = fcw_mon.update(t_s, lead_f)
                ldw = ldw_mon.update(t_s)
                tl_boxes = [(b, c) for _, cl, b, c in ldets if cl == 9]
                tl_in = [{"box": b, "score": c, "state": dv.traffic_light_state(frame, b)} for b, c in tl_boxes]
                rlr = rlr_mon.update(t_s, frame, tl_boxes)
                arec = {"i": i, "t": t_s, "lanes_in": lane_list, "ego": ego,
                        "lead_in": lead_in, "lead": lead, "fcw_lead_in": lead_f, "fcw": fcw,
                        "ldw": dict(ldw, mon={"fail": ldw_mon.fail, "m": ldw_mon.m, "stats": dict(ldw_mon.stats),
                                              "last_v": ldw_mon.last_v, "n_cross": ldw_mon.n_cross}),
                        "tl_in": tl_in, "tl_tracks": tl_tracks_state(rlr_mon.trk), "tl_closed": list(tl_closed),
                        "rlr": rlr}
                adas_f.write(json.dumps(jsonable(arec), separators=(",", ":")) + "\n")
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
    if adas_f:
        adas_f.close()

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
    if adas_meta is not None:
        meta["adas"] = jsonable(adas_meta)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {n_frames} frames -> {out}")


if __name__ == "__main__":
    main()
