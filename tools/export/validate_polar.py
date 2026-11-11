"""Validate the exported Polar R-CNN ONNX model against the reference PolarLaneDetector.

Reference: lanes_polar.PolarLaneDetector (fp32, thr=0.44, as in the pipeline).
Candidate (ONNX Runtime or TensorRT): same crop/resize preprocessing, raw outputs
(pts, endp, o2o) from the backend, o2o >= 0.44 threshold, lanes_polar.decode and the
same conversion to frame pixels.
Frames: `frames.sample` over the 3 test videos plus INTERSECTION_FRAMES frames of
sp1_crop_30fps.mp4 between 58 s and 62 s (intersection, no lane markings).
Gates: same lane count in >= 98% of frames; mean point-to-point distance <= 2 px
(1920x1080 frame) over matched lanes.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames as frames_mod  # noqa: E402
from export_polar import onnx_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference" / "python"))
import lanes_polar as lp  # noqa: E402

MIN_COUNT_MATCH = 98.0   # % of frames with the same lane count
MAX_MEAN_DIST_PX = 2.0   # on a 1920x1080 frame
INTERSECTION_FRAMES = 10
INTERSECTION_RANGE_S = (58.0, 62.0)
N_RESAMPLE = 72


def make_runner(backend: str, model: str = "polar"):
    """Return a callable image(1,3,320,800) fp32 -> (pts, endp, o2o) numpy arrays."""
    if backend == "ort":
        import onnxruntime as ort

        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        try:
            sess = ort.InferenceSession(str(onnx_path(ROOT / "models")), providers=prov)
        except Exception:  # CUDA provider may fail to load: CPU is accepted
            sess = ort.InferenceSession(str(onnx_path(ROOT / "models")), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return lambda x: sess.run(None, {name: x})
    try:
        trt_runner = importlib.import_module("trt_runner")
    except ModuleNotFoundError:
        sys.exit(
            "--backend trt needs tools/export/trt_runner.py, which does not exist yet "
            "(it is created in a later task). Use --backend ort for now."
        )
    return trt_runner.load_runner(model)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """Same as PolarLaneDetector.__call__: top crop, BGR->RGB, linear resize, [0,1]."""
    h = frame_bgr.shape[0]
    y_off = int(round(h * lp.CUT_FRAC))
    rgb = cv2.cvtColor(frame_bgr[y_off:], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (lp.IMG_W, lp.IMG_H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def candidate_lanes(runner, frame_bgr: np.ndarray, thr: float = lp.CONF_NMSFREE):
    h, w = frame_bgr.shape[:2]
    pts, endp, o2o = runner(preprocess(frame_bgr))
    keep = o2o[0] >= thr
    lanes, scores = lp.decode(pts[0][keep], endp[0][keep], o2o[0][keep])
    return [np.stack((p[:, 0] * w, p[:, 1] * h), 1).astype(np.float32) for p in lanes], scores


def _resample(p: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    t = np.linspace(0, len(p) - 1, n)
    return np.stack([np.interp(t, np.arange(len(p)), p[:, k]) for k in range(2)], 1)


def match_distance(ref: list, cand: list) -> list[float]:
    """Greedy nearest matching (mean point distance after resampling); one value per matched pair."""
    if not ref or not cand:
        return []
    R, C = [_resample(p) for p in ref], [_resample(p) for p in cand]
    cost = np.array([[np.linalg.norm(r - c, axis=1).mean() for c in C] for r in R])
    out = []
    for _ in range(min(len(R), len(C))):
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        out.append(float(cost[i, j]))
        cost[i, :] = np.inf
        cost[:, j] = np.inf
    return out


def intersection_frames(videos_dir, n: int):
    path = Path(videos_dir) / "sp1_crop_30fps.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out = []
    for t in np.linspace(*INTERSECTION_RANGE_S, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{path}: could not read frame at {t:.2f}s")
        out.append((f"{path.name}@{t:.1f}s", int(round(t * 1000)), frame))
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["ort", "trt"], required=True)
    ap.add_argument("--frames", type=int, default=40,
                    help=f"total frames incl. {INTERSECTION_FRAMES} intersection frames")
    ap.add_argument("--videos-dir", default=os.environ.get(frames_mod.ENV_VAR),
                    help=f"directory with the test videos (or set ${frames_mod.ENV_VAR})")
    args = ap.parse_args()
    if not args.videos_dir:
        ap.error(f"--videos-dir is required (or set ${frames_mod.ENV_VAR})")

    runner = make_runner(args.backend)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = lp.PolarLaneDetector(device=dev, fp16=False, verbose=False)

    n_inter = min(INTERSECTION_FRAMES, args.frames)
    per_video = max(1, -(-(args.frames - n_inter) // len(frames_mod.VIDEOS)))
    samples = frames_mod.sample(args.videos_dir, per_video) + intersection_frames(args.videos_dir, n_inter)
    print(f"backend={args.backend} frames={len(samples)} (incl. {n_inter} intersection) "
          f"gate: count match >= {MIN_COUNT_MATCH}% and mean dist <= {MAX_MEAN_DIST_PX}px")

    rows = []
    dists: list[float] = []
    by_res = defaultdict(list)
    print(f"{'frame':<34} {'ref':>3} {'cand':>4} {'mean px':>8}")
    for name, idx, frame in samples:
        ref, _ = det(frame)
        cand, _ = candidate_lanes(runner, frame)
        d = match_distance(ref, cand)
        dists += d
        same = len(ref) == len(cand)
        by_res[frame.shape[1::-1]].append(same)
        rows.append(same)
        tag = f"{name}#{idx}" if "@" not in name else name
        print(f"{tag:<34} {len(ref):>3} {len(cand):>4} {(np.mean(d) if d else float('nan')):>8.3f}")

    pct = 100.0 * sum(rows) / len(rows)
    mean_d = float(np.mean(dists)) if dists else 0.0
    max_d = float(np.max(dists)) if dists else 0.0
    for (fw, fh), v in sorted(by_res.items(), reverse=True):
        print(f"  {fw}x{fh}: count match {sum(v)}/{len(v)}")
    ok = pct >= MIN_COUNT_MATCH and mean_d <= MAX_MEAN_DIST_PX
    print(f"count match {pct:.1f}%  matched lanes={len(dists)}  mean dist {mean_d:.4f}px  max lane dist {max_d:.4f}px")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
