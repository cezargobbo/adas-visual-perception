"""Validate the exported Polar R-CNN ONNX model against the reference PolarLaneDetector.

Reference: lanes_polar.PolarLaneDetector (fp32, thr=0.44, as in the pipeline).
Candidate (ONNX Runtime or TensorRT): same crop/resize preprocessing, raw outputs
(pts, endp, o2o) from the backend, o2o >= 0.44 threshold, lanes_polar.decode and the
same conversion to frame pixels.
Frames: `frames.sample` over the 3 test videos plus INTERSECTION_FRAMES frames of
sp1_crop_30fps.mp4 between 58 s and 62 s (intersection, no lane markings).
Lane distance: |dx| at common y values over the overlapping y-range (np.interp, sorted by y).
Gates: same lane count in >= 98% of frames; mean |dx| <= 2 px over matched lanes
(dx is measured in the pixels of each frame's own resolution, not rescaled); per-lane max |dx| <= 5 px; |n_ref - n_cand| <= 1 point for every
matched lane. Lanes with a point-count difference or mean |dx| > 2 px are listed.
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
MAX_MEAN_DIST_PX = 2.0   # dx in each frame's own pixels (no rescaling)
MAX_LANE_DIST_PX = 5.0
MAX_POINT_DIFF = 1
INTERSECTION_FRAMES = 10
INTERSECTION_RANGE_S = (58.0, 62.0)
N_RESAMPLE = 72


def make_runner(backend: str, model: str = "polar", fp16: bool = True):
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
    except ImportError as e:  # e.g. the tensorrt wheel is not installed
        sys.exit(
            f"--backend trt could not import tools/export/trt_runner.py: {e!r}\n"
            "Install the requirements (uv pip install -r requirements.txt) or use --backend ort."
        )
    return trt_runner.load_runner(model, fp16=fp16)


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


def lane_dx(a: np.ndarray, b: np.ndarray):
    """|dx| of two lanes at common y values over their overlapping y-range.

    Both lanes are sorted by y and x is linearly interpolated at the union of both
    lanes' y values that fall inside the overlap. Returns the
    |dx| array, or None if the y-ranges do not overlap.
    """
    a, b = a[np.argsort(a[:, 1])], b[np.argsort(b[:, 1])]
    lo, hi = max(a[0, 1], b[0, 1]), min(a[-1, 1], b[-1, 1])
    if hi <= lo:
        return None
    ys = np.union1d(a[:, 1], b[:, 1])
    ys = ys[(ys >= lo) & (ys <= hi)]
    return np.abs(np.interp(ys, a[:, 1], a[:, 0]) - np.interp(ys, b[:, 1], b[:, 0]))


def match_lanes(ref: list, cand: list) -> list[dict]:
    """Greedy matching by mean |dx| over the common y-range; one record per matched pair."""
    if not ref or not cand:
        return []
    cost = np.full((len(ref), len(cand)), np.inf)
    dxs = {}
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            d = lane_dx(r, c)
            if d is not None:
                dxs[i, j] = d
                cost[i, j] = d.mean()
    out = []
    for _ in range(min(len(ref), len(cand))):
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        if not np.isfinite(cost[i, j]):
            break
        r, c = ref[i], cand[j]
        ext = lambda p: float(p[:, 1].max() - p[:, 1].min())  # noqa: E731
        out.append(dict(n_ref=len(r), n_cand=len(c), mean=float(dxs[i, j].mean()),
                        max=float(dxs[i, j].max()), ext_diff=abs(ext(r) - ext(c))))
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
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                    help="TensorRT engine precision (--backend trt only)")
    args = ap.parse_args()
    if not args.videos_dir:
        ap.error(f"--videos-dir is required (or set ${frames_mod.ENV_VAR})")

    runner = make_runner(args.backend, fp16=args.precision == "fp16")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = lp.PolarLaneDetector(device=dev, fp16=False, verbose=False)

    n_inter = min(INTERSECTION_FRAMES, args.frames)
    per_video = max(1, -(-(args.frames - n_inter) // len(frames_mod.VIDEOS)))
    samples = frames_mod.sample(args.videos_dir, per_video) + intersection_frames(args.videos_dir, n_inter)
    print(f"backend={args.backend} frames={len(samples)} (incl. {n_inter} intersection) "
          f"gate: count match >= {MIN_COUNT_MATCH}% and mean dist <= {MAX_MEAN_DIST_PX}px")

    rows = []
    lanes_rec: list[dict] = []
    flagged: list[str] = []
    by_res = defaultdict(list)
    print(f"{'frame':<34} {'ref':>3} {'cand':>4} {'mean px':>8}")
    for name, idx, frame in samples:
        ref, _ = det(frame)
        cand, _ = candidate_lanes(runner, frame)
        recs = match_lanes(ref, cand)
        tag = f"{name}#{idx}" if "@" not in name else name
        for r in recs:
            if r["n_ref"] != r["n_cand"] or r["mean"] > MAX_MEAN_DIST_PX:
                flagged.append(f"  {tag}: n_ref={r['n_ref']} n_cand={r['n_cand']} "
                               f"mean|dx|={r['mean']:.3f}px max|dx|={r['max']:.3f}px "
                               f"y-extent diff={r['ext_diff']:.3f}px")
        lanes_rec += recs
        same = len(ref) == len(cand)
        by_res[frame.shape[1::-1]].append(same)
        rows.append(same)
        m = np.mean([r["mean"] for r in recs]) if recs else float("nan")
        print(f"{tag:<34} {len(ref):>3} {len(cand):>4} {m:>8.3f}")

    pct = 100.0 * sum(rows) / len(rows)
    means = [r["mean"] for r in lanes_rec]
    mean_d = float(np.mean(means)) if means else 0.0
    max_d = max((r["max"] for r in lanes_rec), default=0.0)
    max_pt = max((abs(r["n_ref"] - r["n_cand"]) for r in lanes_rec), default=0)
    max_ext = max((r["ext_diff"] for r in lanes_rec), default=0.0)
    for (fw, fh), v in sorted(by_res.items(), reverse=True):
        print(f"  {fw}x{fh}: count match {sum(v)}/{len(v)}")
    print(f"lanes with point-count difference or mean |dx| > {MAX_MEAN_DIST_PX}px: {len(flagged)}")
    print("\n".join(flagged))
    ok = (pct >= MIN_COUNT_MATCH and mean_d <= MAX_MEAN_DIST_PX
          and max_d <= MAX_LANE_DIST_PX and max_pt <= MAX_POINT_DIFF)
    print(f"count match {pct:.1f}%  matched lanes={len(lanes_rec)}  mean |dx| {mean_d:.4f}px  "
          f"worst lane max |dx| {max_d:.4f}px  max point diff {max_pt}  max y-extent diff {max_ext:.3f}px")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
