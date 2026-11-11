"""Validate an exported YOLO26 end-to-end model against Ultralytics .pt inference.

Reference: YOLO(.pt).predict(frame, imgsz, conf=0.30, classes=CLASSES) in fp32 is the
pass/fail gate; the quantize=16 reference (as track() runs) is report only. The candidate
(ONNX Runtime or TensorRT) runs on tensors from preproc.letterbox and its
(1,300,6) output is filtered (score > conf, class in CLASSES) and mapped back
to frame coordinates.
"""
from __future__ import annotations

import argparse
import importlib
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames as frames_mod  # noqa: E402
import preproc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLASSES = [0, 1, 2, 3, 5, 7, 9]
CONF = 0.30
IOU_MATCH = 0.9
MAX_SCORE_DIFF = 0.02
ORT_MIN_RP = 0.98  # ORT fp32 vs fp32 reference, per resolution
TRT_POOLED_RP = 0.97  # TRT fp16 vs fp32 reference, pooled over resolutions
TRT_EACH_RP = 0.95  # ... and every resolution


def make_runner(backend: str, model: str):
    if backend == "ort":
        import onnxruntime as ort

        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        sess = ort.InferenceSession(str(ROOT / "models" / f"yolo26{model}.onnx"), providers=prov)
        name = sess.get_inputs()[0].name
        return lambda x: sess.run(None, {name: x})[0]
    # trt: provided by a later task (tools/export/trt_runner.py)
    try:
        trt_runner = importlib.import_module("trt_runner")
    except ModuleNotFoundError:
        sys.exit(
            "--backend trt needs tools/export/trt_runner.py, which does not exist yet "
            "(it is created in a later task). Use --backend ort for now."
        )
    return trt_runner.load_runner(model)


def decode(out: np.ndarray, scale: float, pad: tuple[float, float], shape) -> np.ndarray:
    """(1,300,6) letterbox-space output -> rows [x1,y1,x2,y2,score,cls] in frame coords."""
    d = out[0]
    d = d[(d[:, 4] > CONF) & np.isin(d[:, 5].astype(int), CLASSES)].copy()
    left, top = pad
    d[:, [0, 2]] = ((d[:, [0, 2]] - left) / scale).clip(0, shape[1])
    d[:, [1, 3]] = ((d[:, [1, 3]] - top) / scale).clip(0, shape[0])
    return d


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def match(ref: np.ndarray, cand: np.ndarray):
    """Greedy per-class matching (reference by descending score). Returns score diffs."""
    used = np.zeros(len(cand), bool)
    diffs = []
    for i in np.argsort(-ref[:, 4]):
        r = ref[i]
        ok = (~used) & (cand[:, 5] == r[5])
        if not ok.any():
            continue
        v = np.where(ok, iou(r, cand), -1.0)
        j = int(v.argmax())
        if v[j] >= IOU_MATCH:
            used[j] = True
            diffs.append(abs(float(r[4]) - float(cand[j, 4])))
    return diffs


def rp(nr, nc, nm):
    return (nm / nr if nr else 1.0), (nm / nc if nc else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["ort", "trt"], required=True)
    ap.add_argument("--model", choices=["s", "x"], required=True)
    ap.add_argument("--frames", type=int, default=30, help="frames per video")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280])
    ap.add_argument("--videos-dir", default=os.environ.get(frames_mod.ENV_VAR),
                    help=f"directory with the test videos (or set ${frames_mod.ENV_VAR})")
    args = ap.parse_args()
    if not args.videos_dir:
        ap.error(f"--videos-dir is required (or set ${frames_mod.ENV_VAR})")

    runner = make_runner(args.backend, args.model)
    from ultralytics import YOLO

    det = YOLO(str(ROOT / "models" / f"yolo26{args.model}.pt"))
    samples = frames_mod.sample(args.videos_dir, args.frames)
    print(f"backend={args.backend} model=yolo26{args.model} frames/video={args.frames} conf={CONF} IoU>={IOU_MATCH}")
    if args.backend == "ort":
        print(f"GATE (fp32 reference): per resolution recall,precision >= {ORT_MIN_RP}, median|dscore| <= {MAX_SCORE_DIFF}")
    else:
        print(f"GATE (fp32 reference): pooled recall,precision >= {TRT_POOLED_RP}, every resolution >= {TRT_EACH_RP}, "
              f"median|dscore| <= {MAX_SCORE_DIFF}")
    print("quantize=16 reference tables are REPORT ONLY (fp16 noise in the reference); they never fail the run.")

    def ref_dets(frame, imgsz, quant):
        kw = {"quantize": 16} if quant else {}
        b = det.predict(frame, imgsz=imgsz, conf=CONF, classes=CLASSES, verbose=False, **kw)[0].boxes
        if not len(b):
            return np.zeros((0, 6), np.float32)
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)

    all_ok = True
    for imgsz in args.imgsz:
        stats = {q: defaultdict(lambda: [0, 0, 0, []]) for q in (False, True)}  # nref, ncand, nmatch, diffs
        for _name, _idx, frame in samples:
            tensor, scale, pad = preproc.letterbox(frame, imgsz)
            cand = decode(runner(tensor), scale, pad, frame.shape)
            for q in (False, True):
                ref = ref_dets(frame, imgsz, q)
                diffs = match(ref, cand)
                s = stats[q][(frame.shape[1], frame.shape[0])]
                s[0] += len(ref); s[1] += len(cand); s[2] += len(diffs); s[3] += diffs
        for q in (False, True):
            gate = not q
            label = "quantize=16 reference (REPORT ONLY)" if q else "fp32 reference (GATE)"
            print(f"\nimgsz={imgsz}  vs {label}")
            print(f"{'resolution':>11} {'ref':>6} {'cand':>6} {'match':>6} {'recall':>7} {'prec':>7} {'med|ds|':>8}  result")
            rows = sorted(stats[q].items(), reverse=True)
            tot = [0, 0, 0, []]
            for (w, h), (nr, nc, nm, d) in rows:
                rec, prec = rp(nr, nc, nm)
                med = statistics.median(d) if d else 0.0
                for k in range(3):
                    tot[k] += (nr, nc, nm)[k]
                tot[3] += d
                if not gate:
                    res = "-"
                elif args.backend == "ort":
                    res = "PASS" if min(rec, prec) >= ORT_MIN_RP and med <= MAX_SCORE_DIFF else "FAIL"
                else:
                    res = "PASS" if min(rec, prec) >= TRT_EACH_RP and med <= MAX_SCORE_DIFF else "FAIL"
                if gate:
                    all_ok &= res == "PASS"
                print(f"{w:>5}x{h:<5} {nr:>6} {nc:>6} {nm:>6} {rec:>7.4f} {prec:>7.4f} {med:>8.5f}  {res}")
            rec, prec = rp(*tot[:3])
            med = statistics.median(tot[3]) if tot[3] else 0.0
            pooled = ""
            if gate and args.backend == "trt":
                pooled = "PASS" if min(rec, prec) >= TRT_POOLED_RP and med <= MAX_SCORE_DIFF else "FAIL"
                all_ok &= pooled == "PASS"
            print(f"{'pooled':>11} {tot[0]:>6} {tot[1]:>6} {tot[2]:>6} {rec:>7.4f} {prec:>7.4f} {med:>8.5f}  {pooled or '-'}")
    print("\nOVERALL (fp32 gate):", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
