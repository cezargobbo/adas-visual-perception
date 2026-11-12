"""Validate an exported SegFormer ONNX model against the reference Segmenter.

Reference: Segmenter.__call__ from prototype/python/dashcam_vision.py (fp16 PyTorch
on GPU, as in the pipeline). Candidate (ONNX Runtime or TensorRT): the same
preprocessing, logits from the backend at H/4, then the same bilinear upsample to
the frame size and argmax in torch. Metric: % of pixels with the same class.
Gate: >= 99.5% (ORT) / >= 99% (TRT fp16), per resolution and pooled.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames as frames_mod  # noqa: E402
from export_segformer import PROFILES, onnx_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "python"))
MIN_AGREE = {"ort": 99.5, "trt": 99.0}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_runner(backend: str, model: str, fp16: bool = True):
    if backend == "ort":
        import onnxruntime as ort

        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        sess = ort.InferenceSession(str(onnx_path(model, ROOT / "models")), providers=prov)
        name = sess.get_inputs()[0].name
        return lambda x: sess.run(None, {name: x})[0]
    try:
        trt_runner = importlib.import_module("trt_runner")
    except ImportError as e:  # e.g. the tensorrt wheel is not installed
        sys.exit(
            f"--backend trt could not import tools/export/trt_runner.py: {e!r}\n"
            "Install the requirements (uv pip install -r requirements.txt) or use --backend ort."
        )
    return trt_runner.load_runner(model, fp16=fp16)


def preprocess(frame_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """BGR uint8 frame -> (1,3,h,w) fp32 RGB, bilinear resize, ImageNet-normalised."""
    x = torch.from_numpy(frame_bgr)[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["ort", "trt"], required=True)
    ap.add_argument("--model", choices=list(PROFILES), required=True)
    ap.add_argument("--frames", type=int, default=20, help="total frames, spread over the 3 videos")
    ap.add_argument("--videos-dir", default=os.environ.get(frames_mod.ENV_VAR),
                    help=f"directory with the test videos (or set ${frames_mod.ENV_VAR})")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                    help="TensorRT engine precision (--backend trt only)")
    args = ap.parse_args()
    if not args.videos_dir:
        ap.error(f"--videos-dir is required (or set ${frames_mod.ENV_VAR})")

    runner = make_runner(args.backend, args.model, fp16=args.precision == "fp16")
    from dashcam_vision import Segmenter

    d, w, h = PROFILES[args.model]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = Segmenter(str(ROOT / "models" / d), dev, w, h, fp16=True)

    per_video = max(1, -(-args.frames // len(frames_mod.VIDEOS)))
    samples = frames_mod.sample(args.videos_dir, per_video)
    gate = MIN_AGREE[args.backend]
    print(f"backend={args.backend} model={args.model} seg={w}x{h} frames={len(samples)} gate>={gate}% same-class pixels")

    stats = defaultdict(lambda: [0, 0, 100.0])  # same, total, worst frame
    for _name, _idx, frame in samples:
        fh, fw = frame.shape[:2]
        ref = seg(frame).cpu()
        logits = torch.from_numpy(runner(preprocess(frame, w, h)))
        cand = F.interpolate(logits.float(), size=(fh, fw), mode="bilinear", align_corners=False).argmax(1)[0].to(torch.uint8)
        same = int((ref == cand).sum())
        s = stats[(fw, fh)]
        s[0] += same
        s[1] += ref.numel()
        s[2] = min(s[2], 100.0 * same / ref.numel())

    ok = True
    print(f"{'resolution':>11} {'agree %':>9} {'worst %':>9}  result")
    tot = [0, 0]
    for (fw, fh), (same, total, worst) in sorted(stats.items(), reverse=True):
        pct = 100.0 * same / total
        res = "PASS" if pct >= gate else "FAIL"
        ok &= res == "PASS"
        tot[0] += same
        tot[1] += total
        print(f"{fw:>5}x{fh:<5} {pct:>9.4f} {worst:>9.4f}  {res}")
    pooled = 100.0 * tot[0] / tot[1]
    ok &= pooled >= gate
    print(f"{'pooled':>11} {pooled:>9.4f}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
