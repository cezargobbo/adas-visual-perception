"""Export SegFormer (Cityscapes) to a static-shape fp32 ONNX model.

Input  pixel_values (1,3,H,W): RGB, already resized (bilinear, align_corners=False)
       and ImageNet-normalised (mean 0.485,0.456,0.406 / std 0.229,0.224,0.225).
Output logits (1,19,H/4,W/4). The upsample to the frame size and the argmax stay
outside the graph because the frame size varies.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[2]
# model -> (weights dir, width, height)
PROFILES = {
    "b5": ("segformer-b5-cityscapes", 1280, 720),
    "b1": ("segformer-b1-cityscapes", 1024, 576),
}
OPSET = 17


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).logits


def onnx_path(model: str, models_dir: Path) -> Path:
    d, w, h = PROFILES[model]
    return models_dir / f"{d}-{w}x{h}.onnx"


def export(model: str, models_dir: Path) -> Path:
    d, w, h = PROFILES[model]
    net = SegformerForSemanticSegmentation.from_pretrained(str(models_dir / d)).float().eval()
    dst = onnx_path(model, models_dir)
    x = torch.zeros(1, 3, h, w)
    with torch.inference_mode():
        torch.onnx.export(
            _Wrap(net), (x,), str(dst), opset_version=OPSET,
            input_names=["pixel_values"], output_names=["logits"],
            dynamic_axes=None, dynamo=False,
        )
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=[*PROFILES, "all"], default="all")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    for m in (PROFILES if args.model == "all" else [args.model]):
        p = export(m, args.models_dir)
        print(f"{m}: {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
