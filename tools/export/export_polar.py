"""Export Polar R-CNN (CurveLanes, DLA34, NMS-free) to a static-shape fp32 ONNX model.

Input  image (1,3,320,800): RGB in [0,1]; the top round(H*640/1440) rows of the frame
       are cropped and resized (linear) to 800x320.
Outputs raw per-prior results for the 50 top-scoring priors, BEFORE the confidence
threshold (thresholding and lanes_polar.decode stay on the host):
       pts  (1,50,72,2)  lane points in 800x320 crop pixels, bottom -> top
       endp (1,50,2)     normalised start/end offsets
       o2o  (1,50)       one-to-one confidence
The prior top-k (k=50) is inside the graph.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "python"))
import lanes_polar as lp  # noqa: E402

OPSET = 17
WEIGHTS = ROOT / "models" / "polarrcnn_curvelanes_dla34.pth"


def onnx_path(models_dir: Path) -> Path:
    return models_dir / "polarrcnn_curvelanes_dla34.onnx"


class PolarRCNNRaw(nn.Module):
    """PolarRCNN.forward + GlobalPolarHead.forward without the `keep` filtering.

    Re-uses the loaded submodules (backbone, neck, rpn_head, roi_head.rcnn_head); only the
    final part of GlobalPolarHead.forward is re-expressed with traceable torch ops (the
    reference ends with a boolean mask + .cpu().numpy()). Reference fp32 path (no autocast).
    """

    def __init__(self, net: lp.PolarRCNN):
        super().__init__()
        self.net = net

    def forward(self, img):
        net = self.net
        feats = [f.float() for f in net.neck(net.backbone(img)[1:])]
        emb, ids = net.rpn_head(feats)                       # includes torch.topk(k=50)
        head = net.roi_head
        feat_list = list(feats)[::-1]
        ang = emb[..., 0] * math.pi
        rho = emb[..., 1] * lp.IMG_W
        cy = head.car_y.view(1, 1, -1)
        cx = -torch.tan(ang).unsqueeze(-1) * cy + (rho / torch.cos(ang)).unsqueeze(-1)
        samples_car = torch.stack((cx, cy.expand_as(cx)), -1)
        img_s = lp.car2img(samples_car)
        grid = img_s[..., head.sample_index, :].clone()
        grid[..., 0] /= lp.IMG_W
        grid[..., 1] /= lp.IMG_H
        base = torch.flip(samples_car, dims=[-2])
        _cls, reg, o2o = head.rcnn_head(feat_list, grid, ids, emb)
        endp, xs_off = reg[..., 0:2], reg[..., 2:]
        base_x = base[..., 0] + xs_off * lp.IMG_W
        pts = lp.car2img(torch.stack((base_x, base[..., 1]), -1))
        return pts, endp, o2o


def load_raw(weights: Path = WEIGHTS) -> PolarRCNNRaw:
    return PolarRCNNRaw(lp.load_model(str(weights), torch.device("cpu"))).eval()


def export(models_dir: Path) -> Path:
    model = load_raw(models_dir / WEIGHTS.name)
    dst = onnx_path(models_dir)
    x = torch.zeros(1, 3, lp.IMG_H, lp.IMG_W)
    with torch.inference_mode():
        torch.onnx.export(
            model, (x,), str(dst), opset_version=OPSET,
            input_names=["image"], output_names=["pts", "endp", "o2o"],
            dynamic_axes=None, dynamo=False,
        )
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    p = export(args.models_dir)
    print(f"{p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
