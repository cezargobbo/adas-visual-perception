"""Export YOLO26 .pt weights to dynamic-shape ONNX with a (1,300,6) output.

Output rows are x1,y1,x2,y2,score,cls in letterbox coordinates, padded to 300.

Two heads exist in Ultralytics 8.4.208 (YOLO26 has both):
  nms  (default): one-to-many head + NMS baked into the graph (export nms=True,
       conf=0.25, iou=0.5). This is what YOLO(.pt).predict()/track() run with
       default settings, so it is the variant that reproduces the reference.
  e2e : NMS-free one-to-one head (export nms=False). Saved as yolo26X_e2e.onnx.
       It is exact against predict(nms=False) but is a different head and
       differs from the default reference output.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
CONF = 0.25  # score floor baked into the NMS graph; filter higher at runtime
IOU = 0.5  # Ultralytics 8.4.208 default for predict/track and export


def export(size: str, models_dir: Path, head: str) -> Path:
    pt = models_dir / f"yolo26{size}.pt"
    kw = dict(nms=True, conf=CONF, iou=IOU) if head == "nms" else dict(nms=False)
    dst = models_dir / (f"yolo26{size}.onnx" if head == "nms" else f"yolo26{size}_e2e.onnx")
    # export() writes the .onnx next to the .pt; use a scratch dir so the two
    # heads never overwrite each other, then move the result into models/.
    with tempfile.TemporaryDirectory() as tmp:
        link = Path(tmp) / pt.name
        link.symlink_to(pt.resolve())
        produced = Path(YOLO(str(link)).export(format="onnx", dynamic=True, opset=17, simplify=True, **kw))
        shutil.move(str(produced), dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--sizes", nargs="+", default=["s", "x"], choices=["s", "x"])
    ap.add_argument("--head", choices=["nms", "e2e"], default="nms")
    args = ap.parse_args()
    for s in args.sizes:
        print("exported", export(s, args.models_dir, args.head))


if __name__ == "__main__":
    main()
