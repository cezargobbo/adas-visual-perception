#!/usr/bin/env python3
"""Populate models/ with the weights used by the pipeline.

Copies the YOLO and PolarRCNN weights from a source directory and exports the
SegFormer Cityscapes checkpoints from the Hugging Face hub (or its local cache)
with save_pretrained. Prints the sha256 of every file written.

Usage: .venv/bin/python tools/setup_models.py --from /path/to/weights_dir
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

COPIES = [
    ("yolo26s.pt", "yolo26s.pt"),
    ("yolo26x.pt", "yolo26x.pt"),
    ("weights/polarrcnn_curvelanes_dla34.pth", "polarrcnn_curvelanes_dla34.pth"),
]
SEGFORMERS = {
    "segformer-b1-cityscapes": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "segformer-b5-cityscapes": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="directory holding yolo26{s,x}.pt and weights/polarrcnn_curvelanes_dla34.pth")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)

    written = []
    for rel, name in COPIES:
        src = args.src / rel
        if not src.is_file():
            sys.exit(f"missing source file: {src}")
        dst = MODELS / name
        shutil.copy2(src, dst)
        written.append(dst)

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    for name, hub_id in SEGFORMERS.items():
        out = MODELS / name
        SegformerForSemanticSegmentation.from_pretrained(hub_id).save_pretrained(out)
        try:
            SegformerImageProcessor.from_pretrained(hub_id).save_pretrained(out)
        except Exception as e:  # processor files are optional for the pipeline
            print(f"warning: image processor for {hub_id} not saved: {e}")
        written.extend(sorted(p for p in out.iterdir() if p.is_file()))

    for p in written:
        print(f"{sha256(p)}  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
