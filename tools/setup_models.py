#!/usr/bin/env python3
"""Populate models/ with the weights used by the pipeline.

Sources of each file:
  yolo26s.pt, yolo26x.pt   Ultralytics release assets (downloaded automatically by
                           ultralytics: YOLO("yolo26x.pt")).
  polarrcnn_curvelanes_dla34.pth
                           https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_curvelanes_dla34.pth
  segformer-b{1,5}-cityscapes
                           Hugging Face nvidia/segformer-b{1,5}-finetuned-cityscapes-1024-1024,
                           saved with save_pretrained.

Without --from every file is downloaded and the sha256 of the .pt/.pth files is checked
against the values this project was validated with. With --from DIR the YOLO and Polar
weights are copied from DIR (layout: DIR/yolo26s.pt, DIR/yolo26x.pt,
DIR/weights/polarrcnn_curvelanes_dla34.pth) and checked the same way.
Prints the sha256 of every file written.

Usage: .venv/bin/python tools/setup_models.py [--from /path/to/weights_dir]
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

COPIES = [
    ("yolo26s.pt", "yolo26s.pt"),
    ("yolo26x.pt", "yolo26x.pt"),
    ("weights/polarrcnn_curvelanes_dla34.pth", "polarrcnn_curvelanes_dla34.pth"),
]
POLAR_URL = "https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_curvelanes_dla34.pth"
EXPECTED_SHA256 = {
    "yolo26s.pt": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
    "yolo26x.pt": "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
    "polarrcnn_curvelanes_dla34.pth": "938d0b01656d6576ffb96285c5a1d9da79f2c155e66bffa63edfb1ed7c15af0e",
}
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


def fetch_yolo(name: str, dst: Path) -> None:
    """Let ultralytics download the release asset, then move it to dst."""
    import os
    from ultralytics import YOLO

    cwd = os.getcwd()
    os.chdir(MODELS)  # ultralytics saves the download in the current directory
    try:
        YOLO(name)
    finally:
        os.chdir(cwd)
    if not dst.is_file():
        sys.exit(f"ultralytics did not produce {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, default=None,
                    help="optional directory holding yolo26{s,x}.pt and weights/polarrcnn_curvelanes_dla34.pth; "
                         "without it the files are downloaded")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)

    written = []
    for rel, name in COPIES:
        dst = MODELS / name
        if args.src is not None:
            src = args.src / rel
            if not src.is_file():
                sys.exit(f"missing source file: {src}")
            shutil.copy2(src, dst)
        elif name.endswith(".pth"):
            print(f"downloading {POLAR_URL}")
            urllib.request.urlretrieve(POLAR_URL, dst)
        else:
            fetch_yolo(name, dst)
        got = sha256(dst)
        if got != EXPECTED_SHA256[name]:
            sys.exit(f"sha256 mismatch for {name}: expected {EXPECTED_SHA256[name]}, got {got}")
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
