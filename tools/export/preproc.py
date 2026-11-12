"""YOLO preprocessing (letterbox) in Python: the specification of the C++ one.

This is the specification the C++ preprocessing must copy. It is bit-exact
with Ultralytics 8.4.208 ``predictor.preprocess`` for a ``.pt`` model fed one
BGR frame (rectangular letterbox, ``auto=True``, centered, pad value 114,
``cv2.INTER_LINEAR``), followed by BGR->RGB, HWC->CHW and ``/ 255`` in float32.

Steps, for an image of height h and width w and a square target ``imgsz``:
  1. r = min(imgsz / h, imgsz / w)                     (upscaling allowed)
  2. new_w = max(round(w * r), 1), new_h = max(round(h * r), 1)
     (Python round(): half to even)
  3. resize to (new_w, new_h) with INTER_LINEAR (skipped if already that size)
  4. dw = (imgsz - new_w) % stride, dh = (imgsz - new_h) % stride, each halved
  5. top = round(dh - 0.1), bottom = round(dh + 0.1),
     left = round(dw - 0.1), right = round(dw + 0.1)
  6. pad with constant 114 on all three channels
  7. BGR -> RGB, transpose to CHW, float32 / 255
"""
from __future__ import annotations

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(
    img_bgr: np.ndarray, imgsz: int, stride: int = 32
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Return (NCHW float32 RGB tensor in [0,1], scale, (left, top) padding)."""
    h, w = img_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw = (imgsz - new_w) % stride / 2
    dh = (imgsz - new_h) % stride / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    img = img_bgr
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    chw = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    tensor = (chw.astype(np.float32) / np.float32(255.0))[None]
    return tensor, r, (float(left), float(top))
