"""Deterministic frame sampling shared by the export validation scripts."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

SEED = 1234

# (path relative to videos_dir, expected (width, height))
VIDEOS = (
    ("sp1_crop_30fps.mp4", (1920, 1080)),
    ("redlights/red_crop_5_30fps.mp4", (1280, 720)),
    ("redlights/red_crop_9_30fps.mp4", (854, 474)),
)


ENV_VAR = "ADASVP_VIDEOS_DIR"


def videos_dir_from_env() -> str:
    """Videos directory from $ADASVP_VIDEOS_DIR; fail clearly if unset."""
    d = os.environ.get(ENV_VAR)
    if not d:
        raise SystemExit(f"videos directory not given: pass --videos-dir or set ${ENV_VAR}")
    return d


def sample(videos_dir, n_per_video: int) -> list[tuple[str, int, np.ndarray]]:
    """Return (video_name, frame_index, BGR frame) triples.

    Frame indices are drawn with a fixed seed (sorted, without replacement), so
    the same call always returns the same frames.
    """
    out: list[tuple[str, int, np.ndarray]] = []
    for rel, (w, h) in VIDEOS:
        path = Path(videos_dir) / rel
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rng = np.random.default_rng(SEED)
        idxs = sorted(rng.choice(n, size=min(n_per_video, n), replace=False).tolist())
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{path}: could not read frame {i}")
            if frame.shape[1] != w or frame.shape[0] != h:
                raise ValueError(f"{path}: expected {w}x{h}, got {frame.shape[1]}x{frame.shape[0]}")
            out.append((path.name, i, frame))
        cap.release()
    return out
