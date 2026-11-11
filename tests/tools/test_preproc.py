import os

import numpy as np
import pytest
import torch
from ultralytics import YOLO

import frames
import preproc

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
EXPECTED = {
    ((1080, 1920), 1280): (1, 3, 736, 1280),
    ((474, 854), 1280): (1, 3, 736, 1280),
    ((1080, 1920), 960): (1, 3, 544, 960),
    ((474, 854), 960): (1, 3, 544, 960),
}


@pytest.fixture(scope="module")
def sample_frames():
    if not os.environ.get(frames.ENV_VAR):
        pytest.skip(f"set {frames.ENV_VAR} to the videos directory")
    return frames.sample(frames.videos_dir_from_env(), 2)


def ultralytics_tensor(imgsz, frame):
    """Tensor built by Ultralytics' own predictor.preprocess (.pt, fp32, cpu)."""
    model = YOLO(str(ROOT / "models" / "yolo26s.pt"))
    model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
    return model.predictor.preprocess([frame]).numpy()


@pytest.mark.parametrize("imgsz", [960, 1280])
def test_letterbox_matches_ultralytics(sample_frames, imgsz):
    assert len(sample_frames) == 6
    seen_shapes = set()
    for name, idx, frame in sample_frames:
        ref = ultralytics_tensor(imgsz, frame)
        got, scale, (left, top) = preproc.letterbox(frame, imgsz)
        assert got.dtype == np.float32
        assert got.shape == ref.shape, (name, idx)
        assert np.array_equal(got, ref), (name, idx)
        key = (frame.shape[:2], imgsz)
        if key in EXPECTED:
            assert got.shape == EXPECTED[key]
        seen_shapes.add(key)
        assert scale == min(imgsz / frame.shape[0], imgsz / frame.shape[1])
    assert len(seen_shapes) == 3


def test_letterbox_padding_and_scale_values():
    img = np.zeros((474, 854, 3), np.uint8)
    out, scale, (left, top) = preproc.letterbox(img, 1280)
    assert out.shape == (1, 3, 736, 1280)
    assert (left, top) == (0.0, 13.0) or (left, top) == (0, 13)
    assert scale == pytest.approx(1280 / 854)


def test_frames_sample_is_deterministic():
    if not os.environ.get(frames.ENV_VAR):
        pytest.skip(f"set {frames.ENV_VAR} to the videos directory")
    a = frames.sample(frames.videos_dir_from_env(), 2)
    b = frames.sample(frames.videos_dir_from_env(), 2)
    assert [(n, i) for n, i, _ in a] == [(n, i) for n, i, _ in b]
    assert all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))
