import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference" / "python"))
import lanes_polar  # noqa: E402


def test_decode_empty_returns_empty():
    pts = np.zeros((0, 72, 2), dtype=np.float32)
    endp = np.zeros((0, 2), dtype=np.float32)
    conf = np.zeros((0,), dtype=np.float32)
    assert lanes_polar.decode(pts, endp, conf) == ([], [])
