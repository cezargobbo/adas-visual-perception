#!/usr/bin/env python3
"""Compare a C++ dump (adasvp_dump) against a Python reference dump (dump_reference.py).

Usage: compare_dumps.py REF_DIR CPP_DIR

Gates (per video; exit code 0 only if all pass):
  dets   recall and precision >= 0.97 (greedy IoU >= 0.9, same class)
  tracks id consistency >= 0.95: among tracks matched by IoU >= 0.9, the fraction whose C++ id equals the
         Python id under the video's majority mapping (each C++ id -> its most frequent Python id)
  lanes  same lane count on >= 98% of frames, mean |dx| <= 2 px, point-count difference <= 1
  seg    >= 99% equal pixels in every PNG
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "export"))
from validate_polar import match_lanes  # noqa: E402


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: np.clip(r[:, 2] - r[:, 0], 0, None) * np.clip(r[:, 3] - r[:, 1], 0, None)  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def greedy_match(ref: np.ndarray, cpp: np.ndarray, ref_cls, cpp_cls, thr=0.9):
    """Greedy by descending IoU; returns list of (ref_idx, cpp_idx)."""
    iou = iou_matrix(ref, cpp)
    if iou.size:
        iou = iou * (np.asarray(ref_cls)[:, None] == np.asarray(cpp_cls)[None, :])
    pairs = []
    while iou.size and iou.max() >= thr:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        pairs.append((int(i), int(j)))
        iou[i, :] = 0
        iou[:, j] = 0
    return pairs


def load(d: Path):
    return [json.loads(l) for l in open(d / "frames.jsonl")]


def compare(ref_dir: Path, cpp_dir: Path) -> dict:
    ref, cpp = load(ref_dir), load(cpp_dir)
    n = min(len(ref), len(cpp))
    res = {"frames": f"{len(ref)}/{len(cpp)}"}
    tp = nref = ncpp = 0
    id_pairs = []
    lane_frames_eq = lane_frames = 0
    dx_all, pt_diff_max, pt_diffs = [], 0, []
    for k in range(n):
        r, c = ref[k], cpp[k]
        # dets
        ra = np.array([d[:4] for d in r["dets"]]).reshape(-1, 4); rc = [int(d[5]) for d in r["dets"]]
        ca = np.array([d[:4] for d in c["dets"]]).reshape(-1, 4); cc = [int(d[5]) for d in c["dets"]]
        tp += len(greedy_match(ra, ca, rc, cc)); nref += len(ra); ncpp += len(ca)
        # tracks
        ra = np.array([t[:4] for t in r["tracks"]]).reshape(-1, 4); rc = [int(t[6]) for t in r["tracks"]]
        ca = np.array([t[:4] for t in c["tracks"]]).reshape(-1, 4); cc = [int(t[6]) for t in c["tracks"]]
        for i, j in greedy_match(ra, ca, rc, cc):
            id_pairs.append((int(c["tracks"][j][4]), int(r["tracks"][i][4])))
        # lanes
        lane_frames += 1
        lane_frames_eq += len(r["lanes"]) == len(c["lanes"])
        rl = [np.array(l, dtype=np.float64) for l in r["lanes"]]
        cl = [np.array(l, dtype=np.float64) for l in c["lanes"]]
        for m in match_lanes(rl, cl):
            dx_all.append(m["mean"]); pt_diffs.append(abs(m["n_ref"] - m["n_cand"]))
    res["det_recall"] = tp / nref if nref else 1.0
    res["det_precision"] = tp / ncpp if ncpp else 1.0
    maj = {}
    by = defaultdict(Counter)
    for ci, ri in id_pairs:
        by[ci][ri] += 1
    for ci, cnt in by.items():
        maj[ci] = cnt.most_common(1)[0][0]
    res["id_consistency"] = (sum(maj[ci] == ri for ci, ri in id_pairs) / len(id_pairs)) if id_pairs else 1.0
    res["lane_count_eq"] = lane_frames_eq / lane_frames if lane_frames else 1.0
    res["lane_dx_mean"] = float(np.mean(dx_all)) if dx_all else 0.0
    res["lane_pt_diff_max"] = int(max(pt_diffs)) if pt_diffs else 0
    res["lane_pt_diff_mean"] = float(np.mean(pt_diffs)) if pt_diffs else 0.0
    # seg
    worst, nseg, missing = 1.0, 0, 0
    for p in sorted((ref_dir / "seg").glob("*.png")):
        q = cpp_dir / "seg" / p.name
        if not q.exists():
            missing += 1
            continue
        a, b = cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
        nseg += 1
        worst = min(worst, 0.0 if a.shape != b.shape else float((a == b).mean()))
    res["seg_pngs"] = nseg
    res["seg_min_equal"] = worst if not missing else 0.0
    res["seg_missing"] = missing
    gates = {
        "frames": len(ref) == len(cpp),
        "dets": res["det_recall"] >= 0.97 and res["det_precision"] >= 0.97,
        "tracks": res["id_consistency"] >= 0.95,
        "lanes": res["lane_count_eq"] >= 0.98 and res["lane_dx_mean"] <= 2.0 and res["lane_pt_diff_max"] <= 1,
        "seg": res["seg_min_equal"] >= 0.99,
    }
    res["gates"] = gates
    res["pass"] = all(gates.values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("cpp_dir")
    a = ap.parse_args()
    r = compare(Path(a.ref_dir), Path(a.cpp_dir))
    print(f"frames ref/cpp      {r['frames']}  [{'OK' if r['gates']['frames'] else 'FAIL'}]")
    print(f"dets recall/prec    {r['det_recall']:.4f} / {r['det_precision']:.4f}  (>=0.97) "
          f"[{'OK' if r['gates']['dets'] else 'FAIL'}]")
    print(f"track id consist.   {r['id_consistency']:.4f}  (>=0.95) [{'OK' if r['gates']['tracks'] else 'FAIL'}]")
    print(f"lanes count-eq      {r['lane_count_eq']:.4f}  (>=0.98); mean|dx| {r['lane_dx_mean']:.3f} px (<=2); "
          f"pt diff max {r['lane_pt_diff_max']} mean {r['lane_pt_diff_mean']:.3f} (<=1) "
          f"[{'OK' if r['gates']['lanes'] else 'FAIL'}]")
    print(f"seg min equal       {r['seg_min_equal']:.5f} over {r['seg_pngs']} PNGs, missing {r['seg_missing']} "
          f"(>=0.99) [{'OK' if r['gates']['seg'] else 'FAIL'}]")
    print("RESULT:", "PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
