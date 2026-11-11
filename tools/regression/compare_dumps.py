#!/usr/bin/env python3
"""Compare a C++ dump (adasvp_dump) against a Python reference dump (dump_reference.py).

Usage: compare_dumps.py REF_DIR CPP_DIR [--adas]

Gates (per video; exit code 0 only if all pass):
  dets   recall and precision >= 0.97 (greedy IoU >= 0.9, same class)
  tracks id consistency >= 0.95: among tracks matched by IoU >= 0.9, the fraction whose C++ id equals the
         Python id under the video's majority mapping (each C++ id -> its most frequent Python id)
  lanes  same lane count on >= 98% of frames, mean |dx| <= 2 px (over true pairs). Matched pairs with
         mean |dx| > 20 px are greedy cross-lane mismatches: reported separately, excluded from the
         point-count gate. Point count over the true pairs: >= 99.9% with diff <= 1 and max diff <= 2.
  seg    >= 99% equal pixels in every PNG; the reference must contain at least one PNG
With --adas (both dirs hold adas.jsonl, written by dump_reference.py --adas / adasvp_dump --adas) the alert
events are also compared. Gate: every event (type + side/level: FCW warning|critical, LDW left|right, RLR red)
of one dump has a partner of the other within +-0.5 s (one-to-one, greedy by time difference). Diagnostics, not
gates: ego.state equal on >= 97% of the frames; lead id equal (under the majority C++ -> Python id mapping,
"no lead" in both counts as equal) on >= 95% of the frames.
Gates never pass on an empty comparison: a reference with lanes but no matched lane pair fails the lane
gate, a reference without seg PNGs fails the seg gate, and more than MAX_MISMATCHED (5) cross-lane
mismatched pairs per video fail the lane gate.
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


MAX_MISMATCHED = 5  # greedy cross-lane mismatches tolerated per video


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
    dx_all, pt_diffs, mismatched = [], [], []
    ref_lanes_total = 0
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
        ref_lanes_total += len(r["lanes"])
        lane_frames_eq += len(r["lanes"]) == len(c["lanes"])
        rl = [np.array(l, dtype=np.float64) for l in r["lanes"]]
        cl = [np.array(l, dtype=np.float64) for l in c["lanes"]]
        for li, m in enumerate(match_lanes(rl, cl)):
            if m["mean"] > 20.0:
                mismatched.append((k, li, m["n_ref"], m["n_cand"], round(m["mean"], 1)))
                continue
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
    res["lane_pairs"] = len(pt_diffs)
    res["lane_pt_le1"] = float(np.mean(np.array(pt_diffs) <= 1)) if pt_diffs else 1.0
    res["lane_mismatched"] = mismatched
    res["ref_lanes"] = ref_lanes_total
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
        "lanes": res["lane_count_eq"] >= 0.98 and res["lane_dx_mean"] <= 2.0
                 and res["lane_pt_le1"] >= 0.999 and res["lane_pt_diff_max"] <= 2
                 and len(mismatched) <= MAX_MISMATCHED
                 and not (ref_lanes_total > 0 and res["lane_pairs"] == 0),
        "seg": nseg > 0 and res["seg_min_equal"] >= 0.99,
    }
    res["gates"] = gates
    res["pass"] = all(gates.values())
    return res


EVENT_TOL_S = 0.5


def load_adas(d: Path):
    return [json.loads(l) for l in open(d / "adas.jsonl")]


def events_of(rows):
    ev = []
    for r in rows:
        if r["fcw"]["event"]:
            ev.append(("FCW", r["fcw"]["event"], r["t"], r["i"]))
        if r["ldw"]["event"]:
            ev.append(("LDW", r["ldw"]["event"], r["t"], r["i"]))
        if r["rlr"]["event"]:
            ev.append(("RLR", "red", r["t"], r["i"]))
    return ev


def match_events(ref_ev, cpp_ev, tol=EVENT_TOL_S):
    """Greedy one-to-one matching by |dt| among events with the same (type, side)."""
    cand = sorted((abs(a[2] - b[2]), i, j) for i, a in enumerate(ref_ev) for j, b in enumerate(cpp_ev)
                  if a[:2] == b[:2] and abs(a[2] - b[2]) <= tol)
    used_r, used_c, pairs = set(), set(), []
    for dt, i, j in cand:
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j); pairs.append((i, j, dt))
    return pairs, [i for i in range(len(ref_ev)) if i not in used_r], [j for j in range(len(cpp_ev)) if j not in used_c]


def compare_adas(ref_dir: Path, cpp_dir: Path) -> dict:
    ref, cpp = load_adas(ref_dir), load_adas(cpp_dir)
    n = min(len(ref), len(cpp))
    res = {"frames": f"{len(ref)}/{len(cpp)}"}
    ref_ev, cpp_ev = events_of(ref), events_of(cpp)
    pairs, miss, extra = match_events(ref_ev, cpp_ev)
    res["ref_events"], res["cpp_events"] = ref_ev, cpp_ev
    res["pairs"], res["missing"], res["extra"] = pairs, [ref_ev[i] for i in miss], [cpp_ev[j] for j in extra]
    res["state_agree"] = sum(ref[k]["ego"]["state"] == cpp[k]["ego"]["state"] for k in range(n)) / max(n, 1)
    by = defaultdict(Counter)
    for k in range(n):
        a, b = ref[k]["lead"], cpp[k]["lead"]
        if a is not None and b is not None:
            by[int(b[0])][int(a[0])] += 1
    maj = {ci: cnt.most_common(1)[0][0] for ci, cnt in by.items()}
    agree = 0
    for k in range(n):
        a, b = ref[k]["lead"], cpp[k]["lead"]
        agree += (a is None and b is None) or (a is not None and b is not None and maj.get(int(b[0])) == int(a[0]))
    res["lead_agree"] = agree / max(n, 1)
    res["pass"] = not miss and not extra and len(ref) == len(cpp)
    return res


def print_adas(r: dict) -> None:
    fmt = lambda e: f"{e[0]} {e[1]} @ frame {e[3]} (t={e[2]:.2f}s)"  # noqa: E731
    print(f"adas frames ref/cpp {r['frames']}")
    print(f"adas events         ref {len(r['ref_events'])}, cpp {len(r['cpp_events'])}, matched {len(r['pairs'])} "
          f"(+-{EVENT_TOL_S} s) [{'OK' if r['pass'] else 'FAIL'}]")
    for i, j, dt in r["pairs"]:
        a, b = r["ref_events"][i], r["cpp_events"][j]
        print(f"    {a[0]} {a[1]}: ref frame {a[3]} / cpp frame {b[3]} (dt {dt:.3f} s)")
    for e in r["missing"]:
        print(f"    MISSING in cpp: {fmt(e)}")
    for e in r["extra"]:
        print(f"    EXTRA in cpp:   {fmt(e)}")
    print(f"adas ego.state eq   {r['state_agree']:.4f}  (diagnostic >=0.97) [{'ok' if r['state_agree'] >= 0.97 else 'low'}]")
    print(f"adas lead id eq     {r['lead_agree']:.4f}  (diagnostic >=0.95) [{'ok' if r['lead_agree'] >= 0.95 else 'low'}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("cpp_dir")
    ap.add_argument("--adas", action="store_true", help="also compare adas.jsonl (alert events, diagnostics)")
    a = ap.parse_args()
    r = compare(Path(a.ref_dir), Path(a.cpp_dir))
    print(f"frames ref/cpp      {r['frames']}  [{'OK' if r['gates']['frames'] else 'FAIL'}]")
    print(f"dets recall/prec    {r['det_recall']:.4f} / {r['det_precision']:.4f}  (>=0.97) "
          f"[{'OK' if r['gates']['dets'] else 'FAIL'}]")
    print(f"track id consist.   {r['id_consistency']:.4f}  (>=0.95) [{'OK' if r['gates']['tracks'] else 'FAIL'}]")
    print(f"lanes count-eq      {r['lane_count_eq']:.4f}  (>=0.98); mean|dx| {r['lane_dx_mean']:.3f} px (<=2); "
          f"pt diff<=1 {r['lane_pt_le1']:.5f} (>=0.999) max {r['lane_pt_diff_max']} (<=2) "
          f"mean {r['lane_pt_diff_mean']:.4f} over {r['lane_pairs']} pairs "
          f"[{'OK' if r['gates']['lanes'] else 'FAIL'}]")
    print(f"lane mismatched pairs (mean|dx|>20, excluded, max {MAX_MISMATCHED}): {len(r['lane_mismatched'])}"
          + "".join(f"\n    frame {f} lane {l}: {a} vs {b} pts, mean|dx| {d} px" for f, l, a, b, d in r['lane_mismatched']))
    print(f"seg min equal       {r['seg_min_equal']:.5f} over {r['seg_pngs']} PNGs, missing {r['seg_missing']} "
          f"(>=0.99) [{'OK' if r['gates']['seg'] else 'FAIL'}]")
    ok = r["pass"]
    if a.adas:
        ra = compare_adas(Path(a.ref_dir), Path(a.cpp_dir))
        print_adas(ra)
        ok = ok and ra["pass"]
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
