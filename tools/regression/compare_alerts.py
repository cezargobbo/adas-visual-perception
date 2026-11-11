#!/usr/bin/env python
"""Compares two alert CSVs (Python reference vs C++ video_demo) event by event.

Usage: compare_alerts.py REF.csv CPP.csv [--tol 0.5]

Events are matched by (type, level_side) within +-tol seconds on time_s (greedy, closest pairs first, each
event used once). Greedy matching can only err on the strict side: it may leave a false missing/extra pair when
events are closer together than the tolerance, but it never pairs events that are farther apart than the tolerance
or of a different kind, so it cannot hide a real mismatch. Prints the pairs, the missing events (in REF only) and the extra events (in CPP only), and
exits 0 only when nothing is missing or extra. The tolerance is never loosened: borderline FCW cases are
diagnosed separately (see README.md).
"""
import argparse
import csv
import sys


def load(path):
    with open(path, newline="") as f:
        rows = []
        for n, r in enumerate(csv.DictReader(f), start=2):
            try:
                rows.append((float(r["time_s"]), int(r["frame"]), r["type"], r["level_side"]))
            except (ValueError, TypeError, KeyError) as e:
                sys.exit(f"{path}: line {n}: invalid event row ({e!r}); empty or missing time_s/frame?")
        return rows


def match(ref, cpp, tol):
    cands = sorted(
        (abs(r[0] - c[0]), i, j)
        for i, r in enumerate(ref) for j, c in enumerate(cpp)
        if r[2] == c[2] and r[3] == c[3] and abs(r[0] - c[0]) <= tol
    )
    ui, uj, pairs = set(), set(), []
    for d, i, j in cands:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); pairs.append((i, j))
    pairs.sort()
    missing = [ref[i] for i in range(len(ref)) if i not in ui]
    extra = [cpp[j] for j in range(len(cpp)) if j not in uj]
    return pairs, missing, extra


def fmt(e):
    return f"{e[2]} {e[3]} @ {e[0]:.2f} s (frame {e[1]})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("cpp")
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args(argv)
    ref, cpp = load(a.ref), load(a.cpp)
    pairs, missing, extra = match(ref, cpp, a.tol)
    print(f"ref={len(ref)} cpp={len(cpp)} matched={len(pairs)} missing={len(missing)} extra={len(extra)}")
    for i, j in pairs:
        print(f"  pair    {fmt(ref[i])}  <->  {fmt(cpp[j])}  (dt {cpp[j][0] - ref[i][0]:+.2f} s)")
    for e in missing:
        print(f"  MISSING {fmt(e)}")
    for e in extra:
        print(f"  EXTRA   {fmt(e)}")
    return 0 if not missing and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
