import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "regression"))
import compare_dumps as cd  # noqa: E402

THR = {"warning": 2.7, "critical": 1.5}


def rows(ttcs, t0=10.0):
    return [{"t": t0 + k / 30, "fcw": {"ttc": v}} for k, v in enumerate(ttcs)]


def test_borderline_miss_is_not_a_failure():
    ev = [("FCW", "warning", 10.2, 6)]
    border, fail = cd.classify_unmatched(ev, rows([3.0, 2.8, 2.706, 2.9, 3.1, 3.2, 3.3]), THR)
    assert fail == [] and border == [(ev[0], 2.706)]  # 2.706 <= 1.02 * 2.7


def test_non_borderline_miss_fails():
    ev = [("FCW", "warning", 10.2, 6)]
    border, fail = cd.classify_unmatched(ev, rows([4.0, 3.9, 3.8, 3.9, 4.0, 4.1, 4.2]), THR)
    assert border == [] and fail == ev
    # no ttc at all in the window, and non-FCW events, never count as borderline
    assert cd.classify_unmatched(ev, rows([None] * 7), THR)[1] == ev
    ldw = [("LDW", "left", 10.2, 6)]
    assert cd.classify_unmatched(ldw, rows([2.0] * 7), THR)[1] == ldw


def test_critical_uses_its_own_threshold():
    ev = [("FCW", "critical", 10.2, 6)]
    assert cd.classify_unmatched(ev, rows([2.6] * 7), THR)[1] == ev  # below 2.7 but far above 1.5
    assert cd.classify_unmatched(ev, rows([1.52] * 7), THR)[0]
