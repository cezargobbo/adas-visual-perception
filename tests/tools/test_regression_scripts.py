"""Argument handling of the regression shell scripts (no GPU, no videos needed)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "tools" / "regression" / "run_cpp_regression.sh"
GOLDEN = ROOT / "tools" / "regression" / "make_golden.sh"


def run(script, *args):
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, cwd=ROOT)


def test_run_help_documents_against():
    r = run(RUN, "--help")
    assert r.returncode == 0
    assert "--against" in r.stdout and "golden" in r.stdout and "python" in r.stdout


def test_run_rejects_unknown_baseline(tmp_path):
    r = run(RUN, "--videos-dir", str(tmp_path), "--profile", "fast", "--against", "reference")
    assert r.returncode == 2
    assert "--against must be golden or python" in r.stderr


def test_make_golden_requires_profile(tmp_path):
    r = run(GOLDEN, "--videos-dir", str(tmp_path))
    assert r.returncode == 2


@pytest.mark.parametrize("profile", ["fast", "accurate"])
@pytest.mark.parametrize("baseline", ["golden", "python"])
def test_baseline_dirs_cover_the_video_list(profile, baseline):
    d = ROOT / "tests" / "regression" / baseline / profile
    stems = [Path(line).stem for line in (ROOT / "tools/regression/videos.txt").read_text().split() if line]
    missing = [s for s in stems if not (d / f"{s}_alerts.csv").exists()]
    assert not missing, f"{d}: no alerts CSV for {missing}"
