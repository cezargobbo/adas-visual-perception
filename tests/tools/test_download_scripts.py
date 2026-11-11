import hashlib
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

MODELS = [
    "yolo26s.onnx",
    "yolo26x.onnx",
    "segformer-b1-cityscapes-1024x576.onnx",
    "segformer-b5-cityscapes-1280x720.onnx",
    "polarrcnn_curvelanes_dla34.onnx",
]
VIDEOS = [
    "sp1_crop_30fps.mp4",
    "sp1_crop_2_30fps.mp4",
    "sp1_crop_3_30fps.mp4",
    "sp1_crop_4_30fps.mp4",
    "red_crop_11_30fps.mp4",
    "red_crop_12_30fps.mp4",
    "red_crop_13_30fps.mp4",
]


def make_release(d: Path) -> Path:
    d.mkdir(parents=True)
    lines = []
    for name in MODELS + VIDEOS:
        data = f"fake {name}\n".encode()
        (d / name).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return d


def run(script, *args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *map(str, args)],
        capture_output=True, text=True, env=e,
    )


def test_models_fast_downloads_three(tmp_path):
    rel = make_release(tmp_path / "rel")
    dest = tmp_path / "models"
    r = run("download_models.sh", "--profile", "fast", "--dest", dest, "--base-url", f"file://{rel}")
    assert r.returncode == 0, r.stderr
    got = sorted(p.name for p in dest.iterdir())
    assert got == sorted(
        ["yolo26s.onnx", "segformer-b1-cityscapes-1024x576.onnx", "polarrcnn_curvelanes_dla34.onnx"]
    )
    assert "AGPL-3.0" in r.stdout


def test_models_skip_existing_valid(tmp_path):
    rel = make_release(tmp_path / "rel")
    dest = tmp_path / "models"
    args = ("--profile", "fast", "--dest", dest, "--base-url", f"file://{rel}")
    assert run("download_models.sh", *args).returncode == 0
    before = {p.name: p.stat().st_mtime_ns for p in dest.iterdir()}
    time.sleep(0.05)
    assert run("download_models.sh", *args).returncode == 0
    after = {p.name: p.stat().st_mtime_ns for p in dest.iterdir()}
    assert before == after


def test_models_bad_checksum(tmp_path):
    rel = make_release(tmp_path / "rel")
    (rel / "yolo26s.onnx").write_bytes(b"corrupted")
    dest = tmp_path / "models"
    r = run("download_models.sh", "--profile", "fast", "--dest", dest, "--base-url", f"file://{rel}")
    assert r.returncode == 1
    assert not (dest / "yolo26s.onnx").exists()
    assert not list(dest.glob("*.part"))
    assert "yolo26s.onnx" in r.stderr


def test_models_replaces_invalid_existing(tmp_path):
    rel = make_release(tmp_path / "rel")
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "yolo26s.onnx").write_bytes(b"truncated")
    r = run("download_models.sh", "--profile", "fast", "--dest", dest, "--base-url", f"file://{rel}")
    assert r.returncode == 0, r.stderr
    assert (dest / "yolo26s.onnx").read_bytes() == (rel / "yolo26s.onnx").read_bytes()


def test_videos_layout(tmp_path):
    rel = make_release(tmp_path / "rel")
    dest = tmp_path / "videos"
    r = run("download_videos.sh", "--dest", dest, "--base-url", f"file://{rel}")
    assert r.returncode == 0, r.stderr
    assert (dest / "redlights" / "red_crop_11_30fps.mp4").is_file()
    assert (dest / "sp1_crop_30fps.mp4").is_file()
    assert not (dest / "sp1_crop_4_30fps.mp4").is_dir()
    assert not (dest / "red_crop_11_30fps.mp4").exists()
    assert len(list(dest.rglob("*.mp4"))) == 7
    assert "youtube.com/@QuinnDickinson2610" in r.stdout


def test_usage_errors(tmp_path):
    assert run("download_models.sh", "--profile", "bogus", "--dest", tmp_path / "m").returncode == 2
    assert run("download_models.sh", "--nope").returncode == 2
    assert run("download_videos.sh", "--nope").returncode == 2
    assert run("download_models.sh", "--help").returncode == 0


def test_missing_curl(tmp_path):
    empty = tmp_path / "emptybin"
    empty.mkdir()
    bash = subprocess.run(["bash", "-c", "command -v bash"], capture_output=True, text=True).stdout.strip()
    r = subprocess.run(
        [bash, str(SCRIPTS / "download_models.sh"), "--dest", str(tmp_path / "m")],
        capture_output=True, text=True, env={"PATH": str(empty)},
    )
    assert r.returncode == 2
    assert "curl" in r.stderr


def test_stage_release_sums(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    videos = tmp_path / "vids"
    (videos / "redlights").mkdir(parents=True)
    for m in MODELS:
        (models / m).write_bytes(f"m {m}".encode())
    for v in VIDEOS:
        sub = videos / "redlights" if v.startswith("red_crop") else videos
        (sub / v).write_bytes(f"v {v}".encode())
    out = tmp_path / "out"
    r = run("stage_release.sh", "--videos-dir", videos, "--models-dir", models, "--out", out)
    assert r.returncode == 0, r.stderr
    assert len(list(out.iterdir())) == 13
    chk = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=out, capture_output=True, text=True)
    assert chk.returncode == 0, chk.stdout + chk.stderr

    (videos / "redlights" / "red_crop_13_30fps.mp4").unlink()
    r = run("stage_release.sh", "--videos-dir", videos, "--models-dir", models, "--out", tmp_path / "out2")
    assert r.returncode == 1
    assert "red_crop_13_30fps.mp4" in r.stderr
