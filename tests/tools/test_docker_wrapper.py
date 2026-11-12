"""The Docker wrapper must know which video_demo options take a value (drift guard against examples/cli.cpp)."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "video_demo_docker.sh"
CLI = ROOT / "examples" / "cli.cpp"

# Handled by explicit branches in the wrapper (not by VALUE_OPTS).
SPECIAL = {"-o", "--out-prefix", "--adas-debug", "--rlr-zone", "--show", "--models-dir"}


def cli_options():
    text = CLI.read_text(encoding="utf-8")
    opts, flags = set(), set()
    for kind, names in re.findall(r'app\.(add_option|add_flag)\("([^"]+)"', text):
        if names == "input":
            continue
        (opts if kind == "add_option" else flags).update(names.split(","))
    return opts, flags


def wrapper_value_opts():
    m = re.search(r'VALUE_OPTS="([^"]*)"', WRAPPER.read_text(encoding="utf-8"))
    assert m, "VALUE_OPTS not found in the wrapper"
    return set(m.group(1).split())


def test_every_value_option_is_known_to_the_wrapper():
    opts, _ = cli_options()
    assert len(opts) > 20  # the extraction itself works
    known = wrapper_value_opts() | SPECIAL
    missing = sorted(o for o in opts if o not in known)
    assert not missing, f"video_demo options the wrapper does not know: {missing}"


def test_flags_are_not_listed_as_value_options():
    _, flags = cli_options()
    wrong = sorted(wrapper_value_opts() & flags)
    assert not wrong, f"flags listed in VALUE_OPTS: {wrong}"


def test_special_options_have_explicit_branches():
    text = WRAPPER.read_text(encoding="utf-8")
    for o in SPECIAL - {"-o"}:
        assert f"{o})" in text or f"{o}|" in text or f"|{o}" in text, f"no branch for {o}"


def test_show_is_rejected_with_exit_2():
    r = subprocess.run([str(WRAPPER), "clip.mp4", "--show", "det"], capture_output=True, text=True)
    assert r.returncode == 2
    assert "--show is not supported through Docker" in r.stderr


def run_wrapper(tmp_path, *args):
    """Run the wrapper with a stub `docker` on PATH that records its argv; returns (argv list, cwd)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "docker"
    stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$STUB_OUT"\n')
    stub.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    out = tmp_path / "argv.txt"
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "STUB_OUT": str(out),
        "ADASVP_DOCKER_GPU_FLAGS": "",
    }
    r = subprocess.run([str(WRAPPER), *args], capture_output=True, text=True, cwd=work, env=env)
    assert r.returncode == 0, r.stderr
    return out.read_text().splitlines(), work


def test_default_output_prefix_mirrors_native(tmp_path):
    argv, work = run_wrapper(tmp_path, "clip.mp4", "--profile", "fast")
    out = work / "outputs"
    assert out.is_dir()
    assert f"type=bind,src={out},dst=/out" in argv
    i = argv.index("video_demo")
    assert argv[i + 1:][argv[i + 1:].index("-o") + 1] == "/out/clip"
    assert "/in/clip.mp4" in argv


def test_explicit_output_prefix_is_kept(tmp_path):
    argv, work = run_wrapper(tmp_path, "clip.mp4", "-o", "res/run1")
    assert f"type=bind,src={work / 'res'},dst=/out" in argv
    assert argv.count("-o") == 1 and argv[argv.index("-o") + 1] == "/out/run1"
    assert not (work / "outputs").exists()
