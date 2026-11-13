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


def run_wrapper(tmp_path, *args, env_extra=None, expect_rc=0):
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
    env.update(env_extra or {})
    r = subprocess.run([str(WRAPPER), *args], capture_output=True, text=True, cwd=work, env=env)
    assert r.returncode == expect_rc, r.stderr
    if expect_rc != 0:
        return r.stderr, work
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


def x11_env(tmp_path, display=":1"):
    """A fake X11 socket dir with the socket file for DISPLAY :1."""
    xdir = tmp_path / "x11"
    xdir.mkdir(exist_ok=True)
    (xdir / "X1").touch()
    return {"DISPLAY": display, "ADASVP_X11_DIR": str(xdir)}


def test_show_passes_the_view_and_mounts_the_x11_socket(tmp_path):
    env = x11_env(tmp_path)
    argv, work = run_wrapper(tmp_path, "clip.mp4", "--show", "comb", env_extra=env)
    i = argv.index("video_demo")
    demo = argv[i + 1:]
    assert demo[demo.index("--show") + 1] == "comb"
    assert f"type=bind,src={env['ADASVP_X11_DIR']},dst=/tmp/.X11-unix,readonly" in argv
    assert "DISPLAY=:1" in argv[:i]
    # --show writes nothing, so no default output prefix and no outputs/ folder
    assert "-o" not in demo
    assert not (work / "outputs").exists()


def test_show_without_a_value_keeps_the_next_word_as_input(tmp_path):
    argv, _ = run_wrapper(tmp_path, "--show", "clip.mp4", env_extra=x11_env(tmp_path))
    demo = argv[argv.index("video_demo") + 1:]
    assert demo[:3] == ["--models-dir", "/work/models", "/in/clip.mp4"]  # the input, not a value of --show
    assert "--show" in demo and demo.count("/in/clip.mp4") == 1


def test_show_mounts_the_xauthority_file_when_there_is_one(tmp_path):
    env = x11_env(tmp_path)
    xauth = tmp_path / "xauth"
    xauth.write_text("cookie")
    env["XAUTHORITY"] = str(xauth)
    argv, _ = run_wrapper(tmp_path, "clip.mp4", "--show", env_extra=env)
    assert f"type=bind,src={xauth},dst=/tmp/.Xauthority,readonly" in argv
    assert "XAUTHORITY=/tmp/.Xauthority" in argv


def test_show_needs_a_display(tmp_path):
    env = x11_env(tmp_path)
    env.pop("DISPLAY")
    err, _ = run_wrapper(tmp_path, "clip.mp4", "--show", env_extra=env, expect_rc=2)
    assert "DISPLAY" in err


def test_show_rejects_a_remote_display(tmp_path):
    err, _ = run_wrapper(tmp_path, "clip.mp4", "--show", env_extra=x11_env(tmp_path, "localhost:10.0"), expect_rc=2)
    assert "local display" in err
