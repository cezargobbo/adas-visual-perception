"""Documentation checks: every relative link and image in the git-tracked *.md files must resolve to a file."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FENCE = re.compile(r"^\s*(```|~~~)")
MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_REF = re.compile(r"<(?:img|a)\b[^>]*?\b(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
EXTERNAL = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")  # http:, https:, mailto:, protocol-relative


def doc_files():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / n for n in sorted(out.splitlines()) if (ROOT / n).exists()]


def link_targets(text):
    """Relative link targets of a Markdown text (fenced code blocks and inline code are skipped)."""
    targets = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = re.sub(r"`[^`]*`", "", line)
        for m in list(MD_LINK.finditer(line)) + list(HTML_REF.finditer(line)):
            t = m.group(1).strip()
            if not t or t.startswith("#") or EXTERNAL.match(t):
                continue
            t = t.split("#", 1)[0].split("?", 1)[0]
            if t:
                targets.append(t)
    return targets


def test_link_extraction():
    text = (
        "[a](docs/x.md#sec) ![i](docs/media/a.gif) <img src=\"docs/media/b.jpg\" width=1>\n"
        "[ext](https://example.com/a.md) [mail](mailto:a@b.c) [anchor](#top)\n"
        "`[code](nope.md)`\n```\n[fenced](nope2.md)\n```\n"
    )
    assert link_targets(text) == ["docs/x.md", "docs/media/a.gif", "docs/media/b.jpg"]


def test_relative_links_exist():
    broken = []
    for f in doc_files():
        for t in link_targets(f.read_text(encoding="utf-8")):
            if not (f.parent / t).exists():
                broken.append(f"{f.relative_to(ROOT)}: {t}")
    assert not broken, "broken relative links:\n  " + "\n  ".join(broken)


BINARY = ROOT / "build" / "native-release" / "examples" / "video_demo"
ENGINE_MESSAGE = "[adasvp] building TensorRT engine for "


def help_flags(text):
    """Long option names listed in the option groups of `video_demo --help` (before the prose sections)."""
    head = text.split("\nprofiles (--profile):", 1)[0]
    head = head.split("\nhelp:\n", 1)[1]
    return set(re.findall(r"(?:^|[ ,])(--[a-z][a-z0-9-]*)", "\n".join(l for l in head.splitlines() if l.startswith("  -") or l.startswith("  --"))))


def readme_table_flags():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^\| Option \| Default \| What it does \|\n\|[-| ]+\|\n((?:\|.*\n?)+)", text, re.M)
    assert m, "option table not found in README.md"
    flags = set()
    for row in m.group(1).splitlines():
        cell = row.split("|")[1]
        flags.update(re.findall(r"`(--[a-z][a-z0-9-]*)", cell))
    return flags


def test_flag_table_matches_help():
    if not BINARY.exists():
        pytest.skip("video_demo is not built")
    r = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True, check=True)
    want = help_flags(r.stdout) - {"--help"}
    assert len(want) > 30
    got = readme_table_flags()
    assert not want - got, f"flags in --help but not in the README table: {sorted(want - got)}"
    assert not got - want, f"flags in the README table but not in --help: {sorted(got - want)}"


def test_engine_build_message():
    src = (ROOT / "src" / "trt" / "trt_engine.cpp").read_text(encoding="utf-8")
    assert ENGINE_MESSAGE in src
    assert "(first run only, may take several minutes)" in src
