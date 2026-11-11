"""Documentation checks: every relative link and image in README.md and docs/*.md must resolve to a file."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FENCE = re.compile(r"^\s*(```|~~~)")
MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_REF = re.compile(r"<(?:img|a)\b[^>]*?\b(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
EXTERNAL = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")  # http:, https:, mailto:, protocol-relative


def doc_files():
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    return [f for f in files if f.exists()]


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
