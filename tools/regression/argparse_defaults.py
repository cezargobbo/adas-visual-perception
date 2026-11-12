#!/usr/bin/env python
"""Extracts the argparse defaults of prototype/python/dashcam_vision.py into tests/fixtures/argparse_defaults.json.

The reference builds its parser inside main(), so it cannot be imported on its own. This script imports the
module (main() does not run), swaps argparse.ArgumentParser.parse_args for a hook that captures the parser and
aborts, and calls main(): the parser is fully built by the time parse_args is reached. The defaults are then read
from the parser actions (dest -> default), the "--profile" defaults are resolved from PROFILES, and the result is
written as {"defaults": {dest: value}, "profiles": {name: {dest: value}}}.

Usage: python tools/regression/argparse_defaults.py [OUT_JSON]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "prototype" / "python" / "dashcam_vision.py"


class _Captured(Exception):
    def __init__(self, parser):
        super().__init__("parser captured")
        self.parser = parser


def extract():
    spec = importlib.util.spec_from_file_location("dashcam_vision_proto", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def hook(self, *a, **k):
        raise _Captured(self)

    orig_parse, orig_argv = argparse.ArgumentParser.parse_args, sys.argv
    argparse.ArgumentParser.parse_args = hook
    sys.argv = ["dashcam_vision.py", "input.mp4"]
    try:
        mod.main()
    except _Captured as c:
        parser = c.parser
    else:
        raise RuntimeError("main() returned without parsing arguments")
    finally:
        argparse.ArgumentParser.parse_args = orig_parse
        sys.argv = orig_argv

    defaults = {}
    for act in parser._actions:
        if act.dest in ("help", argparse.SUPPRESS) or act.dest == "input":
            continue
        d = act.default
        defaults[act.dest] = list(d) if isinstance(d, tuple) else d
    profiles = {name: {k: v for k, v in p.items()} for name, p in mod.PROFILES.items()}
    return {"defaults": defaults, "profiles": profiles}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "fixtures" / "argparse_defaults.json"
    data = extract()
    # absolute model paths are machine specific: keep only the file names of the profile entries
    for p in data["profiles"].values():
        for k, v in p.items():
            if isinstance(v, str) and "/" in v:
                p[k] = Path(v).name
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
