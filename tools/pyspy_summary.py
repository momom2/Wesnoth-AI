#!/usr/bin/env python3
"""Summarize a `py-spy record --format raw` file (collapsed stacks,
one `thread;frame;...;frame count` line each): samples per thread,
and the top leaf frames and top project frames per thread.

Usage: python tools/pyspy_summary.py prof_all.txt [--top 12]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_FRAME_RE = re.compile(r"^(?P<fn>.+?) \((?P<file>[^:]+):(?P<line>\d+)\)$")


def parse(path: Path):
    per_thread = Counter()
    leaf = defaultdict(Counter)
    project = defaultdict(Counter)
    total = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        stack, _, count = line.rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        frames = stack.split(";")
        thread = frames[0]
        total += n
        per_thread[thread] += n
        if len(frames) > 1:
            leaf[thread][frames[-1]] += n
            # Innermost frame that lives in this project (not torch/numpy/stdlib).
            for fr in reversed(frames[1:]):
                m = _FRAME_RE.match(fr)
                f = m.group("file") if m else fr
                if ("wesnoth_ai" in f or "/tools/" in f or "\\tools\\" in f) and "site-packages" not in f:
                    project[thread][fr] += n
                    break
            else:
                project[thread]["<no project frame>"] += n
    return total, per_thread, leaf, project


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv[1:])
    total, per_thread, leaf, project = parse(args.path)
    print(f"{args.path.name}: {total} samples")
    for thread, n in per_thread.most_common():
        print(f"\n== {thread}: {n} samples ({100.0 * n / total:.0f}%)")
        print("  leaf frames:")
        for fr, c in leaf[thread].most_common(args.top):
            print(f"    {100.0 * c / n:5.1f}%  {fr}")
        print("  innermost project frames:")
        for fr, c in project[thread].most_common(args.top):
            print(f"    {100.0 * c / n:5.1f}%  {fr}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
