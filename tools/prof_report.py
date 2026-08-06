"""Fleet-wide rollout profile readout (2026-08-06).

Aggregates the per-component timers that `WESNOTH_PROF=1` workers
fold into their heartbeat JSONs (see tools/prof_hooks.py) and prints
one breakdown table for the whole fleet. Run it anytime during a
campaign — locally against a synced stats dir, or on the box:

    python tools/prof_report.py training/spool/stats

Wall-clock attribution note: per-worker wall = updated - started
(heartbeats write at game completion). "unattributed" = wall minus
the five timed components — MCTS tree bookkeeping, the per-action
snapshot deepcopy, spool pickling, and scheduling gaps. On CUDA
workers forward attribution is async-skewed (no synchronize in
production); CPU workers (the fleet's norm) are exact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def aggregate(stats_dir: Path) -> dict:
    comps: dict = {}
    wall = 0.0
    n_workers = 0
    games = 0
    decisions = 0
    for f in sorted(stats_dir.glob("w*.json")):
        try:
            hb = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                    # mid-write or stale file
        prof = hb.get("prof")
        if not prof:
            continue
        n_workers += 1
        games += int(hb.get("games", 0))
        decisions += int(hb.get("decisions", 0))
        wall += max(0.0, float(hb.get("updated", 0))
                    - float(hb.get("started", 0)))
        for label, e in prof.items():
            c = comps.setdefault(label, [0, 0.0])
            c[0] += int(e.get("n", 0))
            c[1] += float(e.get("s", 0.0))
    return {"workers": n_workers, "games": games,
            "decisions": decisions, "wall_s": wall, "components": comps}


def render(agg: dict) -> str:
    out = []
    wall = agg["wall_s"]
    out.append(f"workers={agg['workers']} games={agg['games']} "
               f"decisions={agg['decisions']} "
               f"wall={wall/3600:.2f} worker-hours")
    if not agg["components"]:
        out.append("no prof data (fleet not armed with WESNOTH_PROF=1?)")
        return "\n".join(out)
    total_timed = sum(s for _, s in agg["components"].values())
    out.append(f"{'component':<12}{'calls':>12}{'seconds':>12}"
               f"{'% wall':>9}{'ms/call':>10}")
    for label, (n, s) in sorted(agg["components"].items(),
                                key=lambda kv: -kv[1][1]):
        pct = 100.0 * s / wall if wall else 0.0
        ms = 1000.0 * s / n if n else 0.0
        out.append(f"{label:<12}{n:>12}{s:>12.1f}{pct:>8.1f}%"
                   f"{ms:>10.2f}")
    if wall:
        rest = wall - total_timed
        out.append(f"{'unattributed':<12}{'':>12}{rest:>12.1f}"
                   f"{100.0 * rest / wall:>8.1f}%")
        if agg["decisions"]:
            out.append(f"mean decision wall: "
                       f"{wall / agg['decisions']:.2f}s")
    return "\n".join(out)


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: prof_report.py <spool/stats dir>")
        return 2
    stats_dir = Path(argv[1])
    if not stats_dir.is_dir():
        print(f"not a directory: {stats_dir}")
        return 2
    print(render(aggregate(stats_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
