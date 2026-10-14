"""The workers' decision stages over a match, summed.

Every persistent worker of a match keeps its running total in
`.worker_stages_<pid>.json` in the outdir (tools/elo_eval_game.py,
wesnoth_ai/stage_timing.py); one-process games carry theirs in the
result files under `worker_stages`. `summarize(outdir)` sums them and
reads them per decision: each stage's milliseconds, the decision's wall
and CPU, and the worker's own wall (the wall less the round trip to the
inference server).

    python tools/worker_stages.py OUTDIR [--save PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.stage_timing import StageTimer  # noqa: E402

# The stages of the game loop proper; the others are inside "decide" or per game.
LOOP_STAGES = ("snapshot", "decide", "bounce", "step", "bookkeeping")
GAME_STAGES = ("game_setup", "game_record")


def _timers(outdir: Path) -> List[StageTimer]:
    """The workers' totals, or the result files' when no worker wrote one."""
    files = sorted(outdir.glob(".worker_stages_*.json"))
    if files:
        return [StageTimer.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]
    out = []
    for f in sorted(outdir.glob("game_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8")).get("worker_stages")
        except (OSError, ValueError):
            continue
        if d:
            out.append(StageTimer.from_dict(d))
    return out


def summarize(outdir: Path) -> Dict[str, object]:
    """The match's stages per decision (milliseconds), from its workers."""
    timers = _timers(Path(outdir))
    total = StageTimer()
    for t in timers:
        total.merge(t)
    n = total.decisions
    per = {k: round(1e3 * v / n, 4) for k, v in sorted(total.seconds.items())} if n else {}
    wall = 1e3 * total.decision_wall_s / n if n else 0.0
    cpu = 1e3 * total.decision_cpu_s / n if n else 0.0
    loop = sum(per.get(k, 0.0) for k in LOOP_STAGES if k != "decide")
    return {
        "workers": len(timers), "decisions": n,
        "decision_wall_ms": round(wall, 4), "decision_cpu_ms": round(cpu, 4),
        "own_wall_ms": round(wall - per.get("round_trip", 0.0), 4),
        "decide_other_ms": round(per.get("decide", 0.0) - sum(
            v for k, v in per.items() if k not in LOOP_STAGES and k not in GAME_STAGES), 4),
        "loop_other_ms": round(wall - loop - per.get("decide", 0.0), 4),
        "stage_ms_per_decision": per,
        "totals": total.as_dict(),
    }


def format_summary(s: Dict[str, object]) -> str:
    lines = [f"{s['decisions']} decisions over {s['workers']} worker(s): wall "
             f"{s['decision_wall_ms']:.2f} ms, cpu {s['decision_cpu_ms']:.2f} ms, own wall "
             f"{s['own_wall_ms']:.2f} ms per decision"]
    for k, v in sorted(s["stage_ms_per_decision"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:14s} {v:8.3f} ms")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args(argv[1:])
    s = summarize(args.outdir)
    print(format_summary(s))
    if args.save is not None:
        args.save.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
