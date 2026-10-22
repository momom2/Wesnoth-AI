"""Stage C's wall and cost from measured rates (docs/training_design_20261019/
current_design.md §6; scripts/stage_c_box.sh's sizing): one function, read
by stage C's C0 (scripts/stage_c_box.sh, c0_plan) and by stage 1's rate
measurement (scripts/selfplay_t1_box.sh), so the user prices stage C from
measurements before deciding on it.

The inputs are the gradient pass's decisions a second (tools/pg_pass.py's
`rate`, past its warm-up), the evaluator's forward-only decisions a second
(tools/pg_eval.py time), the decisions a game, and the games labelled for
training and held out. The parts, in minutes:

    pass      N D / t
    stage C   E H D / f, E = FWD_EQ forward-equivalents over the held-out
              games (round11_synthesis_plan.py, stage_c_forwards), or
              FWD_EQ_TRIMMED when the forwards run at the bound: f below
              TRIM_RATIO t, the geometric middle of the bound (f = t) and
              the design's illustration (f = 3t); design §6 then drops the
              heads-only arm and the 5,000-game point
    plant     (2 P + r H) D / t + PLANT_FORWARDS r H D / f: the plant's
              expected gradient and its planted pass over P = min(N_PLANT,
              N) games, grad M and the readout's forwards over r H
              held-out games (tools/pg_plant.py)

with t the pass's rate, f the forwards', D the decisions a game, N and H the
training and held-out games. Each part's step is cut at CUT_FACTOR times
its estimate plus CUT_PAD_MIN minutes. The wall adds FIXED_HOURS (bring-up,
the restore, C0, the uploads) and the cost is the wall at HOST_RATE.

    python tools/stage_c_plan.py --pass-summary pass.json --forwards forwards.json --labels labels.jsonl \\
        --out plan.json [--left-min 600 --line]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

HOST_RATE = 0.577             # dollars an hour (design §4 prices)
FIXED_HOURS = 1.0             # bring-up, the restore, C0 and the uploads (scripts/stage_c_box.sh)
FWD_EQ = 8.65                 # stage C's forward-equivalents over the held-out games
FWD_EQ_TRIMMED = 7.35         # ... without the heads-only arm and the 5,000-game point
N_PLANT = 5000
READOUT_FRACTION = 0.2
PLANT_FORWARDS = 5            # the readout's forwards: theta_k, +-s along d', +-h along P'u
TRIM_RATIO = math.sqrt(3.0)
CUT_FACTOR = 1.5              # a box job past about 1.5 times its estimate is cut (CLAUDE.md, 2026-09-05)
CUT_PAD_MIN = 15


def project(trainer_rate: float, forward_rate: float, decisions_per_game: float, n_train: int, n_held: int,
            n_plant: int = N_PLANT, readout_fraction: float = READOUT_FRACTION, fwd_eq: float = FWD_EQ,
            fwd_eq_trimmed: float = FWD_EQ_TRIMMED) -> Dict:
    """Stage C's parts in minutes, their cuts, the wall and the cost
    (module docstring)."""
    t, f, D = float(trainer_rate), float(forward_rate), float(decisions_per_game)
    if not (t > 0 and f > 0 and D > 0):
        raise ValueError(f"rates and decisions must be positive: trainer {t}, forwards {f}, decisions {D}")
    trim = f < TRIM_RATIO * t
    plant_games = min(int(n_plant), int(n_train))
    readout = readout_fraction * n_held
    minutes = {"pass": n_train * D / t / 60.0,
               "stage_c": (fwd_eq_trimmed if trim else fwd_eq) * n_held * D / f / 60.0,
               "plant": ((2 * plant_games + readout) * D / t + PLANT_FORWARDS * readout * D / f) / 60.0}
    total = sum(minutes.values())
    wall = FIXED_HOURS + total / 60.0
    return {"rates": {"trainer": t, "forwards": f, "decisions_per_game": D, "forward_to_trainer": f / t},
            "games": {"train": int(n_train), "held": int(n_held), "plant": plant_games},
            "trim": bool(trim), "minutes": minutes, "total_minutes": total,
            "cuts": {k: int(CUT_FACTOR * v + CUT_PAD_MIN) for k, v in minutes.items()},
            "wall_hours": wall, "cost": wall * HOST_RATE,
            "constants": {"host_rate": HOST_RATE, "fixed_hours": FIXED_HOURS, "fwd_eq": fwd_eq,
                          "fwd_eq_trimmed": fwd_eq_trimmed, "readout_fraction": readout_fraction,
                          "trim_ratio": TRIM_RATIO}}


def pass_rate(summary: Dict) -> float:
    """The pass's decisions a second: past its warm-up when it read one."""
    rate = summary.get("rate") or {}
    return float(rate.get("positions_per_second") or summary["decisions"] / summary["seconds"])


def label_counts(labels: Path) -> Dict[str, int]:
    """Games labelled for training and held out."""
    train, held = set(), set()
    for line in Path(labels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            (held if row["heldout"] else train).add(row["game"])
    return {"train": len(train), "held": len(held)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pass-summary", type=Path, required=True, help="pg_pass.py's pass.json")
    ap.add_argument("--forwards", type=Path, required=True, help="pg_eval.py time's JSON")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--compare", type=Path, nargs="*", default=[],
                    help="other passes' summaries (bf16, say), their rates recorded beside")
    ap.add_argument("--n-plant", type=int, default=N_PLANT)
    ap.add_argument("--readout-fraction", type=float, default=READOUT_FRACTION)
    ap.add_argument("--fwd-eq", type=float, default=FWD_EQ)
    ap.add_argument("--fwd-eq-trimmed", type=float, default=FWD_EQ_TRIMMED)
    ap.add_argument("--left-min", type=int, default=None, help="the minutes the run may use")
    ap.add_argument("--line", action="store_true",
                    help="print 'trim total left pass_cut stage_c_cut plant_cut train held' for a shell")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = json.loads(args.pass_summary.read_text(encoding="utf-8"))
    forwards = json.loads(args.forwards.read_text(encoding="utf-8"))
    counts = label_counts(args.labels)
    plan = project(pass_rate(summary), forwards["positions_per_second"], summary["decisions"] / summary["games"],
                   counts["train"], counts["held"], args.n_plant, args.readout_fraction, args.fwd_eq,
                   args.fwd_eq_trimmed)
    plan["measured"] = {"pass": {k: summary.get(k) for k in ("games", "decisions", "seconds", "rate", "replay",
                                                             "floor", "config")},
                        "forwards": forwards,
                        "compare": {p.stem: pass_rate(json.loads(p.read_text(encoding="utf-8"))) for p in args.compare}}
    if args.left_min is not None:
        plan["left_minutes"] = args.left_min
        plan["fits"] = plan["total_minutes"] <= args.left_min
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    if args.line:
        c = plan["cuts"]
        print(int(plan["trim"]), int(math.ceil(plan["total_minutes"])), args.left_min if args.left_min is not None
              else -1, c["pass"], c["stage_c"], c["plant"], counts["train"], counts["held"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
