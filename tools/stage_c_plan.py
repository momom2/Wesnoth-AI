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
    stage C   E H D / f, E the forward-equivalents over the held-out games
              of the driver's reads (`read_plan`, priced by
              `forward_equivalents`): ARMS arms and two gain-curve points,
              or, when the forwards run at the bound (f below TRIM_RATIO t,
              the geometric middle of the bound f = t and the design's
              illustration f = 3t), the arms without heads-only and one
              point (design §6)
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
ARMS = ("sgd", "reduced_adamw", "muon", "heads_only", "probe_span")    # tools/pg_stage_c.py's
TRIMMED_ARMS = ("sgd", "reduced_adamw", "muon", "probe_span")
LINEAR_ARMS = ("sgd", "reduced_adamw", "heads_only", "probe_span")    # a critical batch can be read in their metric
CURVE_POINTS, TRIMMED_CURVE_POINTS = 2, 1
LADDER_RUNGS = 3
SPLITS = 8
KL_FRACTION = 0.05            # the KL subsample (tools/pg_eval.py EvalConfig)
# A KL search's readings: 3 each on the toy (tests/test_pg_stage_c.py) and 2 to 3
# on the tiny network's end-to-end run (mean 2.6, 2026-10-22); 4 allows one more.
KL_EVALS = 4
N_PLANT = 5000
READOUT_FRACTION = 0.2
PLANT_FORWARDS = 5            # the readout's forwards: theta_k, +-s along d', +-h along P'u
TRIM_RATIO = math.sqrt(3.0)
CUT_FACTOR = 1.5              # a box job past about 1.5 times its estimate is cut (CLAUDE.md, 2026-09-05)
CUT_PAD_MIN = 15


def read_plan(arms: int, curve_points: int, critical_batches: int, span_forms: int = 0, splits: int = SPLITS,
              rungs: int = LADDER_RUNGS) -> Dict[str, int]:
    """The forwards tools/pg_stage_c.py's run_stage_c makes, by kind, on an
    evaluator that shares theta_k's readings (tools/pg_eval.py):
        full         theta_k over every held-out decision, once
        laws         theta_k over the KL subsample, once
        half         each arm on half A; the ladder's rungs other than the
                     arm's own on half A and its chosen rung on half B; each
                     gain-curve point on half A
        kl_searches  each arm, each of the ladder's other rungs, each point
                     (KL_EVALS readings each, priced, not counted)
        kl_readings  the Fisher forms at +s and -s: each critical batch's
                     whole direction, splits and deviations (1 + 2 splits
                     forms), and the span's forms when the pass lacks its
                     Fisher
    A KL reading forwards the moved network on the KL subsample.
    tests/test_pg_stage_c.py counts the driver's reads on the toy against
    this."""
    return {"full": 1, "laws": 1, "half": arms + (rungs - 1) + 1 + curve_points,
            "kl_searches": arms + (rungs - 1) + curve_points,
            "kl_readings": 2 * (critical_batches * (1 + 2 * splits) + span_forms)}


def forward_equivalents(plan: Dict[str, int], kl_fraction: float = KL_FRACTION, kl_evals: float = KL_EVALS) -> float:
    """A read plan in forwards over every held-out decision."""
    subsample = plan["laws"] + plan["kl_readings"] + plan["kl_searches"] * kl_evals
    return plan["full"] + 0.5 * plan["half"] + kl_fraction * subsample


def configured_plan(trim: bool) -> Dict[str, int]:
    """The read plan of stage C's configured run (scripts/stage_c_box.sh),
    a second critical batch counted when a non-default linear arm could win."""
    arms = TRIMMED_ARMS if trim else ARMS
    critical = 1 + int(any(a in LINEAR_ARMS and a != "reduced_adamw" for a in arms))
    return read_plan(len(arms), TRIMMED_CURVE_POINTS if trim else CURVE_POINTS, critical)


def project(trainer_rate: float, forward_rate: float, decisions_per_game: float, n_train: int, n_held: int,
            n_plant: int = N_PLANT, readout_fraction: float = READOUT_FRACTION) -> Dict:
    """Stage C's parts in minutes, their cuts, the wall and the cost
    (module docstring)."""
    t, f, D = float(trainer_rate), float(forward_rate), float(decisions_per_game)
    if not (t > 0 and f > 0 and D > 0):
        raise ValueError(f"rates and decisions must be positive: trainer {t}, forwards {f}, decisions {D}")
    trim = f < TRIM_RATIO * t
    plan = configured_plan(trim)
    fwd = forward_equivalents(plan)
    plant_games = min(int(n_plant), int(n_train))
    readout = readout_fraction * n_held
    minutes = {"pass": n_train * D / t / 60.0,
               "stage_c": fwd * n_held * D / f / 60.0,
               "plant": ((2 * plant_games + readout) * D / t + PLANT_FORWARDS * readout * D / f) / 60.0}
    total = sum(minutes.values())
    wall = FIXED_HOURS + total / 60.0
    return {"rates": {"trainer": t, "forwards": f, "decisions_per_game": D, "forward_to_trainer": f / t},
            "games": {"train": int(n_train), "held": int(n_held), "plant": plant_games},
            "trim": bool(trim), "minutes": minutes, "total_minutes": total,
            "cuts": {k: int(CUT_FACTOR * v + CUT_PAD_MIN) for k, v in minutes.items()},
            "wall_hours": wall, "cost": wall * HOST_RATE,
            "read_plan": plan, "forward_equivalents": fwd,
            "constants": {"host_rate": HOST_RATE, "fixed_hours": FIXED_HOURS, "kl_fraction": KL_FRACTION,
                          "kl_evals": KL_EVALS, "readout_fraction": readout_fraction, "trim_ratio": TRIM_RATIO}}


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
    ap.add_argument("--left-min", type=int, default=None, help="the minutes the run may use")
    ap.add_argument("--line", action="store_true",
                    help="print 'trim total left pass_cut stage_c_cut plant_cut train held' for a shell")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = json.loads(args.pass_summary.read_text(encoding="utf-8"))
    forwards = json.loads(args.forwards.read_text(encoding="utf-8"))
    counts = label_counts(args.labels)
    plan = project(pass_rate(summary), forwards["positions_per_second"], summary["decisions"] / summary["games"],
                   counts["train"], counts["held"], args.n_plant, args.readout_fraction)
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
