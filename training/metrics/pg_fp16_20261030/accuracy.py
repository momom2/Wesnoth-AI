"""The fp16 evaluator (tools/pg_eval.py `fp16`) against the float32 padded
one, on held-out games of the local stage 1: theta_k's log probabilities,
a moved network's, the KL reading between them and each game-side's
log W = sum_t (log pi' - log pi_k), what the step's exact reading ΔĴ
weighs by (docs/training_design_20261019/current_design.md §1 step 5).

The direction is random (each parameter's entries N(0, 1) times its mean
magnitude), its step found on the fp16 path at KL about `--kl` a decision,
both paths then read at that step.

    set HIP_VISIBLE_DEVICES=1
    <rocm python> training/metrics/pg_fp16_20261030/accuracy.py --checkpoint <own2.pt> \
        --games <run>/games_s00 ... --labels <run>/labels.jsonl --held-games 12 --out accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.pg_eval import EvalConfig, EvalContext  # noqa: E402
from tools.pg_positions import RecordIndex  # noqa: E402
from wesnoth_ai.param_groups import named_model_parameters  # noqa: E402
from wesnoth_ai.pg_records import read_labels  # noqa: E402


def stats(x: np.ndarray) -> dict:
    a = np.abs(x)
    return {"mean": float(x.mean()), "mean_abs": float(a.mean()), "p99_abs": float(np.quantile(a, 0.99)),
            "max_abs": float(a.max()), "n": int(x.size)}


def side_sums(ctx: EvalContext, per_decision: np.ndarray) -> np.ndarray:
    return np.array([per_decision[gs.start:gs.stop].sum() for gs in ctx.sides])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--games", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--held-games", type=int, default=12)
    ap.add_argument("--kl", type=float, default=1e-3, help="the step's target KL a decision")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--streams32", type=int, default=8,
                    help="the float32 path's streams: its padded forward takes about 3 s at 64 rows here, "
                         "the kind of call that reset the display on 2026-10-28")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    rows = read_labels(args.labels)
    chosen = sorted({r.game for r in rows if r.heldout})[:args.held_games]
    index = RecordIndex(args.games)
    records = {g: index.path(g) for g in chosen}
    held = [r for r in rows if r.heldout and r.game in set(chosen)]
    out = {"games": len(chosen)}
    contexts = {}
    for fp16 in (True, False):
        contexts[fp16] = EvalContext(args.checkpoint, records, held, dev,
                                     EvalConfig(streams=args.streams if fp16 else args.streams32,
                                                workers=args.workers, kl_fraction=1.0, fp16=fp16))
    try:
        ctx16, ctx32 = contexts[True], contexts[False]
        g = torch.Generator().manual_seed(20261030)
        direction = {n: torch.randn(p.shape, generator=g) * float(p.detach().abs().mean() + 1e-6)
                     for n, p in named_model_parameters(ctx16.model, ctx16.encoder)}
        ev16, ev32 = ctx16.bind(direction), ctx32.bind(direction)
        # The step on the fp16 path: KL grows as s^2 near 0, so two readings place it.
        s0 = 1e-3
        k0 = ev16.mean_kl(s0)
        s = s0 * (args.kl / k0) ** 0.5 if k0 > 0 else s0
        timings = {}
        readings = {}
        for name, ev in (("fp16", ev16), ("fp32", ev32)):
            t0 = time.perf_counter()
            ev.evaluate([s])
            base, moved = ev.logp(0.0), ev.logp(s)
            kl = ev.mean_kl(s)
            timings[name] = time.perf_counter() - t0
            readings[name] = {"base": base, "moved": moved, "kl": kl}
        r16, r32 = readings["fp16"], readings["fp32"]
        w16 = side_sums(ctx16, r16["moved"] - r16["base"])
        w32 = side_sums(ctx32, r32["moved"] - r32["base"])
        out.update({
            "decisions": int(ctx16.decisions), "sides": len(ctx16.sides), "step": s, "kl_probe": {"s": s0, "kl": k0},
            "logp_theta_k": stats(r16["base"] - r32["base"]),
            "logp_moved": stats(r16["moved"] - r32["moved"]),
            "log_ratio_per_decision": {"fp32": stats(r32["moved"] - r32["base"]),
                                       "fp16_minus_fp32": stats((r16["moved"] - r16["base"])
                                                                - (r32["moved"] - r32["base"]))},
            "kl": {"fp16": r16["kl"], "fp32": r32["kl"], "ratio": r16["kl"] / r32["kl"] if r32["kl"] else None},
            "log_W_per_side": {"fp32": stats(w32), "fp16_minus_fp32": stats(w16 - w32),
                               "W_fp16_over_W_fp32": stats(np.exp(w16 - w32) - 1.0)},
            "seconds": timings,
        })
    finally:
        for ctx in contexts.values():
            ctx.close()
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
