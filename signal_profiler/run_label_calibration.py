#!/usr/bin/env python3
"""Label calibration harvest (arm VG2, 2026-09-02).

"Every parameter becomes a measurement": play N grounding-enabled
games from a checkpoint and, from the PAIRED (search-estimate,
rollout-outcome) labels the capture machinery produces, estimate
what the learner will estimate online -- the search estimate's
systematic bias b and residual variance sigma2 -- plus the
distributions behind them, so the implied mixture can be read
before any training leg runs.

Also reports what the per-state Gaussian gradient will look like
(|gap| after bias correction, over sigma2) next to what the old
categorical term would have charged, so the "no blow-up" claim is
checked in numbers.

    python signal_profiler/run_label_calibration.py \
        --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --games 12 --out calib_seed.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_profiler.experience_harvest import (  # noqa: E402
    harvest_experiences, make_policy,
)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calibrated-checkpoint", type=Path, default=None,
                    help="Write a copy of --checkpoint carrying "
                         "training_meta (lambda0 from the measured "
                         "movement's upper spread, b/sigma2 from the "
                         "paired labels) -- the prior every arm "
                         "starting from this checkpoint loads.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else None)
    # Generous per-game caps: this run exists to collect pairs.
    from tools.value_grounding import GroundingConfig
    factory = make_policy(args.checkpoint, device, grounding=True)
    gen = factory()
    gen._ground_cfg = GroundingConfig(
        enabled=True, capture_prob=0.5,
        max_consist_per_game=16, max_rollout_per_game=8)
    batch, outcomes = harvest_experiences(
        gen, args.games, args.seed, max_turns=args.max_turns)
    base = gen._base

    consist = [e for e in batch
               if getattr(e, "label_kind", "game") == "consist"]
    paired = [e for e in consist if e.z_pair is not None]
    rolls = [e for e in batch
             if getattr(e, "label_kind", "game") == "roll"]
    print(f"experiences={len(batch)} consist={len(consist)} "
          f"paired={len(paired)} rollouts={len(rolls)}")
    if len(paired) < 4:
        print("too few pairs; nothing to calibrate")
        return 1

    # Head predictions on the paired states (what the loss sees).
    with torch.no_grad():
        preds = [float(base._model(base._encoder.encode(e.game_state))
                       .value.squeeze().item()) for e in paired]

    diffs = [e.z - e.z_pair for e in paired]          # search - rollout
    bias = st.fmean(diffs)
    var_diff = st.pvariance(diffs)
    roll_noise = st.fmean(max(0.0, 1.0 - (e.z - bias) ** 2)
                          for e in paired)
    sigma2 = max(var_diff - roll_noise, 1e-3)
    gaps_raw = [e.z - v for e, v in zip(paired, preds)]
    gaps_corr = [(e.z - bias) - v for e, v in zip(paired, preds)]
    # Per-state gradient scale of the Gaussian term vs the old
    # categorical charge (-log p on a confident head ~ up to ~14).
    gauss_grad = [abs(g) / sigma2 for g in gaps_corr]

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    result = {
        "checkpoint": str(args.checkpoint),
        "n_pairs": len(paired), "n_consist": len(consist),
        "n_roll": len(rolls),
        "bias_hat": bias, "var_diff": var_diff,
        "roll_noise_mean": roll_noise, "sigma2_hat": sigma2,
        "gap_raw_mean": st.fmean(gaps_raw),
        "gap_raw_abs_mean": st.fmean(abs(g) for g in gaps_raw),
        "gap_corr_abs_mean": st.fmean(abs(g) for g in gaps_corr),
        "gauss_grad_mean": st.fmean(gauss_grad),
        "gauss_grad_p90": q(gauss_grad, 0.9),
        "search_z_mean": st.fmean(e.z for e in paired),
        "roll_z_mean": st.fmean(e.z_pair for e in paired),
        "pred_mean": st.fmean(preds),
        "roll_win_frac": st.fmean(1.0 if e.z_pair > 0 else 0.0
                                  for e in paired),
        "outcomes": [{"winner": getattr(o, "winner", None),
                      "turns": getattr(o, "turns", None)}
                     for o in outcomes],
    }
    # --- lambda0: the controller's own step, applied OFFLINE -------
    # One production iteration (the real replay-update loop) at
    # lambda=1 on a scratch copy, movement measured on the captured
    # consulted states; lambda0 = p90(dv)/delta -- the upper spread,
    # a risk posture (too much lambda = slower learning for a couple
    # of iterations; too little = collapse), then relaxed by the
    # live controller as iterations confirm dv < delta.
    from tools.signal_telemetry import consult_values
    scratch = factory()
    sb = scratch._base
    sb._trainer.config.grad_clip = 1.0          # production, not profiler
    pre = consult_values(sb, batch, cap=10_000)
    scratch._vg2_prepare(batch, pre)            # anchors + b/sigma2
    sb._trainer.config.trust_lambda = 1.0
    with scratch._lock:
        scratch._queue = list(batch)
    scratch.train_step()
    states, v0 = pre
    with torch.no_grad():
        v1 = [float(sb._model(sb._encoder.encode(s)).value.squeeze().item())
              for s in states]
    dvs = sorted(abs(a - b) for a, b in zip(v0, v1))
    delta = float(sb._trainer.config.trust_delta)
    dv_mean = st.fmean(dvs)
    dv_p90 = q(dvs, 0.9)
    lambda0 = dv_p90 / delta
    result.update({
        "trust_delta": delta, "dv_mean_at_lambda1": dv_mean,
        "dv_p90_at_lambda1": dv_p90, "lambda0": lambda0,
        "policy_bias_hat": float(sb._trainer.config.consist_bias),
        "policy_sigma2_hat": float(sb._trainer.config.consist_sigma2),
        "n_consult": len(states),
    })
    print(f"lambda0: dv at lambda=1 mean {dv_mean:.4f} p90 {dv_p90:.4f} "
          f"on {len(states)} consulted states; delta {delta} -> "
          f"lambda0 = {lambda0:.2f}")
    if args.calibrated_checkpoint is not None:
        fresh = factory()                       # untouched seed weights
        fresh._trust_lambda = lambda0
        fresh._consist_bias = float(sb._trainer.config.consist_bias)
        fresh._consist_sigma2 = float(sb._trainer.config.consist_sigma2)
        fresh._dv_history = [dv_mean]
        fresh._base._trainer.config.grad_clip = 1.0
        fresh.save_checkpoint(args.calibrated_checkpoint)
        print(f"wrote calibrated checkpoint {args.calibrated_checkpoint} "
              f"with training_meta {fresh.training_meta()['vg2']}")

    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(f"bias_hat={bias:+.3f}  var(search-roll)={var_diff:.3f}  "
          f"rollout noise={roll_noise:.3f}  sigma2_hat={sigma2:.3f}")
    print(f"gap search-vs-head: raw mean {result['gap_raw_mean']:+.3f} "
          f"(|.| {result['gap_raw_abs_mean']:.3f}); bias-corrected "
          f"|.| {result['gap_corr_abs_mean']:.3f}")
    print(f"Gaussian per-state grad scale: mean "
          f"{result['gauss_grad_mean']:.2f} p90 "
          f"{result['gauss_grad_p90']:.2f}  (categorical was ~"
          f"{-math.log(1e-6):.0f} on confident misses)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
