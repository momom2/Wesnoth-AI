"""Collect a cliffness histogram from a trained model over real game states.

Cliffness (std of the C51 value distribution) measures the model's
per-state value uncertainty. The MCTS adaptive sim-budget and the
Bayesian-precision bootstrap weighting both consume cliffness, but
both are OFF by default pending empirical calibration -- we needed
to see what the actual distribution looks like before picking
`cliffness_max`, `n_simulations_{min,max}`, and
`cliffness_bootstrap_alpha`.

What this script does:
  1. Loads the trained policy from a checkpoint.
  2. Walks K real replays from `replays_dataset/`, applying each
     command in order.
  3. At each player-action step (move / attack / recruit) BEFORE
     applying the command, computes the model's cliffness on the
     current state via direct encoder + model forward.
  4. Aggregates the cliffness values into a histogram.
  5. Reports percentiles + the empirical max plus a markdown table
     suitable for pasting into docs/cliffness_calibration.md.

Why real replays not self-play: the trained model is competent but
still mid-training, so its self-play states are systematically less
representative than the human-game distribution. The whole point of
the calibration is to set thresholds based on positions the agent
WILL encounter; humans + agent both encounter a similar mix of
"clear" vs "cliff" positions, so this is the right reference
distribution.

Usage:
    python tools/collect_cliffness.py \\
      --checkpoint training/checkpoints/supervised_epoch3.pt \\
      --replays-dir replays_dataset/ \\
      --n-replays 30 \\
      --out docs/cliffness_calibration.md

Dependencies: transformer_policy, encoder, model, tools.replay_dataset.
Dependents: standalone CLI (calibration data collection).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

# Allow running both as a module (`python -m tools.collect_cliffness`)
# and as a script (`python tools/collect_cliffness.py`).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformer_policy import TransformerPolicy
from tools.replay_dataset import (
    _apply_command,
    _build_initial_gamestate,
    _setup_scenario_events,
)

log = logging.getLogger("collect_cliffness")


def _enumerate_replays(replays_dir: Path, n: int, seed: int) -> List[Path]:
    """Pick `n` distinct .json.gz replays from `replays_dir`,
    deterministically given the seed."""
    candidates = sorted(replays_dir.glob("*.json.gz"))
    if not candidates:
        raise SystemExit(f"no .json.gz replays under {replays_dir}")
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def _is_decision_step(cmd) -> bool:
    """Whether `cmd` is a player-action that a policy would have
    chosen (vs an engine bookkeeping step like init_side / end_turn).
    We collect cliffness on these BEFORE the command applies, so the
    state matches what the policy would have seen."""
    if not cmd:
        return False
    return cmd[0] in ("move", "attack", "recruit", "recall")


def collect_from_replay(
    policy: TransformerPolicy,
    replay_path: Path,
    max_decisions: int = 200,
) -> List[float]:
    """Walk one replay, sampling cliffness at decision steps.

    Returns the list of cliffness values observed. Cap at
    `max_decisions` to keep cost bounded on long endgames.
    """
    with gzip.open(replay_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    encoder = policy._inference_encoder
    model = policy._inference_model

    cliffs: List[float] = []
    for cmd in data["commands"]:
        if _is_decision_step(cmd):
            try:
                with torch.no_grad():
                    encoded = encoder.encode(gs)
                    out = model(encoded)
                    cliffs.append(float(out.cliffness.squeeze().item()))
            except Exception as e:
                # Encoder/model errors on rare states (very few units,
                # unusual scenarios) shouldn't kill the sweep -- log
                # and continue.
                log.debug(f"  encode/forward error: {e}")
            if len(cliffs) >= max_decisions:
                break
        try:
            _apply_command(gs, cmd)
        except Exception as e:
            # Cascade failures past the first divergence aren't
            # interesting here; bail on this replay.
            log.debug(f"  apply error at {cmd[0]}: {e}")
            break
    return cliffs


def _format_markdown(
    cliffs: np.ndarray,
    *,
    checkpoint: str,
    n_replays: int,
    n_decisions: int,
) -> str:
    """Build the markdown report. Reports percentiles plus a quick
    histogram suitable for eyeball calibration."""
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(cliffs, pcts)

    # Theoretical max for a C51 value distribution over [-1, +1] is
    # the std of a 2-point {-1, +1} mass = 1.0 (highest possible
    # uncertainty). The "uniform over 51 atoms" baseline std is the
    # std of a uniform distribution on [-1, +1] = 0.577 (1/sqrt(3)).
    # That's the current default `cliffness_max`.
    uniform_std = float(np.sqrt(np.mean(np.linspace(-1, 1, 51) ** 2)))

    pre_c51 = "pre-C51" in checkpoint or "random-init" in checkpoint
    lines = [
        "# Cliffness empirical calibration",
        "",
    ]
    if pre_c51:
        lines += [
            "> **CAVEAT:** This run used a model whose value head was "
            "**randomly initialized** (either no checkpoint, or a "
            "pre-2026-05-10 checkpoint that predates the C51 head). The "
            "histogram below shows the C51 *uniform-prior* baseline, "
            "not a calibrated trained-model distribution. Re-run this "
            "tool after a fresh checkpoint exists from a C51-aware "
            "training run; the meaningful numbers will live in a "
            "follow-up section below this one.",
            "",
        ]
    lines += [
        f"Collected {len(cliffs)} cliffness values from {n_decisions} "
        f"decision steps across {n_replays} replays.",
        "",
        f"- **Checkpoint:** `{checkpoint}`",
        f"- **Atom range:** [-1, +1] over K=51 atoms",
        f"- **Theoretical max (point mass at +/-1):** ~1.0",
        f"- **Uniform-prior baseline (current default `cliffness_max`):** "
        f"{uniform_std:.3f}",
        "",
        "## Percentiles",
        "",
        "| Pct | Cliffness |",
        "|----:|----------:|",
    ]
    for p, v in zip(pcts, pct_vals):
        lines.append(f"| {p:>3} | {v:>9.4f} |")
    lines += [
        "",
        f"- **min:** {float(cliffs.min()):.4f}",
        f"- **mean:** {float(cliffs.mean()):.4f}",
        f"- **std:** {float(cliffs.std()):.4f}",
        f"- **max:** {float(cliffs.max()):.4f}",
        "",
        "## Histogram (16 bins, 0 to max)",
        "",
    ]
    hist, edges = np.histogram(cliffs, bins=16, range=(0, max(0.1, float(cliffs.max()))))
    bar_max = hist.max() if hist.max() > 0 else 1
    for i in range(len(hist)):
        bar = "#" * int(round(40 * hist[i] / bar_max))
        lines.append(
            f"    [{edges[i]:.3f}, {edges[i+1]:.3f}) "
            f"{hist[i]:>5}  {bar}"
        )
    lines += [
        "",
        "## Calibration recommendation",
        "",
        f"- Current default `cliffness_max = {uniform_std:.3f}` "
        f"(uniform-prior baseline).",
        f"- Empirical p99 = {pct_vals[-1]:.3f}. If p99 < default, the "
        f"adaptive-sim-budget rarely saturates and `n_simulations_max` "
        f"is mostly unreached. Consider lowering `cliffness_max` to "
        f"the empirical p95 ({pct_vals[-3]:.3f}) so the budget actually "
        f"varies across positions.",
        f"- If p99 > default, the model has more value uncertainty than "
        f"a uniform prior would predict (suggests the C51 head has "
        f"learned to spread mass on bimodal positions); keep "
        f"`cliffness_max` near the uniform baseline or raise it to "
        f"p99 to avoid clipping.",
        f"- For `cliffness_bootstrap_alpha`: with empirical std at "
        f"{float(cliffs.std()):.3f}, alpha=1.0 (Bayes-optimal) means a "
        f"typical leaf's contribution to ancestor Q is downweighted by "
        f"`1 / (1 + cliffness^2)` ~ "
        f"{1.0 / (1 + float(cliffs.mean())**2):.2f} on average. Start "
        f"with alpha=1.0 on a short calibration run; if MCTS Q-values "
        f"drift noisily, lower alpha; if the search ignores cliff "
        f"signal entirely, raise it.",
    ]
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="path to a .pt checkpoint; if omitted, runs "
                         "with random init (uncalibrated baseline)")
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--n-replays", type=int, default=30,
                    help="how many distinct replays to sweep")
    ap.add_argument("--max-decisions-per-replay", type=int, default=150,
                    help="cap per-replay to keep cost bounded")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/cliffness_calibration.md"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    is_random_init = (
        args.checkpoint is None or str(args.checkpoint) == "-")
    if not is_random_init and not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    if is_random_init:
        log.warning(
            "no checkpoint specified -- using a RANDOM-INIT model. "
            "Output cliffness distribution is the C51 uniform-prior "
            "baseline, not a calibration of trained behavior."
        )
        policy = TransformerPolicy()  # default arch with C51 head.
        ckpt_label = "<random-init>"
    else:
        log.info(f"loading policy from {args.checkpoint}")
        # Peek the checkpoint's saved arch.
        raw = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
        saved_arch = raw.get("arch", {})
        d_model    = int(saved_arch.get("d_model", 512))
        num_layers = int(saved_arch.get("num_layers", 6))
        num_heads  = int(saved_arch.get("num_heads", 8))
        d_ff       = int(saved_arch.get("d_ff", 2048))
        log.info(
            f"  arch: d_model={d_model} num_layers={num_layers} "
            f"num_heads={num_heads} d_ff={d_ff}"
        )
        # Pre-C51 checkpoints have a 1-scalar value_head; the current
        # model has K=51 atoms. Detect and warn -- we can still load
        # everything except the value head, but the cliffness output
        # will be from a randomly-initialized C51 head and not
        # informative.
        ms = raw.get("model_state", raw.get("model", {}))
        vh_shape = (ms.get("value_head.2.weight", None)
                    if isinstance(ms, dict) else None)
        pre_c51 = (vh_shape is not None and vh_shape.shape[0] == 1)
        if pre_c51:
            log.warning(
                "checkpoint predates the C51 value head (scalar "
                "value_head). Continuing with RANDOM-INIT value head; "
                "trunk + actor head come from the checkpoint, but "
                "the cliffness output is uncalibrated."
            )
        policy = TransformerPolicy(
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, d_ff=d_ff,
        )
        try:
            policy.load_checkpoint(args.checkpoint)
            ckpt_label = str(args.checkpoint)
        except RuntimeError as e:
            if pre_c51:
                # Load with strict=False semantics: monkey-patch by
                # stripping pre-C51 value-head keys from the state.
                log.info("falling back to partial load (skipping "
                         "incompatible value_head keys)")
                for k in list(ms.keys()):
                    if k.startswith("value_head."):
                        del ms[k]
                policy._model.load_state_dict(ms, strict=False)
                enc_state = raw.get("encoder_state")
                if enc_state is not None:
                    policy._encoder.load_state_dict(enc_state, strict=False)
                policy._snapshot_inference_weights()
                ckpt_label = f"{args.checkpoint} (pre-C51, partial load)"
            else:
                raise
    # Inference-only: snapshot weights & switch to eval.
    policy._inference_model.eval()
    policy._inference_encoder.eval()

    replays = _enumerate_replays(args.replays_dir, args.n_replays, args.seed)
    log.info(f"sampled {len(replays)} replays")

    all_cliffs: List[float] = []
    t0 = time.perf_counter()
    for i, p in enumerate(replays):
        cliffs = collect_from_replay(
            policy, p, max_decisions=args.max_decisions_per_replay)
        all_cliffs.extend(cliffs)
        if (i + 1) % 5 == 0 or i + 1 == len(replays):
            log.info(
                f"  {i+1}/{len(replays)} replays, "
                f"{len(all_cliffs)} cliffness samples so far, "
                f"elapsed={time.perf_counter()-t0:.1f}s"
            )

    if not all_cliffs:
        raise SystemExit("collected no cliffness samples; aborting")

    arr = np.asarray(all_cliffs, dtype=np.float64)
    md = _format_markdown(
        arr,
        checkpoint=ckpt_label,
        n_replays=len(replays),
        n_decisions=len(arr),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    log.info(f"wrote {args.out}  (n={len(arr)})")
    # NB: don't print() the full md -- Windows console default codec
    # (cp1252) chokes on non-ASCII characters we may add later.
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv)
