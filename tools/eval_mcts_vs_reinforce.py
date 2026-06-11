"""Head-to-head eval: MCTS-policy vs REINFORCE-sampling policy.

Both players share the SAME trained checkpoint -- the only difference
is the decision rule:
  - MCTS: run `mcts_search` for `--mcts-sims` simulations and pick
    `best_action` from the resulting visit counts.
  - REINFORCE: sample once from the policy logits (mirroring
    `TransformerPolicy.select_action` at training time).

We don't need Wesnoth in the loop for this -- both players are pure
sim consumers, so the harness runs entirely in-process.

Output: a Wilson-interval win-rate for MCTS over REINFORCE, plus a
side-balanced breakdown (MCTS-as-side-1 / MCTS-as-side-2 reported
separately to expose any side bias).

Usage:
    python tools/eval_mcts_vs_reinforce.py \\
      --checkpoint training/checkpoints/sim_selfplay.pt \\
      --games 40 \\
      --mcts-sims 64 \\
      --out docs/mcts_vs_reinforce_eval.md

Counts as a calibration result: low simulations + small N produce a
quick smoke; bump both for a real headline number once a trained
checkpoint is available.

Dependencies: transformer_policy, tools.mcts, tools.scenario_pool,
              tools.wesnoth_sim, classes.
Dependents: standalone CLI; reported in BACKLOG / readiness scorecard.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformer_policy import TransformerPolicy
from tools.mcts import MCTSConfig, best_action, mcts_search
from tools.scenario_pool import random_setup, build_scenario_gamestate
from tools.wesnoth_sim import WesnothSim, PvPDefaults

log = logging.getLogger("eval_mcts_vs_reinforce")


@dataclass
class GameOutcome:
    """One game's result. `mcts_side` ∈ {1, 2}; `winner` ∈ {1, 2, 0}
    (0 = draw / timeout). `mcts_won` is the canonical bit we
    aggregate. `turns` and `wallclock` for cost reporting."""
    mcts_side: int
    winner: int
    mcts_won: bool   # True iff `winner == mcts_side`
    draw: bool       # True iff winner == 0
    turns: int
    wallclock: float
    scenario_label: str


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple:
    """Symmetric Wilson interval for a binomial proportion. Returns
    (lo, hi). Used for win-rate confidence in the tiny-N regime
    where the normal approximation is biased."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load_policy(checkpoint: Path) -> TransformerPolicy:
    """Load a TransformerPolicy at the checkpoint's saved arch.
    Handles the pre-C51 -> C51 transition the same way
    `tools/collect_cliffness.py` does (partial load, value_head
    reset to random C51 init if checkpoint is pre-C51)."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_arch = raw.get("arch", {})
    d_model    = int(saved_arch.get("d_model", 512))
    num_layers = int(saved_arch.get("num_layers", 6))
    num_heads  = int(saved_arch.get("num_heads", 8))
    d_ff       = int(saved_arch.get("d_ff", 2048))
    log.info(
        f"checkpoint arch: d_model={d_model} num_layers={num_layers} "
        f"num_heads={num_heads} d_ff={d_ff}"
    )
    policy = TransformerPolicy(
        d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, d_ff=d_ff,
    )
    try:
        policy.load_checkpoint(checkpoint)
    except RuntimeError as e:
        ms = raw.get("model_state", {})
        vh = ms.get("value_head.2.weight", None)
        if vh is not None and vh.shape[0] == 1:
            log.warning(
                "checkpoint predates the C51 value head; partial-load "
                "and run with a random-init value head. MCTS Q-values "
                "will be biased toward zero / uniform priors. The "
                "win-rate measurement is still valid (both sides use "
                "the same value head), but the absolute Q magnitudes "
                "won't be meaningful."
            )
            for k in list(ms.keys()):
                if k.startswith("value_head."):
                    del ms[k]
            policy._model.load_state_dict(ms, strict=False)
            enc_state = raw.get("encoder_state")
            if enc_state is not None:
                policy._encoder.load_state_dict(enc_state, strict=False)
            policy._snapshot_inference_weights()
        else:
            raise
    return policy


def _play_one_game(
    policy:       TransformerPolicy,
    setup,
    mcts_side:    int,
    mcts_config:  MCTSConfig,
    *,
    seed:         int,  # accepted for label uniqueness; sim RNG is
                        # handled internally by WesnothSim
    max_actions:  int = 800,
) -> GameOutcome:
    """Play one game with one side using MCTS and the other using
    REINFORCE sampling. Same policy weights on both sides.

    `mcts_side` is which side number gets MCTS (the other gets
    REINFORCE). We pick this externally per game so the caller can
    balance MCTS-as-1 vs MCTS-as-2 across the run.
    """
    pvp = PvPDefaults(
        starting_gold=100, village_gold=2, village_support=1,
        experience_modifier=70,
    )
    gs = build_scenario_gamestate(
        setup,
        starting_gold=None,  # use the scenario's [side] gold attrs.
        base_income=pvp.base_income,
        village_gold=pvp.village_gold,
        village_upkeep=pvp.village_support,
        experience_modifier=pvp.experience_modifier,
    )
    sim = WesnothSim(gs, scenario_id=setup.scenario_id)

    t0 = time.perf_counter()
    n_actions = 0
    game_label = f"eval_{seed}"

    while not sim.done and n_actions < max_actions:
        pre_state = copy.deepcopy(sim.gs)
        side = pre_state.global_info.current_side
        if side == mcts_side:
            # MCTS branch.
            root = mcts_search(
                sim, policy._inference_model,
                policy._inference_encoder, mcts_config,
            )
            action = best_action(root)
            if action is None:
                # MCTS couldn't expand any legal action -- bail.
                log.warning(
                    "MCTS produced no action; treating as side loss")
                sim.done = True
                sim.winner = 3 - mcts_side
                break
        else:
            action = policy.select_action(
                pre_state, game_label=game_label)
        sim.step(action)
        n_actions += 1

    # Drop any pending REINFORCE transitions so the policy state
    # stays clean across games.
    policy.drop_pending(game_label)

    winner = sim.winner if sim.done else 0
    return GameOutcome(
        mcts_side=mcts_side,
        winner=winner,
        mcts_won=(winner == mcts_side),
        draw=(winner == 0),
        turns=sim.gs.global_info.turn_number,
        wallclock=time.perf_counter() - t0,
        scenario_label=setup.label(),
    )


def _format_report(
    games: list,
    *,
    checkpoint: str,
    mcts_sims: int,
    n_games: int,
) -> str:
    """Build the markdown report. Reports overall WR with Wilson
    bounds + per-side breakdown."""
    n = len(games)
    if n == 0:
        return "# MCTS vs REINFORCE\n\nno games completed"

    mcts_wins = sum(1 for g in games if g.mcts_won)
    draws     = sum(1 for g in games if g.draw)
    re_wins   = n - mcts_wins - draws
    overall_lo, overall_hi = _wilson(mcts_wins, n)
    overall_wr = mcts_wins / n if n else 0.0

    # Per-side: MCTS-as-side-1 vs MCTS-as-side-2.
    side1 = [g for g in games if g.mcts_side == 1]
    side2 = [g for g in games if g.mcts_side == 2]
    s1_wins = sum(1 for g in side1 if g.mcts_won)
    s2_wins = sum(1 for g in side2 if g.mcts_won)
    s1_lo, s1_hi = _wilson(s1_wins, len(side1))
    s2_lo, s2_hi = _wilson(s2_wins, len(side2))

    avg_turns = sum(g.turns for g in games) / n
    avg_wall  = sum(g.wallclock for g in games) / n

    pre_c51 = "pre-C51" in checkpoint or "random-init" in checkpoint
    too_passive = (draws / n > 0.8)
    lines = [
        "# MCTS vs REINFORCE eval",
        "",
    ]
    if pre_c51 or too_passive:
        notes = []
        if pre_c51:
            notes.append(
                "the checkpoint predates the C51 value head (pre-2026-"
                "05-10), so MCTS Q-values are biased toward a random-"
                "init head"
            )
        if too_passive:
            notes.append(
                f"{draws}/{n} games ended in draws (likely both sides "
                f"sitting still until `--max-actions` -- passive policy "
                f"behavior is a known supervised_epoch3 limitation, see "
                f"BACKLOG `Out-of-scope`)"
            )
        lines += [
            "> **CAVEAT:** " + "; ".join(notes) + ". Re-run after the "
            "next training cycle produces an aggressive C51-aware "
            "checkpoint for a meaningful head-to-head number.",
            "",
        ]
    lines += [
        f"- **Checkpoint:** `{checkpoint}`",
        f"- **MCTS simulations / decision:** {mcts_sims}",
        f"- **Games played:** {n} (target {n_games})",
        f"- **Avg turns/game:** {avg_turns:.1f}",
        f"- **Avg wallclock/game:** {avg_wall:.1f}s",
        "",
        "## Overall",
        "",
        f"- **MCTS wins:** {mcts_wins} / {n}",
        f"- **REINFORCE wins:** {re_wins} / {n}",
        f"- **Draws / timeouts:** {draws} / {n}",
        f"- **MCTS win-rate:** {overall_wr:.3f}  "
        f"(95% Wilson [{overall_lo:.3f}, {overall_hi:.3f}])",
        "",
        "## Side breakdown (controls for side bias)",
        "",
        f"- **MCTS as side 1:** {s1_wins} / {len(side1)} "
        f"= {(s1_wins / max(1, len(side1))):.3f}  "
        f"[{s1_lo:.3f}, {s1_hi:.3f}]",
        f"- **MCTS as side 2:** {s2_wins} / {len(side2)} "
        f"= {(s2_wins / max(1, len(side2))):.3f}  "
        f"[{s2_lo:.3f}, {s2_hi:.3f}]",
        "",
        "## Per-game log",
        "",
        "| # | scenario | MCTS side | winner | turns | wallclock |",
        "|--:|---|--:|--:|--:|--:|",
    ]
    for i, g in enumerate(games):
        winner_lbl = ("draw/to" if g.draw else f"side {g.winner}")
        lines.append(
            f"| {i+1} | {g.scenario_label} | {g.mcts_side} | "
            f"{winner_lbl} | {g.turns} | {g.wallclock:.1f}s |"
        )
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/checkpoints/supervised_epoch3.pt"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--mcts-sims", type=int, default=32,
                    help="simulations per MCTS decision")
    ap.add_argument("--mcts-batch", type=int, default=4)
    ap.add_argument("--max-actions", type=int, default=800,
                    help="hard cap per game; counts as draw if hit")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/mcts_vs_reinforce_eval.md"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    policy = _load_policy(args.checkpoint)
    policy._inference_model.eval()
    policy._inference_encoder.eval()
    # Record whether the checkpoint had a C51-shaped value head; we
    # tag the label so `_format_report` can emit the right caveat.
    raw = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)
    ms = raw.get("model_state", {})
    vh = ms.get("value_head.2.weight", None) if isinstance(ms, dict) else None
    ckpt_label = (
        f"{args.checkpoint} (pre-C51, partial load)"
        if (vh is not None and vh.shape[0] == 1)
        else str(args.checkpoint)
    )
    mcts_config = MCTSConfig(
        n_simulations=args.mcts_sims,
        batch_size=args.mcts_batch,
    )

    rng = random.Random(args.seed)
    games: list = []
    t_run = time.perf_counter()
    for i in range(args.games):
        setup = random_setup(rng)
        # Alternate MCTS-side across games so the per-side breakdown
        # has a balanced denominator.
        mcts_side = 1 if i % 2 == 0 else 2
        game_seed = args.seed + i + 1
        try:
            outcome = _play_one_game(
                policy, setup, mcts_side, mcts_config,
                seed=game_seed, max_actions=args.max_actions,
            )
        except Exception as e:
            log.warning(f"game {i+1} crashed: {e!r}; skipping")
            continue
        games.append(outcome)
        log.info(
            f"  game {i+1}/{args.games}: MCTS=side{mcts_side}  "
            f"winner={'draw' if outcome.draw else outcome.winner}  "
            f"turns={outcome.turns}  "
            f"wall={outcome.wallclock:.1f}s  ({setup.scenario_id})"
        )

    md = _format_report(
        games,
        checkpoint=ckpt_label,
        mcts_sims=args.mcts_sims,
        n_games=args.games,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    log.info(
        f"wrote {args.out}  (total {time.perf_counter()-t_run:.1f}s, "
        f"{len(games)} games)"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv)
