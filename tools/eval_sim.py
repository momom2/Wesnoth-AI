"""Sim-based evaluation: latest checkpoint vs reference checkpoint.

Both policies play through the simulator (~1000× faster than spawning
a Wesnoth subprocess per game, which `eval_vs_builtin.py` does).
Trade-off:

  + Fast: a 30-game eval finishes in ~30-60s on DML/CUDA vs ~7-15min
    on Wesnoth. Makes "after every cluster pull" daily evals
    actually quick.
  + Stable: no Wesnoth subprocess management, no IPC, no log-file
    tailing. The sim is the bit-exact simulator we already trust.
  + Free of the "0% WR vs RCA" floor while self-play hasn't
    learned to win: comparing latest-vs-reference still tells us
    "did training improve the policy this day?" via a non-zero
    win-rate delta -- the most direct daily-progress signal.

  - Doesn't test our policy against ANY external opponent (RCA AI
    in Wesnoth, real humans, etc.). The sim-eval can be gamed by a
    policy that beats its prior self by exploiting sim quirks.
    Periodic Wesnoth-eval (`--backend wesnoth` in eval_daily) is
    the cross-check.

The reference policy is whichever checkpoint we want to call
"yesterday's version." Default selection logic (in `eval_daily`):

  1. The freshest `sim_selfplay_archive_*.pt` (typically 6h old,
     thanks to the archive interval). Direct "did the last 6h of
     training help?" comparison.
  2. Fallback: `supervised_epoch*.pt` (the warmstart anchor).
  3. Fallback: random init (the model's lowest-bar baseline).

Output JSON schema matches `eval_vs_builtin.py` so `eval_daily.py`'s
history-writing code consumes both backends uniformly:

    {
      "n_games":     N,
      "overall":     {"win": ..., "loss": ..., "draw": ..., ...},
      "by_faction":  {...},
      "by_matchup":  {...},
      ...
      "backend":     "sim",
      "reference":   "<path>",
      "wall_seconds": ...
    }

The players and the game between two of them are tools/eval_players.py.

Usage:
    python tools/eval_sim.py \\
        --checkpoint    training/checkpoints/sim_selfplay.pt \\
        --reference     training/checkpoints/sim_selfplay_archive_X.pt \\
        --games         30
    python tools/eval_sim.py --reference random   # untrained baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.device_select import select_inference_device, describe_device
from tools.eval_players import GameResult, _PolicyPair, _load_policy, _play_one_eval_game
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
from tools.selfplay_game import _recruit_cost_lookup
from tools.wesnoth_sim import WesnothSim


log = logging.getLogger("eval_sim")


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def _bucket() -> Dict[str, int]:
    return {"win": 0, "loss": 0, "draw": 0, "timeout": 0, "errored": 0}


def _summarize(
    results: List[GameResult], checkpoint: Path,
    reference: Optional[Path], wall_seconds: float,
) -> Dict:
    """Roll up per-game records into the same JSON schema
    eval_vs_builtin emits."""
    overall = _bucket()
    by_faction: Dict[str, Dict[str, int]] = defaultdict(_bucket)
    by_matchup: Dict[str, Dict[str, int]] = defaultdict(_bucket)
    for r in results:
        overall[r.outcome] += 1
        by_faction[r.our_faction][r.outcome] += 1
        by_matchup[f"{r.our_faction}__vs__{r.opp_faction}"][r.outcome] += 1
    return {
        "n_games":      len(results),
        "overall":      overall,
        "by_faction":   dict(by_faction),
        "by_matchup":   dict(by_matchup),
        "results":      [asdict(r) for r in results],
        "checkpoint":   str(checkpoint),
        "reference":    str(reference) if reference else None,
        "backend":      "sim",
        "wall_seconds": wall_seconds,
    }


def _print_summary(summary: Dict) -> None:
    """Human-readable text summary printed to stdout."""
    n = summary["n_games"]
    o = summary["overall"]
    decisive = o["win"] + o["loss"]
    print()
    print("=" * 72)
    print(f"Sim eval: {Path(summary['checkpoint']).name} vs "
          f"{Path(summary['reference']).name if summary['reference'] else 'random'}")
    print("=" * 72)
    print(f"Games played: {n}")
    print(f"  win:     {o['win']}")
    print(f"  loss:    {o['loss']}")
    print(f"  draw:    {o['draw']}")
    print(f"  timeout: {o['timeout']}")
    print(f"  errored: {o['errored']}")
    print()
    if decisive > 0:
        wr = 100.0 * o["win"] / decisive
        print(f"Win rate (decisive only): {wr:.1f}% "
              f"({o['win']} / {decisive})")
    else:
        print("Win rate: n/a (no decisive games yet -- expected "
              "in the no-kills phase)")
    # Sim-specific diagnostic block: closest approach + attack rate.
    # These are the most useful signals during no-kills training,
    # since WR can't distinguish progress when all games are draws.
    results = summary.get("results", [])
    if results:
        n_with_attack = sum(1 for r in results if r["attack_count_ours"] > 0)
        ca = [r["closest_approach_ours"] for r in results
              if r["closest_approach_ours"] is not None]
        ca_opp = [r["closest_approach_opp"] for r in results
                  if r["closest_approach_opp"] is not None]
        if ca:
            print(f"Closest approach (ours):   mean={sum(ca)/len(ca):.1f}, "
                  f"min={min(ca)}")
        if ca_opp:
            print(f"Closest approach (oppo):   mean={sum(ca_opp)/len(ca_opp):.1f}, "
                  f"min={min(ca_opp)}")
        print(f"Games where we attacked:   {n_with_attack}/{len(results)}")
    print()
    print("=" * 72)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="The CANDIDATE checkpoint (the one we want "
                         "to evaluate).")
    ap.add_argument("--reference", type=str, default="auto",
                    help="The REFERENCE checkpoint to play against. "
                         "Special values: 'auto' -> the freshest "
                         "sim_selfplay_archive_*.pt, falling back "
                         "to the freshest supervised epoch; "
                         "'random' -> random-init baseline.")
    ap.add_argument("--games", type=int, default=30,
                    help="Total games. Half played candidate-on-"
                         "side-1, half candidate-on-side-2 (side "
                         "swap) to neutralize first-mover bias.")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="Torch device for BOTH policies. 'auto' "
                         "= DML (discrete) > CUDA > CPU.")
    ap.add_argument("--forced-faction", default=None,
                    help="If set, every game has at least one side "
                         "playing this faction. 'none' disables "
                         "the module default. Same semantics as "
                         "sim_self_play.")
    ap.add_argument("--save-json", type=Path, default=None,
                    help="Write structured results to PATH.json.")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.checkpoint.exists():
        log.error(f"checkpoint not found: {args.checkpoint}")
        return 1

    # Resolve --reference.
    ref_path: Optional[Path] = None
    if args.reference == "random":
        ref_path = None
    elif args.reference == "auto":
        ckpt_dir = args.checkpoint.parent
        archives = sorted(
            ckpt_dir.glob("sim_selfplay_archive_*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        # Skip the freshest archive if it's the SAME file as the
        # candidate (would compare a checkpoint against itself).
        archives = [p for p in archives if p.resolve() != args.checkpoint.resolve()]
        if archives:
            ref_path = archives[0]
        else:
            sup = sorted(ckpt_dir.glob("supervised_epoch*.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            if sup:
                ref_path = sup[0]
            else:
                sup_rolling = ckpt_dir / "supervised.pt"
                if sup_rolling.exists():
                    ref_path = sup_rolling
        if ref_path is None:
            log.warning("no reference checkpoint found under "
                        f"{ckpt_dir}; falling back to random init")
        else:
            log.info(f"auto-picked reference: {ref_path}")
    else:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            log.error(f"reference not found: {ref_path}")
            return 1

    # Build both policies on the resolved device.
    device = select_inference_device(args.device)
    log.info(f"device: {describe_device(device)}")
    cand = _load_policy(args.checkpoint, device, label="cand")
    ref  = _load_policy(ref_path, device, label="ref")

    # Forced-faction handling. Same translation as sim_self_play.
    forced_faction_arg: object = ...
    if args.forced_faction is not None:
        if args.forced_faction.lower() == "none":
            forced_faction_arg = None
        else:
            forced_faction_arg = args.forced_faction

    cost_lookup = _recruit_cost_lookup()
    _ = cost_lookup  # touched to surface missing unit_stats.json early

    rng = random.Random(args.seed)
    results: List[GameResult] = []
    t_start = time.perf_counter()
    half = args.games // 2
    # First half: candidate on side 1. Second half: candidate on side 2.
    # Same set of scenarios is sampled fresh by rng -- not a literal
    # swap of the same N games, but a fair statistical control on
    # first-mover bias.
    for g_idx in range(args.games):
        our_side = 1 if g_idx < half else 2
        setup = random_setup(rng, forced_faction=forced_faction_arg)
        game_label = f"eval{g_idx}"
        try:
            gs = build_scenario_gamestate(setup)
            sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                             max_turns=args.max_turns)
            # Each game its own dice (the 2026-09-13 luck-stream fix of
            # tools/elo_eval_game.py): unsalted, every game replayed one
            # luck vector.
            sim._seed_salt = f"eval_sim:{args.seed}:g{g_idx}"
        except Exception as e:
            log.warning(f"skipping {setup.label()}: {e}")
            continue
        if hasattr(cand, "reset_game"):
            cand.reset_game(game_label)
        if hasattr(ref,  "reset_game"):
            ref.reset_game(game_label)
        pair_cand = _PolicyPair(policy=cand, label="cand",
                                side=our_side)
        pair_ref  = _PolicyPair(policy=ref,  label="ref",
                                side=(3 - our_side))
        try:
            r = _play_one_eval_game(sim, pair_cand, pair_ref,
                                    game_label=game_label)
        except Exception as e:
            log.exception(f"game {g_idx} crashed: {e}")
            cand.drop_pending(game_label)
            ref.drop_pending(game_label)
            results.append(GameResult(
                scenario_id=setup.scenario_id,
                our_faction="", opp_faction="", our_side=our_side,
                outcome="errored", turns=0, our_actions=0,
                wall_seconds=0.0,
            ))
            continue
        results.append(r)
        sys.stderr.write(
            f"  game {g_idx+1}/{args.games}  side={our_side}  "
            f"{r.our_faction[:6]:>6} v {r.opp_faction[:6]:<6}  "
            f"-> {r.outcome:8}  turns={r.turns}  "
            f"approach={r.closest_approach_ours}\n"
        )
    wall = time.perf_counter() - t_start

    summary = _summarize(results, args.checkpoint, ref_path, wall)
    _print_summary(summary)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8")
        log.info(f"results written to {args.save_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
