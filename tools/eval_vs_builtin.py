"""Evaluate a trained checkpoint against Wesnoth's default RCA AI.

Generates a matchup grid (maps x faction pairs x side-swap), runs each
game with `wesnoth --test eval_<id>`, aggregates outcomes, prints a
report.

Usage:
    python tools/eval_vs_builtin.py --checkpoint training/checkpoints/supervised.pt
    python tools/eval_vs_builtin.py --checkpoint training/checkpoints/supervised_epoch3.pt \\
        --maps caves den --pairs cross --no-swap --parallel 2
    python tools/eval_vs_builtin.py --checkpoint X --report-only results.json

Tradeoffs of the default config:
  - 6 maps x 21 faction pairs (incl. mirrors) x 2 side-swaps = 252 games.
    At ~5 min/game wall, that's ~21h on 4 parallel processes. Use the
    flags below to cut this down for a quick read.
  - --pairs cross drops mirrors -> 15 pairs -> 180 games (~15h @ 4-way).
  - --maps caves -> 1 map -> 42 games (~3.5h @ 4-way).
  - --no-swap -> halves it again.

Recommended quick eval: --maps caves den --pairs cross --no-swap
  -> 2 maps x 15 pairs = 30 games, ~2h on 4 parallel.

Outcomes:
  win    = the side our policy played on won (leader of opposing side
           died OR the opposing side surrendered all units)
  loss   = the OTHER side's leader survived; our leader died
  draw   = mutual elimination (rare with end_level=victory_when_enemies_defeated)
  timeout= game ran past max_actions actions without resolving
  errored= game crashed / Wesnoth failed to launch / state parse failed

Reported metrics:
  Overall win rate (W / (W + L)), excluding draws/timeouts.
  Per-faction win rate (when our policy played that faction).
  Per-map win rate.
  Per-matchup detail (how often a specific faction beats another).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Make project root + tools/ importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from constants import ADDONS_PATH, MAX_ACTIONS_PER_GAME, SCENARIOS_PATH
from encoder import GameStateEncoder
from eval_runner import GameResult, play_many
from eval_scenarios import (
    FACTIONS,
    FACTION_BY_NAME,
    MAPS,
    MAP_BY_SHORT,
    all_pairs,
    build_matchup_grid,
    cross_pairs,
    generate_eval_scenarios,
)
from model import WesnothModel
from state_converter import StateConverter


log = logging.getLogger("eval_vs_builtin")


# ---------------------------------------------------------------------
# Model + converter setup
# ---------------------------------------------------------------------

def _load_checkpoint(
    ckpt_path: Path,
    device:    torch.device,
):
    """Load encoder + model + their vocabs from a supervised .pt file.

    Mirrors transformer_policy.load_checkpoint (we don't go through the
    Policy registry here -- this is pure inference, no rollout buffers).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    d_model    = arch.get("d_model",    128)
    num_layers = arch.get("num_layers", 3)
    num_heads  = arch.get("num_heads",  4)
    d_ff       = arch.get("d_ff",       256)
    log.info(
        f"checkpoint arch: d_model={d_model} layers={num_layers} "
        f"heads={num_heads} d_ff={d_ff}"
    )

    encoder = GameStateEncoder(d_model=d_model).to(device)
    model   = WesnothModel(
        d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, d_ff=d_ff,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.unit_type_to_id = dict(ckpt.get("unit_type_to_id", {}))
    if "faction_to_id" in ckpt:
        encoder.faction_to_id = dict(ckpt["faction_to_id"])
    encoder.eval(); model.eval()
    log.info(
        f"loaded {ckpt_path.name}: "
        f"{len(encoder.unit_type_to_id)} unit types, "
        f"{len(encoder.faction_to_id)} factions in vocab"
    )
    return encoder, model


# ---------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------

def _bucket() -> Dict[str, int]:
    """Empty per-outcome counter. Called with no args by defaultdict."""
    return {"win": 0, "loss": 0, "draw": 0, "timeout": 0, "errored": 0}


def _wr(b: Dict[str, int]) -> str:
    """Win rate over decisive games (excludes draws/timeouts/errors)."""
    decisive = b["win"] + b["loss"]
    if decisive == 0:
        return "n/a"
    return f"{100.0 * b['win'] / decisive:5.1f}% ({b['win']}/{decisive})"


def aggregate_and_report(
    results:    List[GameResult],
    matchups:   List[Dict],   # {map_short, our_faction, opp_faction, our_side, scenario_id}
) -> Dict:
    """Roll up per-game results into multiple cuts: overall, per-faction
    (when we played as that faction), per-map, per-matchup. Prints a
    text summary and returns the structured roll-up for JSON dump."""
    overall    = _bucket()
    by_faction = defaultdict(_bucket)         # our_faction -> bucket
    by_map     = defaultdict(_bucket)         # map_short    -> bucket
    by_matchup = defaultdict(_bucket)         # (our, opp)   -> bucket
    by_outcome = defaultdict(int)

    by_id = {m["scenario_id"]: m for m in matchups}
    for r in results:
        m = by_id.get(r.scenario_id)
        if m is None:
            log.warning(f"result for unknown scenario {r.scenario_id}; skipping")
            continue
        overall[r.outcome] += 1
        by_faction[m["our_faction"]][r.outcome] += 1
        by_map[m["map_short"]][r.outcome] += 1
        by_matchup[(m["our_faction"], m["opp_faction"])][r.outcome] += 1
        by_outcome[r.outcome] += 1

    # ---- Print human-readable summary ----
    print()
    print("=" * 72)
    print("Eval vs Wesnoth built-in AI — summary")
    print("=" * 72)
    n = len(results)
    decisive = overall["win"] + overall["loss"]
    print(f"Games played: {n}")
    print(f"  win:     {overall['win']}")
    print(f"  loss:    {overall['loss']}")
    print(f"  draw:    {overall['draw']}")
    print(f"  timeout: {overall['timeout']}")
    print(f"  errored: {overall['errored']}")
    print()
    if decisive > 0:
        wr = 100.0 * overall["win"] / decisive
        print(f"Overall win rate (decisive only): {wr:.1f}% "
              f"({overall['win']} / {decisive})")
    else:
        print("Overall win rate: n/a (no decisive games)")
    print()

    print("Per-faction (our policy playing AS this faction):")
    print(f"  {'faction':22s} {'win rate':22s}")
    for fname in sorted(by_faction):
        print(f"  {fname:22s} {_wr(by_faction[fname])}")
    print()

    print("Per-map:")
    print(f"  {'map':22s} {'win rate':22s}")
    for m in MAPS:
        if m.short not in by_map:
            continue
        print(f"  {m.name:22s} {_wr(by_map[m.short])}")
    print()

    print("Per-matchup (our_faction vs opp_faction):")
    print(f"  {'matchup':45s} {'win rate':22s}")
    for (a, b) in sorted(by_matchup):
        print(f"  {a:20s} vs {b:20s}  {_wr(by_matchup[(a,b)])}")
    print()
    print("=" * 72)

    # ---- Return structured data for JSON dump ----
    return {
        "n_games":   n,
        "overall":   overall,
        "by_faction": dict(by_faction),
        "by_map":     dict(by_map),
        "by_matchup": {f"{a}__vs__{b}": v
                       for (a, b), v in by_matchup.items()},
        "results":   [asdict(r) for r in results],
        "matchups":  matchups,
    }


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Trained .pt checkpoint to evaluate.")
    ap.add_argument("--maps", nargs="+", default=None,
                    help=f"Map shorts to use. Default = all "
                         f"({', '.join(m.short for m in MAPS)}). "
                         f"Pass a subset like `--maps caves den`.")
    ap.add_argument("--pairs", choices=("all", "cross"), default="all",
                    help="Faction pairs. 'all' includes mirrors (21); "
                         "'cross' is non-mirror only (15).")
    ap.add_argument("--no-swap", action="store_true",
                    help="Skip the side-swap rerun. Halves game count "
                         "but introduces first-mover bias.")
    ap.add_argument("--parallel", type=int, default=4,
                    help="Max parallel Wesnoth processes (default 4).")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS_PER_GAME,
                    help=f"Per-game timeout in actions "
                         f"(default {MAX_ACTIONS_PER_GAME}).")
    ap.add_argument("--state-timeout", type=float, default=90.0,
                    help="Seconds to wait for the next state frame "
                         "(default 90). Higher than training's 30s "
                         "default because the opposing default-RCA "
                         "side runs autonomously inside Wesnoth and "
                         "can take a while between our turns, "
                         "especially when GUI animations are on.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device (default cpu).")
    ap.add_argument("--save-json", type=Path, default=None,
                    help="Write structured results to PATH.json.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N games (0 = full grid).")
    args = ap.parse_args(argv[1:])

    if not args.checkpoint.exists():
        log.error(f"checkpoint not found: {args.checkpoint}")
        return 1

    # -- 1. Load model + encoder --
    device = torch.device(args.device)
    encoder, model = _load_checkpoint(args.checkpoint, device)
    converter = StateConverter()

    # -- 2. Build the matchup grid --
    if args.maps is None:
        chosen_maps = list(MAPS)
    else:
        unknown = set(args.maps) - set(MAP_BY_SHORT)
        if unknown:
            log.error(f"unknown map shorts: {sorted(unknown)}; "
                      f"available: {sorted(MAP_BY_SHORT)}")
            return 1
        chosen_maps = [MAP_BY_SHORT[s] for s in args.maps]

    pairs = all_pairs() if args.pairs == "all" else cross_pairs()
    grid = build_matchup_grid(chosen_maps, pairs,
                              swap_sides=not args.no_swap)
    if args.limit:
        grid = grid[: args.limit]
    log.info(f"matchup grid: {len(grid)} games "
             f"({len(chosen_maps)} maps × {len(pairs)} pairs"
             f"{' × 2 swaps' if not args.no_swap else ''})")

    # -- 3. Generate scenario .cfg files --
    eval_dir = SCENARIOS_PATH / "eval"
    main_cfg = ADDONS_PATH / "_main.cfg"
    sids = generate_eval_scenarios(
        matchups=grid, out_dir=eval_dir, main_cfg=main_cfg,
    )
    matchup_meta: List[Dict] = []
    for (m, fa, fb, our_side), sid in zip(grid, sids):
        our_faction = fa.name if our_side == 1 else fb.name
        opp_faction = fb.name if our_side == 1 else fa.name
        matchup_meta.append({
            "scenario_id": sid,
            "map_short":   m.short,
            "map_name":    m.name,
            "our_faction": our_faction,
            "opp_faction": opp_faction,
            "our_side":    our_side,
        })

    # -- 4. Run --
    t0 = time.perf_counter()
    progress_count = [0]
    def _on_done(r: GameResult) -> None:
        progress_count[0] += 1
        log.info(
            f"  [{progress_count[0]}/{len(grid)}] {r.scenario_id} "
            f"-> {r.outcome} (turns={r.turns}, our_acts={r.our_actions}, "
            f"{r.wall_seconds:.1f}s)"
        )

    log.info(f"launching eval ({args.parallel} parallel)...")
    runner_inputs = [
        {"scenario_id": m["scenario_id"], "our_side": m["our_side"]}
        for m in matchup_meta
    ]
    results = asyncio.run(play_many(
        matchups=runner_inputs,
        encoder=encoder, model=model, converter=converter,
        parallel=args.parallel,
        max_actions=args.max_actions,
        state_timeout=args.state_timeout,
        progress_cb=_on_done,
    ))
    wall = time.perf_counter() - t0
    log.info(f"all games done in {wall/60:.1f}m")

    # -- 5. Aggregate + report --
    summary = aggregate_and_report(results, matchup_meta)
    summary["checkpoint"] = str(args.checkpoint)
    summary["wall_seconds"] = wall
    summary["parallel"] = args.parallel
    summary["max_actions"] = args.max_actions

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        log.info(f"results written to {args.save_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
