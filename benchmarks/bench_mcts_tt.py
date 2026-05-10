"""How often does the MCTS transposition table actually fire?

Runs N games' worth of MCTS searches, captures every search's
`(tt_hits, tt_misses)` from the root node, and prints aggregate
hit rate. Wesnoth's per-action state mutations (HP, MP, gold,
village ownership all change every action) make true state-key
collisions rare in practice; we expect the table to mostly fire
on intra-turn move-reorderings (move A then B vs B then A landing
in identical end-of-turn states).

Usage:
    python benchmarks/bench_mcts_tt.py [--games N] [--sims K]

Dependencies: tools.mcts, tools.scenario_pool, transformer_policy
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=20,
                    help="MCTS sims per move; lower = faster bench, "
                         "but fewer chances for paths to converge.")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    from transformer_policy import TransformerPolicy
    from tools.mcts import mcts_search, MCTSConfig
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions,
    )
    from tools.wesnoth_sim import WesnothSim, PvPDefaults

    rng = random.Random(args.seed)

    # Tiny model: TT hit-rate is a property of the SEARCH (sim
    # state geometry + PUCT visiting same states), not the
    # priors. Whatever the network outputs, paths can converge
    # or not based on which game states are reachable.
    print("Building tiny policy (d=64, L=2) ...")
    policy = TransformerPolicy(
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        device=torch.device("cpu"),
    )
    config = MCTSConfig(
        n_simulations=args.sims,
        c_puct=1.5,
        add_root_noise=False,    # cleaner stats without noise jitter
        use_transposition_table=True,
    )

    pvp = PvPDefaults()
    factions = load_factions()
    if not factions:
        print("ERROR: no factions loaded -- run scrape or check working dir")
        return 2

    total_hits = 0
    total_misses = 0
    n_searches = 0

    print(f"\nRunning {args.games} games × {args.sims} sims/move "
          f"× max {args.max_turns} turns ...")
    t0 = time.perf_counter()

    for g in range(args.games):
        try:
            setup = random_setup(rng, forced_faction=None)
            gs = build_scenario_gamestate(
                setup,
                starting_gold=pvp.starting_gold,
                base_income=pvp.base_income,
                village_gold=pvp.village_gold,
                village_upkeep=pvp.village_support,
                experience_modifier=pvp.experience_modifier,
            )
            sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                             max_turns=args.max_turns)
        except Exception as e:
            print(f"  game {g}: setup failed: {e}")
            continue

        game_hits = 0
        game_misses = 0
        game_searches = 0
        moves = 0
        while not sim.done and sim.gs.global_info.turn_number <= args.max_turns:
            root = mcts_search(sim, policy._inference_model,
                               policy._inference_encoder, config)
            game_hits += root.tt_hits
            game_misses += root.tt_misses
            game_searches += 1
            moves += 1
            best = None
            best_n = -1
            for e in root.edges:
                if e.n_visits > best_n:
                    best_n = e.n_visits
                    best = e
            if best is None:
                break
            try:
                sim.step(best.action)
            except Exception as e:
                print(f"  game {g}: step error after {moves} moves: {e}")
                break

        total_hits += game_hits
        total_misses += game_misses
        n_searches += game_searches
        hr = (game_hits / max(1, game_hits + game_misses))
        print(f"  game {g:2d} ({setup.scenario_id[:30]:30s}): "
              f"{moves:3d} moves, "
              f"{game_searches:3d} searches, "
              f"hits={game_hits:5d} misses={game_misses:5d} "
              f"hit_rate={hr:.1%}")

    dt = time.perf_counter() - t0

    print()
    print("=" * 72)
    print(f"MCTS TT measurement: {args.games} games × {args.sims} sims/move "
          f"in {dt:.1f}s")
    print("=" * 72)
    print(f"  searches:        {n_searches}")
    print(f"  total hits:      {total_hits}")
    print(f"  total misses:    {total_misses}")
    if total_hits + total_misses == 0:
        print("  hit rate:        n/a (no searches reached child creation)")
    else:
        print(f"  hit rate:        "
              f"{total_hits / (total_hits + total_misses):.2%}")
    print(f"  hits/search:     {total_hits / max(1, n_searches):.2f}")
    print(f"  misses/search:   {total_misses / max(1, n_searches):.2f}")
    print()
    print("Reading: a hit means PUCT descended to a child state that")
    print("ANOTHER edge in the same search had already reached. With")
    print("Wesnoth's per-action state churn (HP/MP/gold/villages all")
    print("change), most paths produce unique state_keys; intra-turn")
    print("move reordering is the main legitimate hit source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
