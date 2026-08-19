"""Mask/sim reachability-disagreement fuzzer (2026-08-17).

The A4 probe run surfaced `move target not landable ... mask/sim
reachability disagreement` at volume (1,121 warnings in one game
stream) -- a contract violation (CLAUDE.md: the mask is a pure
function of observable state, and mask-offers => sim-routes must
hold). Policy-independent by construction, so this fuzzer drives
the mask with UNIFORM random picks (no network, fast) until
rejections fire, then autopsies each: rebuilds BOTH sides' reach
contexts on the same pre-step snapshot and diffs them component by
component (playable / occupied / ally / enemy / zoc / landable).

Usage:
    python tools/mask_sim_fuzz.py [--games 30] [--seed 7]
        [--incidents 10]
"""
from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("mask_sim_fuzz")


def autopsy(gs, action, sim) -> None:
    """Rebuild both contexts on the SAME state and diff."""
    from tools.pathfind_sim import ReachContext, unit_reach

    side = gs.global_info.current_side
    start = action["start_hex"]
    target = action["target_hex"]
    mover = next((u for u in gs.map.units
                  if u.position.x == start.x
                  and u.position.y == start.y and u.side == side),
                 None)
    if mover is None:
        print("  autopsy: mover missing on snapshot?!")
        return
    tpos = (target.x, target.y)
    # Sim-side context (what _action_to_command builds).
    sim_ctx = ReachContext.for_side(gs, side, exclude_unit=mover)
    sim_reach = unit_reach(mover, gs, sim_ctx)
    # Sim-side WITHOUT the mover exclusion (the mask's convention).
    sim_ctx_noex = ReachContext.for_side(gs, side)
    sim_reach_noex = unit_reach(mover, gs, sim_ctx_noex)
    print(f"  mover {mover.id}@{(start.x, start.y)} "
          f"mp={mover.current_moves} -> {tpos}")
    print(f"    sim landable (exclude mover): "
          f"{tpos in sim_reach.landable}")
    print(f"    sim landable (no exclusion):  "
          f"{tpos in sim_reach_noex.landable}")
    occ = (mover.position.x, mover.position.y) in sim_ctx.occupied_visible
    print(f"    target occupied(sim view): "
          f"{tpos in sim_ctx.occupied_visible} | mover-hex occ: {occ}")
    print(f"    target in zoc(sim view): {tpos in sim_ctx.zoc_hexes}")
    # Who is on the target hex, and is it visible to `side`?
    from wesnoth_ai.visibility import units_visible_to
    vis_ids = {u.id for u in units_visible_to(gs, side)}
    on_t = [u for u in gs.map.units
            if (u.position.x, u.position.y) == tpos]
    for u in on_t:
        print(f"    unit on target: {u.id} side={u.side} "
              f"name={u.name} visible_to_mover_side="
              f"{u.id in vis_ids}")
    if not on_t:
        print("    target hex empty (god view)")
    # Also check the LIVE sim state in case snapshot != live.
    live_mover = next((u for u in sim.gs.map.units
                       if u.id == mover.id), None)
    if live_mover is not None and (
            live_mover.current_moves != mover.current_moves
            or (live_mover.position.x, live_mover.position.y)
            != (start.x, start.y)):
        print(f"    LIVE DIVERGES from snapshot: live mp="
              f"{live_mover.current_moves} live pos="
              f"{(live_mover.position.x, live_mover.position.y)}")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--incidents", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-turns", type=int, default=40)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.WARNING)

    import torch
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.turn_search import forward_state
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.transformer_policy import TransformerPolicy

    rng = random.Random(args.seed)
    torch.manual_seed(0)
    # Tiny throwaway net: the fuzzer needs SOME ModelOutput for the
    # enumerator's prior plumbing; the bug is policy-independent.
    policy = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                               num_layers=1, num_heads=4, d_ff=64)
    incidents = 0
    total_steps = 0
    total_rejects = 0
    for g in range(args.games):
        setup = random_setup(rng)
        sim = WesnothSim(build_scenario_gamestate(setup),
                         scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        guard = 0
        while not sim.done and guard < 4000:
            guard += 1
            snap = copy.deepcopy(sim.gs)
            try:
                _, _, legal = forward_state(policy, snap, 0)
            except Exception as e:                      # noqa: BLE001
                print(f"game {g}: enumerate failed: {e!r}")
                break
            if not legal:
                break
            action = rng.choice(legal).action
            sim.step(action)
            total_steps += 1
            if getattr(sim, "last_step_rejected", False):
                total_rejects += 1
                if action.get("type") == "move":
                    incidents += 1
                    print(f"\nINCIDENT {incidents} (game {g}, "
                          f"turn {snap.global_info.turn_number}, "
                          f"scenario {setup.scenario_id}):")
                    autopsy(snap, action, sim)
                    if incidents >= args.incidents:
                        print(f"\n{incidents} incidents from "
                              f"{total_steps} steps "
                              f"({total_rejects} rejects total)")
                        return 0
        # game done
    print(f"\ndone: {incidents} move-reject incidents from "
          f"{total_steps} steps across {args.games} games "
          f"({total_rejects} rejects of any type)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
