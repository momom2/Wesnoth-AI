#!/usr/bin/env python3
"""Mask <-> sim pathfinding contract (2026-07-17).

The two-level contract:
  - The MASK offers what the policy may ATTEMPT, computed from the
    acting side's observable state via tools/pathfind_sim.
  - The SIM translates orders with the SAME planner on the SAME
    observable state, so every mask-offered (actor, move-target)
    must be accepted by `_action_to_command` (no silent end_turn, no
    reject loop). God-view divergence (hidden units) resolves at
    EXECUTION (walk_move_path), never at translation.

These tests drive REAL scenario states (production builders), per
the tests-drive-real-code rule.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _advance(sim, n):
    for _ in range(n):
        if sim.done:
            break
        sim.step({"type": "end_turn"})


def test_every_masked_move_target_is_sim_landable():
    """For each own unit, every hex the mask offers as a MOVE target
    must be in the sim planner's landable set for that unit --
    mask/sim disagreement here means either a burned re-decide loop
    (mask-consulting policy) or a wrong mask. Checked across several
    turns of a fogged ladder game so mid-game occupancy/ZoC shapes
    are exercised."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.action_sampler import _build_legality_masks
    from tools.pathfind_sim import ReachContext, unit_reach
    from tools.scenario_pool import random_setup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim

    rng = random.Random(11)
    setup = random_setup(rng, forced_faction=None, mini_maps=False)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=12)
    enc = GameStateEncoder()

    checked = 0
    for _turn in range(8):
        state = sim.gs
        side = state.global_info.current_side
        encoded = enc.encode(state)
        masks = _build_legality_masks(encoded, state)
        U = encoded.unit_tokens.size(1)
        by_id = {u.id: u for u in state.map.units}
        for i in range(U):
            u = by_id.get(encoded.unit_ids[i])
            if u is None or u.side != side:
                continue
            row = masks.target_valid_move[i]
            if not row.any():
                continue
            # The sim's translation-side reach (exclude_unit mirrors
            # _action_to_command).
            ctx = ReachContext.for_side(state, side, exclude_unit=u)
            reach = unit_reach(u, state, ctx)
            for j in row.nonzero().flatten().tolist():
                pos = encoded.hex_positions[j]
                assert (pos.x, pos.y) in reach.landable, (
                    f"mask offers ({pos.x},{pos.y}) for {u.id} "
                    f"(mp={u.current_moves}) but sim planner says "
                    f"not landable")
                checked += 1
        _advance(sim, 1)
        if sim.done:
            break
    assert checked > 50, f"too few targets exercised ({checked})"


def test_masked_moves_never_crowfly_unreachable():
    """The reachability mask must be a SUBSET of the old crow-flies
    bound (dist <= moves): true reachability can only shrink the
    offer, never extend it past raw MP range."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.rewards import hex_distance
    from tools.scenario_pool import random_setup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim

    rng = random.Random(3)
    setup = random_setup(rng, forced_faction=None, mini_maps=False,
                         category="fogless")
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=8)
    _advance(sim, 4)
    state = sim.gs
    side = state.global_info.current_side
    enc = GameStateEncoder()
    encoded = enc.encode(state)
    masks = _build_legality_masks(encoded, state)
    U = encoded.unit_tokens.size(1)
    by_id = {u.id: u for u in state.map.units}
    for i in range(U):
        u = by_id.get(encoded.unit_ids[i])
        if u is None or u.side != side:
            continue
        row = masks.target_valid_move[i]
        for j in row.nonzero().flatten().tolist():
            pos = encoded.hex_positions[j]
            d = hex_distance(u.position.x, u.position.y, pos.x, pos.y)
            assert d <= u.current_moves, (
                f"{u.id}: masked move target ({pos.x},{pos.y}) at "
                f"hex-distance {d} > mp {u.current_moves}")


def test_mask_less_caller_cannot_hang_the_sim():
    """A caller that ignores the mask and re-emits the same doomed
    action must NOT hang: after _MAX_CONSECUTIVE_REJECTS rejected
    steps the sim degrades to end_turn (loop guard, 2026-07-17 --
    caught live as an infinite no-op loop with DummyPolicy)."""
    from tools.scenario_pool import random_setup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.classes import Position

    rng = random.Random(5)
    setup = random_setup(rng, forced_faction=None, mini_maps=False,
                         category="fogless")
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=4)
    side0 = sim.gs.global_info.current_side
    leader = next(u for u in sim.gs.map.units
                  if u.side == side0 and u.is_leader)
    # A move order the planner can never accept: target = own hex's
    # far-off unreachable coordinate (off any reasonable reach).
    doomed = {"type": "move",
              "start_hex": leader.position,
              "target_hex": Position(x=leader.position.x,
                                     y=leader.position.y)}
    for i in range(sim._MAX_CONSECUTIVE_REJECTS + 2):
        sim.step(doomed)
        if sim.gs.global_info.current_side != side0:
            break
    assert sim.gs.global_info.current_side != side0, (
        "side never advanced: loop guard did not fire")
