#!/usr/bin/env python3
"""Terrain-morph copy-on-write isolation (adversarial review
2026-07-18, HIGH finding).

`Map.__deepcopy__` / `GlobalInfo.__deepcopy__` deliberately ALIAS
`map.hexes` and `_terrain_codes` across `WesnothSim.fork()` (terrain
was assumed immutable). `_terrain_action` used to mutate both in
place, so an MCTS rollout fork that crossed Aethermaw's morph turns
(4-6) morphed the LIVE game's terrain too -- reproduced as 22 live
hexes changing from a fork stepped to turn 13, with the live
`_terrain_epoch` left stale (planner/mask serving pre-morph costs
against post-morph combat). `_terrain_action` is now copy-on-write:
new containers, rebound on the morphing gs only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _terrain_snapshot(gs):
    codes = dict(getattr(gs.global_info, "_terrain_codes", {}) or {})
    hexes = {
        (h.position.x, h.position.y): (frozenset(h.terrain_types),
                                       frozenset(h.modifiers))
        for h in gs.map.hexes
    }
    return codes, hexes


def test_fork_morph_does_not_touch_live_sim():
    """Advance a FORK past Aethermaw's morph turns; the live sim's
    terrain codes, Hex set, and epoch stamp must be untouched, and
    the fork's epoch must have moved (its planner cache refreshes)."""
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(0, max_turns=20,
                             scenario_id="multiplayer_Aethermaw")

    live_codes0, live_hexes0 = _terrain_snapshot(sim.gs)
    live_epoch0 = getattr(sim.gs.global_info, "_terrain_epoch", None)

    fork = sim.fork()
    # Burn turns on the fork until well past the morph window.
    for _ in range(2 * 14):
        if fork.done:
            break
        fork.step({"type": "end_turn"})

    fork_codes, _ = _terrain_snapshot(fork.gs)
    live_codes1, live_hexes1 = _terrain_snapshot(sim.gs)
    live_epoch1 = getattr(sim.gs.global_info, "_terrain_epoch", None)

    # The fork must have actually morphed (else this test is vacuous).
    changed_in_fork = {
        k for k in fork_codes
        if fork_codes.get(k) != live_codes0.get(k)
    }
    assert changed_in_fork, (
        "Aethermaw fork never morphed -- scenario events not firing?")

    # And the LIVE sim must be untouched.
    assert live_codes1 == live_codes0, (
        f"live terrain codes mutated by fork: "
        f"{ {k: (live_codes0.get(k), live_codes1.get(k)) for k in live_codes1 if live_codes1.get(k) != live_codes0.get(k)} }")
    assert live_hexes1 == live_hexes0, "live Hex set mutated by fork"
    assert live_epoch1 == live_epoch0, "live terrain epoch bumped by fork"


def test_morph_bumps_epoch_on_the_morphing_sim():
    """The sim that DOES cross the morph gets a fresh epoch (its
    reach-planner cache must not serve pre-morph costs)."""
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(0, max_turns=20,
                             scenario_id="multiplayer_Aethermaw")
    epoch0 = getattr(sim.gs.global_info, "_terrain_epoch", None)
    codes0 = dict(getattr(sim.gs.global_info, "_terrain_codes", {}) or {})
    for _ in range(2 * 14):
        if sim.done:
            break
        sim.step({"type": "end_turn"})
    codes1 = dict(getattr(sim.gs.global_info, "_terrain_codes", {}) or {})
    assert codes1 != codes0, "morph never fired"
    epoch1 = getattr(sim.gs.global_info, "_terrain_epoch", None)
    assert epoch1 != epoch0, (
        "terrain epoch not bumped on morph -- planner would serve "
        "stale movement/defense maps")
