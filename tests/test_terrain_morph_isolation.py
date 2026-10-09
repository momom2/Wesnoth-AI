#!/usr/bin/env python3
"""Terrain-morph copy-on-write isolation (adversarial review
2026-07-18, HIGH finding).

`Map.__deepcopy__` / `GlobalInfo.__deepcopy__` deliberately ALIAS
`map.hexes` and `_terrain_codes` across `WesnothSim.fork()` (terrain
was assumed immutable). `_terrain_action` used to mutate both in
place, so an MCTS rollout fork that crossed Aethermaw's morph turns
(4-6) morphed the LIVE game's terrain too -- reproduced as 22 live
hexes changing from a fork stepped to turn 13. A fork now clones the
Rust core, and a view applies the core's terrain writes copy-on-write
(`scenario_events.terrain_writes_applied`): new containers, on the
morphing view only.
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
    terrain codes and Hex set must be untouched."""
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(0, max_turns=20,
                             scenario_id="multiplayer_Aethermaw")

    live_codes0, live_hexes0 = _terrain_snapshot(sim.gs)

    fork = sim.fork()
    # Burn turns on the fork until well past the morph window.
    for _ in range(2 * 14):
        if fork.done:
            break
        fork.step({"type": "end_turn"})

    fork_codes, _ = _terrain_snapshot(fork.gs)
    live_codes1, live_hexes1 = _terrain_snapshot(sim.gs)

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
