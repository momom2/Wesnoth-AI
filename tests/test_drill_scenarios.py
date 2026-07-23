"""Capability-drill scenarios (tools/scenario_pool.DRILL_SCENARIO_IDS).

Drills are hand-authored [multiplayer] cfgs in
add-ons/wesnoth_ai/scenarios/drills/ with purpose-built micro maps.
These tests pin the full path: cfg discovery (project add-on tree),
map resolution, pre-placed army instantiation in the sim, per-drill
gold, sim playability, and save export carrying the placed units.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.dummy_policy import DummyPolicy   # noqa: E402
from sim_test_helpers import fresh_scenario_sim, scenario_setup   # noqa: E402
from tools.scenario_pool import (   # noqa: E402
    DRILL_SCENARIO_IDS, build_scenario_gamestate,
)

# (scenario_id, expected gold, expected placed units per player side)
_DRILL_SHAPE = [
    ("drill_duel",          0, 3),
    ("drill_village_rush", 40, 0),
    ("drill_chokepoint",    0, 3),
]


def test_drill_shape_constants_cover_the_pool():
    assert {sid for sid, _, _ in _DRILL_SHAPE} == set(DRILL_SCENARIO_IDS)


def test_drill_gamestates_build_with_armies_and_gold():
    """Every drill builds from its cfg: two leaders on their keep
    starts, the cfg's gold=, and the fixed armies instantiated as
    ordinary (trait-less) units on the right sides."""
    for sid, gold, army in _DRILL_SHAPE:
        setup = scenario_setup(seed=5, scenario_id=sid)
        gs = build_scenario_gamestate(setup)
        leaders = [u for u in gs.map.units if u.is_leader]
        assert len(leaders) == 2, f"{sid}: {len(leaders)} leaders"
        assert {u.side for u in leaders} == {1, 2}
        for side in (1, 2):
            placed = [u for u in gs.map.units
                      if u.side == side and not u.is_leader]
            assert len(placed) == army, (
                f"{sid} side {side}: {len(placed)} placed units, "
                f"expected {army}")
            for u in placed:
                assert not u.traits, f"{sid}: placed unit has traits"
                assert u.attacks, f"{sid}: placed unit has no attacks"
            assert gs.sides[side - 1].current_gold == gold, (
                f"{sid} side {side}: gold "
                f"{gs.sides[side - 1].current_gold} != {gold}")


def test_drill_village_rush_has_eight_villages():
    from wesnoth_ai.classes import Terrain
    setup = scenario_setup(seed=5, scenario_id="drill_village_rush")
    gs = build_scenario_gamestate(setup)
    n_villages = sum(
        1 for h in gs.map.hexes
        if Terrain.VILLAGE in h.terrain_types)
    assert n_villages == 8, f"expected 8 villages, found {n_villages}"


def test_drill_duel_plays_under_dummy_policy():
    """A drill game steps without raising; with gold=0 nothing can
    be recruited, so the unit count can only shrink (combat)."""
    sim = fresh_scenario_sim(seed=7, max_turns=6,
                             scenario_id="drill_duel")
    start_units = len(sim.gs.map.units)
    assert start_units == 8     # 2 leaders + 2x3 army
    pol = DummyPolicy()
    for _ in range(120):
        if sim.done:
            break
        sim.step(pol.select_action(sim.gs, game_label="drill"))
    assert len(sim.gs.map.units) <= start_units


def test_drill_export_carries_placed_units():
    """The exported save must re-create the fixed armies at
    [replay_start]: six [unit] blocks (3 per side) with
    random_traits=no, and the cfg's gold=0 on both sides --
    otherwise Wesnoth playback runs the recorded commands against
    a different starting state than the sim played."""
    from tools.sim_to_replay import build_save_wml

    sim = fresh_scenario_sim(seed=7, max_turns=6,
                             scenario_id="drill_duel")
    pol = DummyPolicy()
    for _ in range(30):
        if sim.done:
            break
        sim.step(pol.select_action(sim.gs, game_label="drillx"))
    wml = build_save_wml(sim)
    assert wml.count("random_traits=no") >= 6, (
        "exported save lost the pre-placed armies")
    for utype in ("Spearman", "Bowman", "Mage"):
        assert wml.count(f'type="{utype}"') >= 2, (
            f"both sides' {utype} must be emitted")
    assert 'gold="0"' in wml, "drill gold=0 not exported"
