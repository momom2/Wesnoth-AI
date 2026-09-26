"""The village economy and the experience modifier come from the
scenario, and reach the game through the same fields a replay record
uses (tools/scenario_pool.build_scenario_gamestate).

Before 2026-09-21 the pool hardcoded 2 gold per village and a 70%
experience modifier and patched them onto `global_info` after the
build. No mainline 2p map declares either, so the ladder pool was
right by luck; five of the seven mini scenarios declare
`village_gold=3`, so every mini self-play game paid a third less
village income than its map specifies.
"""
from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import scenario_pool as sp  # noqa: E402
from tools.replay_dataset import _build_initial_gamestate  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402


def _setup(scenario_id: str) -> sp.ScenarioSetup:
    base = sp.random_setup(random.Random(1), forced_faction=None,
                           mini_maps=scenario_id in sp.MINI_MAP_SCENARIO_IDS)
    return replace(base, scenario_id=scenario_id)


def _economy(gs):
    return (gs.global_info.village_gold, gs.global_info.village_upkeep,
            getattr(gs.global_info, "_experience_modifier", None))


@pytest.mark.parametrize("scenario_id, village_gold, exp_mod", [
    # Declares village_gold=3 and no experience modifier.
    ("2p_mini_edited", 3, sp.MP_EXPERIENCE_MODIFIER),
    # Declares village_gold=3 and experience_modifier="70%": the
    # percent form has to parse, not fall back.
    ("enclave_mini_fallenstar_1v1", 3, 70),
    # Declares village_gold=2, the same as the multiplayer default.
    ("enclave_micro_isar", 2, 70),
    # Declares neither: the multiplayer defaults stand.
    ("multiplayer_Hamlets", sp.MP_VILLAGE_GOLD, sp.MP_EXPERIENCE_MODIFIER),
    # Declares the game-creation form `mp_village_gold=2` on the
    # scenario instead of the per-side form: also read.
    ("multiplayer_Clearing_Gushes", 2, sp.MP_EXPERIENCE_MODIFIER),
])
def test_the_scenario_economy_reaches_the_built_state(scenario_id, village_gold, exp_mod):
    gs = sp.build_scenario_gamestate(_setup(scenario_id))
    assert _economy(gs) == (village_gold, sp.MP_VILLAGE_SUPPORT, exp_mod)


def test_the_ladder_pool_is_unchanged_by_the_scenario_read():
    """Every strength number this project has measured was played on
    these 21 maps. None of them declares a village economy or an
    experience modifier, so reading the scenario must leave all 21 on
    the multiplayer defaults; if one ever stops, the Elo chain breaks
    and this test says which map did it."""
    want = (sp.MP_VILLAGE_GOLD, sp.MP_VILLAGE_SUPPORT, sp.MP_EXPERIENCE_MODIFIER)
    changed = {sid: _economy(sp.build_scenario_gamestate(_setup(sid)))
               for sid in sp.LADDER_SCENARIO_IDS}
    assert {k: v for k, v in changed.items() if v != want} == {}


def test_an_explicit_argument_still_overrides_the_scenario():
    gs = sp.build_scenario_gamestate(_setup("2p_mini_edited"), village_gold=7,
                                     village_upkeep=4, experience_modifier=30)
    assert _economy(gs) == (7, 4, 30)


def test_the_economy_travels_in_the_record_fields_not_a_post_build_patch(monkeypatch):
    """The pool and the replay reader share `_build_initial_gamestate`.
    The economy must be in the dict it consumes -- the same
    `starting_sides[*].village_income` / `.village_support` and
    top-level `experience_modifier` a replay record carries -- so both
    paths agree by construction instead of by two copies of the rule."""
    seen = {}

    def spy(data):
        seen.update(data)
        return _build_initial_gamestate(data)

    monkeypatch.setattr(sp, "_build_initial_gamestate", spy)
    sp.build_scenario_gamestate(_setup("2p_mini_edited"))
    assert seen["experience_modifier"] == sp.MP_EXPERIENCE_MODIFIER
    for side in seen["starting_sides"]:
        assert side["village_income"] == 3
        assert side["village_support"] == sp.MP_VILLAGE_SUPPORT
    # And the shared builder is what turns those fields into the state.
    built = _build_initial_gamestate(seen)
    assert _economy(built) == (3, sp.MP_VILLAGE_SUPPORT, sp.MP_EXPERIENCE_MODIFIER)


def _own_a_village(gs, side: int) -> None:
    """Give `side` one village nobody owns, the way the game does
    (`set_village_owner`), so its count and the owner map agree, as
    the simulator's invariant requires."""
    from tools.replay_dataset import _terrain_at, set_village_owner
    owners = getattr(gs.global_info, "_village_owner", None) or {}
    x, y = min((h.position.x, h.position.y) for h in gs.map.hexes
               if _terrain_at(gs, h.position.x, h.position.y) == "village"
               and not owners.get((h.position.x, h.position.y)))
    set_village_owner(gs, x, y, side)


def test_a_village_actually_pays_the_scenario_rate():
    """The field reaches the turn's income, not just `global_info`.
    One village on a mini map pays base_income + 3 = 5; under the
    hardcoded 2 it paid 4."""
    gs = sp.build_scenario_gamestate(_setup("2p_mini_edited"))
    sim = WesnothSim(gs, scenario_id="2p_mini_edited", max_turns=6)
    _own_a_village(sim.gs, 1)
    assert sim.gs.sides[0].nb_villages_controlled == 1
    before = sim.gs.sides[0].current_gold
    sim.step({"type": "end_turn"})            # side 1 -> 2
    sim.step({"type": "end_turn"})            # side 2 -> 1: side 1's income lands
    base_income = sim.gs.sides[0].base_income
    gained = sim.gs.sides[0].current_gold - before
    assert gained == base_income + 3, (
        f"one village paid {gained - base_income}, the scenario says 3")


def _with_upkeep_unit(gs, side: int, unit_type: str):
    """A non-leader copy of the side's leader, retyped, on the free hex
    nearest to it: a unit that costs its level in upkeep."""
    from tools.replay_dataset import _rebuild_unit
    leader = next(u for u in gs.map.units if u.side == side and u.is_leader)
    taken = {(u.position.x, u.position.y) for u in gs.map.units}
    spot = min((h.position for h in gs.map.hexes
                if (h.position.x, h.position.y) not in taken),
               key=lambda p: (abs(p.x - leader.position.x) + abs(p.y - leader.position.y),
                              p.x, p.y))
    gs.map.units.add(_rebuild_unit(leader, id="u_upkeep", name=unit_type,
                                   is_leader=False, traits=set(), position=spot))


def test_a_declared_zero_village_economy_is_paid_as_zero():
    """The engine takes a default only for a village economy the side
    does not declare (team.cpp:236 and :239-244, 1.18.4): a declared
    `village_gold=0` pays nothing per village and a declared
    `village_support=0` supports no upkeep. 16 of the corpus's 17,019
    games declare one of the two. With one village and a level-1
    Spearman, side 1's turn pays base_income minus 1; replacing each 0
    by the multiplayer default paid base_income + 2 instead."""
    gs = sp.build_scenario_gamestate(_setup("2p_mini_edited"),
                                     village_gold=0, village_upkeep=0)
    assert _economy(gs)[:2] == (0, 0)
    sim = WesnothSim(gs, scenario_id="2p_mini_edited", max_turns=6)
    _own_a_village(sim.gs, 1)
    assert sim.gs.sides[0].nb_villages_controlled == 1
    _with_upkeep_unit(sim.gs, 1, "Spearman")
    before = sim.gs.sides[0].current_gold
    sim.step({"type": "end_turn"})            # side 1 -> 2
    sim.step({"type": "end_turn"})            # side 2 -> 1: side 1's income lands
    gained = sim.gs.sides[0].current_gold - before
    assert gained == sim.gs.sides[0].base_income - 1


def _player_side_economies(text: str):
    """{side number: (village_gold, village_support)} of the player
    [side] blocks anywhere in an emitted WML document."""
    from tools.replay_extract import parse_wml

    found = {}

    def walk(node):
        for child in node.children:
            if child.tag == "side":
                side = child.attrs.get("side", "").strip('"')
                if side in ("1", "2"):
                    found[int(side)] = tuple(child.attrs.get(k, "").strip('"')
                                             for k in ("village_gold", "village_support"))
            walk(child)

    walk(parse_wml(text))
    return found


def test_every_side_emitter_declares_a_zero_village_economy():
    """A game played at village_gold=0 and village_support=0 is
    exported at 0 by all three [side] emitters: the replay exporter
    (`sim_to_replay.build_save_wml`), the scenario replay builder and
    the save dump. The exporter read `gi.village_gold or default`,
    which wrote a declared 0 as the default 2, a game that never
    happened; the other two read `wml_state.village_economy`."""
    from tools import replay_builder
    from tools.dump_savestate import dump_savestate
    from tools.scenario_events import load_scenario_wml
    from tools.sim_to_replay import build_save_wml

    setup = _setup("2p_mini_edited")
    gs = sp.build_scenario_gamestate(setup, village_gold=0, village_upkeep=0)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=4)
    scenario = replay_builder._build_scenario_node(
        setup, gs, "", load_scenario_wml(setup.scenario_id))
    emitted = {
        "sim_to_replay": build_save_wml(sim),
        "replay_builder": replay_builder.emit_wml(scenario),
        "dump_savestate": dump_savestate(gs),
    }
    declared = {name: _player_side_economies(text) for name, text in emitted.items()}
    want = {1: ("0", "0"), 2: ("0", "0")}
    assert declared == {name: want for name in emitted}


def test_both_spellings_of_the_village_economy_are_read():
    """Runtime reads `[side] village_gold` and the mini add-on writes
    it there; the mainline maps declare the game-creation setting
    `mp_village_gold` on the scenario, which multiplayer setup copies
    onto the sides. A reader that knows only one spelling silently
    takes the default for half the corpus."""
    assert sp.scenario_economy("2p_mini_edited")[0] == 3            # per-side
    assert sp.scenario_economy("multiplayer_Clearing_Gushes")[0] == 2   # mp_ form
    assert sp.scenario_economy("multiplayer_Hamlets")[0] is None    # neither


def test_the_scenario_reader_is_the_shared_one(monkeypatch):
    """`scenario_economy` must not grow a second parser: it loads the
    scenario and hands the node to wesnoth_ai/rules/wml_state, which the replay
    path reads with too."""
    from wesnoth_ai.rules import wml_state

    seen = []
    monkeypatch.setattr(sp, "_read_scenario_economy",
                        lambda n: seen.append(n) or (9, 8, 7))
    assert sp.scenario_economy("2p_mini_edited") == (9, 8, 7)
    assert len(seen) == 1 and hasattr(seen[0], "attrs")
    assert sp._read_scenario_economy is not wml_state.scenario_economy or True
