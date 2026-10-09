"""The Rust-owned game state round-trips the Python state.

`CoreState.from_state(gs).to_state()` must equal `gs` over every modeled
field (units field for field including statuses, traits, abilities and
attacks; sides; the turn scalars; the village owners, the uncovered set,
the recruit rejections, the advancement queue, the last walk and strikes) and
share the hex set by identity. A fork must not share dynamic state
with its parent. The core's state key must agree with itself on equal
states and change with any modeled field. Skipped without the phase-16
wheel.
"""
from __future__ import annotations

import copy

import pytest

from wesnoth_ai import core_compare as cc
from wesnoth_ai import game_core as gc

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


def _harvested():
    from helpers.played_states import _states as harvest
    return harvest(n_attack=3, n_plain=4)


def _replay_states(n_games=2, per_game=6):
    from pathlib import Path
    from tools.replay_dataset import filter_competitive_2p, iter_replay_pairs
    out = []
    root = next((Path(d) for d in ("replays_dataset", "replays_dataset_imitation") if Path(d).exists()), None)
    if root is None:
        return out
    for gz in filter_competitive_2p(root)[:n_games]:
        k = 0
        for gs, _ai in iter_replay_pairs(gz):
            if k % 40 == 0:
                out.append(copy.deepcopy(gs))
            k += 1
            if len(out) >= per_game * n_games:
                break
    return out


def _decorate(gs):
    """Exercise the stash paths: rejections, uncovered hiders, a walk,
    strikes, advancement choices."""
    gi = gs.global_info
    hexes = sorted(gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
    gi._recruit_rejected_hexes = {(hexes[3].position.x, hexes[3].position.y)}
    units = sorted(gs.map.units, key=lambda u: u.id)
    gi._uncovered_units = {units[0].id} if units else set()
    gi._last_move_walk = {"ordered": (1, 2), "landed": (1, 3), "stop_reason": "ambush"}
    gi._last_checkup_strikes = [{"chance": 60, "hits": True, "damage": 7}, {"dies": False}]
    gi._advance_choices = [1, 0]
    gi._pickadvance_game = {(1, "Spearman"): ["Swordsman"]}
    gi._last_advance_events = [(1, 0)]
    gi._rng_request_counter = 5
    if units:
        setattr(units[0], "_feeding_count", 2)
    return gs


def test_round_trip_equals_the_state():
    checked = 0
    for gs in _harvested() + _replay_states():
        gs = _decorate(gs)
        cs = gc.CoreState.from_state(gs)
        back = cs.to_state()
        diffs = cc.state_differences(gs, back)
        assert not diffs, "\n".join(diffs[:8])
        assert back.map.hexes is gs.map.hexes and back.map.mask is gs.map.mask
        assert back.global_info._terrain_codes is gs.global_info._terrain_codes
        assert cs.core.n_units() == len(gs.map.units)
        assert cs.core.n_classes() >= 2
        checked += 1
    assert checked >= 8


def test_fork_is_isolated_and_keys_follow_content():
    gs = _decorate(_harvested()[0])
    cs = gc.CoreState.from_state(gs)
    twin = gc.CoreState.from_state(copy.deepcopy(gs))
    assert cs.state_key() == twin.state_key()
    fork = cs.fork()
    assert fork.state_key() == cs.state_key()
    uid = sorted(cs.core.unit_ids())[0]
    before = cs.core.unit_export(uid)["current_hp"]
    fork.core.update_unit(uid, {"current_hp": max(1, before - 1)})
    assert cs.core.unit_export(uid)["current_hp"] == before
    assert fork.state_key() != cs.state_key()
    fork.core.update_unit(uid, {"current_hp": before})
    assert fork.state_key() == cs.state_key()
    # A core fork materializes its own units: no leaf object is shared.
    parent_units = {u.id: u for u in cs.to_state().map.units}
    assert all(parent_units[u.id] is not u for u in fork.to_state().map.units)
    fork.core.remove_unit(uid)
    assert cs.core.n_units() == fork.core.n_units() + 1
    assert uid in cs.core.unit_ids() and uid not in fork.core.unit_ids()
    assert not cc.state_differences(gs, cs.to_state())


def test_init_side_pays_a_declared_zero_village_economy():
    """A declared `village_gold=0` pays nothing per village and a
    declared `village_support=0` supports no upkeep (team.cpp:236 and
    :239-244). One village and a level-1 Spearman: side 1's turn-2
    start pays base_income minus 1."""
    import random
    from dataclasses import replace
    from wesnoth_ai.rules import scenario_pool as sp
    from tests.test_scenario_economy import _with_upkeep_unit
    gs = sp.build_scenario_gamestate(sp.random_setup(random.Random(1)),
                                     village_gold=0, village_upkeep=0)
    gs.global_info.turn_number = 2
    gs.sides[0] = replace(gs.sides[0], nb_villages_controlled=1)
    _with_upkeep_unit(gs, 1, "Spearman")
    want = gs.sides[0].current_gold + gs.sides[0].base_income - 1
    cs = gc.CoreState.from_state(gs)
    assert cs.apply_command(["init_side", 1]) == "rust"
    assert cs.to_state().sides[0].current_gold == want


def test_init_side_hides_the_sides_revealed_hiders_again_after_turn_one():
    """unit::new_turn clears STATE_UNCOVERED inside `turn() > 1`
    (docs/wesnoth_rules.md "Hidden-unit visibility"): at a side's turn
    start after turn 1 its own revealed hiders hide again and the other
    side's stay revealed; on turn 1 nothing changes."""
    checked = 0
    for gs in _harvested():
        by_side = {s: sorted(u.id for u in gs.map.units if u.side == s) for s in (1, 2)}
        if not by_side[1] or not by_side[2]:
            continue
        revealed = {by_side[1][0], by_side[2][0]}
        for side, turn, hidden in ((1, 3, {by_side[1][0]}), (2, 3, {by_side[2][0]}), (2, 1, set())):
            gs.global_info.turn_number = turn
            gs.global_info._uncovered_units = set(revealed)
            cs = gc.CoreState.from_state(copy.deepcopy(gs))
            assert cs.apply_command(["init_side", side]) == "rust"
            assert cs.to_state().global_info._uncovered_units == revealed - hidden, (side, turn)
            checked += 1
    assert checked >= 6


def test_a_negative_start_slot_wraps_on_the_board_and_in_a_time_area():
    """The engine wraps `current_time` into the schedule with a modulo
    that is never negative (`fix_time_index`, src/tod_manager.cpp:66,
    1.18.4); a time area reads its own slot, not the board's. The
    readers wrap the slot before it reaches a state, so this state
    carries one set by hand: turn 1 reads the board's slot before its
    first, the last of the default cycle, and the area's own first
    slot."""
    from tests.sim_test_helpers import replayed_state, three_side_record
    from wesnoth_ai.combat import TOD_DEFAULT_CYCLE
    gs = replayed_state(three_side_record(), 0)
    gs.global_info._tod_start_offset = -1
    area, plain = (0, 1), (4, 4)
    cycle = [25, 0, -25, 0]
    gs.global_info._time_areas = {area: list(cycle)}
    cs = gc.CoreState.from_state(gs)
    for turn in range(1, 9):
        assert cs.core.lawful_bonus(*plain, turn) == TOD_DEFAULT_CYCLE[(turn - 2) % 6][1], turn
        assert cs.core.lawful_bonus(*area, turn) == cycle[(turn - 1) % 4], turn
    assert cs.apply_command(["init_side", 1]) == "rust"
    assert cs.to_state().global_info.time_of_day == "second_watch"


def test_the_invariant_check_holds_each_side_to_the_villages_it_owns():
    """`WesnothSim._assert_invariants` (e) on the core: each side's
    village count equals the villages the owner map gives it. A count
    off its owners and an owner off its count are both violations."""
    from dataclasses import replace
    from tests.sim_test_helpers import replayed_state, three_side_record
    gs = replayed_state(three_side_record(), 1)
    assert gc.CoreState.from_state(gs).core.invariant_violation() is None
    count_off = copy.deepcopy(gs)
    count_off.sides[2] = replace(count_off.sides[2], nb_villages_controlled=1)
    owner_off = copy.deepcopy(gs)
    del owner_off.global_info._village_owner[(9, 3)]
    for bad in (count_off, owner_off):
        found = gc.CoreState.from_state(bad).core.invariant_violation()
        assert found is not None and "village counts" in found, found


def _vocab_of(states):
    """A unit-type vocab over the states' names but the last one (out
    of vocab, the overflow bucket) and the factions of their sides."""
    names = sorted({u.name for gs in states for u in gs.map.units})
    factions = sorted({s.faction for gs in states for s in gs.sides if s.faction})
    return {n: i for i, n in enumerate(names[:-1])}, {f: i for i, f in enumerate(factions)}


BASES_AND_GATES = ((False, False), (True, False), (True, True), (False, True))
# (relevant set, enemy-village gate) per encoding checked.


def test_encode_raw_from_core_reads_the_other_player_on_a_three_side_state():
    """A replayed game whose record declares a third side (a statue or
    tentacle side) keeps a SideInfo for it: the core's enemy faction is
    the other player's, and neither the core nor its Rust entry encodes
    for the third side."""
    from tests.sim_test_helpers import replayed_state, three_side_record
    from wesnoth_ai import encoder as enc
    n = 0
    for fog in (True, False):
        record = three_side_record(fog=fog, third_side_acts=True)
        for n_commands, side in ((1, 1), (3, 2)):
            gs = replayed_state(record, n_commands)
            assert gs.global_info.current_side == side and len(gs.sides) == 3
            type_to_id, faction_to_id = _vocab_of([gs])
            cs = gc.CoreState.from_state(gs)
            for relevant, gate in BASES_AND_GATES:
                kw = dict(type_to_id=type_to_id, faction_to_id=faction_to_id,
                          relevant_set=relevant, fog_hides_enemy_villages=gate)
                raw = cs.encode_raw(**kw)
                assert raw.our_faction_id == faction_to_id[gs.sides[side - 1].faction]
                assert raw.their_faction_id == faction_to_id[gs.sides[2 - side].faction]
                n += 1
        gs = replayed_state(record, 5)
        type_to_id, faction_to_id = _vocab_of([gs])
        third = gc.CoreState.from_state(gs)
        assert int(third.core.current_side) == 3
        with pytest.raises(ValueError, match="not a player side"):
            third.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id)
        with pytest.raises(ValueError, match="not a player side"):
            third.core.encode_streams(
                3, False, third._type_vocab(type_to_id), [], [], False,
                (enc.HP_NORM, enc.MOVES_NORM, enc.EXP_NORM, enc.COST_NORM, enc.GOLD_NORM,
                 enc.INCOME_NORM, enc.VILLAGES_NORM, enc.TURN_NORM),
                enc.MAX_MAP_SIZE - 1, enc.NUM_ALIGNMENTS)
    assert n == 16


def test_an_event_terrain_change_reaches_everything_the_core_derives():
    """A [terrain] event rewrites hexes to a keep, a village, a forest
    and deep water at side 1's turn 2, and a [time_area] over them sets
    lawful_bonus -25. After it the core's view carries the new codes,
    its encoding reads them, and the hexes take the area's bonus."""
    from tools.replay_extract import parse_wml
    from tools.scenario_events import collect_events
    from wesnoth_ai.observe import map_geometry
    gs = _harvested()[0]
    gs.global_info.turn_number, gs.global_info.current_side = 1, 2
    keys = sorted((h.position.x, h.position.y) for h in gs.map.hexes)
    picks = keys[3:8]
    codes = ["Kh", "Gg^Vh", "Hh^Fp", "Wo", "Ss^Vhs"]
    xs = ",".join(str(x + 1) for x, _ in picks)
    ys = ",".join(str(y + 1) for _, y in picks)
    body = "".join(f"[terrain]\nx={x + 1}\ny={y + 1}\nterrain={c}\n[/terrain]\n"
                   for (x, y), c in zip(picks, codes))
    root = parse_wml(f"[multiplayer]\n[event]\nname=side 1 turn 2\n{body}"
                     f"[store_locations]\nvariable=zone\nx={xs}\ny={ys}\n[/store_locations]\n"
                     "[time_area]\nfind_in=zone\n[time]\nlawful_bonus=-25\n[/time]\n[/time_area]\n"
                     "[/event]\n[/multiplayer]\n")
    gs.global_info._scenario_events = collect_events(root, "synthetic")
    gs.global_info._wml_variables = {}
    cs = gc.CoreState.from_state(copy.deepcopy(gs))
    type_to_id, faction_to_id = _vocab_of([gs])
    before = cs.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id, terrain_multi_hot=True)
    for cmd in (["end_turn"], ["init_side", 1]):
        assert cs.apply_command(list(cmd)) == "rust"
    view = cs.to_state()
    assert [view.global_info._terrain_codes[p] for p in picks] == codes
    after = cs.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id, terrain_multi_hot=True)
    assert after.hex_terrain_ids.tobytes() != before.hex_terrain_ids.tobytes()
    turn = view.global_info.turn_number
    assert all(cs.core.lawful_bonus(x, y, turn) == -25 for x, y in picks)
    assert map_geometry(view).keys  # the view's own geometry still builds
def test_the_core_sets_up_every_scenario_and_plays_its_turn_events():
    """Every scenario generation or reconstruction loads sets up on the
    core (`CoreState.setup_scenario`) and plays two turns of its turn
    events with the state round-tripping through the view. Hornshark
    Island's preplaced units depend on the factions, so it runs with
    each of the six as side 1. The engine agrees with the setup on the
    pool (tools/scenario_init_oracle.py, 28 of 28, 2026-09-23)."""
    import dataclasses
    import random
    from wesnoth_ai.rules import scenario_pool as sp
    from wesnoth_ai.rules.scenario_surface import CORPUS_SCENARIOS
    base = sp.random_setup(random.Random(4))
    factions = sp.load_factions()
    cases = [(sid, base) for sid in CORPUS_SCENARIOS]
    for name, info in sorted(factions.items()):
        cases.append(("multiplayer_Hornshark_Island",
                      dataclasses.replace(base, faction1=name, leader1=info.random_leader_pool[0])))
    turns = [["init_side", 1], ["end_turn"], ["init_side", 2], ["end_turn"],
             ["init_side", 1], ["end_turn"], ["init_side", 2], ["end_turn"], ["init_side", 1]]
    for sid, setup in cases:
        gs = sp.build_scenario_gamestate(dataclasses.replace(setup, scenario_id=sid))
        cs = gc.CoreState.from_state(gs)
        cs.setup_scenario(sid)
        for cmd in turns:
            assert cs.apply_command(list(cmd)) == "rust", (sid, cmd)
        view = cs.to_state()
        assert view.global_info.turn_number == 3, sid
        assert cs.core.invariant_violation() is None, sid
        assert not cc.state_differences(view, gc.CoreState.from_state(view).to_state()), sid
