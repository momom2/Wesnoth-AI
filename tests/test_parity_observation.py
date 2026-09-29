"""The parity observation (docs/parity_memory_design_20260929.md "What the
network observes"), which the Rust core builds under `observation_parity`:
each new column against values computed here from unit_stats.json and the
engine's rules on hand-built positions; the recruit rows; the hex columns,
terrain classes and village bit; the globals; the relevant set version 2;
the refusal of the paths that do not build it; and the eval path's
encoding of a core fork, byte for byte the deep copy's for obs8's flags."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from wesnoth_ai import encoder as enc
from wesnoth_ai.game_core import game_core_class

pytestmark = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core.GameCore phase 23 not available")

from helpers.parity_games import (  # noqa: E402
    core_of, parity_raw, record, seed_rolling, unit_id_at, unit_row, unit_types, vocab_of,
)

W, C = enc.PARITY_WEAPON_AT, enc.PARITY_WEAPON_COLS
NIGHT = 4                     # tod_start_index of first watch: turn 1 at lawful bonus -25


def expected_weapon(attack: dict, damage: int) -> np.ndarray:
    """One weapon slot from a unit_stats.json attack and its damage."""
    cols = np.zeros(C, dtype=np.float32)
    cols[0] = 1.0
    cols[1] = damage / enc.WEAPON_DAMAGE_NORM
    cols[2] = attack["number"] / enc.WEAPON_STRIKES_NORM
    cols[3] = 1.0 if attack["range"] == "ranged" else 0.0
    cols[4 + enc.PARITY_DAMAGE_TYPES.index(attack["type"])] = 1.0
    for s in attack.get("specials") or ():
        if s in enc.PARITY_SPECIALS:
            cols[4 + len(enc.PARITY_DAMAGE_TYPES) + enc.PARITY_SPECIALS.index(s)] = 1.0
    return cols


def expected_resistances(unit_type: dict) -> np.ndarray:
    res = unit_type.get("resistance") or {}
    return np.array([(100 - res.get(t, 100)) / 100 for t in enc.PARITY_DAMAGE_TYPES], dtype=np.float32)


def bits_on(row: np.ndarray, start: int, names) -> set:
    return {n for n, b in zip(names, row[start:start + len(names)]) if b}


def combat_modifier(alignment: str, lawful_bonus: int, fearless: bool) -> int:
    """src/actions/attack.cpp `generic_combat_modifier` (1.18.4)."""
    bonus = {"lawful": lawful_bonus, "neutral": 0, "chaotic": -lawful_bonus,
             "liminal": 25 - abs(lawful_bonus)}[alignment]
    return max(bonus, 0) if fearless else bonus


def illuminated(lawful_bonus: int) -> int:
    """The [illuminates] ability: +25, bounded at 25 (tod_manager.cpp:265-281)."""
    return min(lawful_bonus + 25, max(lawful_bonus, 25))


def test_a_units_weapons_resistances_traits_and_abilities_are_its_own():
    """A strong Spearman's spear hits for one more than its type's, a
    dextrous Elvish Archer's bow likewise; every other column follows
    the unit database and the rolled traits."""
    import wesnoth_core
    types = unit_types()
    data = record([("Lieutenant", 1, 0, 3, True), ("Lieutenant", 2, 19, 3, True)], fog=False)
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    strong, dextrous = seed_rolling("Spearman", "strong"), seed_rolling("Elvish Archer", "dextrous")
    cs.apply_command(["recruit", "Spearman", 5, 2, strong])
    cs.apply_command(["recruit", "Elvish Archer", 6, 2, dextrous])
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Elvish Archer"]))
    assert raw.unit_feats.shape[1] == enc.UNIT_FEAT_DIM_PARITY
    for name, (x, y), seed in (("Spearman", (5, 2), strong), ("Elvish Archer", (6, 2), dextrous)):
        row = unit_row(raw, unit_id_at(cs, x, y))
        traits = set(wesnoth_core.roll_type_traits(name, seed))
        for k, attack in enumerate(types[name]["attacks"]):
            # strong: +1 melee damage; dextrous: +1 ranged (data/core/macros/traits.cfg)
            bump = (attack["range"] == "melee" and "strong" in traits) or \
                   (attack["range"] == "ranged" and "dextrous" in traits)
            assert np.array_equal(row[W + k * C:W + (k + 1) * C],
                                  expected_weapon(attack, attack["damage"] + bump)), (name, k)
        assert not row[W + 2 * C:W + 3 * C].any()
        assert np.array_equal(row[enc.PARITY_RESIST_AT:enc.PARITY_TRAIT_AT], expected_resistances(types[name]))
        assert bits_on(row, enc.PARITY_TRAIT_AT, enc.PARITY_TRAITS) == traits & set(enc.PARITY_TRAITS)
        assert not bits_on(row, enc.PARITY_ABILITY_AT, enc.PARITY_ABILITIES)
    leader = unit_row(raw, unit_id_at(cs, 0, 3))
    assert bits_on(leader, enc.PARITY_ABILITY_AT, enc.PARITY_ABILITIES) == {"leadership"}


def test_statuses_time_of_day_and_leadership():
    """At first watch: a lawful Spearman under an adjacent Lieutenant
    fights at -25 with +25 leadership; a fearless Heavy Infantryman at 0;
    a chaotic Troll Whelp at +25; a Spearman in an always-day time area at
    +25; a Mage of Light and its neighbour lit back to 0. A poisoned and
    slowed enemy shows both. The hex columns carry the time area and the
    light as the difference from the board's -25."""
    data = record([("Lieutenant", 1, 5, 1, True), ("Spearman", 1, 5, 2, False),
                   ("Heavy Infantryman", 1, 9, 2, False), ("Troll Whelp", 1, 11, 2, False),
                   ("Spearman", 1, 12, 5, False), ("Mage of Light", 1, 2, 4, False),
                   ("Spearman", 1, 2, 5, False), ("Lieutenant", 2, 19, 6, True),
                   ("Elvish Archer", 2, 15, 2, False)],
                  fog=False, tod_start_index=NIGHT)
    cs = core_of(data, time_areas={(12, 5): [25] * 6})
    cs.apply_command(["init_side", 1])
    cs.core.update_unit(unit_id_at(cs, 9, 2), {"traits": ["fearless"]})
    cs.core.update_unit(unit_id_at(cs, 15, 2), {"statuses": ["poisoned", "slowed"]})
    raw = parity_raw(cs, vocab_of(u["type"] for u in data["starting_units"]))
    board = -25
    cases = {  # hex: (alignment, lawful bonus at the hex, fearless, leadership)
        (5, 2): ("lawful", board, False, 25), (9, 2): ("lawful", board, True, 0),
        (11, 2): ("chaotic", board, False, 0), (12, 5): ("lawful", 25, False, 0),
        (2, 4): ("lawful", illuminated(board), False, 0), (2, 5): ("lawful", illuminated(board), False, 0),
        (5, 1): ("lawful", board, False, 0), (15, 2): ("neutral", board, False, 0),
    }
    for (x, y), (alignment, bonus, fearless, lead) in cases.items():
        row = unit_row(raw, unit_id_at(cs, x, y))
        assert row[enc.PARITY_TOD_AT] == np.float32(combat_modifier(alignment, bonus, fearless) / 25), (x, y)
        assert row[enc.PARITY_LEADERSHIP_AT] == np.float32(lead / enc.LEADERSHIP_NORM), (x, y)
        statuses = (row[enc.PARITY_POISONED_AT], row[enc.PARITY_SLOWED_AT])
        assert statuses == ((1.0, 1.0) if (x, y) == (15, 2) else (0.0, 0.0)), (x, y)
    from tools.abilities import hex_neighbors
    tod = {(p.x, p.y): v for p, v in zip(raw.hex_positions, raw.hex_dynamic_flags[:, enc.PARITY_HEX_TOD_AT])}
    assert tod[(12, 5)] == np.float32((25 - board) / 25)
    for p in {(2, 4)} | set(hex_neighbors(2, 4)):      # the Mage of Light's hex and its ring
        assert tod[p] == np.float32((illuminated(board) - board) / 25), p
    assert tod[(9, 2)] == 0.0


def test_recruit_rows_take_the_types_base_values():
    """A recruit row reads as a fresh recruit of the type: full hit points,
    no moves, no attack left, the experience cap the game's 70% gives it,
    the alignment coded as board units code it, the type's weapons,
    resistances and abilities. Without the flag the row is obs8's."""
    types = unit_types()
    data = record([("Lieutenant", 1, 0, 3, True), ("Lieutenant", 2, 19, 3, True)],
                  recruits={1: ["Spearman", "Mage"]}, experience_modifier=70)
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    names = ["Lieutenant", "Spearman", "Mage"]
    raw = parity_raw(cs, vocab_of(names))
    assert raw.recruit_types == ["Spearman", "Mage"]
    for row, name in zip(raw.recruit_feats, raw.recruit_types):
        t = types[name]
        xp = max(1, (t["experience"] * 70 + 50) // 100)
        align = np.zeros(enc.NUM_ALIGNMENTS, dtype=np.float32)
        align[{"lawful": 0, "neutral": 1, "chaotic": 2, "liminal": 3}[t["alignment"]]] = 1.0
        head = np.array([t["hitpoints"] / enc.HP_NORM, 1.0, t["moves"] / enc.MOVES_NORM, 0.0,
                         xp / enc.EXP_NORM, 0.0, t["cost"] / enc.COST_NORM, 0.0, 1.0], dtype=np.float32)
        assert np.array_equal(row[:enc.UNIT_FEAT_DIM], np.concatenate([head, align])), name
        for k, attack in enumerate(t["attacks"]):
            assert np.array_equal(row[W + k * C:W + (k + 1) * C], expected_weapon(attack, attack["damage"]))
        assert np.array_equal(row[enc.PARITY_RESIST_AT:enc.PARITY_TRAIT_AT], expected_resistances(t))
        assert not row[enc.PARITY_TRAIT_AT:enc.PARITY_ABILITY_AT].any()
        assert not row[enc.PARITY_POISONED_AT:].any()
    obs8 = parity_raw(cs, vocab_of(names), parity=False)
    for row, name in zip(obs8.recruit_feats, obs8.recruit_types):
        assert np.array_equal(row, enc._recruit_features_for(name))


def _economy_record(fog: bool) -> dict:
    villages = {(3, 0): 1, (3, 6): 1, (16, 0): 2}
    return record([("Lieutenant", 1, 0, 3, True), ("Spearman", 1, 4, 3, False), ("Spearman", 1, 5, 3, False),
                   ("Spearman", 1, 6, 3, False), ("Spearman", 1, 7, 2, False), ("Lieutenant", 1, 8, 3, False),
                   ("Lieutenant", 2, 19, 3, True), ("Mage", 2, 18, 3, False), ("Spearman", 2, 17, 3, False)],
                  special={p: "Gg^Vh" for p in villages}, fog=fog, gold=(100, 150),
                  villages={s: [p for p, o in villages.items() if o == s] for s in (1, 2)},
                  village_gold=3, village_support=2)


@pytest.mark.parametrize("fog", [True, False])
def test_globals_carry_the_economy_the_status_table_shows(fog):
    """Side 1: base income 2, two villages at 3 gold, upkeep 1+1+1+2 (its
    loyal Spearman and its leader pay none) against a support of 2x2, so a
    net income of 2 + 6 - 1 = 7. Side 2: 150 gold, net 2 + 3 - 0 = 5,
    upkeep 2; shown only with fog off."""
    cs = core_of(_economy_record(fog))
    cs.apply_command(["init_side", 1])
    cs.core.update_unit(unit_id_at(cs, 7, 2), {"traits": ["loyal"]})
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Mage"]))
    g = raw.global_feats
    assert g.shape == (enc.GLOBAL_FEAT_DIM_PARITY,)
    want = [3 / enc.VILLAGE_GOLD_NORM, 2 / enc.VILLAGE_SUPPORT_NORM, 7 / enc.INCOME_NORM, 1.0 if fog else 0.0]
    want += [0.0, 0.0, 0.0] if fog else [150 / enc.GOLD_NORM, 5 / enc.INCOME_NORM, 2 / enc.INCOME_NORM]
    assert np.array_equal(g[enc.GLOBAL_FEAT_DIM:], np.array(want, dtype=np.float32))
    obs8 = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Mage"]), parity=False)
    assert np.array_equal(g[:enc.GLOBAL_FEAT_DIM], obs8.global_feats)


@pytest.mark.parametrize("fog", [True, False])
def test_hex_columns_terrain_classes_and_the_village_bit(fog):
    """The fog overlay column is the side's seen hexes (every hex with fog
    off); an unowned merfolk village far in the fog carries its village
    bit; mushroom grove and reef have their own classes; the recruit
    rejection column is written 0. Without the flag the village reads as
    water and the classes are cave and shallow water."""
    special = {(12, 1): "Ww^Vm", (2, 2): "Tb^Tf", (2, 4): "Wwr"}
    data = record([("Lieutenant", 1, 1, 3, True), ("Lieutenant", 2, 19, 3, True)], special=special, fog=fog)
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    cs.core.add_recruit_rejected(1, 2)
    names = ["Lieutenant"]
    raw = parity_raw(cs, vocab_of(names))
    obs8 = parity_raw(cs, vocab_of(names), parity=False)
    assert raw.hex_dynamic_flags.shape[1] == enc.NUM_HEX_DYNAMIC_FLAGS_PARITY
    keys = cs.geometry().keys
    seen = {keys[j] for j in np.flatnonzero(cs.core.seen_export(1))}
    tok = {(p.x, p.y): t for t, p in enumerate(raw.hex_positions)}
    for p, t in tok.items():
        assert raw.hex_dynamic_flags[t, enc.PARITY_HEX_SEEN_AT] == (1.0 if (not fog or p in seen) else 0.0), p
    assert (12, 1) not in seen or not fog
    old = {(p.x, p.y): t for t, p in enumerate(obs8.hex_positions)}
    # The unowned merfolk village: its village bit with the flag, none in
    # obs8's encoding, fog or not (obs8 marks a village by its owner).
    assert raw.hex_modifier_flags[tok[(12, 1)], 0] == 1.0
    assert obs8.hex_modifier_flags[old[(12, 1)], 0] == 0.0
    fungus, reef = 1 << 14, 1 << 15
    cave, shallow = 1 << int(enc.Terrain.CAVE), 1 << int(enc.Terrain.SHALLOWWATER)
    assert raw.hex_terrain_ids[tok[(2, 2)]] & fungus and not raw.hex_terrain_ids[tok[(2, 2)]] & cave
    assert raw.hex_terrain_ids[tok[(2, 4)]] & reef and not raw.hex_terrain_ids[tok[(2, 4)]] & shallow
    assert obs8.hex_terrain_ids[old[(2, 2)]] & cave and obs8.hex_terrain_ids[old[(2, 4)]] & shallow
    assert raw.hex_dynamic_flags[tok[(1, 2)], 0] == 0.0 and obs8.hex_dynamic_flags[old[(1, 2)], 0] == 1.0
    both = [p for p in tok if p in old]
    assert np.array_equal(raw.hex_dynamic_flags[[tok[p] for p in both], 1:3],
                          obs8.hex_dynamic_flags[[old[p] for p in both], 1:3])


def _hex_distance_by_steps(a, b, limit):
    """Hex distance up to `limit` by walking neighbours on an open grid."""
    from tools.abilities import hex_neighbors
    frontier, seen = {a}, {a}
    for d in range(limit + 1):
        if b in frontier:
            return d
        frontier = {n for p in frontier for n in hex_neighbors(*p)} - seen
        seen |= frontier
    return limit + 1


def test_the_relevant_set_version_2():
    """obs8's set, plus the six neighbours of each own unit that is not
    scenery, plus every hex the side does not see within 6 of such a
    unit; the hex tokens are that set in row-major order. A deep-water
    channel stops the Spearman's vision two hexes out."""
    channel = {(x, y): "Wo" for x in (8, 9, 10) for y in range(9)}
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 6, 1, False),
                   ("Lieutenant", 2, 19, 3, True)], fog=True, width=24, height=9, special=channel)
    data["starting_units"].append({"uid": 9, "type": "Dwarvish Fighter", "side": 1, "x": 20, "y": 8,
                                   "is_leader": False, "petrified": True})
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    names = ["Lieutenant", "Spearman", "Dwarvish Fighter"]
    raw = parity_raw(cs, vocab_of(names))
    v1 = parity_raw(cs, vocab_of(names), parity=False)
    keys = cs.geometry().keys
    base = {keys[j] for j in np.flatnonzero(v1.observation.relevant)}
    seen = {keys[j] for j in np.flatnonzero(cs.core.seen_export(1))}
    own = [(1, 3), (6, 1)]
    from tools.abilities import hex_neighbors
    board = set(keys)
    near = {p for p in board - seen if any(_hex_distance_by_steps(u, p, 6) <= 6 for u in own)}
    want = base | {n for u in own for n in hex_neighbors(*u) if n in board} | near
    got = {keys[j] for j in np.flatnonzero(raw.observation.relevant)}
    assert near and not near <= base
    assert got == want
    assert [(p.x, p.y) for p in raw.hex_positions] == sorted(want, key=lambda p: (p[1], p[0]))
    statue_ring = set(hex_neighbors(20, 8)) & board
    assert not (statue_ring - base - near) & got


def test_the_paths_that_do_not_build_it_refuse_the_flag():
    """The Python builders (a state not bound to a core), the kernel
    `encode_raw_streams`, and the core without the terrain set view."""
    import wesnoth_core
    data = record([("Lieutenant", 1, 1, 3, True), ("Lieutenant", 2, 19, 3, True)])
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    view = cs.to_state()
    with pytest.raises(ValueError, match="Rust core only"):
        enc.encode_raw(copy.deepcopy(view), type_to_id={}, faction_to_id={}, relevant_set=True,
                       terrain_multi_hot=True, observation_parity=True)
    with pytest.raises(ValueError, match="observation_parity"):
        wesnoth_core.encode_raw_streams(
            np.zeros((0, enc.NUM_HEX_MODIFIERS), dtype=np.float32), np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), 0, 0, [0.0] * enc.GLOBAL_FEAT_DIM,
            [1.0] * 8, enc.MAX_MAP_SIZE - 1, enc.NUM_ALIGNMENTS, observation_parity=True)
    with pytest.raises(ValueError, match="terrain_multi_hot"):
        cs.encode_raw(type_to_id={}, faction_to_id={}, relevant_set=True, terrain_multi_hot=False,
                      observation_parity=True)


def test_the_eval_path_encodes_a_core_fork_as_it_encoded_the_deep_copy():
    """The eval player, the pool's root decisions and the holdout probe
    keep a view of a fork of the simulator's core (`snapshot_view`) where
    they kept a deep copy; for obs8's flags its encoding equals the deep
    copy's, field for field, at every decision of a played game."""
    from sim_test_helpers import scenario_setup
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.core_compare import encoding_differences
    from wesnoth_ai.dummy_policy import DummyPolicy
    from wesnoth_ai.game_core import core_of as bound_core, snapshot_view
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    obs8 = dict(relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True)
    compared = 0
    for seed in (3, 5):
        setup = scenario_setup(seed, mini=seed == 5)
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=4)
        assert sim.core is not None
        policy = DummyPolicy()
        names = sorted({u.name for u in sim.gs.map.units} | {r for s in sim.gs.sides for r in s.recruits})
        type_to_id = vocab_of(names)
        faction_to_id = {f: i for i, f in enumerate(sorted({s.faction for s in sim.gs.sides}))}
        step = 0
        while not sim.done and step < 60:
            if step % 3 == 0:
                snap = snapshot_view(sim.gs)
                assert bound_core(snap) is not None and bound_core(copy.deepcopy(sim.gs)) is None
                kw = dict(type_to_id=type_to_id, faction_to_id=faction_to_id, **obs8)
                assert encoding_differences(enc.encode_raw(copy.deepcopy(sim.gs), **kw),
                                            enc.encode_raw(snap, **kw)) == [], (seed, step)
                compared += 1
            sim.step(policy.select_action(sim.gs, game_label="fork"))
            step += 1
    assert compared >= 20


def test_unknown_unit_names_are_counted_on_every_encode_path(caplog):
    """A unit type the vocabulary lacks takes the overflow row, as before,
    and is now counted and warned about once: on the core's path and on the
    deep copy's. A type the unit database lacks is counted there too."""
    import logging
    from wesnoth_ai.game_core import CoreState, unit_db_fallbacks
    from helpers.parity_games import state_of
    data = record([("Lieutenant", 1, 1, 3, True), ("Lieutenant", 2, 19, 3, True),
                   ("Spearman", 1, 2, 3, False)])
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    before = enc.unknown_type_counts().get("Spearman", 0)
    with caplog.at_level(logging.WARNING, logger="encoder"):
        core_raw = parity_raw(cs, {"Lieutenant": 0}, parity=False)
        py_raw = enc.encode_raw(copy.deepcopy(cs.to_state()), type_to_id={"Lieutenant": 0},
                                faction_to_id={}, relevant_set=True, fog_hides_enemy_villages=True,
                                terrain_multi_hot=True)
    assert enc.unknown_type_counts()["Spearman"] >= before + 2
    overflow = enc.MAX_UNIT_TYPES - 1
    assert core_raw.unit_type_ids[core_raw.unit_ids.index(unit_id_at(cs, 2, 3))] == overflow
    assert py_raw.unit_type_ids[py_raw.unit_ids.index(unit_id_at(cs, 2, 3))] == overflow
    if before == 0:
        assert sum("'Spearman'" in r.getMessage() for r in caplog.records) == 1
    gs = state_of(data)
    odd = next(u for u in gs.map.units if u.name == "Spearman")
    odd.name = "Parity Test Phantom"
    with caplog.at_level(logging.WARNING, logger="game_core"):
        CoreState.from_state(gs)
    assert unit_db_fallbacks().get("Parity Test Phantom", 0) >= 1
