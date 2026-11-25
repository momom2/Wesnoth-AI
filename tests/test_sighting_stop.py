"""A player's move under fog stops where it brings an enemy into view
(docs/wesnoth_rules.md, "A player's move under fog stops where it brings
an enemy into view"; rust/wesnoth_core/src/core_move.rs).

The board is grass, 20 by 7. A Spearman has 5 movement points and grass
costs 1, so from a hex it sees every hex within 6. Along one row the hexes
of consecutive columns touch, so the Spearman walking east from (1, 3)
first sees a unit on (10, 3) from (4, 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai.game_core import game_core_class  # noqa: E402

pytestmark = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core.GameCore not available")

from helpers.parity_games import core_of, record, unit_id_at  # noqa: E402

ROW = 3
EAST = (["move", [1, 2, 3, 4, 5, 6], [ROW] * 6, 1])


def game(enemy_at=(10, ROW), enemy="Skeleton", extra=(), stops=True, **record_kw):
    """Side 1: a Lieutenant on (0, 3) and a Spearman on (1, 3). Side 2: a
    Lieutenant on (19, 3) and `enemy` on `enemy_at`. Side 1 to move, with
    the stop rule set as the simulator sets it."""
    units = [("Lieutenant", 1, 0, ROW, True), ("Spearman", 1, 1, ROW, False),
             ("Lieutenant", 2, 19, ROW, True), (enemy, 2, *enemy_at, False), *extra]
    cs = core_of(record(units, **record_kw))
    if stops:
        cs.core.set_global_int("sighting_stops", 1)
    cs.apply_command(["init_side", 1])
    return cs


def spearman(cs) -> dict:
    return next(u for u in cs.core.units_export() if u["name"] == "Spearman" and u["side"] == 1)


def walk(cs) -> tuple:
    """(the hex the last move landed on, why it stopped)."""
    w = cs.core.last_move_walk_export()
    return (w[2], w[3]), w[4]


def seen_by_side_1(cs) -> set:
    """The ids of the enemy units side 1 has seen since its last end of turn."""
    return {row[0] for row in cs.core.sightings_export(1)}


def test_the_move_stops_on_the_hex_the_enemy_is_sighted_from():
    cs = game()
    skeleton = unit_id_at(cs, 10, ROW)
    assert skeleton not in seen_by_side_1(cs), "the Skeleton starts in fog"
    cs.apply_command(EAST)
    u = spearman(cs)
    assert (u["x"], u["y"]) == (4, ROW)
    assert u["current_moves"] == 2, "three hexes of grass spent, two left"
    assert walk(cs) == ((4, ROW), "sighted")
    assert skeleton in seen_by_side_1(cs), "and it is in view now"
    # The unit decides again with what it has left: it can walk on.
    cs.apply_command(["move", [4, 5, 6], [ROW] * 3, 1])
    u = spearman(cs)
    assert (u["x"], u["y"], u["current_moves"]) == (6, ROW, 0)
    assert walk(cs)[1] == "end", "nothing new comes into view on the way"


def test_without_the_rule_the_move_runs_to_its_end():
    """The core's default, which reconstruction and the records of earlier
    formats play under."""
    cs = game(stops=False)
    cs.apply_command(EAST)
    u = spearman(cs)
    assert (u["x"], u["y"], u["current_moves"]) == (6, ROW, 0)
    assert walk(cs) == ((6, ROW), "end")


def test_a_sighting_from_the_last_hex_stops_nothing():
    cs = game()
    cs.apply_command(["move", [1, 2, 3, 4], [ROW] * 4, 1])
    u = spearman(cs)
    assert (u["x"], u["y"], u["current_moves"]) == (4, ROW, 2)
    assert walk(cs) == ((4, ROW), "end")


def test_an_enemy_already_in_view_stops_nothing():
    cs = game(enemy_at=(7, ROW))
    assert unit_id_at(cs, 7, ROW) in seen_by_side_1(cs)
    cs.apply_command(EAST)
    u = spearman(cs)
    assert (u["x"], u["y"]) == (6, ROW)
    assert walk(cs)[1] == "end"


def test_the_stop_waits_for_a_hex_the_unit_stands_on_alone():
    """The sighting is made while passing over a friend on (4, 3); the unit
    cannot stop there, so it stops on the next hex. The friend is a Heavy
    Infantryman, whose 4 movement points see one hex less far than the
    Spearman's 5, so the Skeleton on (10, 3) is in fog until the Spearman
    looks from that hex."""
    cs = game(extra=[("Heavy Infantryman", 1, 4, ROW, False)])
    assert unit_id_at(cs, 10, ROW) not in seen_by_side_1(cs)
    cs.apply_command(EAST)
    mover = spearman(cs)
    assert (mover["x"], mover["y"], mover["current_moves"]) == (5, ROW, 1)
    assert walk(cs) == ((5, ROW), "sighted")


def test_the_stop_passes_a_village_the_side_does_not_own():
    """Stopping on it would capture it and end the unit's movement, which
    the order did not ask for: the engine walks on to the next hex."""
    cs = game(special={(4, ROW): "Gg^Vh"})
    cs.apply_command(EAST)
    u = spearman(cs)
    assert (u["x"], u["y"], u["current_moves"]) == (5, ROW, 1)
    assert walk(cs) == ((5, ROW), "sighted")


def test_the_stop_is_made_on_a_village_the_side_owns():
    cs = game(special={(4, ROW): "Gg^Vh"}, villages={1: [(4, ROW)]})
    cs.apply_command(EAST)
    u = spearman(cs)
    assert (u["x"], u["y"], u["current_moves"]) == (4, ROW, 2)
    assert walk(cs) == ((4, ROW), "sighted")


def test_a_hidden_ambusher_is_not_a_sighting():
    """An Elvish Ranger in forest is not visible on a hex newly seen, so it
    stops nothing from a distance."""
    cs = game(enemy="Elvish Ranger", special={(10, ROW): "Gs^Fp"})
    cs.apply_command(EAST)
    u = spearman(cs)
    assert (u["x"], u["y"]) == (6, ROW)
    assert walk(cs)[1] == "end"


def test_a_petrified_unit_is_not_a_sighting():
    cs = game()
    cs.core.update_unit(unit_id_at(cs, 10, ROW), {"statuses": ["petrified"]})
    cs.apply_command(EAST)
    assert (spearman(cs)["x"], walk(cs)[1]) == (6, "end")


def test_without_fog_nothing_is_sighted():
    cs = game(fog=False)
    cs.apply_command(EAST)
    assert (spearman(cs)["x"], walk(cs)[1]) == (6, "end")


def test_the_rule_survives_the_round_trip_through_a_state():
    """A simulator's fork and a rebuilt core carry it."""
    from wesnoth_ai.game_core import CoreState
    cs = game()
    again = CoreState.from_state(cs.to_state())
    assert again.core.globals_export()["sighting_stops"] is True
    again.apply_command(EAST)
    assert walk(again) == ((4, ROW), "sighted")
    plain = CoreState.from_state(game(stops=False).to_state())
    assert plain.core.globals_export()["sighting_stops"] is False


def test_the_simulator_plays_under_the_rule_and_its_record_says_so():
    """A game the simulator plays stops its moves on sightings, the record
    carries the rule, and the record rebuilds the game it was written
    from; a record without the field rebuilds with every move run to its
    end."""
    import random
    from tools import game_record
    from tools.inference_seam import RemoteEncoder
    from tools.random_positions import random_legal_action
    from tools.selfplay_game import _would_recruit_bounce
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    encoder = RemoteEncoder({}, {}, relevant_set=True, fog_hides_enemy_villages=True, server_priors=True,
                            terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    stopped = old_rule_mismatches = 0
    for seed in range(3):
        rng = random.Random(seed)
        setup = random_setup(rng)
        sim = WesnothSim(build_scenario_gamestate(setup), setup.scenario_id, max_turns=6)
        assert sim.gs.global_info._sighting_stops is True
        for _ in range(4000):
            if sim.done:
                break
            side = sim.gs.global_info.current_side
            leaders = {u.side: (u.position.x, u.position.y) for u in sim.gs.map.units if u.is_leader}
            action = random_legal_action(encoder.encode(sim.gs), rng, aim=leaders.get(3 - side))
            if _would_recruit_bounce(action, sim.gs):
                sim.reject_recruit_hex(action["target_hex"].x, action["target_hex"].y)
            else:
                sim.step(action)
        stopped += sum(1 for rc in sim.command_history if rc.extras.get("sighted_stop") is not None)
        rec = game_record.game_record(sim, setup, game_label=f"g{seed}")
        assert rec["format"] == game_record.FORMAT == 7 and rec["sighting_stops"] is True
        game_record.rebuild(json_round_trip(rec))                 # verifies every turn's fingerprint
        try:
            game_record.rebuild(dict(json_round_trip(rec), sighting_stops=False))
        except game_record.RecordMismatch:
            old_rule_mismatches += 1
    assert stopped >= 3, "too few moves of three games stopped on a sighting: the fixture tests nothing"
    # A stopped unit that dies before its side's next turn leaves no trace in
    # the fingerprints, so not every game need differ; some must.
    assert old_rule_mismatches >= 1, "no game rebuilt differently with every move run to its end"


def json_round_trip(rec: dict) -> dict:
    import json
    return json.loads(json.dumps(rec))


def test_a_random_draw_sets_the_game_up_as_two_random_picks():
    """`random_setup` draws factions as a Random pick does, so neither side
    is told the other's faction: the faction input starts at the era's six
    and is cut by what the side sees. A setup that names its factions is
    two open picks, as every record written before the field."""
    import dataclasses
    import random
    import numpy as np
    from helpers.parity_games import FACTION_IDS
    from wesnoth_ai.game_core import CoreState
    from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate, random_setup
    setup = random_setup(random.Random(4))
    assert setup.random_factions is True
    gs = build_scenario_gamestate(setup)
    assert [s.chose_random for s in gs.sides[:2]] == [True, True]
    cs = CoreState.from_state(gs)
    cs.setup_scenario(setup.scenario_id)
    cs.apply_command(["init_side", 1])
    types = {u["name"] for u in cs.core.units_export()}
    raw = cs.encode_raw(type_to_id={t: i for i, t in enumerate(sorted(types))}, faction_to_id=FACTION_IDS,
                        relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                        observation_parity=True, relevant_set_version=2)
    probs = np.asarray(raw.their_faction_probs)
    assert np.allclose(sorted(probs[probs > 0]), [1 / 6] * 6), "all six factions, before any enemy is seen"
    # The same game as two open picks: the faction is certain at once.
    open_setup = dataclasses.replace(setup, random_factions=False)
    assert ScenarioSetup(**dataclasses.asdict(open_setup)) == open_setup
    gs = build_scenario_gamestate(open_setup)
    assert [s.chose_random for s in gs.sides[:2]] == [False, False]
    cs = CoreState.from_state(gs)
    cs.setup_scenario(setup.scenario_id)
    cs.apply_command(["init_side", 1])
    raw = cs.encode_raw(type_to_id={t: i for i, t in enumerate(sorted(types))}, faction_to_id=FACTION_IDS,
                        relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                        observation_parity=True, relevant_set_version=2)
    assert np.asarray(raw.their_faction_probs).max() == 1.0
    assert ScenarioSetup("multiplayer_Hamlets", "Rebels", "Elvish Captain", "Drakes", "Drake Flare").random_factions \
        is False
