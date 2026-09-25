"""The players are sides 1 and 2, whatever else a scenario declares.

A replayed game keeps a SideInfo for every side its record declares:
the statues of Caves of the Basilisk, Sullas Ruins and Thousand Stings
Garrison, Silverhead Crossing's Shapeshifter, a mini map's tentacles
(7,118 of the 17,019 corpus games, 2026-09-25). The encoder's enemy is
the other player's side (`classes.opponent_of`), the simulator's side
order follows the census of which extra sides take turns
(docs/wesnoth_rules.md "Side order within a turn"), and the value
corpus counts player sides. The encoder tests fail when the enemy is
chosen by counting SideInfos, the turn-order tests when the next side
is, and the value-corpus test when the gate counts [side] blocks.
"""
from __future__ import annotations

import bz2
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import (THREE_SIDE_FACTIONS, replayed_state,  # noqa: E402
                              three_side_record)
from wesnoth_ai.encoder import encode_raw  # noqa: E402

OUR_VILLAGES, THEIR_VILLAGES = 4, 5          # global feature slots

# The record's commands per turn: init_side 1, end_turn, init_side 2, end_turn.
SIDE_1_TO_MOVE, SIDE_2_TO_MOVE = 1, 3        # commands applied


def _vocab():
    names = ("Dwarvish Fighter", "Elvish Captain", "Elvish Fighter", "Lieutenant", "Spearman")
    factions = sorted(set(THREE_SIDE_FACTIONS.values()))
    return ({n: i for i, n in enumerate(names)},
            {"": 0, **{f: i + 1 for i, f in enumerate(factions)}})


def _encode(gs, gate: bool):
    type_to_id, faction_to_id = _vocab()
    return encode_raw(gs, type_to_id=type_to_id, faction_to_id=faction_to_id,
                      fog_hides_enemy_villages=gate)


@pytest.mark.parametrize("fog, gate", [(False, True), (True, False)])
def test_the_encoder_reads_the_other_player_as_the_enemy(fog, gate):
    """The fog gate on a fog-off game (the reference's setting) and no
    gate on a fogged one (the seed's lineage) both read feature 5 from
    the enemy's SideInfo. From each player's seat, the enemy faction and
    feature 5 are what the other player's seat reads as its own."""
    record = three_side_record(fog=fog)
    raws = {}
    for side, n in ((1, SIDE_1_TO_MOVE), (2, SIDE_2_TO_MOVE)):
        gs = replayed_state(record, n)
        assert gs.global_info.current_side == side and len(gs.sides) == 3
        raws[side] = _encode(gs, gate)
    for side in (1, 2):
        mine, theirs = raws[side], raws[3 - side]
        assert mine.their_faction_id == theirs.our_faction_id != mine.our_faction_id
        assert (float(mine.global_feats[THEIR_VILLAGES]) == float(theirs.global_feats[OUR_VILLAGES])
                != float(mine.global_feats[OUR_VILLAGES]))


def test_a_side_that_is_not_a_player_has_no_encoding():
    """The tentacle side to move: there is no opponent to encode."""
    gs = replayed_state(three_side_record(third_side_acts=True), 5)
    assert gs.global_info.current_side == 3
    with pytest.raises(ValueError, match="not a player side"):
        _encode(gs, gate=True)


def _continued(record):
    """The record cut at side 2's turn start of turn 2 and continued in
    the simulator, as tools/turn_gap continues its positions."""
    from tools.bench_pipeline import reconstruct_boundary
    from tools.turn_gap import sim_from_state
    gs, begin_side = reconstruct_boundary(record, 2)
    assert begin_side == 2 and len(gs.sides) == 3
    return sim_from_state(gs, record["scenario_id"], 20, "player_sides")


def _played(sim, steps: int):
    start = len(sim.command_history)
    for _ in range(steps):
        sim.step({"type": "end_turn"})
    return [(c.kind, c.side) for c in sim.command_history[start:]]


def test_a_third_side_that_never_took_a_turn_gets_none_when_continued():
    """Caves of the Basilisk's statue side (controller=null): the
    players alternate and the turn advances at side 1."""
    sim = _continued(three_side_record())
    assert sim.turn_number == 2
    assert _played(sim, 4) == [("end_turn", 2), ("init_side", 1),
                               ("end_turn", 1), ("init_side", 2)] * 2
    assert (sim.current_side, sim.turn_number) == (2, 4)


def test_a_third_side_that_took_turns_keeps_them_after_side_2_when_continued():
    """A mini map's tentacle side (controller=ai): its turn runs inside
    side 2's end_turn step, and side 1 opens the next turn."""
    sim = _continued(three_side_record(third_side_acts=True))
    assert _played(sim, 2) == [("end_turn", 2), ("init_side", 3), ("end_turn", 3),
                               ("init_side", 1), ("end_turn", 1), ("init_side", 2)]
    assert (sim.current_side, sim.turn_number) == (2, 3)


def test_the_turn_gap_hp_margin_counts_the_two_players_only():
    """tools/turn_gap's HP-margin pre-grader: the statue's hit points
    belong to neither player."""
    from tools.turn_gap import _hp_margin
    gs = replayed_state(three_side_record(), SIDE_1_TO_MOVE)
    hp = {u.side: u.current_hp for u in gs.map.units}      # one unit per side
    assert hp[3] > 0
    assert (_hp_margin(gs, 1), _hp_margin(gs, 2)) == (hp[1] - hp[2], hp[2] - hp[1])


class _EndTurns:
    """A policy that ends every turn."""

    def select_action(self, gs, **kw):
        return {"type": "end_turn"}

    def drop_pending(self, game_label):
        pass


def test_the_eval_loop_counts_and_reports_a_turn_no_policy_plays(caplog):
    """A side to move that neither policy plays is a defect: its turn is
    ended, counted in the result and logged."""
    from tools.eval_sim import _PolicyPair, _play_one_eval_game
    from tools.wesnoth_sim import WesnothSim
    gs = replayed_state(three_side_record(third_side_acts=True), 5)
    sim = WesnothSim(gs, "", max_turns=2, apply_scenario_events=False, begin_turn=False)
    with caplog.at_level(logging.WARNING, logger="eval_sim"):
        result = _play_one_eval_game(sim, _PolicyPair(_EndTurns(), "a", 1),
                                     _PolicyPair(_EndTurns(), "b", 2), game_label="sides")
    assert result.unplayed_side_turns == 1
    assert "side 3 is to move" in caplog.text


def _raw_replay(directory: Path, name: str, sides) -> None:
    """A raw replay whose header passes the value corpus's version, era
    and map gates, with the given [side] blocks."""
    lines = ['version="1.18.4"', 'era_id="era_default"',
             "[replay_start]", '    id="multiplayer_Basilisk"']
    for s in sides:
        lines += ["    [side]"] + [f'        {k}="{v}"' for k, v in s.items()] + ["    [/side]"]
    lines.append("[/replay_start]")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(bz2.compress("\n".join(lines).encode("utf-8")))


def test_the_value_corpus_admits_a_scenario_side_beyond_the_two_players(tmp_path):
    """A statue side (controller=null) passes the header gates and
    reaches extraction, which finds no commands in this header-only
    file; an AI opponent and a third human side stop at the side gate."""
    from tools.build_value_corpus import build
    rebels = {"side": 1, "controller": "human", "faction": "Rebels"}
    loyalists = {"side": 2, "controller": "human", "faction": "Loyalists"}
    raw = tmp_path / "raw"
    _raw_replay(raw, "statues.bz2", [rebels, loyalists,
                                     {"side": 3, "controller": "null", "faction": "Custom"}])
    _raw_replay(raw, "ai_opponent.bz2", [rebels, {**loyalists, "controller": "ai"}])
    _raw_replay(raw, "third_human.bz2", [rebels, loyalists,
                                         {"side": 3, "controller": "human", "faction": "Drakes"}])
    stats = build(raw, tmp_path / "out", min_turns=8)
    assert (stats["reject_sides"], stats["reject_no_commands"]) == (2, 1), dict(stats)
