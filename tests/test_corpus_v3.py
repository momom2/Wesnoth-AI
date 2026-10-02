"""The imitation corpus's version-3 rules, on replays written from scratch
(tests/helpers/synthetic_replay.py), so they run where no corpus is:

  - a move the engine makes at a side's turn start for a unit's standing
    order is applied and never paired as a decision; a move the player
    makes first, or toward an order an interruption cancelled, is paired;
  - the end_turn of a turn that ran out of time is applied and never
    paired, when the recorded time can only follow a timeout;
  - each side records whether its player picked Random, and the record
    its era.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.synthetic_replay import (countdown_update, move, side_block, turn,  # noqa: E402
                                      two_sides, write_replay)
from tools.replay_dataset import iter_record_pairs  # noqa: E402
from tools.replay_engine_actions import TurnTimer, engine_goto_moves  # noqa: E402
from tools.replay_extract import extract_replay  # noqa: E402

CAVALRY = [("Cavalryman", 3, 5, False)]
ORDER = [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]
TIMER = {"mp_countdown": "yes", "mp_countdown_init_time": 240, "mp_countdown_turn_bonus": 240,
         "mp_countdown_action_bonus": 0, "mp_countdown_reservoir_time": 360}


def _pairs(record):
    stats = Counter()
    pairs = [(gs, ai) for gs, ai in iter_record_pairs(record, stats=stats)]
    return pairs, stats


def _cavalryman_hex(gs):
    return next((u.position.x, u.position.y) for u in gs.map.units if u.name == "Cavalryman")


def _multi_turn_game(tmp_path, second_turn):
    """Side 1 orders its Cavalryman to (9, 5); the turn's movement ends at
    (6, 5) (stopped_early no: a standing order remains). Side 2 passes;
    side 1's second turn plays `second_turn`."""
    first = move(1, ORDER, final=(6, 5), stopped_early="no")
    return extract_replay(write_replay(tmp_path / "g.bz2", two_sides(extra1=CAVALRY),
                                       [*turn(1, first), *turn(2), *turn(1, *second_turn), *turn(2)]))


def test_a_standing_order_moved_at_turn_start_is_not_a_decision(tmp_path):
    record = _multi_turn_game(tmp_path, [move(1, ORDER[3:])])
    assert record["engine_issued"]["goto"] == [6]
    pairs, stats = _pairs(record)
    assert stats["engine_goto"] == 1
    side1 = [(gs, ai) for gs, ai in pairs if gs.global_info.current_side == 1]
    assert [ai.action_type for _gs, ai in side1] == ["move", "end_turn", "end_turn"]
    # applied all the same: side 1's second end_turn sees the unit at the order's end
    assert _cavalryman_hex(side1[-1][0]) == (8, 4)


def test_a_move_the_player_makes_first_ends_the_turn_start(tmp_path):
    leader_step = move(1, [(2, 4), (2, 5)])
    record = _multi_turn_game(tmp_path, [leader_step, move(1, ORDER[3:])])
    assert record["engine_issued"]["goto"] == []
    pairs, stats = _pairs(record)
    assert stats["engine_goto"] == 0
    assert [ai.action_type for gs, ai in pairs if gs.global_info.current_side == 1] == \
        ["move", "end_turn", "move", "move", "end_turn"]


def test_an_interrupted_move_leaves_no_standing_order(tmp_path):
    """stopped_early yes: the engine interrupted the move (a sighting), so no
    goto is set; the next turn's move is the player's."""
    first = move(1, ORDER, final=(6, 5), stopped_early="yes")
    record = extract_replay(write_replay(tmp_path / "g.bz2", two_sides(extra1=CAVALRY),
                                         [*turn(1, first), *turn(2), *turn(1, move(1, ORDER[3:])),
                                          *turn(2)]))
    assert record["engine_issued"]["goto"] == []


def test_standing_orders_chain_across_turns():
    """A continuation that again ends its turn short keeps the order."""
    commands = [["init_side", 1], ["move", [2, 3, 4], [4, 4, 4], 1, {"clicked": [9, 4], "stopped_early": False}],
                ["end_turn"], ["init_side", 2], ["end_turn"],
                ["init_side", 1], ["move", [4, 5, 6], [4, 4, 4], 1, {"clicked": [9, 4], "stopped_early": False}],
                ["end_turn"], ["init_side", 2], ["end_turn"],
                ["init_side", 1], ["move", [6, 7, 8, 9], [4, 4, 4, 4], 1], ["end_turn"]]
    assert engine_goto_moves(commands) == [6, 11]


def test_a_turn_that_ran_out_is_not_an_end_turn_decision(tmp_path):
    sides = two_sides(extra1=CAVALRY)
    commands = [*turn(1, countdown_update(1, 240_000)),          # nothing left: a timeout
                *turn(2, countdown_update(2, 300_000)),          # 60 s left
                *turn(1, countdown_update(1, 360_000)),          # the reservoir
                *turn(2)]
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, commands, multiplayer=TIMER))
    assert record["turn_timer"] == [240, 240, 0, 360]
    assert record["engine_issued"]["timeout"] == [1]
    pairs, stats = _pairs(record)
    assert stats["engine_timeout"] == 1
    assert [gs.global_info.current_side for gs, ai in pairs if ai.action_type == "end_turn"] == [2, 1, 2]
    # On request, the position the player was deciding in comes with the
    # TIMEOUT label, which names no action; the end_turns are as before.
    labelled = list(iter_record_pairs(record, timeouts=True))
    assert [(gs.global_info.current_side, gs.global_info.turn_number) for gs, ai in labelled
            if ai.action_type == "timeout"] == [(1, 1)]
    assert [ai.actor_idx for gs, ai in labelled if ai.action_type == "timeout"] == [-1]
    assert [ai.action_type for gs, ai in labelled if ai.action_type != "timeout"] == \
        [ai.action_type for gs, ai in pairs]


def test_a_timeout_is_certain_only_below_the_reservoir_and_without_action_bonus():
    assert TurnTimer(240, 240, 0, 360).is_timeout(240_000)
    assert not TurnTimer(240, 240, 0, 360).is_timeout(241_000)
    assert not TurnTimer(120, 120, 0, 120).is_timeout(120_000)     # capped: a player's time reads the same
    assert not TurnTimer(240, 240, 30, 600).is_timeout(240_000)    # the action bonus is unknown here


def test_each_side_records_whether_its_player_picked_random(tmp_path):
    sides = [side_block(1, "alice", [("Lieutenant", 2, 4, True)], chose_random=True),
             side_block(2, "bob", [("Lieutenant", 11, 4, True)], chose_random=False)]
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, [*turn(1), *turn(2)]))
    assert [s["chose_random"] for s in record["starting_sides"]] == [True, False]
    assert record["era_id"] == "era_default"


def test_the_winners_action_count_leaves_out_the_engines_moves(tmp_path):
    from tools.build_imitation_dataset import _winner_action_count
    from tools.replay_dataset import engine_issued_of
    record = _multi_turn_game(tmp_path, [move(1, ORDER[3:])])
    assert _winner_action_count(record["commands"], 1) == 2
    assert _winner_action_count(record["commands"], 1, engine_issued_of(record)) == 1
