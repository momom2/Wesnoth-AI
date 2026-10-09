"""Delayed shroud updates (docs/wesnoth_rules.md "Delayed shroud updates"):
a side that turned "delay shroud updates" on sees the fog it has
committed, not the fog its moves and recruits would have cleared, until
an action that cannot be undone, `[update_shroud]` or `[auto_shroud]
active=yes` commits them. Through the Rust core that applies the
commands, and through the extractor that keeps the two commands.

Each core test fails when the moves and recruits clear fog at once,
which is what the simulator did until 2026-09-30.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers.synthetic_replay import (auto_shroud, move, side_block, turn,  # noqa: E402
                                      two_sides, update_shroud, write_replay)
from tools.replay_dataset import _build_initial_gamestate, iter_record_pairs  # noqa: E402
from tools.replay_extract import extract_replay  # noqa: E402
from wesnoth_ai import delayed_shroud  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")

# A Cavalryman at x=1 sees up to x=10 at turn start; after riding to x=5 it
# would see the enemy Spearman at x=14. The Walking Corpse (level 0, no
# zone of control) stands on the Spearman's path at (2, 1).
UNITS = [("Spearman", 1, 0, 0), ("Cavalryman", 1, 1, 0), ("Spearman", 2, 14, 0),
         ("Spearman", 2, 19, 0), ("Walking Corpse", 2, 2, 1)]
RIDE = ["move", [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 1]
RIDE_BACK = ["move", [6, 5, 4], [0, 0, 0], 1]
FAR_ENEMY = "u3"


def _game(units=UNITS, plan_unit_advance=False):
    """The core of the board's start."""
    row = ", ".join(["Gg"] * 20)
    border = ", ".join(["Xv"] * 22)
    lines = [border] + [f"Xv, {row}, Xv"] * 2 + [border]
    gs = _build_initial_gamestate({
        "game_id": "delayed_shroud",
        "map_data": "\n".join(lines),
        "starting_units": [{"uid": i + 1, "type": t, "side": s, "x": x, "y": y, "is_leader": False}
                           for i, (t, s, x, y) in enumerate(units)],
        "starting_sides": [{"side": k, "gold": 100, "recruit": ["Spearman", "Skeleton"], "fog": True}
                           for k in (1, 2)],
        "plan_unit_advance": plan_unit_advance,
    })
    return gc.CoreState.from_state(gs)


def _sees(cs, uid, side=1) -> bool:
    return uid in cs.core.visible_ids(side)


def _apply(cs, cmd) -> None:
    assert cs.apply_command(list(cmd)) == "rust", cmd


def _play(commands, units=UNITS, plan_unit_advance=False):
    cs = _game(units, plan_unit_advance)
    for cmd in [["init_side", 1], *commands]:
        _apply(cs, cmd)
    return cs


def test_a_move_clears_fog_at_once_unless_the_side_delays():
    assert _sees(_play([RIDE]), FAR_ENEMY)
    delayed = _play([["auto_shroud", 0], RIDE])
    assert not _sees(delayed, FAR_ENEMY)
    assert _sees(_play([["auto_shroud", 0], RIDE, ["update_shroud"]]), FAR_ENEMY)


def test_turning_the_updates_back_on_commits_the_pending_vision():
    cs = _play([["auto_shroud", 0], RIDE, ["auto_shroud", 1]])
    assert _sees(cs, FAR_ENEMY)
    assert not delayed_shroud.delaying_sides(cs.to_state())


def test_an_attack_commits_the_pending_vision():
    """The Spearman at (0, 0) fights the Walking Corpse moved next to it."""
    units = UNITS[:4] + [("Walking Corpse", 2, 0, 1)]
    cs = _play([["auto_shroud", 0], RIDE], units)
    assert not _sees(cs, FAR_ENEMY)
    _apply(cs, ["attack", 0, 0, 0, 1, 0, 0, "00c0ffee"])
    assert _sees(cs, FAR_ENEMY)


def test_an_attack_aborted_before_its_draw_still_commits():
    """No seed follows the attack (a disconnect): the engine's handler had
    cleared the stack before the fight."""
    units = UNITS[:4] + [("Walking Corpse", 2, 0, 1)]
    cs = _play([["auto_shroud", 0], RIDE, ["attack", 0, 0, 0, 1, 0, 0, ""]], units)
    assert _sees(cs, FAR_ENEMY)


STEP = ["move", [0, 0], [0, 1], 1]          # the Spearman's first move: it reveals nothing


def test_the_plan_unit_advance_modification_makes_a_side_turns_first_move_final():
    first = _play([["auto_shroud", 0], RIDE], plan_unit_advance=True)
    assert _sees(first, FAR_ENEMY), "the turn's first move is committed with its own vision"
    second = _play([["auto_shroud", 0], STEP, RIDE], plan_unit_advance=True)
    assert not _sees(second, FAR_ENEMY), "later moves wait"
    _apply(second, ["menu_item", "pickadvance"])
    assert _sees(second, FAR_ENEMY), "the menu's event commits"
    assert not _sees(_play([["auto_shroud", 0], RIDE]), FAR_ENEMY), "without the modification"


def test_a_blocked_move_commits_the_pending_vision():
    cs = _play([["auto_shroud", 0], RIDE])
    _apply(cs, ["move", [0, 0, 1, 2, 3], [0, 1, 1, 1, 1], 1])
    spearman = cs.core.unit_export("u1")
    assert (spearman["x"], spearman["y"]) == (1, 1), "stopped before the corpse"
    assert _sees(cs, FAR_ENEMY)


# The record of the blocked move above: the route cut at the stop, the
# hex after it kept (replay_extract), where the corpse blocked it.
CUT_BEFORE_CORPSE = ["move", [0, 0, 1], [0, 1, 1], 1, {"clicked": [3, 1], "stopped_early": True, "next": [2, 1]}]
# The same route with the checkup saying it ran out of this turn's moves.
RAN_OUT = [*CUT_BEFORE_CORPSE[:4], {**CUT_BEFORE_CORPSE[4], "stopped_early": False}]


def test_a_route_cut_before_an_enemy_is_a_blocked_move():
    cs = _play([["auto_shroud", 0], RIDE, CUT_BEFORE_CORPSE])
    assert _sees(cs, FAR_ENEMY), "the block made the move final"
    assert "u5" in (cs.to_state().global_info._uncovered_units or set()), "the blocker is revealed"
    ended = _play([["auto_shroud", 0], RIDE, CUT_BEFORE_CORPSE[:4]])
    assert not _sees(ended, FAR_ENEMY), "a route that simply ended commits nothing"
    ran_out = _play([["auto_shroud", 0], RIDE, RAN_OUT])
    assert not _sees(ran_out, FAR_ENEMY), "a route that ran out of this turn's moves was not blocked"


def test_a_recruit_commits_only_when_it_drew_random_numbers():
    """A Skeleton's recruit draws nothing (empty seed) and waits; a
    Spearman's draws its traits and commits both."""
    cs = _play([["auto_shroud", 0], ["recruit", "Skeleton", 9, 1, ""]])
    assert not _sees(cs, FAR_ENEMY)
    _apply(cs, ["recruit", "Spearman", 2, 1, "0badc0de"])
    assert _sees(cs, FAR_ENEMY)


def test_an_advancement_on_the_delaying_sides_turn_clears_nothing():
    """The Cavalryman at x=1 sees up to x=10; advanced to a 9-move
    Dragoon it sees the Spearman at x=11, unless its side delays."""
    units = UNITS + [("Spearman", 2, 11, 0)]
    for delay, expected in ((False, True), (True, False)):
        cs = _play([["auto_shroud", 0]] if delay else [], units)
        assert not _sees(cs, "u6")
        cav = cs.core.unit_export("u2")
        cs.core.update_unit("u2", {"current_exp": cav["max_exp"]})
        assert cs.core.advance_unit_id("u2")
        assert cs.core.unit_export("u2")["name"] == "Dragoon"
        assert _sees(cs, "u6") is expected


def test_the_setting_outlives_the_turn_and_the_turn_end_empties_the_stack():
    view = _play([["auto_shroud", 0], RIDE, ["end_turn"], ["init_side", 2], ["end_turn"],
                  ["init_side", 1]]).to_state()
    assert not delayed_shroud.pending_vision(view)
    assert delayed_shroud.vision_delayed(view, 1)


def test_the_policy_takes_over_a_delaying_side_with_updates_on():
    """A mid-game start from a game whose player delayed: the simulator
    records `[auto_shroud] active=yes` at the side's first turn, as the
    engine does when an AI takes control."""
    from tools.wesnoth_sim import WesnothSim
    gs = _game().to_state()
    gs.global_info._shroud_delayed = frozenset({1})
    sim = WesnothSim(gs, scenario_id="", apply_scenario_events=False)
    assert [c.cmd for c in sim.command_history] == [["init_side", 1], ["auto_shroud", 1]]
    assert not delayed_shroud.delaying_sides(sim.gs)


def test_a_save_that_delays_starts_the_side_delayed(tmp_path):
    sides = [side_block(1, "alice", [("Lieutenant", 2, 4, True)], auto_shroud=False),
             two_sides()[1]]
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, [*turn(1), *turn(2)]))
    assert [s["auto_shroud"] for s in record["starting_sides"]] == [False, True]
    assert delayed_shroud.delaying_sides(_build_initial_gamestate(record)) == {1}


def test_the_extractor_records_the_modification_and_its_menu_events(tmp_path):
    from helpers.synthetic_replay import menu_item
    commands = [*turn(1, menu_item(1, 2, 4)), *turn(2)]
    record = extract_replay(write_replay(tmp_path / "g.bz2", two_sides(), commands,
                                         header=['active_mods="plan_unit_advance"']))
    assert record["plan_unit_advance"] is True
    assert ["menu_item", "pickadvance"] in record["commands"]
    plain = extract_replay(write_replay(tmp_path / "h.bz2", two_sides(), [*turn(1), *turn(2)]))
    assert plain["plan_unit_advance"] is False


def test_the_extractor_keeps_the_commands_and_the_labels_skip_them(tmp_path):
    ride = [(3, 5), (4, 5), (5, 5), (6, 5)]
    commands = [*turn(1, auto_shroud(False), move(1, ride), update_shroud(), auto_shroud(True)),
                *turn(2)]
    record = extract_replay(write_replay(tmp_path / "g.bz2",
                                         two_sides(extra1=[("Cavalryman", 3, 5, False)]), commands))
    kinds = [c[0] for c in record["commands"]]
    assert kinds[:5] == ["init_side", "auto_shroud", "move", "update_shroud", "auto_shroud"]
    assert [record["commands"][1], record["commands"][4]] == [["auto_shroud", 0], ["auto_shroud", 1]]
    stats = Counter()
    labels = [ai.action_type for _gs, ai in iter_record_pairs(record, stats=stats)]
    assert labels == ["move", "end_turn", "end_turn"] and not stats["unpaired"]
