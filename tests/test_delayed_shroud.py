"""Delayed shroud updates (docs/wesnoth_rules.md "Delayed shroud updates"):
a side that turned "delay shroud updates" on sees the fog it has
committed, not the fog its moves and recruits would have cleared, until
an action that cannot be undone, `[update_shroud]` or `[auto_shroud]
active=yes` commits them. Through the command applier replay
reconstruction uses, against the Rust core over the same commands, and
through the extractor that keeps the two commands.

Each applier test fails when the moves and recruits clear fog at once,
which is what the simulator did until 2026-09-30.
"""
from __future__ import annotations

import copy
import dataclasses
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers.synthetic_replay import (auto_shroud, move, side_block, turn,  # noqa: E402
                                      two_sides, update_shroud, write_replay)
from tools.replay_dataset import (_apply_command, _build_initial_gamestate,  # noqa: E402
                                  _clear_fog_if_advanced, _setup_scenario_events,
                                  iter_record_pairs)
from tools.replay_extract import extract_replay  # noqa: E402
from wesnoth_ai import delayed_shroud  # noqa: E402
from wesnoth_ai.visibility import units_visible_to  # noqa: E402

# A Cavalryman at x=1 sees up to x=10 at turn start; after riding to x=5 it
# would see the enemy Spearman at x=14. The Walking Corpse (level 0, no
# zone of control) stands on the Spearman's path at (2, 1).
UNITS = [("Spearman", 1, 0, 0), ("Cavalryman", 1, 1, 0), ("Spearman", 2, 14, 0),
         ("Spearman", 2, 19, 0), ("Walking Corpse", 2, 2, 1)]
RIDE = ["move", [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 1]
FAR_ENEMY = "u3"


def _game(units=UNITS):
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
    })
    _setup_scenario_events(gs, "")
    return gs


def _sees(gs, uid, side=1) -> bool:
    return any(u.id == uid for u in units_visible_to(gs, side))


def _play(commands, units=UNITS):
    gs = _game(units)
    for cmd in [["init_side", 1], *commands]:
        _apply_command(gs, list(cmd))
    return gs


def test_a_move_clears_fog_at_once_unless_the_side_delays():
    assert _sees(_play([RIDE]), FAR_ENEMY)
    delayed = _play([["auto_shroud", 0], RIDE])
    assert not _sees(delayed, FAR_ENEMY)
    assert _sees(_play([["auto_shroud", 0], RIDE, ["update_shroud"]]), FAR_ENEMY)


def test_turning_the_updates_back_on_commits_the_pending_vision():
    gs = _play([["auto_shroud", 0], RIDE, ["auto_shroud", 1]])
    assert _sees(gs, FAR_ENEMY)
    assert not delayed_shroud.delaying_sides(gs)


def test_an_attack_commits_the_pending_vision():
    """The Spearman at (0, 0) fights the Walking Corpse moved next to it."""
    units = UNITS[:4] + [("Walking Corpse", 2, 0, 1)]
    gs = _play([["auto_shroud", 0], RIDE], units)
    assert not _sees(gs, FAR_ENEMY)
    _apply_command(gs, ["attack", 0, 0, 0, 1, 0, 0, "00c0ffee"])
    assert _sees(gs, FAR_ENEMY)


def test_a_blocked_move_commits_the_pending_vision():
    gs = _play([["auto_shroud", 0], RIDE])
    _apply_command(gs, ["move", [0, 0, 1, 2, 3], [0, 1, 1, 1, 1], 1])
    spearman = next(u for u in gs.map.units if u.id == "u1")
    assert (spearman.position.x, spearman.position.y) == (1, 1), "stopped before the corpse"
    assert _sees(gs, FAR_ENEMY)


def test_a_recruit_commits_only_when_it_drew_random_numbers():
    """A Skeleton's recruit draws nothing (empty seed) and waits; a
    Spearman's draws its traits and commits both."""
    gs = _play([["auto_shroud", 0], ["recruit", "Skeleton", 9, 1, ""]])
    assert not _sees(gs, FAR_ENEMY)
    _apply_command(gs, ["recruit", "Spearman", 2, 1, "0badc0de"])
    assert _sees(gs, FAR_ENEMY)


def test_an_advancement_on_the_delaying_sides_turn_clears_nothing():
    for delay, expected in ((False, True), (True, False)):
        gs = _play([["auto_shroud", 0]] if delay else [])
        before = next(u for u in gs.map.units if u.id == "u1")
        _clear_fog_if_advanced(gs, before, dataclasses.replace(before, max_moves=14))
        assert _sees(gs, FAR_ENEMY) is expected


def test_the_setting_outlives_the_turn_and_the_turn_end_empties_the_stack():
    gs = _play([["auto_shroud", 0], RIDE, ["end_turn"], ["init_side", 2], ["end_turn"],
                ["init_side", 1]])
    assert not delayed_shroud.pending_vision(gs)
    assert delayed_shroud.vision_delayed(gs, 1)


@pytest.mark.parametrize("use_core", [False, True])
def test_the_policy_takes_over_a_delaying_side_with_updates_on(use_core):
    """A mid-game start from a game whose player delayed: the simulator
    records `[auto_shroud] active=yes` at the side's first turn, as the
    engine does when an AI takes control."""
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai import game_core as gc
    if use_core and gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    gs = _game()
    gs.global_info._shroud_delayed = frozenset({1})
    sim = WesnothSim(gs, scenario_id="", apply_scenario_events=False, use_core=use_core)
    assert [c.cmd for c in sim.command_history] == [["init_side", 1], ["auto_shroud", 1]]
    assert not delayed_shroud.delaying_sides(sim.gs)


def test_a_save_that_delays_starts_the_side_delayed(tmp_path):
    sides = [side_block(1, "alice", [("Lieutenant", 2, 4, True)], auto_shroud=False),
             two_sides()[1]]
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, [*turn(1), *turn(2)]))
    assert [s["auto_shroud"] for s in record["starting_sides"]] == [False, True]
    assert delayed_shroud.delaying_sides(_build_initial_gamestate(record)) == {1}


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


# ---- the Rust core against the applier --------------------------------

SEQUENCES = {
    "update": [["auto_shroud", 0], RIDE, ["update_shroud"]],
    "blocked": [["auto_shroud", 0], RIDE, ["move", [0, 0, 1, 2, 3], [0, 1, 1, 1, 1], 1]],
    "recruits": [["auto_shroud", 0], ["recruit", "Skeleton", 9, 1, ""], RIDE,
                 ["recruit", "Spearman", 2, 1, "0badc0de"]],
    "switch": [["auto_shroud", 0], RIDE, ["auto_shroud", 1], ["move", [0, 1], [0, 0], 1]],
    "turns": [["auto_shroud", 0], RIDE, ["end_turn"], ["init_side", 2], ["end_turn"],
              ["init_side", 1], ["move", [5, 6], [0, 0], 1], ["update_shroud"]],
}


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_the_core_and_the_applier_agree_command_by_command(name):
    from wesnoth_ai import game_core as gc
    from wesnoth_ai.core_compare import state_differences
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    gs = _game()
    cs = gc.CoreState.from_state(copy.deepcopy(gs))
    pending_seen = False
    for cmd in [["init_side", 1], *SEQUENCES[name]]:
        _apply_command(gs, list(cmd))
        assert cs.apply_command(list(cmd)) == "rust"
        view = cs.to_state()
        assert state_differences(gs, view, stash=False, map_and_events=False) == [], (name, cmd)
        pending_seen |= bool(view.global_info._pending_vision)
    assert pending_seen, "the sequence never left vision pending"
