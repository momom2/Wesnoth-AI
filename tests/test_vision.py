"""What a side sees under fog, by the engine's rule (docs/wesnoth_rules.md
"Vision and fog"), and when a revealed hider hides again, through the
Rust core that the simulator and replay reconstruction share.

The first three tests fail under the disc of radius max_moves the
simulator drew until 2026-09-24; the recruit and death tests fail when
a command does not clear or recalculate the side's fog.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from tools.replay_dataset import _build_initial_gamestate  # noqa: E402
from wesnoth_ai import visibility  # noqa: E402
from wesnoth_ai.game_core import CoreState  # noqa: E402


def _game(rows, units, *, recruits=()):
    """The core of a fogged game on `rows` of terrain codes (0-indexed,
    no border); `units` as (type, side, x, y, extra fields)."""
    width = len(rows[0].split())
    border = ", ".join(["Xv"] * (width + 2))
    lines = [border] + ["Xv, " + ", ".join(r.split()) + ", Xv" for r in rows] + [border]
    return CoreState.from_state(_build_initial_gamestate({
        "game_id": "vision",
        "map_data": "\n".join(lines),
        "starting_units": [{"uid": i + 1, "type": t, "side": s, "x": x, "y": y, "is_leader": False, **extra}
                           for i, (t, s, x, y, extra) in enumerate(units)],
        "starting_sides": [{"side": k, "gold": 100, "recruit": list(recruits), "fog": True} for k in (1, 2)],
    }))


def _sees(cs, side, uid) -> bool:
    return uid in set(cs.core.visible_ids(side))


def _seen(cs, side) -> set:
    """The hexes `side` sees."""
    keys = cs.geometry().keys
    return {keys[j] for j in np.flatnonzero(cs.core.seen_export(side))}


def test_terrain_the_unit_cannot_cross_limits_what_it_sees():
    """A Spearman (5 MP) at x=1 facing mountains (3 MP each): it reaches
    x=3 and sees the ring hex x=4, not x=5 and x=6, which its old disc
    covered. The enemy at x=6 stays hidden."""
    cs = _game(["Gg Gg Gg Mm Mm Mm Mm Mm Mm Mm"],
               [("Spearman", 1, 1, 0, {}), ("Spearman", 2, 6, 0, {})])
    seen = _seen(cs, 1)
    assert (4, 0) in seen
    assert (5, 0) not in seen and (6, 0) not in seen
    assert not _sees(cs, 1, "u2")


def test_a_unit_sees_one_hex_beyond_what_it_could_reach():
    """On open ground a 5-MP unit reaches 5 hexes and sees the sixth."""
    cs = _game(["Gg " * 12], [("Spearman", 1, 1, 0, {}), ("Spearman", 2, 7, 0, {})])
    seen = _seen(cs, 1)
    assert (7, 0) in seen and (8, 0) not in seen
    assert _sees(cs, 1, "u2")


def test_what_a_side_clears_during_its_turn_stays_clear_until_the_turn_ends():
    """A Cavalryman (8 MP) at x=5 sees the enemy at x=13 at turn start,
    then rides back to x=2, from where it sees only up to x=11. The side
    keeps seeing the enemy until its turn ends, then loses it."""
    cs = _game(["Gg " * 20], [("Spearman", 1, 0, 0, {}), ("Cavalryman", 1, 5, 0, {"max_moves": 8}),
                              ("Spearman", 2, 13, 0, {}), ("Spearman", 2, 19, 0, {})])
    cs.apply_command(["init_side", 1])
    assert _sees(cs, 1, "u3")
    cs.apply_command(["move", [5, 4, 3, 2], [0, 0, 0, 0], 1])
    cavalryman = cs.core.unit_export("u2")
    assert (cavalryman["x"], cavalryman["y"]) == (2, 0)
    refogged = cs.fork()
    refogged.core.refog_side(1)
    assert (13, 0) not in _seen(refogged, 1), "from x=2 the side's units do not see x=13"
    assert _sees(cs, 1, "u3"), "cleared at turn start, still clear after the move"
    cs.apply_command(["end_turn"])
    assert not _sees(cs, 1, "u3"), "refogged at the end of the turn"


def test_a_recruit_clears_fog_around_its_hex():
    """A Cavalryman recruited beside the keep sees farther than the
    leader: the enemy at x=9 appears the moment it is recruited."""
    cs = _game(["Ke Ce Gg Gg Gg Gg Gg Gg Gg Gg Gg Gg"],
               [("Spearman", 1, 0, 0, {"is_leader": True}), ("Spearman", 2, 9, 0, {})],
               recruits=("Cavalryman",))
    cs.apply_command(["init_side", 1])
    assert not _sees(cs, 1, "u2")
    cs.apply_command(["recruit", "Cavalryman", 1, 0, ""])
    assert _sees(cs, 1, "u2")


def test_the_defenders_side_refogs_when_its_unit_dies():
    """Side 2's Cavalryman at x=10 sees x=19; killed by side 1, it takes
    that view with it (attack.cpp:1456-1458)."""
    cs = _game(["Gg " * 22], [("Spearman", 1, 9, 0, {}), ("Cavalryman", 2, 10, 0, {"hp": 1, "max_moves": 8}),
                              ("Spearman", 2, 0, 0, {})])
    cs.apply_command(["init_side", 1])
    assert (19, 0) in _seen(cs, 2)
    cs.apply_command(["attack", 9, 0, 10, 0, 0, 0, "00000007"])
    assert "u2" not in set(cs.core.unit_ids()), "the seed must give a kill; pick another"
    assert (19, 0) not in _seen(cs, 2)


def test_the_types_with_their_own_vision_are_the_listed_ones():
    """`OWN_VISION_TYPES` is every unit type whose cfg declares `vision=`
    or `[vision_costs]`."""
    units_dir = ROOT / "wesnoth_src" / "data" / "core" / "units"
    if not units_dir.is_dir():
        pytest.skip("wesnoth_src/data/core/units is not in this checkout")
    declared = set()
    for cfg in units_dir.rglob("*.cfg"):
        text = cfg.read_text(encoding="utf-8", errors="replace")
        for block in re.findall(r"\[unit_type\](.*?)\[/unit_type\]", text, flags=re.S):
            type_id = re.search(r"^\s*id=(.+)$", block, flags=re.M)
            if type_id and (re.search(r"^\s*vision=", block, flags=re.M) or "[vision_costs]" in block):
                declared.add(type_id.group(1).strip())
    assert declared == visibility.OWN_VISION_TYPES


def test_a_revealed_hider_hides_again_when_its_side_starts_a_turn_after_turn_1():
    """unit::new_turn clears STATE_UNCOVERED (unit.cpp:1277) inside
    board_.new_turn's `turn() > 1` gate. A Ranger revealed on turn 1
    stays revealed through its own turn-1 start and side 1's turn 2,
    then hides at its turn-2 start: side 1's Spearman, which sees its
    forest hex, no longer sees it on turn 3."""
    cs = _game(["Gs^Fp " * 10], [("Spearman", 1, 0, 0, {}), ("Elvish Ranger", 2, 3, 0, {})])
    cs.apply_command(["init_side", 1])
    assert not _sees(cs, 1, "u2"), "control: the Ranger is under cover"
    cs.core.set_uncovered(["u2"])                      # revealed during side 1's turn
    for cmd in (["end_turn"], ["init_side", 2]):
        cs.apply_command(cmd)
    assert "u2" in cs.core.uncovered_export(), "no re-hide on turn 1"
    for cmd in (["end_turn"], ["init_side", 1]):
        cs.apply_command(cmd)
    assert _sees(cs, 1, "u2"), "still revealed until its own side's turn starts"
    for cmd in (["end_turn"], ["init_side", 2], ["end_turn"], ["init_side", 1]):
        cs.apply_command(cmd)
    assert "u2" not in cs.core.uncovered_export()
    assert not _sees(cs, 1, "u2")
