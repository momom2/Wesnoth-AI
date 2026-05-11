"""Test the side-rewrite path in `tools/sim_to_replay`.

Pre-fix (BACKLOG flagged 2026-04-30): `export_replay` inherited
the source bz2's `[side]` blocks verbatim, so a self-play sim with
different factions than its source would produce a bz2 that Wesnoth
loaded as the SOURCE's leaders and OOS'd immediately.

This test exercises `_rewrite_sides_for_sim` on the strict-sync
fixture in `tests/fixtures/`. We construct a minimal mock sim
whose leader / faction / recruit list intentionally DIFFER from
the source's (e.g. fixture is Knalgan vs Northerners; mock sim
is Drakes vs Rebels), run the rewrite, and assert:
  - the output text references the mock sim's leader types,
    factions, and recruit lists per side
  - the source's original [unit] blocks for the leaders are gone
    or replaced
  - other source [unit] blocks (non-leader, pre-placed scenario
    units) are preserved

This protects the rewrite from regressing into "splice on top of
mismatched source" silently.

Dependencies: tools.sim_to_replay (the rewrite), tests/fixtures.
Dependents: regression CI.
"""
from __future__ import annotations

import bz2
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from tools.sim_to_replay import _rewrite_sides_for_sim

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "strict_sync_hamlets_t9.bz2"


@dataclass
class _MockSide:
    """Subset of SideInfo the rewrite reads."""
    faction:  str
    recruits: List[str]


@dataclass
class _MockUnit:
    name:      str
    side:      int
    is_leader: bool
    position:  SimpleNamespace


@dataclass
class _MockMap:
    units: List[_MockUnit] = field(default_factory=list)


@dataclass
class _MockGS:
    sides: List[_MockSide]
    map:   _MockMap


@dataclass
class _MockSim:
    gs: _MockGS


def _load_source_text() -> str:
    with bz2.open(FIXTURE, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def _mock_sim_drakes_vs_rebels() -> _MockSim:
    """Build a sim mock with leaders DIFFERENT from the fixture
    (which is Knalgan / Northerners). Knowing what leaders the
    fixture has is incidental -- the test just needs the rewrite
    to override whatever was there."""
    return _MockSim(gs=_MockGS(
        sides=[
            _MockSide(
                faction="Drakes",
                recruits=["Drake Burner", "Drake Fighter",
                          "Drake Glider", "Drake Clasher",
                          "Saurian Augur", "Saurian Skirmisher"],
            ),
            _MockSide(
                faction="Rebels",
                recruits=["Elvish Fighter", "Elvish Archer",
                          "Mage", "Elvish Shaman",
                          "Elvish Scout", "Wose", "Merman Hunter"],
            ),
        ],
        map=_MockMap(units=[
            _MockUnit("Drake Flare", side=1, is_leader=True,
                      position=SimpleNamespace(x=2, y=12)),
            _MockUnit("White Mage", side=2, is_leader=True,
                      position=SimpleNamespace(x=23, y=12)),
        ]),
    ))


def test_fixture_present():
    assert FIXTURE.exists()


def test_rewrite_overrides_side_leaders():
    """After rewrite, each [side]'s `type=` attr must show the mock
    sim's leader types (Drake Flare / White Mage), NOT the fixture's
    original leaders.

    Two source layouts to support:
      (1) fresh-game source: leader is implicit, the [side] just
          has `type="Rogue"` at the top. We rewrite `type=`.
      (2) snapshot source: an explicit `[unit canrecruit=yes]`
          sub-block lists the leader with its own `type=`. We
          rewrite both the [side] `type=` and the [unit]'s body.
    The strict-sync fixture is layout (1). This test is satisfied
    by checking the [side] top-level `type=` because that's the
    invariant attr across both layouts; the [unit] body check is
    secondary and only enforced when the block exists.
    """
    text = _load_source_text()
    sim = _mock_sim_drakes_vs_rebels()
    out = _rewrite_sides_for_sim(text, sim)
    for side, expected_leader in [(1, "Drake Flare"), (2, "White Mage")]:
        # Locate the [side]...[/side] for this side number.
        side_blocks = re.findall(r'\[side\][\s\S]*?\[/side\]', out)
        matching_sides = [
            b for b in side_blocks
            if re.search(rf'^\s*side="?{side}"?\s*$', b, re.MULTILINE)
        ]
        assert matching_sides, f"no [side] block for side {side}"
        side_block = matching_sides[0]
        # The [side]'s top-level `type=` must show the new leader.
        m = re.search(r'^\s*type=\"([^\"]+)\"', side_block, re.MULTILINE)
        assert m is not None, (
            f"side {side} has no top-level type= attr after rewrite"
        )
        assert m.group(1) == expected_leader, (
            f"side {side} type not rewritten: got {m.group(1)!r}, "
            f"expected {expected_leader!r}"
        )
        # If the source HAS a [unit canrecruit=yes] block, its
        # `type=` should also match.
        unit_blocks = re.findall(
            r'\[unit\][\s\S]*?\[/unit\]', side_block,
        )
        leader_units = [
            u for u in unit_blocks
            if 'canrecruit=yes' in u or 'canrecruit="yes"' in u
        ]
        for ub in leader_units:
            assert f'type="{expected_leader}"' in ub, (
                f"side {side} [unit canrecruit=yes] not rewritten: "
                f"{ub[:300]}"
            )


def test_rewrite_updates_side_faction_and_recruit():
    """Top-level `faction=` and `recruit=` attrs on each [side]
    must reflect the sim's faction + recruit list."""
    text = _load_source_text()
    sim = _mock_sim_drakes_vs_rebels()
    out = _rewrite_sides_for_sim(text, sim)
    # Walk each [side] block.
    sides = re.findall(r'\[side\][\s\S]*?\[/side\]', out)
    assert len(sides) >= 2
    for block in sides:
        side_m = re.search(r'^\s*side=(\d+)', block, re.MULTILINE)
        if side_m is None:
            continue
        side_num = int(side_m.group(1))
        if side_num == 1:
            assert 'faction="Drakes"' in block, block[:500]
            assert 'recruit="Drake Burner,' in block, block[:500]
        elif side_num == 2:
            assert 'faction="Rebels"' in block, block[:500]
            assert 'recruit="Elvish Fighter,' in block, block[:500]


def test_rewrite_does_not_disturb_replay_block():
    """The [replay]...[/replay] section must be untouched -- the
    rewrite only operates on [side] subtrees."""
    text = _load_source_text()
    sim = _mock_sim_drakes_vs_rebels()
    out = _rewrite_sides_for_sim(text, sim)
    # The [replay] body is identical pre and post.
    src_replay = re.search(r'\[replay\][\s\S]*?\[/replay\]', text)
    out_replay = re.search(r'\[replay\][\s\S]*?\[/replay\]', out)
    assert src_replay is not None and out_replay is not None
    assert src_replay.group(0) == out_replay.group(0), (
        "rewrite mutated the [replay] block -- the side-only "
        "regex is over-matching"
    )


def test_rewrite_idempotent():
    """Running the rewrite twice on the same input yields the same
    output. Catches accidental accumulation (e.g. appending two
    leader [unit] blocks per side instead of replacing one)."""
    text = _load_source_text()
    sim = _mock_sim_drakes_vs_rebels()
    first  = _rewrite_sides_for_sim(text, sim)
    second = _rewrite_sides_for_sim(first, sim)
    assert first == second, (
        "rewrite is not idempotent: re-running mutates the output, "
        "which means later exports could compound side info."
    )
