#!/usr/bin/env python3
"""Tests for `classes.state_key` -- the canonical content hash MCTS
uses to dedupe transposition-table entries.

We're checking two contracts at once:

  1. **Discrimination**: a state with a meaningful difference (one
     unit's HP / position / MP / status / XP / has_attacked, side
     gold, village ownership, turn number, RNG counter) MUST hash
     to a different key. If state_key collides on any of these, two
     positions MCTS treats as identical actually aren't, and visit
     counts / value backups corrupt each other.

  2. **Order independence**: the same state encoded with units in a
     different iteration order MUST hash to the SAME key. `set`
     iteration is hash-randomized; if state_key were sensitive to
     that, two semantically-equal states could miss the
     transposition match and waste search.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from classes import (
    Alignment, GameState, GlobalInfo, Map, Position, SideInfo,
    Unit, state_key,
)


def _u(uid: str, side: int, x: int, y: int, *,
       hp=40, mp=5, xp=0, statuses=None, has_attacked=False,
       is_leader=False, name="Spearman") -> Unit:
    """Minimal Unit with default fields filled."""
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=40, max_moves=5, max_exp=32,
        cost=14, alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=hp, current_moves=mp, current_exp=xp,
        has_attacked=has_attacked,
        attacks=[], resistances=[1.0]*6, defenses=[0.5]*14,
        movement_costs=[1]*14, abilities=set(), traits=set(),
        statuses=set(statuses or set()),
    )


def _gs(units, *, current_side=1, turn=1, side_gold=(100, 100),
        side_villages=(0, 0)) -> GameState:
    sides = [
        SideInfo(player=f"Side {i+1}", recruits=[],
                 current_gold=side_gold[i], base_income=2,
                 nb_villages_controlled=side_villages[i])
        for i in range(2)
    ]
    return GameState(
        game_id="t",
        map=Map(size_x=20, size_y=20, mask=set(), fog=set(),
                hexes=set(), units=set(units)),
        global_info=GlobalInfo(
            current_side=current_side, turn_number=turn,
            time_of_day="day", village_gold=2,
            village_upkeep=1, base_income=2),
        sides=sides,
    )


# ---------------------------------------------------------------------
# Discrimination: any meaningful change -> different key
# ---------------------------------------------------------------------

def test_state_key_changes_with_unit_position():
    a = _gs([_u("u1", 1, 5, 5)])
    b = _gs([_u("u1", 1, 6, 5)])
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_hp():
    a = _gs([_u("u1", 1, 5, 5, hp=40)])
    b = _gs([_u("u1", 1, 5, 5, hp=39)])
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_mp():
    a = _gs([_u("u1", 1, 5, 5, mp=5)])
    b = _gs([_u("u1", 1, 5, 5, mp=4)])
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_xp():
    a = _gs([_u("u1", 1, 5, 5, xp=10)])
    b = _gs([_u("u1", 1, 5, 5, xp=11)])
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_has_attacked():
    a = _gs([_u("u1", 1, 5, 5, has_attacked=False)])
    b = _gs([_u("u1", 1, 5, 5, has_attacked=True)])
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_status_set():
    a = _gs([_u("u1", 1, 5, 5, statuses=set())])
    b = _gs([_u("u1", 1, 5, 5, statuses={"slowed"})])
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_side_gold():
    a = _gs([_u("u1", 1, 5, 5)], side_gold=(100, 100))
    b = _gs([_u("u1", 1, 5, 5)], side_gold=(95, 100))
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_villages_controlled():
    a = _gs([_u("u1", 1, 5, 5)], side_villages=(0, 0))
    b = _gs([_u("u1", 1, 5, 5)], side_villages=(1, 0))
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_turn_number():
    a = _gs([_u("u1", 1, 5, 5)], turn=5)
    b = _gs([_u("u1", 1, 5, 5)], turn=6)
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_current_side():
    a = _gs([_u("u1", 1, 5, 5)], current_side=1)
    b = _gs([_u("u1", 1, 5, 5)], current_side=2)
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_village_owner_dict():
    """gs.global_info._village_owner is a stash attr the sim keeps
    on side-changes; state_key must include it so two states with
    same unit positions but different village owners differ."""
    a = _gs([_u("u1", 1, 5, 5)], side_villages=(1, 0))
    b = _gs([_u("u1", 1, 5, 5)], side_villages=(1, 0))
    setattr(a.global_info, "_village_owner", {(5, 5): 1})
    setattr(b.global_info, "_village_owner", {(5, 5): 2})
    assert state_key(a) != state_key(b)


def test_state_key_changes_with_rng_counter():
    """Two states with the same unit layout but different
    `_rng_request_counter` are NOT the same MCTS node -- the next
    RNG-using command in each will produce a different seed."""
    a = _gs([_u("u1", 1, 5, 5)])
    b = _gs([_u("u1", 1, 5, 5)])
    setattr(a.global_info, "_rng_request_counter", 5)
    setattr(b.global_info, "_rng_request_counter", 7)
    assert state_key(a) != state_key(b)


# ---------------------------------------------------------------------
# Order-independence: same content, different iteration order -> same key
# ---------------------------------------------------------------------

def test_state_key_order_invariant_2units():
    """Two units, swapped iteration order: keys must match."""
    u1 = _u("u1", 1, 5, 5)
    u2 = _u("u2", 2, 6, 5)
    a = _gs([u1, u2])
    b = _gs([u2, u1])
    assert state_key(a) == state_key(b)


def test_state_key_order_invariant_many_units():
    """Six units in two different orderings: keys must match."""
    units_fwd = [
        _u("u1", 1, 1, 1), _u("u2", 1, 2, 2), _u("u3", 1, 3, 3),
        _u("u4", 2, 4, 4), _u("u5", 2, 5, 5), _u("u6", 2, 6, 6),
    ]
    units_rev = list(reversed(units_fwd))
    a = _gs(units_fwd)
    b = _gs(units_rev)
    assert state_key(a) == state_key(b)


def test_state_key_unaffected_by_set_iteration_seed():
    """The sim builds a fresh `set(units)` from a list; iteration
    order then depends on Python hash randomization. Build the same
    state via two independent set constructions and confirm the
    keys match."""
    units_a = {_u("u1", 1, 1, 1), _u("u2", 2, 2, 2),
               _u("u3", 1, 3, 3), _u("u4", 2, 4, 4)}
    units_b = set(units_a)   # rebuilt via shallow copy; same elems
    a = _gs(list(units_a))
    b = _gs(list(units_b))
    assert state_key(a) == state_key(b)


# ---------------------------------------------------------------------
# Stability: same input -> same key (deterministic)
# ---------------------------------------------------------------------

def test_state_key_deterministic_across_calls():
    """Two calls on the same state must return the exact same int.
    state_key is built with `hash(...)` which uses Python's per-
    interpreter hash seed; within one process the hash is stable."""
    units = [_u("u1", 1, 5, 5), _u("u2", 2, 6, 5)]
    state = _gs(units)
    k1 = state_key(state)
    k2 = state_key(state)
    assert k1 == k2
