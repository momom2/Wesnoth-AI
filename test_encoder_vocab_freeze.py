"""Verify `GameStateEncoder.freeze_vocab()` makes subsequent
`register_names` calls inert + warn-once per new name.

Why this matters: post-pretrain the encoder's `unit_type_to_id`
mapping IS the embedding table layout. Adding an entry post-train
shifts every subsequent id, which silently breaks any saved
checkpoint that lifted the vocab off the wrong state. Freeze
makes the failure mode loud rather than silent.

Dependencies: encoder, classes.
Dependents: regression CI.
"""
from __future__ import annotations

import logging
from typing import List

import pytest

from classes import (
    Alignment, GameState, GlobalInfo, Hex, Map, Position, SideInfo, Unit,
)
from encoder import GameStateEncoder


def _u(uid: str, side: int, x: int, y: int, name: str = "Foo") -> Unit:
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=False, position=Position(x, y),
        max_hp=40, max_moves=5, max_exp=32, cost=14,
        alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=40, current_moves=5, current_exp=0,
        has_attacked=False, attacks=[],
        resistances=[1.0] * 6, defenses=[0.5] * 14,
        movement_costs=[1] * 14,
        abilities=set(), traits=set(), statuses=set(),
    )


def _gs(units: List[Unit]) -> GameState:
    return GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=set(), units=set(units)),
        global_info=GlobalInfo(
            current_side=1, turn_number=1, time_of_day="morning",
            village_gold=2, village_upkeep=1, base_income=2,
        ),
        sides=[
            SideInfo(player="p1", recruits=["Foo"],
                     current_gold=100, base_income=2,
                     nb_villages_controlled=0, faction="Drakes"),
            SideInfo(player="p2", recruits=["Bar"],
                     current_gold=100, base_income=2,
                     nb_villages_controlled=0, faction="Rebels"),
        ],
    )


def test_register_names_grows_vocab_when_not_frozen():
    enc = GameStateEncoder(d_model=32)
    gs = _gs([_u("u1", 1, 0, 0, name="Foo")])
    enc.register_names(gs)
    assert "Foo" in enc.unit_type_to_id
    # New types are added.
    gs2 = _gs([_u("u1", 1, 0, 0, name="Bar")])
    enc.register_names(gs2)
    assert "Bar" in enc.unit_type_to_id


def test_freeze_prevents_new_unit_type(caplog):
    enc = GameStateEncoder(d_model=32)
    gs = _gs([_u("u1", 1, 0, 0, name="Foo")])
    enc.register_names(gs)
    enc.freeze_vocab()
    before = dict(enc.unit_type_to_id)

    gs_new = _gs([_u("u2", 1, 0, 0, name="MisteryUnit")])
    with caplog.at_level(logging.WARNING, logger="encoder"):
        enc.register_names(gs_new)
    # Vocab unchanged.
    assert enc.unit_type_to_id == before
    # And we got a warning.
    assert any("frozen" in r.message and "MisteryUnit" in r.message
               for r in caplog.records), (
        f"expected a frozen-vocab warning for MisteryUnit; got "
        f"{[r.message for r in caplog.records]}"
    )


def test_freeze_warning_is_one_shot_per_name(caplog):
    """Repeated calls with the same unseen name fire the warning
    once, not every iteration -- otherwise a rollout encountering
    the unknown type would flood logs."""
    enc = GameStateEncoder(d_model=32)
    enc.register_names(_gs([_u("u1", 1, 0, 0, name="Foo")]))
    enc.freeze_vocab()

    gs_new = _gs([_u("u2", 1, 0, 0, name="Mystery")])
    with caplog.at_level(logging.WARNING, logger="encoder"):
        for _ in range(5):
            enc.register_names(gs_new)
    mystery_warnings = [
        r for r in caplog.records
        if "Mystery" in r.message and "frozen" in r.message
    ]
    assert len(mystery_warnings) == 1, (
        f"expected 1 warning, got {len(mystery_warnings)}: "
        f"{[r.message for r in mystery_warnings]}"
    )


def test_freeze_prevents_new_faction(caplog):
    enc = GameStateEncoder(d_model=32)
    enc.register_names(_gs([_u("u1", 1, 0, 0)]))
    enc.freeze_vocab()
    # Mutate the gs to have a new faction name.
    gs = _gs([_u("u2", 1, 0, 0)])
    gs.sides[0].faction = "InventedFaction"
    before = dict(enc.faction_to_id)
    with caplog.at_level(logging.WARNING, logger="encoder"):
        enc.register_names(gs)
    assert enc.faction_to_id == before
    assert any("InventedFaction" in r.message for r in caplog.records)
