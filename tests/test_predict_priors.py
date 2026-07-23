#!/usr/bin/env python3
"""Tests for `action_sampler.predict_priors` — the legality-masked,
normalized distribution API that MCTS / PUCT consumes.

`predict_priors` mirrors `sample_action`'s decision chain
(actor -> type -> target -> weapon) but exposes the *distributions*
rather than sampling one. Verified here:

  1. Output shapes match the documented contract.
  2. Each conditional sums to 1 over its legal support; degenerate
     conditionals (no legal options for a slice) sum to 0, never NaN.
  3. Probability mass on illegal slots is exactly 0.
  4. End_turn always has non-zero actor probability.
  5. Non-UNIT actors get zero type_/target_attack/target_move mass;
     non-RECRUIT actors get zero target_recruit mass.
  6. Passing in pre-built masks reproduces the no-args result exactly
     (the cache-handoff used by MCTS doesn't introduce drift).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import torch

from wesnoth_ai.action_sampler import (
    _build_legality_masks, predict_priors,
)
from wesnoth_ai.classes import (
    Alignment, Attack, DamageType, GameState, GlobalInfo, Hex, Map, Position,
    SideInfo, Terrain, TerrainModifiers, Unit,
)
from wesnoth_ai.encoder import GameStateEncoder
from wesnoth_ai.model import ActorKind, UnitActionType, WesnothModel


# ---------------------------------------------------------------------
# Builders (kept in sync with test_action_type_head.py)
# ---------------------------------------------------------------------

def _u(uid, side, x, y, *, hp=40, mp=5, max_hp=40, max_mp=5,
       has_attacked=False, is_leader=False, name="Spearman",
       attacks=None):
    if attacks is None:
        attacks = [Attack(type_id=DamageType.PIERCE, number_strikes=3,
                          damage_per_strike=7, is_ranged=False,
                          weapon_specials=set())]
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=max_hp, max_moves=max_mp, max_exp=32, cost=14,
        alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=hp, current_moves=mp, current_exp=0,
        has_attacked=has_attacked,
        attacks=attacks, resistances=[1.0]*6, defenses=[0.5]*14,
        movement_costs=[1]*14, abilities=set(), traits=set(),
        statuses=set(),
    )


def _hex(x, y, terrain=Terrain.FLAT, mods=None):
    return Hex(position=Position(x, y), terrain_types={terrain},
               modifiers=set(mods or []))


def _gs():
    """Side-1 spearman + leader, side-2 spearman + leader, 10x10 flat
    map, both sides have gold + recruits available so we exercise
    the RECRUIT path too."""
    units = {
        _u("u1", 1, 3, 3),
        _u("u2", 2, 4, 3),
        _u("ldr1", 1, 0, 0, is_leader=True),
        _u("ldr2", 2, 8, 8, is_leader=True),
    }
    # Castle around ldr1's keep so recruit hexes exist.
    hexes = set()
    for x in range(10):
        for y in range(10):
            if (x, y) == (0, 0):
                hexes.add(_hex(x, y, Terrain.CASTLE,
                               mods=[TerrainModifiers.KEEP]))   # the keep
            elif (x, y) in {(1, 0), (0, 1), (1, 1)}:
                hexes.add(_hex(x, y, Terrain.CASTLE,
                               mods=[TerrainModifiers.CASTLE])) # castle ring
            else:
                hexes.add(_hex(x, y))
    sides = [
        SideInfo(player="S1", recruits=["Spearman"], current_gold=100,
                 base_income=2, nb_villages_controlled=0),
        SideInfo(player="S2", recruits=["Spearman"], current_gold=100,
                 base_income=2, nb_villages_controlled=0),
    ]
    return GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=hexes, units=units),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=sides,
    )


def _build():
    gs = _gs()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    model = WesnothModel()
    encoded = encoder.encode(gs)
    with torch.no_grad():
        out = model(encoded)
    return gs, encoded, out


# ---------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------

def test_predict_priors_shapes():
    gs, encoded, out = _build()
    A = out.actor_logits.shape[1]
    H = out.target_logits.shape[2]
    T = UnitActionType.COUNT
    W = out.weapon_logits.shape[2]

    priors = predict_priors(out, encoded, gs)

    assert priors.actor.shape          == (1, A)
    assert priors.type_.shape          == (1, A, T)
    assert priors.target_attack.shape  == (1, A, H)
    assert priors.target_move.shape    == (1, A, H)
    assert priors.target_recruit.shape == (1, A, H)
    assert priors.weapon.shape         == (1, A, W)
    assert priors.value.shape          == (1, 1)


# ---------------------------------------------------------------------
# Normalization invariants
# ---------------------------------------------------------------------

def test_actor_distribution_sums_to_one():
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    s = float(priors.actor.sum().item())
    assert abs(s - 1.0) < 1e-5, f"actor sum = {s}"


def test_type_distribution_sums_to_one_or_zero():
    """For UNIT actors with at least one legal type, P(type | actor)
    sums to 1. For non-UNIT actors (RECRUIT, END_TURN), no type is
    legal -- the row sums to 0 (all-(-inf) -> safe_softmax = 0)."""
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    A = priors.type_.shape[1]
    actor_kind = out.actor_kind[0]
    for a in range(A):
        row = priors.type_[0, a]
        s = float(row.sum().item())
        kind = int(actor_kind[a].item())
        if kind == ActorKind.UNIT:
            # u1 is a side-1 spearman with mp=5 and an enemy adjacent;
            # both ATTACK and MOVE should be legal -> sum = 1.
            # Other UNIT actors are leaders / enemies; could differ.
            assert abs(s - 1.0) < 1e-5 or abs(s) < 1e-5, (
                f"type row a={a} sum={s}")
        else:
            assert abs(s) < 1e-5, f"non-UNIT type row a={a} sum={s}"


def test_target_attack_sums_to_one_for_attacking_unit():
    """u1 has one enemy adjacent (u2 at (4,3)); P(target | u1, ATTACK)
    is concentrated on u2's hex: sum=1 over the row."""
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    u1_slot = encoded.unit_ids.index("u1")
    s = float(priors.target_attack[0, u1_slot].sum().item())
    assert abs(s - 1.0) < 1e-5, f"u1 attack-target sum={s}"


def test_target_move_sums_to_one_for_mobile_unit():
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    u1_slot = encoded.unit_ids.index("u1")
    s = float(priors.target_move[0, u1_slot].sum().item())
    assert abs(s - 1.0) < 1e-5, f"u1 move-target sum={s}"


def test_target_recruit_sums_to_one_for_recruit_actor():
    """The single recruit slot (Spearman) on side 1 has at least one
    legal castle hex around the leader's keep -> sum=1."""
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    U = out.num_units
    R = out.num_recruits
    assert R >= 1, "fixture should have at least one recruit slot"
    # First recruit slot for the current side. Find one whose row
    # has any mass -- side-2 recruit slots have zero mass.
    found = False
    for r in range(R):
        a = U + r
        s = float(priors.target_recruit[0, a].sum().item())
        if s > 1e-5:
            assert abs(s - 1.0) < 1e-5, (
                f"recruit slot a={a} sum={s} (expected 0 or 1)")
            found = True
    assert found, "no recruit row had legal mass"


# ---------------------------------------------------------------------
# Illegal-action mass = 0
# ---------------------------------------------------------------------

def test_illegal_actor_has_zero_mass():
    """Side-2 units that are visible to side-1 still appear in the
    encoder's unit_ids -- and the actor mask should zero them out
    because it isn't their turn.

    Per the fog-of-war contract (visibility.py + encoder.py), enemy
    units OUTSIDE side-1's sight discs are filtered out of unit_ids
    entirely, so they never appear as slots to mask. That's a
    stronger guarantee than the actor mask alone, so we only assert
    here on the visible enemy `u2` (at (4,3) adjacent to u1). ldr2
    at (8,8) is fog-hidden from side 1 and absent from unit_ids by
    construction -- no slot to check, no failure mode to test.
    """
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    u2_slot = encoded.unit_ids.index("u2")
    assert float(priors.actor[0, u2_slot].item()) < 1e-9
    # Fog-hidden enemies have no slot at all: stronger contract.
    assert "ldr2" not in encoded.unit_ids


def test_illegal_attack_target_has_zero_mass():
    """An empty hex is not a valid attack target -- mass on it is 0."""
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    u1_slot = encoded.unit_ids.index("u1")
    j_empty = encoded.pos_to_hex[(2, 3)]
    p = float(priors.target_attack[0, u1_slot, j_empty].item())
    assert p < 1e-9


def test_illegal_move_target_has_zero_mass():
    """The enemy hex is not a valid MOVE target -- mass on it is 0."""
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    u1_slot = encoded.unit_ids.index("u1")
    j_enemy = encoded.pos_to_hex[(4, 3)]
    p = float(priors.target_move[0, u1_slot, j_enemy].item())
    assert p < 1e-9


# ---------------------------------------------------------------------
# End-turn always has positive mass
# ---------------------------------------------------------------------

def test_end_turn_prior_follows_idle_gate():
    # Gate default flipped to False 2026-07-18; this test exercises
    # the gate MECHANISM, so pin it ON for the test's scope.
    from wesnoth_ai import constants
    _orig_gate = constants.FORBID_IDLE_END_TURN
    constants.FORBID_IDLE_END_TURN = True
    try:
        """Old contract: end_turn prior always positive. Superseded by
        FORBID_IDLE_END_TURN (user rule 2026-07-15): while units still
        have legal moves the end_turn prior is exactly 0; once the side
        is exhausted it must be positive again."""
        gs, encoded, out = _build()
        priors = predict_priors(out, encoded, gs)
        A = priors.actor.shape[1]
        assert float(priors.actor[0, A - 1].item()) == 0.0
        from dataclasses import replace as _replace
        for u in list(gs.map.units):
            if u.side == gs.global_info.current_side:
                gs.map.units.discard(u)
                gs.map.units.add(_replace(u, current_moves=0))
        # ... and drain the gold: with the leader on its keep and 100
        # gold, condition (b) (affordable recruit) keeps the gate on.
        si = gs.sides[gs.global_info.current_side - 1]
        gs.sides[gs.global_info.current_side - 1] = _replace(
            si, current_gold=0)
        from wesnoth_ai.encoder import GameStateEncoder
        enc = GameStateEncoder()
        enc.register_names(gs)
        encoded2 = enc.encode(gs)
        import torch as _torch
        from wesnoth_ai.model import WesnothModel
        model = WesnothModel()
        with _torch.no_grad():
            out2 = model(encoded2)
        priors2 = predict_priors(out2, encoded2, gs)
        A2 = priors2.actor.shape[1]
        assert float(priors2.actor[0, A2 - 1].item()) > 0.0


    # ---------------------------------------------------------------------
    # Caching contract: passed-in masks == freshly built
    # ---------------------------------------------------------------------
    finally:
        constants.FORBID_IDLE_END_TURN = _orig_gate

def test_passing_masks_matches_no_args():
    gs, encoded, out = _build()
    p1 = predict_priors(out, encoded, gs)
    masks = _build_legality_masks(encoded, gs)
    p2 = predict_priors(out, encoded, gs, masks=masks)
    assert torch.allclose(p1.actor,          p2.actor)
    assert torch.allclose(p1.type_,          p2.type_)
    assert torch.allclose(p1.target_attack,  p2.target_attack)
    assert torch.allclose(p1.target_move,    p2.target_move)
    assert torch.allclose(p1.target_recruit, p2.target_recruit)
    assert torch.allclose(p1.weapon,         p2.weapon)


# ---------------------------------------------------------------------
# No NaN anywhere
# ---------------------------------------------------------------------

def test_no_nan_in_any_distribution():
    gs, encoded, out = _build()
    priors = predict_priors(out, encoded, gs)
    for name in ("actor", "type_", "target_attack",
                 "target_move", "target_recruit", "weapon"):
        t = getattr(priors, name)
        assert not torch.isnan(t).any(), f"{name} has NaN"
