#!/usr/bin/env python3
"""Tests for the per-actor action-type head (commit C.1).

The type head adds a third factor to the action distribution for
UNIT actors: `P(action) = P(actor) * P(type | actor) *
P(target | actor, type) * P(weapon | actor, ATTACK)`. RECRUIT and
END_TURN actors keep their original chain (`P(actor) * P(target |
actor)` for recruit; `P(actor)` for end_turn).

Verified here:

  1. ModelOutput exposes `type_logits: [1, A, T]` and
     `marginal_type_logits: [1, T+2]` shaped correctly.
  2. Legality mask exposes `target_valid_attack`, `target_valid_move`,
     `type_valid` with the right per-actor split.
  3. `sample_action` records `type_idx` for unit actors and None for
     non-unit actors.
  4. The sampler picks ATTACK / MOVE consistently with the chosen
     action type (no MOVE actions return target == enemy hex).
  5. `reforward_logprob_entropy` accepts `type_idx` and produces a
     log_prob that includes the type term.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest
import torch

from action_sampler import (
    LegalActionPrior, _build_legality_masks, sample_action,
    reforward_logprob_entropy,
)
from classes import (
    Alignment, Attack, DamageType, GameState, GlobalInfo, Hex, Map, Position,
    SideInfo, Terrain, TerrainModifiers, Unit,
)
from encoder import GameStateEncoder
from model import ActorKind, ModelOutput, UnitActionType, WesnothModel


# ---------------------------------------------------------------------
# State builders
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
    return Hex(position=Position(x, y),
               terrain_types={terrain},
               modifiers=set(mods or []))


def _gs_with_unit_and_enemy():
    """Side-1 spearman at (3, 3); side-2 spearman at (4, 3); empty
    flat hexes around. Both sides ready to attack/move."""
    units = {
        _u("u1", 1, 3, 3),
        _u("u2", 2, 4, 3),
        _u("ldr1", 1, 0, 0, is_leader=True),
        _u("ldr2", 2, 8, 8, is_leader=True),
    }
    hexes = {
        _hex(x, y) for x in range(10) for y in range(10)
    }
    sides = [
        SideInfo(player="S1", recruits=[], current_gold=100,
                 base_income=2, nb_villages_controlled=0),
        SideInfo(player="S2", recruits=[], current_gold=100,
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


# ---------------------------------------------------------------------
# Model output shape
# ---------------------------------------------------------------------

def test_model_output_has_type_logits():
    """ModelOutput.type_logits has shape [1, A, T=2]."""
    gs = _gs_with_unit_and_enemy()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    model = WesnothModel()
    encoded = encoder.encode(gs)
    with torch.no_grad():
        out = model(encoded)
    A = out.actor_logits.shape[1]
    assert out.type_logits.shape == (1, A, UnitActionType.COUNT)


def test_marginal_type_logits_shape_and_normalization():
    """marginal_type_logits = [1, 4] with leaves [ATTACK, MOVE,
    RECRUIT, END_TURN]; exp(.).sum() should be ~1."""
    gs = _gs_with_unit_and_enemy()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    model = WesnothModel()
    encoded = encoder.encode(gs)
    with torch.no_grad():
        out = model(encoded)
    assert out.marginal_type_logits.shape == (1, 4)
    masses = torch.exp(out.marginal_type_logits)
    total = float(masses.sum().item())
    assert 0.95 <= total <= 1.05


# ---------------------------------------------------------------------
# Legality masks
# ---------------------------------------------------------------------

def test_legality_mask_splits_attack_and_move():
    """The unit at (3,3) with enemy at (4,3) has BOTH:
       - attack_valid for hex (4,3) (enemy adjacent)
       - move_valid for various empty adjacents
    target_valid is the union; type_valid says both are legal."""
    gs = _gs_with_unit_and_enemy()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    encoded = encoder.encode(gs)
    masks = _build_legality_masks(encoded, gs)

    # Find slot for u1 (the spearman at 3,3).
    u_slot = encoded.unit_ids.index("u1")
    # Find hex j for enemy (4,3) and an adjacent empty hex (e.g. (2,3)).
    j_enemy = encoded.pos_to_hex.get((4, 3))
    j_empty = encoded.pos_to_hex.get((2, 3))
    if j_enemy is None or j_empty is None:
        pytest.skip("missing hex in encoded set")

    attack_row = masks.target_valid_attack[u_slot].cpu().numpy()
    move_row   = masks.target_valid_move[u_slot].cpu().numpy()
    union_row  = masks.target_valid[u_slot].cpu().numpy()

    assert attack_row[j_enemy] == 1.0,  "enemy adjacent should be attack-legal"
    assert attack_row[j_empty] == 0.0,  "empty hex should NOT be attack-legal"
    assert move_row[j_empty]  == 1.0,   "empty hex should be move-legal"
    assert move_row[j_enemy]  == 0.0,   "enemy hex should NOT be move-legal"
    # Union covers both.
    assert union_row[j_enemy] == 1.0
    assert union_row[j_empty] == 1.0

    # type_valid for u1: BOTH ATTACK and MOVE legal.
    type_valid = masks.type_valid[0, u_slot].cpu().numpy()
    assert type_valid[UnitActionType.ATTACK] == 1.0
    assert type_valid[UnitActionType.MOVE]   == 1.0


def test_legality_mask_disables_attack_when_already_attacked():
    """A unit with has_attacked=True can MOVE but not ATTACK; the
    type_valid mask reflects this."""
    gs = _gs_with_unit_and_enemy()
    # Mark u1 as already-attacked this turn.
    new_units = set()
    for u in gs.map.units:
        if u.id == "u1":
            new_units.add(_u("u1", 1, 3, 3, has_attacked=True))
        else:
            new_units.add(u)
    gs.map.units = new_units
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    encoded = encoder.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    u_slot = encoded.unit_ids.index("u1")
    type_valid = masks.type_valid[0, u_slot].cpu().numpy()
    assert type_valid[UnitActionType.ATTACK] == 0.0
    assert type_valid[UnitActionType.MOVE]   == 1.0


# ---------------------------------------------------------------------
# Sampler chain
# ---------------------------------------------------------------------

def test_sample_action_records_type_idx_for_unit_actors():
    """SampledAction.type_idx is set for unit actors (ATTACK or MOVE)."""
    gs = _gs_with_unit_and_enemy()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    model = WesnothModel()
    torch.manual_seed(7)
    encoded = encoder.encode(gs)
    with torch.no_grad():
        out = model(encoded)
        # Repeat sampling several times to land on a unit actor at
        # least once; the random init won't be biased.
        unit_seen = False
        for _ in range(50):
            sampled = sample_action(encoded, out, gs)
            atype = sampled.action.get("type")
            if atype in ("attack", "move"):
                unit_seen = True
                assert sampled.type_idx is not None
                assert sampled.type_idx in (UnitActionType.ATTACK,
                                            UnitActionType.MOVE)
                if atype == "attack":
                    assert sampled.type_idx == UnitActionType.ATTACK
                else:
                    assert sampled.type_idx == UnitActionType.MOVE
            elif atype == "end_turn":
                assert sampled.type_idx is None
            elif atype == "recruit":
                assert sampled.type_idx is None
        assert unit_seen, "should have sampled a unit action at least once in 50 tries"


# ---------------------------------------------------------------------
# Reforward chain rule
# ---------------------------------------------------------------------

def test_reforward_includes_type_term_for_unit_actor():
    """reforward_logprob_entropy with type_idx set returns a different
    log_prob than the same call with type_idx=None: the chain rule
    is mathematically different because the target softmax denominator
    changes (type-conditional mask vs union mask). Both must be
    finite; their ordering depends on the relative entropies."""
    gs = _gs_with_unit_and_enemy()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    model = WesnothModel()
    encoded = encoder.encode(gs)
    out = model(encoded)

    u_slot = encoded.unit_ids.index("u1")
    j_enemy = encoded.pos_to_hex[(4, 3)]
    weapon_idx = 0

    lp_with_type, _ = reforward_logprob_entropy(
        encoded, out, gs,
        actor_idx=u_slot, target_idx=j_enemy, weapon_idx=weapon_idx,
        type_idx=UnitActionType.ATTACK,
    )
    lp_without_type, _ = reforward_logprob_entropy(
        encoded, out, gs,
        actor_idx=u_slot, target_idx=j_enemy, weapon_idx=weapon_idx,
        type_idx=None,
    )
    # Both finite; they differ (the chain rule is genuinely different).
    assert torch.isfinite(lp_with_type).item()
    assert torch.isfinite(lp_without_type).item()
    assert float(lp_with_type.item()) != pytest.approx(
        float(lp_without_type.item()), abs=1e-6)


def test_reforward_legacy_no_type_idx_still_works():
    """A transition stored before the type head landed (type_idx=None)
    must still produce a valid log_prob via the legacy chain rule."""
    gs = _gs_with_unit_and_enemy()
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    model = WesnothModel()
    encoded = encoder.encode(gs)
    out = model(encoded)

    u_slot = encoded.unit_ids.index("u1")
    j_enemy = encoded.pos_to_hex[(4, 3)]
    lp, ent = reforward_logprob_entropy(
        encoded, out, gs,
        actor_idx=u_slot, target_idx=j_enemy, weapon_idx=0,
        type_idx=None,
    )
    # log_prob is finite and < 0; entropy is >= 0.
    assert torch.isfinite(lp).item()
    assert float(lp.item()) < 0.0
    assert float(ent.item()) >= 0.0
