#!/usr/bin/env python3
"""end_turn legality gate (user rule 2026-07-15).

end_turn is masked ILLEGAL while (a) some unit that is not the
leader-standing-on-a-keep still has a legal move, or (b) a recruit
is affordable+placeable. Motivated by watched decisive replays
where the policies mostly idled while a tentacle killed a leader.
Knob: constants.FORBID_IDLE_END_TURN.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from dataclasses import replace

from action_sampler import _build_legality_masks
from classes import (GameState, GlobalInfo, Hex, Map, Position,
                     SideInfo, TerrainModifiers, Terrain)
from encoder import GameStateEncoder
from test_action_type_head import _gs_with_unit_and_enemy, _u


def _masks_for(gs):
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    encoded = encoder.encode(gs)
    return encoded, _build_legality_masks(encoded, gs)


def _end_turn_valid(encoded, masks) -> bool:
    return bool(masks.actor_valid[0, -1].item() > 0)


def _zero_all_mp(gs, side):
    for u in list(gs.map.units):
        if u.side == side:
            gs.map.units.discard(u)
            gs.map.units.add(replace(u, current_moves=0))


def test_end_turn_masked_while_units_can_move():
    gs = _gs_with_unit_and_enemy()
    encoded, masks = _masks_for(gs)
    assert not _end_turn_valid(encoded, masks), \
        "units have MP and legal moves -> end_turn must be illegal"


def test_end_turn_legal_when_side_exhausted():
    gs = _gs_with_unit_and_enemy()
    _zero_all_mp(gs, 1)
    encoded, masks = _masks_for(gs)
    assert _end_turn_valid(encoded, masks), \
        "no MP anywhere, nothing to recruit -> end_turn legal"


def test_leader_on_keep_with_mp_does_not_block_end_turn():
    """The exception: a leader HOLDING a keep with MP left (and no
    affordable recruit, no other movable unit) may end the turn."""
    gs = _gs_with_unit_and_enemy()
    _zero_all_mp(gs, 1)
    # put the (full-MP) leader on a keep hex; empty recruit list ->
    # no recruit actor regardless of gold.
    ldr = next(u for u in gs.map.units if u.side == 1 and u.is_leader)
    gs.map.units.discard(ldr)
    gs.map.units.add(replace(ldr, current_moves=5))
    khex = next(h for h in gs.map.hexes
                if h.position.x == ldr.position.x
                and h.position.y == ldr.position.y)
    khex.modifiers.add(TerrainModifiers.KEEP)
    khex.terrain_types.add(Terrain.CASTLE)
    encoded, masks = _masks_for(gs)
    assert _end_turn_valid(encoded, masks), \
        "leader holding its keep must be allowed to end the turn"


def test_leader_off_keep_with_mp_blocks_end_turn():
    gs = _gs_with_unit_and_enemy()
    _zero_all_mp(gs, 1)
    ldr = next(u for u in gs.map.units if u.side == 1 and u.is_leader)
    gs.map.units.discard(ldr)
    gs.map.units.add(replace(ldr, current_moves=5))   # NOT on a keep
    encoded, masks = _masks_for(gs)
    assert not _end_turn_valid(encoded, masks), \
        "a movable leader off-keep counts like any other unit"
