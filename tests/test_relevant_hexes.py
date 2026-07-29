"""Contract tests for visibility.relevant_hex_positions /
relevant_hexes_in_slot_order (T2-B relevant-set encoder core).

Pins the three adopted requirements at the source-of-truth layer:
  1. SUPERSET of every mask-offerable target hex (checked against
     _build_legality_masks on real pool states).
  2. DETERMINISTIC, STABLE ORDERING across repeated calls on equal
     states (the trainer replays stored target indices).
  3. Degenerate states (no units / no leader) don't crash and still
     return the static components.
"""
import copy
import random

import pytest
import torch

from tools.scenario_pool import (random_setup, build_scenario_gamestate,
                                 load_factions)
from wesnoth_ai.visibility import (hexes_in_slot_order,
                                   relevant_hex_positions,
                                   relevant_hexes_in_slot_order)


@pytest.fixture(scope="module")
def pool_state():
    load_factions()
    setup = random_setup(random.Random(77), forced_faction=None)
    return build_scenario_gamestate(setup)


def test_superset_of_mask_targets(pool_state):
    """Every hex the legality mask offers must be in the set."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.action_sampler import _build_legality_masks
    gs = pool_state
    side = gs.global_info.current_side
    enc = GameStateEncoder()
    with torch.no_grad():
        encoded = enc.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    tv = masks.target_valid.numpy()
    offered = {(encoded.hex_positions[j].x, encoded.hex_positions[j].y)
               for j in range(tv.shape[1]) if tv[:, j].any()}
    rel = relevant_hex_positions(gs, side)
    missing = offered - rel
    assert not missing, f"mask offers hexes outside relevant set: {missing}"


def test_ordering_stable_and_row_major(pool_state):
    gs = pool_state
    a = relevant_hexes_in_slot_order(gs)
    b = relevant_hexes_in_slot_order(copy.deepcopy(gs))
    assert [(h.position.x, h.position.y) for h in a] \
        == [(h.position.x, h.position.y) for h in b]
    # Ordering is the row-major slot order, filtered (never re-sorted).
    full = [(h.position.x, h.position.y)
            for h in hexes_in_slot_order(gs)]
    keep = {(h.position.x, h.position.y) for h in a}
    assert [(x, y) for (x, y) in full if (x, y) in keep] \
        == [(h.position.x, h.position.y) for h in a]


def test_statics_always_included(pool_state):
    """Villages and castles/keeps are in the set regardless of reach."""
    from wesnoth_ai.classes import Terrain, TerrainModifiers
    gs = pool_state
    rel = relevant_hex_positions(gs, gs.global_info.current_side)
    for h in gs.map.hexes:
        p = (h.position.x, h.position.y)
        mods = h.modifiers or set()
        if (Terrain.VILLAGE in h.terrain_types
                or TerrainModifiers.CASTLE in mods
                or TerrainModifiers.KEEP in mods):
            assert p in rel, f"static hex {p} missing"


def test_degenerate_no_units(pool_state):
    """A unit-less board: no crash, statics still present, subset
    strictly smaller than the full board."""
    gs = copy.deepcopy(pool_state)
    gs.map.units = set()
    rel = relevant_hex_positions(gs, 1)
    assert rel, "statics should remain on an empty board"
    assert len(rel) < len(gs.map.hexes)
    assert len(relevant_hexes_in_slot_order(gs)) == len(
        {p for p in rel
         if p in {(h.position.x, h.position.y) for h in gs.map.hexes}})
