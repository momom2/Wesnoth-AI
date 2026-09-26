"""encode_raw's static per-map hex cache (wesnoth_ai/encoder.py, plan
1.2): the encoding is a function of the state, not of cache warmth,
ownership and fog bits still track the state, and a replaced hex set
invalidates the cache."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _state(seed=3):
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    return build_scenario_gamestate(random_setup(random.Random(seed)))


def _encode(gs):
    from wesnoth_ai.encoder import encode_raw
    return encode_raw(gs, type_to_id={}, faction_to_id={}, relevant_set=False)


def _same(a, b):
    import dataclasses
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb), f.name
        else:
            assert va == vb, f.name


def test_warm_cache_matches_cold_and_is_hit():
    from wesnoth_ai import encoder
    gs = _state()
    encoder._STATIC_HEX_CACHE.clear()
    cold = _encode(gs)
    assert len(encoder._STATIC_HEX_CACHE) == 1
    warm = _encode(copy.deepcopy(gs))         # forks alias the hex set
    assert len(encoder._STATIC_HEX_CACHE) == 1
    _same(cold, warm)


def test_ownership_and_rejection_bits_follow_the_state():
    from wesnoth_ai.sim.classes import Terrain
    gs = _state()
    side = gs.global_info.current_side
    villages = [h for h in gs.map.hexes if Terrain.VILLAGE in h.terrain_types]
    assert len(villages) >= 2
    v_own, v_enemy = villages[0].position, villages[1].position
    gs.global_info._village_owner = {(v_own.x, v_own.y): side, (v_enemy.x, v_enemy.y): 3 - side}
    gs.global_info._fog = False
    gs.global_info._recruit_rejected_hexes = {(v_enemy.x, v_enemy.y)}
    raw = _encode(gs)
    idx = {p: i for i, p in enumerate(raw.hex_positions)}
    own, enemy = idx[v_own], idx[v_enemy]
    assert raw.hex_modifier_flags[own, 0] == 1.0 and raw.hex_dynamic_flags[own, 1] == 1.0
    assert raw.hex_modifier_flags[enemy, 0] == 1.0 and raw.hex_dynamic_flags[enemy, 2] == 1.0
    assert raw.hex_dynamic_flags[enemy, 0] == 1.0          # recruit-rejected this turn
    assert raw.hex_dynamic_flags[own, 0] == 0.0
    # Same map, ownership gone: the cached static arrays must not leak it.
    gs2 = copy.deepcopy(gs)
    gs2.global_info._village_owner = {}
    gs2.global_info._recruit_rejected_hexes = set()
    raw2 = _encode(gs2)
    assert raw2.hex_modifier_flags[own, 0] == 0.0 and raw2.hex_dynamic_flags[own].sum() == 0.0
    assert raw2.hex_dynamic_flags[enemy].sum() == 0.0


def test_fog_hides_enemy_ownership_outside_vision():
    from wesnoth_ai.sim.classes import Terrain
    gs = _state()
    side = gs.global_info.current_side
    # The village farthest from every own unit is on a hex the side does not see.
    own_units = [u for u in gs.map.units if u.side == side]
    villages = [h for h in gs.map.hexes if Terrain.VILLAGE in h.terrain_types]
    far = max(villages, key=lambda h: min(abs(h.position.x - u.position.x)
                                          + abs(h.position.y - u.position.y) for u in own_units))
    gs.global_info._village_owner = {(far.position.x, far.position.y): 3 - side}
    gs.global_info._fog = True
    raw = _encode(gs)
    i = {p: k for k, p in enumerate(raw.hex_positions)}[far.position]
    assert raw.hex_modifier_flags[i, 0] == 0.0 and raw.hex_dynamic_flags[i, 2] == 0.0
    gs.global_info._fog = False
    raw = _encode(gs)
    assert raw.hex_modifier_flags[i, 0] == 1.0 and raw.hex_dynamic_flags[i, 2] == 1.0


def test_replaced_hex_set_invalidates_cache():
    from wesnoth_ai import encoder
    from wesnoth_ai.sim.classes import Terrain
    encoder._STATIC_HEX_CACHE.clear()
    gs = _state()
    before = _encode(gs)
    new_hexes = {copy.deepcopy(h) for h in gs.map.hexes}
    target = next(h for h in new_hexes if Terrain.VILLAGE not in h.terrain_types
                  and Terrain.CASTLE not in h.terrain_types)
    target.terrain_types = {Terrain.CASTLE}
    gs.map.hexes = new_hexes                  # how terrain-morph events publish
    after = _encode(gs)
    i = {p: k for k, p in enumerate(after.hex_positions)}[target.position]
    assert after.hex_terrain_ids[i] == Terrain.CASTLE.value
    assert before.hex_terrain_ids[i] != after.hex_terrain_ids[i]
    assert len(encoder._STATIC_HEX_CACHE) == 2


def test_cache_entry_holds_its_hex_set():
    """The entry keeps a strong reference to the set it was built
    from, so its id cannot be recycled by another map; a hit is an
    identity match."""
    from wesnoth_ai import encoder
    encoder._STATIC_HEX_CACHE.clear()
    gs = _state()
    _encode(gs)
    entry = encoder._STATIC_HEX_CACHE[id(gs.map.hexes)]
    assert entry.hex_set is gs.map.hexes
    gs2 = _state(seed=4)                         # another map, same process
    _encode(gs2)
    assert encoder._STATIC_HEX_CACHE[id(gs2.map.hexes)].hex_set is gs2.map.hexes
    assert entry.hex_set is not gs2.map.hexes
