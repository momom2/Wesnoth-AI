"""The observation kernel (wesnoth_ai/observe.py, rust/wesnoth_core/src/
observe.rs) against the Python originals it replaces, on harvested
states: the seen hexes it was given, unit visibility, the reach-context
flags and the recruit row must be identical for both sides, with fog on
and off and with rejected recruit hexes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai import observe as obs_mod  # noqa: E402

pytestmark = pytest.mark.skipif(obs_mod.kernel() is None,
                                reason="wesnoth_core.observe_side not available")


def _states():
    from helpers.played_states import _states as harvest
    return harvest(n_attack=3, n_plain=4)


def _python_reach_sets(state, side):
    """The mask builder's reach context, rebuilt its way (action_sampler
    ._build_legality_masks): visible units by position, ally/enemy/
    occupied/ZoC as coordinate sets."""
    from tools.abilities import hex_neighbors
    from tools.replay_dataset import _stats_for
    from wesnoth_ai.visibility import is_scenery_unit, units_visible_to
    unit_at = {}
    for u in units_visible_to(state, side):
        unit_at[(u.position.x, u.position.y)] = u
    occ, ally, enemy, zoc = set(), set(), set(), set()
    for pos, u in unit_at.items():
        occ.add(pos)
        if u.side == side:
            ally.add(pos)
            continue
        enemy.add(pos)
        if is_scenery_unit(u) or "petrified" in (u.statuses or set()):
            continue
        if int(_stats_for(u.name).get("level", 1)) < 1:
            continue
        zoc.update(hex_neighbors(pos[0], pos[1]))
    return unit_at, occ, ally, enemy, zoc


def _python_recruit_set(state, side, unit_at, rejected):
    from wesnoth_ai.visibility import leader_castle_network
    leader = next((u for u in state.map.units if u.side == side and u.is_leader), None)
    if leader is None:
        return set(), False
    on_keep, network = leader_castle_network(state, leader)
    return {p for p in network if p not in unit_at and p not in rejected}, on_keep


def _as_set(obs, arr):
    keys = obs.geometry.keys
    return set(map(keys.__getitem__, np.nonzero(arr)[0].tolist()))


@pytest.mark.parametrize("fog_on", [True, False])
def test_observation_equals_the_python_originals(fog_on):
    from wesnoth_ai.visibility import units_visible_to, visible_hexes_for
    checked = 0
    for state in _states():
        state.global_info._fog = fog_on
        for side in (1, 2):
            obs = obs_mod.observe(state, side)
            assert obs is not None
            keys_on_map = set(obs.geometry.keys)
            # the seen hexes, carried into map space
            assert _as_set(obs, obs.seen) == visible_hexes_for(state, side)
            # unit visibility, by id and as the same objects
            ref = units_visible_to(state, side)
            assert obs.visible_ids() == {u.id for u in ref}
            assert [u.id for u in obs.visible_units()] == [u.id for u in ref]
            # the reach context (map-space flags against coordinate sets)
            unit_at, occ, ally, enemy, zoc = _python_reach_sets(state, side)
            assert _as_set(obs, obs.occupied) == occ & keys_on_map
            assert _as_set(obs, obs.ally) == ally & keys_on_map
            assert _as_set(obs, obs.enemy) == enemy & keys_on_map
            assert _as_set(obs, obs.zoc) == zoc & keys_on_map
            from wesnoth_ai.visibility import is_scenery_unit
            inert = {p for p, u in unit_at.items() if is_scenery_unit(u)}
            assert _as_set(obs, obs.inert) == inert & keys_on_map
            # the recruit row
            rejected = getattr(state.global_info, "_recruit_rejected_hexes", None) or set()
            want, on_keep = _python_recruit_set(state, side, unit_at, rejected)
            assert obs.leader_on_keep == on_keep
            assert _as_set(obs, obs.recruit_row) == want & keys_on_map
            checked += 1
    assert checked >= 14


def test_rejected_recruit_hexes_leave_the_row():
    for state in _states():
        for side in (1, 2):
            base = obs_mod.observe(state, side)
            row = _as_set(base, base.recruit_row)
            if not row:
                continue
            victim = sorted(row)[0]
            state.global_info._recruit_rejected_hexes = {victim}
            try:
                again = obs_mod.observe(state, side)
            finally:
                state.global_info._recruit_rejected_hexes = set()
            assert _as_set(again, again.recruit_row) == row - {victim}
            return
    pytest.skip("no harvested state with a recruitable leader")


def _masks_equal(a, b) -> bool:
    import torch
    for name in ("actor_valid", "target_valid", "target_valid_attack", "target_valid_move",
                 "type_valid", "type_bias", "attack_bias"):
        if not torch.equal(getattr(a, name), getattr(b, name)):
            return False
    return True


def test_mask_builder_on_the_observation_equals_the_python_path():
    """End to end: the encoder attaches the observation, the legality
    masks built from it equal the masks built by the Python passes on
    the same encoded state (the observation stripped)."""
    import dataclasses
    import torch
    from helpers.priors_parity import _policy
    from wesnoth_ai.action_sampler import _build_legality_masks
    policy = _policy()
    enc = policy._inference_encoder
    compared = 0
    for state in _states():
        for fog_on in (True, False):
            state.global_info._fog = fog_on
            encoded = enc.encode(state)
            assert encoded.observation is not None
            with torch.no_grad():
                with_obs = _build_legality_masks(encoded, state)
                stripped = dataclasses.replace(encoded, observation=None)
                without = _build_legality_masks(stripped, state)
            assert _masks_equal(with_obs, without)
            assert float(with_obs.actor_valid.sum()) >= 1.0
            compared += 1
    assert compared >= 14


def test_detached_observation_pickles_without_units():
    import pickle
    state = _states()[0]
    obs = obs_mod.observe(state, 1)
    d = obs.detached()
    blob = pickle.dumps(d)
    back = pickle.loads(blob)
    assert np.array_equal(back.seen, obs.seen) and back.visible_ids() == obs.visible_ids()
    with pytest.raises(ValueError):
        back.visible_units()
