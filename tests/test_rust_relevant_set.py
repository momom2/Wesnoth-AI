"""The relevant-set basis on the Rust path equals the Python originals.

`observe(state, side, reach=True)` (wesnoth_ai/observe.py, kernels
`reach_rows` and `observe_side`'s castle network) must reproduce
`visibility.relevant_hex_positions` and every acting unit's
`pathfind_sim.unit_reach(...).landable`; `encode_raw(relevant_set=True)`
on that path must produce the record the Python subset path produces,
byte for byte; and the legality masks in the subset basis through
`rows_from_reach` must equal the Python mask path's. Harvested states
from dummy self-play, both sides, fog on and off, with engagement
counters so a run where the kernel never fired cannot pass.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from wesnoth_ai import observe as _obs

pytestmark = pytest.mark.skipif(_obs.kernel() is None or _obs.kernel_rows_from_reach() is None,
                                reason="wesnoth_core phase 5 kernels not available")


def _states():
    from helpers.played_states import _states as harvest
    return harvest(n_attack=3, n_plain=4)


def _both_sides(states):
    for gs in states:
        for fog in (True, False):
            g = copy.deepcopy(gs)
            g.global_info._fog = fog
            for side in (1, 2):
                yield g, side, fog


def _acting(gs, side):
    return [u for u in gs.map.units
            if u.side == side and "petrified" not in (u.statuses or set())
            and (u.current_moves > 0 or not u.has_attacked)]


def test_relevant_set_and_landable_rows_equal_the_python_originals():
    from tools.pathfind_sim import ReachContext, unit_reach
    from wesnoth_ai.visibility import relevant_hex_positions
    checked_sets = checked_rows = nonempty_rows = 0
    for gs, side, _fog in _both_sides(_states()):
        obs = _obs.observe(gs, side, reach=True)
        assert obs is not None and obs.relevant is not None
        assert obs.relevant_set() == relevant_hex_positions(gs, side)
        checked_sets += 1
        ctx = ReachContext.for_side(gs, side)
        index = {uid: k for k, uid in enumerate(obs.unit_ids)}
        keys = obs.geometry.keys
        for u in _acting(gs, side):
            k = index[u.id]
            assert obs.acting[k] == 1
            row = set(map(keys.__getitem__, np.flatnonzero(obs.landable[k]).tolist()))
            expected = unit_reach(u, gs, ctx).landable if u.current_moves > 0 else set()
            assert row == expected, (u.id, side)
            checked_rows += 1
            nonempty_rows += bool(row)
        for k, uid in enumerate(obs.unit_ids):
            if not obs.acting[k]:
                assert not obs.landable[k].any()
    assert checked_sets >= 20 and checked_rows >= 40 and nonempty_rows >= 20


def _encode_subset_both(gs, side_vocab):
    from wesnoth_ai.encoder import encode_raw
    type_to_id, faction_to_id = side_vocab
    kw = dict(type_to_id=type_to_id, faction_to_id=faction_to_id, relevant_set=True)
    rust = encode_raw(gs, **kw)
    saved = _obs._KERNELS
    _obs._KERNELS = {}
    try:
        py = encode_raw(gs, **kw)
    finally:
        _obs._KERNELS = saved
    return py, rust


def test_subset_records_byte_identical_to_the_python_path():
    from wesnoth_ai.encoder import RawEncoded
    states = _states()
    names = sorted({u.name for gs in states for u in gs.map.units}
                   | {r for gs in states for s in gs.sides for r in s.recruits})
    vocab = ({n: i for i, n in enumerate(names)}, {"": 0, states[0].sides[1].faction: 1})
    checked = 0
    for gs, side, _fog in _both_sides(states):
        gs.global_info.current_side = side
        py, rust = _encode_subset_both(gs, vocab)
        assert py.observation is None and rust.observation is not None
        assert rust.hex_subset and py.hex_subset
        for f in dataclasses.fields(RawEncoded):
            if f.name == "observation":
                continue
            a, b = getattr(py, f.name), getattr(rust, f.name)
            if isinstance(a, np.ndarray):
                assert a.dtype == b.dtype and a.shape == b.shape, (f.name, side)
                assert a.tobytes() == b.tobytes(), (f.name, side)
            else:
                assert a == b, (f.name, side)
        assert 0 < len(rust.hex_positions) < len(gs.map.hexes)
        checked += 1
    assert checked >= 20


_FIELDS = ("actor_valid", "target_valid", "target_valid_attack", "target_valid_move",
           "type_valid", "type_bias", "attack_bias")


def test_subset_masks_equal_the_python_mask_path():
    import wesnoth_ai.action_sampler as sampler
    from tools import pathfind_sim as pf
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.encoder import GameStateEncoder
    enc = GameStateEncoder(relevant_set_hexes=True)
    engaged = [0]
    original = sampler._rows_from_observation

    def counting(*a, **k):
        out = original(*a, **k)
        engaged[0] += out is not None
        return out

    sampler._rows_from_observation = counting
    try:
        checked = 0
        for gs, side, _fog in _both_sides(_states()):
            gs.global_info.current_side = side
            encoded = enc.encode(gs)
            assert encoded.observation is not None and encoded.observation.tok_of_hex is not None
            rust = _build_legality_masks(encoded, gs)
            saved = pf._RUST
            pf._RUST = None
            try:
                py = _build_legality_masks(encoded, gs)
            finally:
                pf._RUST = saved
            for f in _FIELDS:
                assert torch.equal(getattr(py, f), getattr(rust, f)), (f, side)
            checked += 1
    finally:
        sampler._rows_from_observation = original
    assert checked >= 20 and engaged[0] >= 20
