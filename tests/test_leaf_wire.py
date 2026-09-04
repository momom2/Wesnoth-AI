"""Packed inference requests (wesnoth_ai/leaf_wire.py): a request's
numeric streams and masks survive the buffer round trip exactly, the
server's padded encode sees the same tensors, and the buffer is
smaller than pickling the objects."""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _leaves(n=3):
    from tools.inference_seam import build_light_encoded
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.server_priors import pack_masks
    cpu = torch.device("cpu")
    items = []
    for seed in range(n):
        gs = build_scenario_gamestate(random_setup(random.Random(seed)))
        raw = encode_raw(gs, type_to_id={}, faction_to_id={}, relevant_set=False)
        enc = build_light_encoded(raw, cpu)
        items.append((raw, pack_masks(enc, gs)))
    return items


def test_round_trip_is_exact_and_smaller():
    from wesnoth_ai.leaf_wire import MASK_FIELDS, RAW_FIELDS, pack_request, unpack_request
    items = _leaves()
    items[1][1].type_bias = np.full_like(items[1][1].type_valid, 0.25, dtype=np.float32)
    req = pack_request(items)
    assert len(req) == len(items)
    back = unpack_request(pickle.loads(pickle.dumps(req)))
    for (raw, masks), (raw2, masks2) in zip(items, back):
        for name, dt in RAW_FIELDS:
            a, b = getattr(raw, name), getattr(raw2, name)
            assert b.dtype == dt and a.shape == b.shape and np.array_equal(a, b), name
        for name, _dt in MASK_FIELDS:
            a, b = getattr(masks, name), getattr(masks2, name)
            assert (a is None) == (b is None), name
            if a is not None:
                assert np.array_equal(a, b), name
        assert (raw2.our_faction_id, raw2.their_faction_id, raw2.hex_subset) == \
            (raw.our_faction_id, raw.their_faction_id, raw.hex_subset)
        assert (masks2.end_turn_bias, masks2.n_units, masks2.n_recruits, masks2.n_hexes) == \
            (masks.end_turn_bias, masks.n_units, masks.n_recruits, masks.n_hexes)
    assert back[1][1].type_bias is not None and back[0][1].type_bias is None
    # Fewer bytes than the object pickle (the metadata lists are gone);
    # the point is one numpy object per request instead of ~40 per leaf.
    assert len(pickle.dumps(req)) < len(pickle.dumps(items))


def test_padded_encode_matches_the_originals():
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.leaf_wire import pack_request, unpack_request
    items = _leaves()
    enc = GameStateEncoder(d_model=32)
    cpu = torch.device("cpu")
    raws = [r for r, _ in items]
    with torch.no_grad():
        a = enc.encode_from_raw_padded(raws, device=cpu)
        b = enc.encode_from_raw_padded([r for r, _ in unpack_request(pack_request(items))],
                                       device=cpu)
    assert a[-1] == b[-1]                                   # sizes
    for x, y in zip(a[:-1], b[:-1]):
        assert torch.equal(x, y)
