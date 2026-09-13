"""The batch reaches the device in two transfers, with the same values.

`GameStateEncoder._embed_streams` coalesces every field of a batch into
one buffer per dtype instead of one transfer per field (the server pays
this once per batch and it was most of its fixed host cost). The
embeddings it produces must be BIT-IDENTICAL to the per-field path:
the server's batched priors and the single-sample eval path must not
disagree about what a position embeds to, or two runs of the same match
would differ.

The reference here is the arithmetic the embedding tables do on the
fields taken straight from the RawEncodeds, computed field by field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.encoder import GameStateEncoder  # noqa: E402


def _raws(n_states=3, relevant_set=False):
    """RawEncodeds from real scenario states, one per side."""
    from sim_test_helpers import scenario_setup
    from tools.scenario_pool import build_scenario_gamestate
    from wesnoth_ai.encoder import encode_raw
    enc = GameStateEncoder(d_model=32, relevant_set_hexes=relevant_set)
    raws = []
    for seed in range(n_states):
        gs = build_scenario_gamestate(scenario_setup(seed, mini=True))
        enc.register_names(gs)
        for side in (1, 2):
            gs.global_info.current_side = side
            raws.append(encode_raw(gs, type_to_id=enc.unit_type_to_id,
                                   faction_to_id=enc.faction_to_id,
                                   relevant_set=relevant_set,
                                   fog_hides_enemy_villages=True))
    return enc, raws


def _reference(enc, raws, device):
    """The per-field path this replaced: one transfer per field."""
    def cat(fields, dtype):
        arrays = [getattr(r, fields) for r in raws]
        return torch.from_numpy(np.concatenate(arrays)).to(device)
    out = {}
    if sum(r.hex_xs.shape[0] for r in raws):
        out["hex"] = enc._hex_embedding(
            cat("hex_xs", np.int64), cat("hex_ys", np.int64),
            cat("hex_terrain_ids", np.int64),
            cat("hex_modifier_flags", np.float32), cat("hex_dynamic_flags", np.float32))
    if sum(r.unit_xs.shape[0] for r in raws):
        out["unit"] = enc._unit_embedding(
            cat("unit_type_ids", np.int64), cat("unit_side_ids", np.int64),
            cat("unit_xs", np.int64), cat("unit_ys", np.int64), cat("unit_feats", np.float32))
        out["unit_is"] = cat("unit_is_ours", np.float32)
    if sum(r.recruit_type_ids.shape[0] for r in raws):
        out["recruit"] = enc._unit_embedding(
            cat("recruit_type_ids", np.int64), cat("recruit_side_ids", np.int64),
            cat("recruit_xs", np.int64), cat("recruit_ys", np.int64),
            cat("recruit_feats", np.float32))
        out["recruit_is"] = cat("recruit_is_ours", np.float32)
    gf = torch.from_numpy(np.stack([r.global_feats for r in raws])).to(device)
    ours = torch.tensor([r.our_faction_id for r in raws], device=device, dtype=torch.long)
    theirs = torch.tensor([r.their_faction_id for r in raws], device=device, dtype=torch.long)
    out["global"] = enc._global_embedding(gf, ours, theirs)
    return out


@pytest.mark.parametrize("relevant_set", [False, True])
def test_coalesced_transfer_embeds_identically(relevant_set):
    enc, raws = _raws(relevant_set=relevant_set)
    device = torch.device("cpu")
    with torch.no_grad():
        got = enc._embed_streams(raws, device)
        want = _reference(enc, raws, device)
    assert got["Hs"] == [r.hex_xs.shape[0] for r in raws]
    assert got["Us"] == [r.unit_xs.shape[0] for r in raws]
    assert got["Rs"] == [r.recruit_type_ids.shape[0] for r in raws]
    checked = 0
    for key, ref in want.items():
        assert got[key] is not None, key
        assert got[key].shape == ref.shape, (key, got[key].shape, ref.shape)
        assert got[key].dtype == ref.dtype, key
        assert torch.equal(got[key], ref), f"{key} differs after coalescing"
        checked += 1
    assert checked >= 4, "the harvest must exercise hexes, units, recruits and globals"


def test_padded_streams_still_match_the_batch_path():
    """The two consumers of _embed_streams agree: what the server pads
    per sample equals what the batch path concatenates."""
    enc, raws = _raws()
    with torch.no_grad():
        hex_b, unit_b, recruit_b, global_b, _end, sizes = \
            enc.encode_from_raw_padded(raws, device=torch.device("cpu"))
        emb = enc._embed_streams(raws, torch.device("cpu"))
    off = 0
    for i, (u, r, h) in enumerate(sizes):
        assert torch.equal(hex_b[i, :h], emb["hex"][off:off + h])
        off += h
    assert global_b.shape[0] == len(raws) and global_b.shape[1] == 1


def test_an_empty_stream_does_not_break_the_buffer():
    """A batch whose recruit stream is empty for every sample must not
    mis-slice the coalesced buffer."""
    enc, raws = _raws(n_states=1)
    for r in raws:
        r.recruit_type_ids = np.zeros(0, dtype=np.int64)
        r.recruit_side_ids = np.zeros(0, dtype=np.int64)
        r.recruit_xs = np.zeros(0, dtype=np.int64)
        r.recruit_ys = np.zeros(0, dtype=np.int64)
        r.recruit_feats = np.zeros((0, enc.unit_feat_dim if hasattr(enc, "unit_feat_dim") else 13),
                                   dtype=np.float32)
        r.recruit_is_ours = np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        emb = enc._embed_streams(raws, torch.device("cpu"))
        want = _reference(enc, raws, torch.device("cpu"))
    assert emb["recruit"] is None and emb["Rs"] == [0] * len(raws)
    assert torch.equal(emb["hex"], want["hex"])
    assert torch.equal(emb["global"], want["global"])
