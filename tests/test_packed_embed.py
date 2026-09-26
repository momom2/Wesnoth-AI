"""Packed embed (design note docs/gpu_forward_design_20260904.md section
14): the vectorized packed layout equals a per-sample reference, the
stream-ordered embeddings of encode_from_raw_embedded reach the trunk as
the same tokens the padded encode produces (forward_embedded equals
forward_streams to the bit on CPU, on both trunks), the inference
server's switch changes nothing in the replies, and the priors'
batched capacity equals the per-leaf count."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness")
CPU = torch.device("cpu")
D = 32


def _layout_reference(sizes, H_max, U_max, R_max, token_kind, actor_kind):
    """The per-sample loop build_packed_layout replaced (padded source)."""
    B = len(sizes)
    L = H_max + U_max + R_max + 2
    A_max = U_max + R_max + 1
    lengths = np.array([U + R + H + 2 for U, R, H in sizes], dtype=np.int64)
    cu = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(lengths, out=cu[1:])
    src = np.empty(int(cu[-1]), dtype=np.int64)
    kind = np.empty(int(cu[-1]), dtype=np.int64)
    actor = np.empty((B, A_max), dtype=np.int64)
    hexes = np.empty((B, H_max), dtype=np.int64)
    unit = np.empty((B, U_max), dtype=np.int64)
    glob = np.empty(B, dtype=np.int64)
    kinds = np.full((B, A_max), actor_kind.END_TURN, dtype=np.int64)
    for b, (U, R, H) in enumerate(sizes):
        o, row, n = int(cu[b]), b * L, int(lengths[b])
        seg = src[o:o + n]
        seg[:H] = row + np.arange(H)
        seg[H:H + U] = row + H_max + np.arange(U)
        seg[H + U:H + U + R] = row + H_max + U_max + np.arange(R)
        seg[n - 2] = row + H_max + U_max + R_max
        seg[n - 1] = row + H_max + U_max + R_max + 1
        kseg = kind[o:o + n]
        kseg[:H] = token_kind.HEX
        kseg[H:H + U] = token_kind.UNIT
        kseg[H + U:H + U + R] = token_kind.RECRUIT
        kseg[n - 2] = token_kind.GLOBAL
        kseg[n - 1] = token_kind.END_TURN
        end = o + n - 1
        actor[b] = end
        actor[b, :U + R] = o + H + np.arange(U + R)
        hexes[b] = end
        hexes[b, :H] = o + np.arange(H)
        unit[b] = end
        unit[b, :U] = o + H + np.arange(U)
        glob[b] = end - 1
        kinds[b, :U] = actor_kind.UNIT
        kinds[b, U:U + R] = actor_kind.RECRUIT
    return dict(src=src, kind=kind, actor=actor.reshape(-1), hex=hexes.reshape(-1),
                unit=unit.reshape(-1), glob=glob, cu_seqlens=cu, actor_kind=kinds)


SIZE_SETS = [
    [(3, 0, 7), (1, 2, 5), (4, 1, 0)],      # a row without recruits, one without hexes
    [(2, 1, 0), (1, 2, 0)],                 # no hex stream at all
    [(0, 0, 3)],                            # a single row with no actors but end_turn
    [(2, 0, 30), (3, 0, 12), (1, 0, 30)],   # no recruit stream at all
]


def test_vectorized_layout_equals_the_per_sample_reference():
    from wesnoth_ai.model import ActorKind, TokenKind
    from wesnoth_ai.packed_trunk import build_packed_layout
    for sizes in SIZE_SETS:
        H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))
        got = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind)
        ref = _layout_reference(sizes, H_max, U_max, R_max, TokenKind, ActorKind)
        for name, a in ref.items():
            b = getattr(got, name)
            assert a.dtype == b.dtype == np.int64, name
            assert np.array_equal(a, b), (name, sizes)
        assert got.max_len == max(U + R + H + 2 for U, R, H in sizes)


def _stream_order(sizes, hex_b, unit_b, rec_b, glob_b, end_b):
    """EmbeddedStreams' row order built by hand from padded streams."""
    rows = [hex_b[b, :H] for b, (_, _, H) in enumerate(sizes)]
    rows += [unit_b[b, :U] for b, (U, _, _) in enumerate(sizes)]
    rows += [rec_b[b, :R] for b, (_, R, _) in enumerate(sizes)]
    rows += [glob_b[:, 0], end_b[:1, 0]]
    return torch.cat(rows)


def test_streams_source_and_padded_gather_index_agree_with_the_padded_source():
    from wesnoth_ai.model import ActorKind, TokenKind
    from wesnoth_ai.packed_trunk import build_packed_layout, padded_gather_index
    g = torch.Generator().manual_seed(3)
    for sizes in SIZE_SETS:
        B = len(sizes)
        H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))
        L = H_max + U_max + R_max + 2
        hex_b, unit_b, rec_b, glob_b = (torch.randn(B, n, D, generator=g)
                                        for n in (H_max, U_max, R_max, 1))
        end_b = torch.randn(1, 1, D, generator=g).expand(B, 1, D)   # one shared end_turn row
        padded = torch.cat([hex_b, unit_b, rec_b, glob_b, end_b], dim=1).reshape(-1, D)
        streams = _stream_order(sizes, hex_b, unit_b, rec_b, glob_b, end_b)
        from_padded = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind)
        from_streams = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind,
                                           source="streams")
        assert torch.equal(padded[torch.from_numpy(from_padded.src)],
                           streams[torch.from_numpy(from_streams.src)])
        for name in ("kind", "actor", "hex", "unit", "glob", "cu_seqlens", "actor_kind"):
            assert np.array_equal(getattr(from_padded, name), getattr(from_streams, name)), name
        idx = torch.from_numpy(padded_gather_index(sizes, H_max, U_max, R_max))
        rebuilt = torch.cat([streams, torch.zeros(1, D)])[idx].view(B, L, D)
        zeroed = padded.view(B, L, D).clone()
        for b, (U, R, H) in enumerate(sizes):
            zeroed[b, H:H_max] = 0
            zeroed[b, H_max + U:H_max + U_max] = 0
            zeroed[b, H_max + U_max + R:H_max + U_max + R_max] = 0
        assert torch.equal(rebuilt, zeroed), sizes


def _model_and_encoder():
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(7)
    return WesnothModel(d_model=D, num_layers=2, num_heads=2, d_ff=64).eval(), \
        GameStateEncoder(d_model=D)


def _items(n=4):
    """(RawEncoded, PackedMasks) of scenario starts on distinct maps."""
    from tools.inference_seam import build_light_encoded
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.server_priors import pack_masks
    items = []
    for seed in range(n):
        gs = build_scenario_gamestate(random_setup(random.Random(seed)))
        raw = encode_raw(gs, type_to_id={}, faction_to_id={}, relevant_set=False)
        items.append((raw, pack_masks(build_light_encoded(raw, CPU), gs)))
    return items


def test_forward_embedded_equals_forward_streams_on_both_trunks():
    model, enc = _model_and_encoder()
    items = _items()
    raws = [r for r, _ in items]
    assert len({len(r.hex_xs) for r in raws}) > 1, "batch must mix map sizes"
    with torch.no_grad():
        padded_streams = enc.encode_from_raw_padded(raws, device=CPU)
        embedded = enc.encode_from_raw_embedded(raws, device=CPU)
        assert embedded.tokens.shape == (sum(len(r.hex_xs) + len(r.unit_xs) + len(r.recruit_type_ids)
                                             for r in raws) + len(raws) + 1, D)
        assert embedded.sizes == padded_streams[-1]
        for packed in (True, False):
            ref = model.forward_streams(*padded_streams, packed=packed)
            got = model.forward_embedded(embedded, packed=packed)
            assert got.sizes == ref.sizes
            assert torch.equal(got.actor_kind, ref.actor_kind)
            for f in FIELDS:
                assert torch.equal(getattr(got, f), getattr(ref, f)), (packed, f)


def test_server_packed_embed_switch_changes_no_reply():
    from tools.inference_seam import InferenceServer
    model, enc = _model_and_encoder()
    items = _items()
    ref = InferenceServer(model, enc).infer_batch(items)
    got = InferenceServer(model, enc, packed_embed=True).infer_batch(items)
    assert sum(len(o.legal_compact.actor) for o in got) > 0
    for x, y in zip(ref, got):
        assert torch.equal(x.value, y.value) and torch.equal(x.value_logits, y.value_logits)
        assert torch.equal(x.cliffness, y.cliffness)
        for f in ("actor", "kind", "target", "weapon", "prior"):
            assert np.array_equal(getattr(x.legal_compact, f), getattr(y.legal_compact, f)), f


def test_staged_capacity_equals_the_per_leaf_count():
    """On harvested packs, and on packs whose actor and hex counts vary
    so the staged buffer holds pad rows and pad columns."""
    from wesnoth_ai.server_priors import _legal_capacity, _stage_masks
    packs = [m for _, m in _items()]
    rng = np.random.default_rng(0)
    T, W = packs[0].type_valid.shape[1], 4
    for k, (U, R, H) in enumerate([(5, 2, 40), (1, 0, 9), (12, 6, 100), (3, 3, 17)]):
        A = U + R + 1
        p = copy.deepcopy(packs[0])
        p.n_units, p.n_recruits, p.n_hexes = U, R, H
        p.actor_mask = (rng.random(A) < 0.7).astype(np.uint8)
        p.type_valid = (rng.random((A, T)) < 0.8).astype(np.uint8)
        p.attack_valid = np.packbits(rng.random((A, H)) < 0.2, axis=1)
        p.move_valid = np.packbits(rng.random((A, H)) < 0.3, axis=1)
        p.union_valid = np.packbits(rng.random((A, H)) < 0.3, axis=1)
        p.n_attacks = rng.integers(0, 6, A).astype(np.int8)     # some above W
        p.type_bias = None
        p.attack_bias = rng.random((A, H)).astype(np.float32) if k % 2 else None
        packs.append(p)
    A_max = max(p.n_units + p.n_recruits + 1 for p in packs)
    H_max = max(p.n_hexes for p in packs)
    _, _, capacity = _stage_masks(packs, A_max, H_max, T, W, pin=False)
    assert capacity == sum(_legal_capacity(p, W) for p in packs)
    assert capacity > 0
