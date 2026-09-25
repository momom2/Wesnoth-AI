"""Packed varlen trunk (wesnoth_ai/packed_trunk.py, design note
docs/gpu_forward_design_20260904.md section 5.3), CPU tier: the packed
layout round-trips the padded streams, the packed forward equals the
padded forward (on CPU the same bookkeeping runs with per-segment
SDPA), its PaddedOutput feeds batched_priors unchanged, and the switch
leaves CPU and fp32 callers on the padded trunk. The kernel itself is
covered by tests/test_packed_trunk_cuda.py on the box."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness")
D = 32
CPU = torch.device("cpu")


def _model(**kw):
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(5)
    return WesnothModel(d_model=D, num_layers=2, num_heads=2, d_ff=64, **kw).eval()


def _streams(sizes, seed=0):
    """Random padded streams for per-sample sizes (U, R, H). Pad
    positions hold random values, not zeros: a trunk that reads them
    would show in the comparison."""
    g = torch.Generator().manual_seed(seed)
    B = len(sizes)
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))

    def stream(n):
        return torch.randn(B, n, D, generator=g)
    return stream(H_max), stream(U_max), stream(R_max), stream(1), stream(1), list(sizes)


def _assert_samples_equal(a, b, atol=1e-5, rtol=1e-4):
    assert len(a.sizes) == len(b.sizes)
    for x, y in zip(a.samples(), b.samples()):
        assert (x.num_units, x.num_recruits) == (y.num_units, y.num_recruits)
        assert torch.equal(x.actor_kind, y.actor_kind)
        for f in FIELDS:
            assert getattr(x, f).shape == getattr(y, f).shape, f
            assert torch.allclose(getattr(x, f), getattr(y, f), atol=atol, rtol=rtol), f


def test_packed_layout_round_trips_the_padded_streams():
    from wesnoth_ai.model import ActorKind, TokenKind
    from wesnoth_ai.packed_trunk import build_packed_layout
    sizes = [(3, 0, 7), (1, 2, 5), (4, 1, 0)]          # (U, R, H): a row without recruits, one without hexes
    hex_b, unit_b, rec_b, glob_b, end_b, _ = _streams(sizes)
    B, H_max, U_max, R_max = len(sizes), hex_b.size(1), unit_b.size(1), rec_b.size(1)
    A_max = U_max + R_max + 1
    layout = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind)
    index = layout.to_device(CPU)
    padded = torch.cat([hex_b, unit_b, rec_b, glob_b, end_b], dim=1).reshape(-1, D)
    packed = padded.index_select(0, index.src)

    lengths = [U + R + H + 2 for U, R, H in sizes]
    assert index.cu_host == [0] + list(np.cumsum(lengths))
    assert index.cu_seqlens.dtype == torch.int32 and index.max_len == max(lengths)
    assert packed.shape == (sum(lengths), D)
    actor = packed.index_select(0, index.actor).view(B, A_max, D)
    hexes = packed.index_select(0, index.hex).view(B, H_max, D)
    units = packed.index_select(0, index.unit).view(B, U_max, D)
    glob = packed.index_select(0, index.glob)
    for b, (U, R, H) in enumerate(sizes):
        lo, hi = index.cu_host[b], index.cu_host[b + 1]
        segment = torch.cat([hex_b[b, :H], unit_b[b, :U], rec_b[b, :R], glob_b[b], end_b[b]])
        assert torch.equal(packed[lo:hi], segment)
        assert index.kind[lo:hi].tolist() == ([TokenKind.HEX] * H + [TokenKind.UNIT] * U
                                              + [TokenKind.RECRUIT] * R
                                              + [TokenKind.GLOBAL, TokenKind.END_TURN])
        assert torch.equal(hexes[b, :H], hex_b[b, :H])
        assert torch.equal(actor[b, :U], unit_b[b, :U])
        assert torch.equal(actor[b, U:U + R], rec_b[b, :R])
        assert torch.equal(actor[b, U + R], end_b[b, 0])
        assert torch.equal(units[b, :U], unit_b[b, :U])
        assert torch.equal(glob[b], glob_b[b, 0])
        assert layout.actor_kind[b].tolist() == ([ActorKind.UNIT] * U + [ActorKind.RECRUIT] * R
                                                 + [ActorKind.END_TURN] * (A_max - U - R))


@pytest.mark.parametrize("sizes", [
    [(3, 2, 40), (1, 0, 25), (5, 3, 33), (2, 1, 40)],   # mixed padding on every stream
    [(2, 0, 30), (3, 0, 12)],                            # no recruit stream at all
    [(2, 1, 0), (1, 2, 0)],                              # no hex stream at all
])
def test_packed_forward_equals_padded_forward(sizes):
    model = _model()
    streams = _streams(sizes, seed=1)
    with torch.no_grad():
        padded = model.forward_streams(*streams, packed=False)
        packed = model.forward_streams(*streams, packed=True)
    _assert_samples_equal(padded, packed)
    assert packed.target_logits.shape == padded.target_logits.shape


def test_packed_forward_on_real_states_matches_single_forward_and_priors():
    """Through forward_padded / forward_batch on encoded states: every
    per-sample view equals the single-sample forward, and the compact
    priors computed from the packed output equal those from the padded
    output (same legal entries, same order)."""
    from helpers.priors_parity import _policy, _states
    from wesnoth_ai.server_priors import batched_priors, pack_masks
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    states = _states()
    encs = [enc.encode(gs) for gs in states]
    packs = [pack_masks(e, gs) for e, gs in zip(encs, states)]
    assert len({(e.unit_tokens.size(1), e.recruit_tokens.size(1), e.hex_tokens.size(1))
                for e in encs}) > 1, "batch must mix sizes"
    with torch.no_grad():
        singles = [model(e) for e in encs]
        for out, single in zip(model.forward_batch(encs, packed=True), singles):
            for f in FIELDS:
                assert torch.allclose(getattr(out, f), getattr(single, f), atol=1e-5, rtol=1e-4), f
        padded = model.forward_padded(encs, packed=False)
        packed = model.forward_padded(encs, packed=True)
        ref, got = batched_priors(padded, packs), batched_priors(packed, packs)
    assert sum(len(c.actor) for c in ref) > 0
    for x, y in zip(ref, got):
        for f in ("actor", "kind", "target", "weapon"):
            assert np.array_equal(getattr(x, f), getattr(y, f)), f
        assert np.allclose(x.prior, y.prior, rtol=1e-5, atol=1e-9)


def test_switch_keeps_cpu_and_training_calls_on_the_padded_trunk(monkeypatch):
    model = _model()
    model.infer_packed_trunk = True
    streams = _streams([(2, 1, 20), (1, 0, 15)])

    def _refuse(*a, **k):
        raise AssertionError("packed trunk selected off the flash kernel's domain")
    monkeypatch.setattr(model, "_forward_streams_packed", _refuse)
    with torch.no_grad():
        out = model.forward_streams(*streams)                 # CPU: padded trunk
    assert out.actor_logits.shape[0] == 2
    model.train()
    assert not model._packed_trunk_applies(streams[0])
    model.eval()
    model.infer_packed_trunk = False
    assert not model._packed_trunk_applies(streams[0])


def test_unsupported_layer_options_are_refused():
    from wesnoth_ai.packed_trunk import check_packed_trunk_supported
    layer = torch.nn.TransformerEncoderLayer(D, 2, 64, batch_first=True, norm_first=True)
    with pytest.raises(NotImplementedError, match="norm_first"):
        check_packed_trunk_supported(torch.nn.TransformerEncoder(layer, 1, enable_nested_tensor=False))
    check_packed_trunk_supported(_model().encoder)
