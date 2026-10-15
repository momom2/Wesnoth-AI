"""The packed trunk's training variants (wesnoth_ai/trunk_modes.py) on a
small encoder over packed segments of differing lengths: the explicit-cast
trunk is the autocast trunk to the bit, the fused ReLU is the unfused layer
to the bit on the CPU, the bf16 residual runs its stream and norms in
bfloat16 near the float32 trunk, and the compiled block computes what the
eager block computes with one graph for every token count, under activation
checkpointing too."""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wesnoth_ai import trunk_modes
from wesnoth_ai.packed_trunk import PackedIndex, _dynamo_cache_entries, packed_trunk
from wesnoth_ai.trunk_modes import CompiledBlock, TrunkMode, layer_block, run_trunk

D, HEADS, FF, LAYERS = 32, 4, 64, 2


def _encoder(seed=0) -> nn.TransformerEncoder:
    torch.manual_seed(seed)
    layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)


def _index(lengths) -> PackedIndex:
    cu = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return PackedIndex(src=None, kind=None, actor=None, hex=None, unit=None, glob=None,
                       cu_seqlens=torch.tensor(cu, dtype=torch.int32), cu_host=cu.tolist(), max_len=max(lengths))


def _run(encoder, fn, lengths=(7, 11, 5), seed=1):
    """fn(x, index)'s output on random tokens and the gradients of a fixed
    projection of it: the input's and every parameter's."""
    torch.manual_seed(seed)
    x = torch.randn(sum(lengths), D, requires_grad=True)
    for p in encoder.parameters():
        p.grad = None
    y = fn(x, _index(lengths))
    (y.float() * torch.linspace(-1.0, 1.0, y.numel()).view_as(y)).sum().backward()
    return y.detach().float(), [x.grad.clone()] + [p.grad.clone() for p in encoder.parameters()]


def _equal(a, b) -> bool:
    return torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


@pytest.mark.parametrize("autocast", [False, True], ids=["float32", "bf16 autocast"])
def test_the_explicit_trunk_is_the_autocast_trunk_to_the_bit(autocast):
    """Every cast written out with the autocast off inside the trunk (the
    variants' common path, no variant on) gives packed_trunk's output and
    gradients to the bit, in float32 and under a bfloat16 autocast."""
    encoder = _encoder()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        base = _run(encoder, lambda x, i: packed_trunk(encoder, x, i))
        explicit = _run(encoder, lambda x, i: run_trunk(encoder, x, i, TrunkMode()))
    assert _equal(base, explicit)


@pytest.mark.parametrize("autocast", [False, True], ids=["float32", "bf16 autocast"])
def test_the_fused_relu_is_the_unfused_layer_to_the_bit_on_the_cpu(monkeypatch, autocast):
    """The first feed-forward linear and its ReLU in one _addmm_activation
    call, its backward written out: packed_trunk's output and every gradient
    to the bit on the CPU, the fused call made once a layer."""
    calls = []
    real = torch._addmm_activation
    monkeypatch.setattr(torch, "_addmm_activation", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    encoder = _encoder()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        base = _run(encoder, lambda x, i: packed_trunk(encoder, x, i))
        fused = _run(encoder, lambda x, i: run_trunk(encoder, x, i, TrunkMode(fused_relu=True)))
    assert len(calls) == LAYERS
    assert _equal(base, fused)


def _cosine(a, b) -> float:
    return float(F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


def test_the_bf16_residual_holds_its_stream_and_norms_in_bfloat16(monkeypatch):
    """Every LayerNorm reads and writes bfloat16 (its weights cast to it);
    the output and every gradient stay near the float32 trunk's without
    being it."""
    seen = []
    real = F.layer_norm

    def spy(x, shape, weight=None, bias=None, eps=1e-5):
        out = real(x, shape, weight, bias, eps)
        seen.append((x.dtype, None if weight is None else weight.dtype, out.dtype))
        return out
    monkeypatch.setattr(trunk_modes.F, "layer_norm", spy)
    encoder = _encoder()
    base = _run(encoder, lambda x, i: packed_trunk(encoder, x, i))
    seen.clear()
    bf16 = _run(encoder, lambda x, i: run_trunk(encoder, x, i, TrunkMode(bf16_residual=True)))
    assert seen == [(torch.bfloat16,) * 3] * (2 * LAYERS)
    assert not _equal(base, bf16)
    scale = base[0].abs().max()
    assert (bf16[0] - base[0]).abs().max() < 0.05 * scale
    assert min(_cosine(a, b) for a, b in zip(base[1], bf16[1])) > 0.995


def _block_inputs(encoder, n, seed):
    torch.manual_seed(seed)
    h = torch.randn(n, D, requires_grad=True)
    ctx = torch.randn(n, D, requires_grad=True)
    layer = encoder.layers[0]
    return h, ctx, trunk_modes._block_weights(layer), layer.norm1.eps


def _block_gradients(encoder, fn, n, seed):
    h, ctx, weights, eps = _block_inputs(encoder, n, seed)
    for p in encoder.parameters():
        p.grad = None
    out = fn(h, ctx, weights, eps)
    (out * torch.linspace(-1.0, 1.0, out.numel()).view_as(out)).sum().backward()
    return out.detach(), [h.grad, ctx.grad] + [w.grad.clone() for w in weights if w is not None]


def _compiled_matches_eager(backend: str, rtol: float, atol: float) -> None:
    """Three token counts through one CompiledBlock, plain and under
    non-reentrant checkpointing: each the eager block's output and
    gradients within (rtol, atol), and dynamo's cache for the block's code
    holds one graph more than before."""
    torch._dynamo.reset()
    encoder = _encoder()
    block = CompiledBlock(backend)
    before = _dynamo_cache_entries(layer_block)

    def compiled(h, ctx, w, eps):
        return block(h, ctx, *w, eps=eps, fused_relu=False)

    def eager(h, ctx, w, eps):
        return layer_block(h, ctx, *w, eps, False)

    def recomputed(h, ctx, w, eps):
        return checkpoint(lambda a, b: block(a, b, *w, eps=eps, fused_relu=False), h, ctx, use_reentrant=False)

    for n, seed in ((37, 1), (53, 2), (61, 3)):
        want = _block_gradients(encoder, eager, n, seed)
        for fn in (compiled, recomputed):
            got = _block_gradients(encoder, fn, n, seed)
            assert torch.allclose(got[0], want[0], rtol=rtol, atol=atol)
            assert all(torch.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(got[1], want[1]))
    assert _dynamo_cache_entries(layer_block) - before == 1
    assert block.calls >= 6 and block.first_call_seconds is not None


@pytest.mark.slow
def test_the_compiled_block_is_the_eager_block_with_one_graph_for_every_token_count():
    """aot_eager: dynamo's capture and AOTAutograd's split into a forward
    and a backward graph, run by the eager kernels."""
    _compiled_matches_eager("aot_eager", rtol=1e-6, atol=1e-7)


def _no_cpp_compiler() -> bool:
    """No compiler inductor's CPU backend would call: cl on Windows, else
    $CXX or g++ (torch/_inductor/config.py, cpp.cxx)."""
    if sys.platform == "win32":
        return shutil.which("cl") is None
    return shutil.which(os.environ.get("CXX", "g++")) is None


@pytest.mark.slow
@pytest.mark.skipif(_no_cpp_compiler(), reason="inductor's CPU code needs a C++ compiler (none on this host)")
@pytest.mark.skipif(torch.cuda.is_available(), reason="a GPU host runs inductor's CUDA path in tests/test_trunk_modes_cuda.py "
                                                      "(on Linux, 2.5.1's CPU path may first fetch gcc through conda)")
def test_the_inductor_block_computes_the_eager_block_with_one_graph_for_every_token_count():
    """inductor's generated C++ kernels on the CPU (CI): the eager block's
    numbers within float32 reassociation."""
    _compiled_matches_eager("inductor", rtol=1e-4, atol=1e-5)


def test_the_variants_refuse_what_they_do_not_rewrite():
    from wesnoth_ai.trunk_modes import check_trunk_mode_supported
    encoder = _encoder()
    with pytest.raises(ValueError, match="fused ReLU"):
        check_trunk_mode_supported(encoder, TrunkMode(fused_relu=True, compile_backend="inductor"))
    gelu = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, FF, activation="gelu", batch_first=True), 1,
                                 enable_nested_tensor=False)
    with pytest.raises(NotImplementedError, match="ReLU"):
        check_trunk_mode_supported(gelu, TrunkMode(bf16_residual=True))
    with pytest.raises(ValueError, match="CompiledBlock"):
        run_trunk(encoder, torch.randn(5, D), _index([5]), TrunkMode(compile_backend="inductor"))


def test_a_mode_names_its_variants():
    assert TrunkMode().describe() == "none" and not TrunkMode().active
    mode = TrunkMode(bf16_residual=True, compile_backend="inductor")
    assert mode.active and mode.describe() == "bf16_residual+compiled:inductor"
