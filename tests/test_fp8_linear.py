"""The trunk's fp8 products (wesnoth_ai/fp8_linear.py) on the CPU's
emulation: the scaling is exact and the cast rounds within the format's
precision; the linear and its three gradients stay within fp8 rounding of
F.linear's, for token counts on and off the padding's multiple; every
product takes the operands torch._scaled_mm takes on CUDA, and the check
refuses the layouts it refuses; the packed trunk under --fp8-trunk runs near
the autocast trunk, recomputes to the bit under activation checkpointing, and
refuses what it cannot run; the compiled block holds the fp8 products under
the box's torch's strict mode."""
from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wesnoth_ai import fp8_linear as f8
from wesnoth_ai.packed_trunk import PackedIndex, packed_trunk
from wesnoth_ai.trunk_modes import TrunkMode, check_trunk_mode_supported, run_trunk

# Half the spacing of the format's significands: the largest relative
# rounding error of a normal value (3 mantissa bits in e4m3, 2 in e5m2).
HALF_ULP = {f8.E4M3: 2.0 ** -4, f8.E5M2: 2.0 ** -3}


@pytest.mark.parametrize("dtype", [f8.E4M3, f8.E5M2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 3e4])
def test_the_scale_is_exact_and_the_cast_rounds_within_the_format(dtype, magnitude):
    torch.manual_seed(0)
    t = torch.randn(64, 48) * magnitude
    t8, inverse = f8.quantize(t, dtype)
    assert t8.dtype == dtype and t8.is_contiguous() and inverse.dtype == torch.float32
    scale = 1.0 / float(inverse)
    assert scale == 2.0 ** round(np.log2(scale)), "a power of two"
    top = float(torch.finfo(dtype).max)
    assert top / 2 < float(t.abs().max()) * scale <= top, "the largest magnitude in the top binade"
    back = t8.float() * inverse
    normal = t.abs() * scale >= torch.finfo(dtype).smallest_normal
    rel = ((back - t).abs() / t.abs())[normal]
    assert float(rel.max()) <= HALF_ULP[dtype] * (1 + 1e-6)
    zero8, zero_inverse = f8.quantize(torch.zeros(16, 16), dtype)
    assert torch.equal(zero8.float(), torch.zeros(16, 16)) and torch.isfinite(zero_inverse)


def _linear_and_grads(fn, x, w, b, probe):
    x, w, b = (t.detach().clone().requires_grad_() for t in (x, w, b))
    y = fn(x, w, b)
    (y.float() * probe).sum().backward()
    return y.detach().float(), x.grad.float(), w.grad.float(), b.grad.float()


def _cos(a, b):
    return float(F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


@pytest.mark.parametrize("tokens", [37, 48], ids=["padded", "a multiple of 16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_fp8_linear_is_the_linear_within_fp8_rounding(tokens, dtype):
    torch.manual_seed(1)
    x = torch.randn(tokens, 32).to(dtype)
    w = (torch.randn(48, 32) * 0.2).to(dtype)
    b = (torch.randn(48) * 0.1).to(dtype)
    probe = torch.randn(tokens, 48)
    ref = _linear_and_grads(F.linear, x, w, b, probe)
    got = _linear_and_grads(f8.fp8_linear, x, w, b, probe)
    for name, a, c in zip(("output", "x grad", "weight grad", "bias grad"), ref, got):
        assert c.shape == a.shape, name
        rms = float((c - a).pow(2).mean().sqrt() / a.pow(2).mean().sqrt())
        assert rms < 0.1 and _cos(a, c) > 0.995, (name, rms, _cos(a, c))
    assert got[0].dtype == torch.float32 and f8.fp8_linear(x, w, b).dtype == dtype


def test_every_product_takes_the_operands_the_cuda_kernel_takes(monkeypatch):
    """The forward's product and the backward's two, recorded: float8, the
    forward's both e4m3, the backward's gradient e5m2 and never two e5m2,
    the first operand row-major, the second column-major, the inner
    dimension and the second operand's a multiple of 16 (the weight
    gradient's inner dimension is the padded token count)."""
    seen = []
    real = f8.scaled_mm

    def recording(a8, b8, *args, **kw):
        seen.append((a8.dtype, b8.dtype, tuple(a8.shape), tuple(b8.shape), a8.is_contiguous(),
                     b8.t().is_contiguous() and not b8.is_contiguous()))
        return real(a8, b8, *args, **kw)
    monkeypatch.setattr(f8, "scaled_mm", recording)
    x = torch.randn(37, 32, requires_grad=True)
    w = torch.randn(48, 32, requires_grad=True)
    b = torch.randn(48, requires_grad=True)
    f8.fp8_linear(x, w, b).sum().backward()
    assert [s[:4] for s in seen] == [(f8.E4M3, f8.E4M3, (37, 32), (32, 48)),
                                     (f8.E5M2, f8.E4M3, (37, 48), (48, 32)),
                                     (f8.E5M2, f8.E4M3, (48, 48), (48, 32))]
    assert all(row_major and column_major for *_, row_major, column_major in seen)


def test_the_check_refuses_what_the_cuda_kernel_refuses():
    a8, _ = f8.quantize(torch.randn(20, 32), f8.E4M3)
    w8, _ = f8.quantize(torch.randn(48, 32), f8.E4M3)
    f8.check_operands(a8, w8.t())
    with pytest.raises(ValueError, match="column-major"):
        f8.check_operands(a8, w8.t().contiguous())
    with pytest.raises(ValueError, match="row-major"):
        f8.check_operands(a8.t().contiguous().t(), w8.t())
    g8, _ = f8.quantize(torch.randn(20, 32), f8.E5M2)
    w8b, _ = f8.quantize(torch.randn(48, 32), f8.E5M2)
    with pytest.raises(ValueError, match="two e5m2"):
        f8.check_operands(g8, w8b.t())
    odd8, _ = f8.quantize(torch.randn(40, 32), f8.E4M3)
    with pytest.raises(ValueError, match="multiples of 16"):
        f8.check_operands(a8, odd8.t())
    with pytest.raises(ValueError, match="float8"):
        f8.check_operands(torch.randn(20, 32), w8.t())


D, HEADS, FF, LAYERS = 32, 4, 64, 2


def _encoder(d_ff=FF) -> nn.TransformerEncoder:
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(D, HEADS, d_ff, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)


def _index(lengths) -> PackedIndex:
    cu = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return PackedIndex(src=None, kind=None, actor=None, hex=None, unit=None, glob=None,
                       cu_seqlens=torch.tensor(cu, dtype=torch.int32), cu_host=cu.tolist(), max_len=max(lengths))


def _trunk(encoder, fn, lengths=(17, 23, 9), recompute=False):
    torch.manual_seed(2)
    x = torch.randn(sum(lengths), D, requires_grad=True)
    for p in encoder.parameters():
        p.grad = None
    index = _index(lengths)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = checkpoint(fn, x, index, use_reentrant=False) if recompute else fn(x, index)
    (y.float() * torch.linspace(-1.0, 1.0, y.numel()).view_as(y)).sum().backward()
    return y.detach().float(), [x.grad.clone()] + [p.grad.clone() for p in encoder.parameters()]


def test_the_fp8_trunk_runs_near_the_autocast_trunk_and_recomputes_to_the_bit():
    encoder = _encoder()
    mode = TrunkMode(fp8=True)
    check_trunk_mode_supported(encoder, mode)
    base_y, base_g = _trunk(encoder, lambda x, i: packed_trunk(encoder, x, i))
    y, g = _trunk(encoder, lambda x, i: run_trunk(encoder, x, i, mode))
    assert float((y - base_y).pow(2).mean().sqrt() / base_y.pow(2).mean().sqrt()) < 0.1
    assert min(_cos(a, c) for a, c in zip(base_g, g)) > 0.98
    y2, g2 = _trunk(encoder, lambda x, i: run_trunk(encoder, x, i, mode), recompute=True)
    assert torch.equal(y, y2) and all(torch.equal(a, c) for a, c in zip(g, g2))
    assert mode.describe() == "fp8" and mode.active


def test_the_fp8_trunk_refuses_what_it_cannot_run():
    with pytest.raises(ValueError, match="fused ReLU"):
        check_trunk_mode_supported(_encoder(), TrunkMode(fp8=True, fused_relu=True))
    with pytest.raises(ValueError, match="multiples of 16"):
        check_trunk_mode_supported(_encoder(d_ff=72), TrunkMode(fp8=True))


# What torch 2.5.1 (the box's) bans in the backward pass of an
# autograd.Function that dynamo traces (torch/_dynamo/config.py:443-453);
# later versions ban less, so the test restores 2.5.1's list.
BOX_STRICT_MODE_BANS = ["stride", "requires_grad", "storage_offset", "layout", "data"] + [
    name for name, _ in inspect.getmembers(torch.Tensor) if re.match(r"^is_.*", name)]


def test_the_fp8_block_compiles_whole_under_the_box_torch_s_strict_mode(monkeypatch):
    """The compiled block with fp8 products (`--compiled-trunk --fp8-trunk`)
    under fullgraph and torch 2.5.1's strict-mode bans, aot_eager on the CPU:
    one graph for token counts on and off the padding's multiple, its output
    and gradients the eager fp8 block's within a few fp8 roundings."""
    import torch._dynamo.config as dynamo_config
    from wesnoth_ai.packed_trunk import _dynamo_cache_entries
    from wesnoth_ai.trunk_modes import CompiledBlock, _block_weights, layer_block
    if not hasattr(dynamo_config, "_autograd_backward_strict_mode_banned_ops"):
        pytest.skip("this torch has no strict-mode ban list")
    monkeypatch.setattr(dynamo_config, "_autograd_backward_strict_mode_banned_ops", BOX_STRICT_MODE_BANS)
    torch._dynamo.reset()
    layer = _encoder().layers[0]
    weights = _block_weights(layer)
    block = CompiledBlock("aot_eager")
    for n in (37, 48, 61):
        runs = []
        for fn in (lambda h, c: layer_block(h, c, *weights, layer.norm1.eps, False, True),
                   lambda h, c: block(h, c, *weights, eps=layer.norm1.eps, fused_relu=False, fp8=True)):
            torch.manual_seed(n)
            h = torch.randn(n, D, requires_grad=True)
            ctx = torch.randn(n, D, requires_grad=True)
            for p in layer.parameters():
                p.grad = None
            y = fn(h, ctx)
            (y * torch.linspace(-1.0, 1.0, y.numel()).view_as(y)).sum().backward()
            runs.append([y.detach(), h.grad, ctx.grad] + [p.grad for p in weights if p is not None])
        for a, c in zip(*runs):
            assert float((c - a).pow(2).mean().sqrt()) <= 0.02 * float(a.pow(2).mean().sqrt()), n
    assert _dynamo_cache_entries(layer_block) == 1, "one graph for every token count"
