"""The packed trunk's training variants on CUDA (wesnoth_ai/trunk_modes.py);
skipped without a GPU (CI, the laptop), run by scripts/trainer_speed_box.sh.
Under the bfloat16 autocast, on the flash varlen kernel: every variant's
output and gradients near packed_trunk's (bfloat16 rounding, the fused
epilogue's GEMM, inductor's kernels), the compiled block one graph for every
token count, under activation checkpointing too, and the variants' kept
activations fewer bytes a token than the autocast trunk's."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wesnoth_ai.packed_trunk import PackedIndex, _dynamo_cache_entries, packed_trunk
from wesnoth_ai.trunk_modes import CompiledBlock, TrunkMode, layer_block, run_trunk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

CUDA = torch.device("cuda")
D, HEADS, FF, LAYERS = 128, 4, 512, 2           # head_dim 32, as production's


def _encoder() -> nn.TransformerEncoder:
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False).to(CUDA)


def _index(lengths) -> PackedIndex:
    cu = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return PackedIndex(src=None, kind=None, actor=None, hex=None, unit=None, glob=None,
                       cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=CUDA), cu_host=cu.tolist(),
                       max_len=int(max(lengths)))


def _run(encoder, fn, lengths, seed=1, recompute=False):
    """fn's output under the bf16 autocast and the gradients of a fixed
    projection of it (the input's and every parameter's); the memory its
    forward keeps for the backward pass."""
    torch.manual_seed(seed)
    x = torch.randn(sum(lengths), D, device=CUDA, requires_grad=True)
    index = _index(lengths)
    for p in encoder.parameters():
        p.grad = None
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = checkpoint(fn, x, index, use_reentrant=False) if recompute else fn(x, index)
    kept = torch.cuda.memory_allocated() - before
    (y.float() * torch.linspace(-1.0, 1.0, y.numel(), device=CUDA).view_as(y)).sum().backward()
    return y.detach().float(), [x.grad.clone()] + [p.grad.clone() for p in encoder.parameters()], kept


def _cosines(a, b):
    return [float(F.cosine_similarity(x.flatten().double(), y.flatten().double(), dim=0)) for x, y in zip(a, b)]


LENGTHS = ((700, 650, 820, 90), (1300, 40, 777), (512, 513, 300, 301, 299))


@pytest.mark.parametrize("mode", [TrunkMode(fused_relu=True), TrunkMode(bf16_residual=True),
                                  TrunkMode(compile_backend="inductor"),
                                  TrunkMode(bf16_residual=True, compile_backend="inductor")],
                         ids=lambda m: m.describe())
def test_a_variant_runs_near_the_autocast_trunk_on_cuda(mode):
    torch._dynamo.reset()
    encoder = _encoder()
    block = CompiledBlock(mode.compile_backend) if mode.compile_backend else None
    for lengths in LENGTHS:
        base_y, base_g, base_kept = _run(encoder, lambda x, i: packed_trunk(encoder, x, i), lengths)
        y, g, kept = _run(encoder, lambda x, i: run_trunk(encoder, x, i, mode, block), lengths)
        scale = float(base_y.abs().max())
        print(mode.describe(), lengths, "output max diff / scale", float((y - base_y).abs().max()) / scale,
              "min grad cosine", min(_cosines(base_g, g)), "kept bytes", base_kept, kept,
              "bit-equal", torch.equal(y, base_y) and all(torch.equal(a, b) for a, b in zip(base_g, g)))
        assert float((y - base_y).abs().max()) <= 0.05 * scale
        assert min(_cosines(base_g, g)) > 0.99
        if mode.bf16_residual:
            assert kept < base_kept
    if block is not None:
        assert _dynamo_cache_entries(layer_block) == 1, "one graph for every token count"


def test_the_compiled_block_recomputes_under_checkpointing_on_cuda():
    torch._dynamo.reset()
    encoder = _encoder()
    mode = TrunkMode(bf16_residual=True, compile_backend="inductor")
    block = CompiledBlock("inductor")
    for lengths in LENGTHS:
        y, g, _ = _run(encoder, lambda x, i: run_trunk(encoder, x, i, mode, block), lengths)
        y2, g2, kept = _run(encoder, lambda x, i: run_trunk(encoder, x, i, mode, block), lengths, recompute=True)
        assert torch.equal(y, y2)
        assert min(_cosines(g, g2)) > 0.9999
    assert _dynamo_cache_entries(layer_block) == 1


def test_the_fused_relu_is_one_lt_call_near_the_plain_linear_on_cuda():
    """_addmm_activation against relu(F.linear) in bfloat16 on production's
    feed-forward shape: equal within one bfloat16 rounding (printed: whether
    cuBLASLt chose the same algorithm, i.e. equal to the bit)."""
    from wesnoth_ai.trunk_modes import linear_relu
    torch.manual_seed(0)
    x = torch.randn(12_000, 384, device=CUDA, dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(1536, 384, device=CUDA) * 0.05).to(torch.bfloat16).requires_grad_()
    b = (torch.randn(1536, device=CUDA) * 0.05).to(torch.bfloat16).requires_grad_()
    grads = []
    for fn in (lambda: F.relu(F.linear(x, w, b)), lambda: linear_relu(x, w, b)):
        x.grad = w.grad = b.grad = None
        y = fn()
        y.float().sum().backward()
        grads.append((y.detach().float(), x.grad.float(), w.grad.float(), b.grad.float()))
    (y0, *g0), (y1, *g1) = grads
    print("fused relu bit-equal:", torch.equal(y0, y1) and all(torch.equal(a, c) for a, c in zip(g0, g1)))
    assert float((y1 - y0).abs().max()) <= 2 ** -7 * float(y0.abs().max())
    assert min(_cosines(g0, g1)) > 0.999
