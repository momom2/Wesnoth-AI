"""The HIP kernels (wesnoth_ai/hip_kernels.py): their dispatch off ROCm,
and on a ROCm GPU the varlen attention against
F.scaled_dot_product_attention per segment, the fallback when a gradient
is wanted, the fused residual add and layer norm against F.layer_norm,
and the packed fp16 trunk (the model's fused loop) against the padded
fp32 trunk over the real slots. The GPU checks skip without a ROCm build and a GPU the
kernel serves (CI has neither); on the local RX 6600 they run under the
ROCm venv (training/metrics/local_gpu_20261028/README.md)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.hip_kernels import (  # noqa: E402
    HEAD_DIMS, hip_add_layer_norm, hip_add_layer_norm_applies, hip_varlen_applies, hip_varlen_attention,
)
from wesnoth_ai.packed_trunk import (  # noqa: E402
    PackedIndex, flash_varlen_applies, packed_attention, varlen_attention_applies,
)

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits")


def test_dispatch_off_rocm():
    """No HIP kernel on a CPU device or off a ROCm build; no flash kernel
    on a ROCm build."""
    assert not hip_varlen_applies(torch.device("cpu"), torch.float16)
    assert not varlen_attention_applies(torch.device("cpu"), torch.float16)
    assert not hip_add_layer_norm_applies(torch.device("cpu"), 384)
    if torch.version.hip is None:
        assert not hip_varlen_applies(torch.device("cuda"), torch.float16)
    else:
        assert not flash_varlen_applies(torch.device("cuda"), torch.float16)


def _rocm_kernel() -> bool:
    return (torch.version.hip is not None and torch.cuda.is_available()
            and hip_varlen_applies(torch.device("cuda"), torch.float16, 32))


rocm = pytest.mark.skipif(not _rocm_kernel(), reason="needs a ROCm GPU the HIP kernel serves")
DEV = torch.device("cuda")


def _segments(lens, heads, d, seed):
    """Packed q, k, v as the trunk's in-projection makes them (unbind views
    of [total, 3, heads, d]) and the int32 offsets."""
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    g = torch.Generator().manual_seed(seed)
    qkv = (torch.randn(cu[-1], 3, heads, d, generator=g) * 2).to(DEV, torch.float16)
    q, k, v = qkv.unbind(1)
    return q, k, v, cu, torch.tensor(cu, dtype=torch.int32, device=DEV)


def _reference(q, k, v, cu):
    outs = []
    for lo, hi in zip(cu[:-1], cu[1:]):
        seg = [t[lo:hi].float().transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
        outs.append(F.scaled_dot_product_attention(*seg)[0].transpose(0, 1))
    return torch.cat(outs)


@rocm
@pytest.mark.parametrize("d", HEAD_DIMS)
def test_kernel_matches_sdpa(d):
    """Segments of 1 token, on and around the tile and block sizes, and up
    to the corpus's longest (692 tokens with the memory, 2026-10-28)."""
    rng = random.Random(d)
    lens = [1, 31, 32, 33, 63, 64, 65, 129, 692] + [rng.randint(141, 692) for _ in range(8)]
    rng.shuffle(lens)
    q, k, v, cu, cu_dev = _segments(lens, 384 // d, d, seed=d)
    out = hip_varlen_attention(q, k, v, cu_dev, max(lens))
    assert out.shape == q.shape and out.is_contiguous()
    err = (out.float() - _reference(q, k, v, cu)).abs().max().item()
    assert err < 1e-2, err


@rocm
def test_kernel_copies_a_foreign_layout():
    """q, k, v whose heads are not adjacent go through a contiguous copy
    and give the same result."""
    q, k, v, cu, cu_dev = _segments([5, 70, 300], 12, 32, seed=3)
    qt = q.transpose(0, 1).contiguous().transpose(0, 1)       # [total, heads, d], token stride 32
    assert qt.stride(1) != 32
    out = hip_varlen_attention(qt, k, v, cu_dev, 300)
    assert (out.float() - _reference(q, k, v, cu)).abs().max().item() < 1e-2


@rocm
def test_gradient_falls_back_to_sdpa():
    """With a gradient wanted the packed attention takes the per-segment
    SDPA, which has a backward."""
    q, k, v, cu, cu_dev = _segments([7, 40], 12, 32, seed=4)
    q = q.detach().requires_grad_(True)
    index = PackedIndex(src=None, kind=None, actor=None, hex=None, unit=None, glob=None,
                        cu_seqlens=cu_dev, cu_host=cu, max_len=40)
    out = packed_attention(q, k, v, index)
    assert out.grad_fn is not None
    out.float().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


@rocm
@pytest.mark.parametrize("d", [128, 384])
def test_add_layer_norm_matches_torch(d):
    """float32 to its rounding (the order of the sums differs), and the
    fp16 copy within one fp16 step of the reference's rounding."""
    g = torch.Generator().manual_seed(d)
    x = (torch.randn(1001, d, generator=g) * 3).to(DEV)
    y = torch.randn(1001, d, generator=g).to(DEV, torch.float16)
    w, b = (torch.rand(d, generator=g) + 0.5).to(DEV), torch.randn(d, generator=g).to(DEV)
    assert hip_add_layer_norm_applies(DEV, d)
    o32, o16 = hip_add_layer_norm(x, y, w, b, 1e-5)
    ref = F.layer_norm(x + y.float(), (d,), w, b, 1e-5)
    assert (o32 - ref).abs().max().item() < 1e-4
    assert o16.dtype == torch.float16
    assert (o16.float() - ref.half().float()).abs().max().item() <= 2 ** -10 * ref.abs().max().item()


@rocm
def test_packed_fp16_trunk_matches_padded_fp32():
    """A small random network (head_dim 32, width 128): the packed trunk in
    fp16 (the fused loop: the HIP attention and add-and-norm) against the
    padded trunk in fp32, over every sample's real slots."""
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(5)
    model = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=256).to(DEV).eval()
    calls = []
    fused = model._run_hip_fp16_trunk
    model._run_hip_fp16_trunk = lambda x, index: calls.append(1) or fused(x, index)
    rng = random.Random(6)
    sizes = [(rng.randint(1, 30), rng.randint(0, 7), rng.randint(100, 600)) for _ in range(12)]
    g = torch.Generator(device=DEV).manual_seed(7)
    B, d = len(sizes), 128
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))

    def stream(n):
        return torch.randn(B, n, d, generator=g, device=DEV)
    streams = (stream(H_max), stream(U_max), stream(R_max), stream(1), stream(1), sizes)
    with torch.no_grad():
        ref = model.forward_streams(*streams, packed=False)
        with torch.autocast("cuda", dtype=torch.float16):
            got = model.forward_streams(*streams, packed=True).float32()
    assert calls, "the fp16 forward did not take the fused loop"
    for f in FIELDS:
        diff = scale = 0.0
        for x, y in zip(got.samples(), ref.samples()):
            xf, yf = getattr(x, f), getattr(y, f)
            fin = torch.isfinite(yf)
            assert torch.equal(fin, torch.isfinite(xf)), f
            if fin.any():
                diff = max(diff, (xf - yf).abs()[fin].max().item())
                scale = max(scale, yf.abs()[fin].max().item())
        assert diff <= 0.02 * max(scale, 1.0), (f, diff, scale)
