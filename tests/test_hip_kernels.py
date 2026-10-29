"""The HIP kernels (wesnoth_ai/hip_kernels.py, wesnoth_ai/hip_attention_grad.py):
their dispatch off ROCm, and on a ROCm GPU the varlen attention against
F.scaled_dot_product_attention per segment, its backward against SDPA's
fp32 autograd and its dispatch when a gradient is wanted, the fused
residual add and layer norm against F.layer_norm, and the packed fp16
trunk (the model's fused loop; packed_trunk with the backward when a
gradient is wanted) against the padded fp32 trunk. The GPU checks skip
without a ROCm build and a GPU the kernel serves (CI has neither); on the
local RX 6600 they run under the ROCm venv
(training/metrics/local_gpu_20261028/README.md)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.hip_attention_grad import (  # noqa: E402
    attention_errors, hip_varlen_attention_trainable, hip_varlen_backward_applies,
)
from wesnoth_ai.hip_kernels import (  # noqa: E402
    HEAD_DIMS, hip_add_layer_norm, hip_add_layer_norm_applies, hip_varlen_applies, hip_varlen_attention,
    hip_varlen_attention_lse,
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
    assert not hip_varlen_backward_applies(torch.device("cpu"), torch.float16)
    if torch.version.hip is None:
        assert not hip_varlen_applies(torch.device("cuda"), torch.float16)
        assert not hip_varlen_backward_applies(torch.device("cuda"), torch.float16)
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


def _index(cu, cu_dev):
    return PackedIndex(src=None, kind=None, actor=None, hex=None, unit=None, glob=None,
                       cu_seqlens=cu_dev, cu_host=cu, max_len=max(b - a for a, b in zip(cu[:-1], cu[1:])))


# The backward's checks (wesnoth_ai/hip_attention_grad.py): each tensor's
# largest error over its reference's largest magnitude. fp16 operands and
# fp32 sums read 3e-4 to 7e-4 on these problems (2026-10-29,
# training/metrics/hip_backward_20261029/README.md); a wrong index, sign or
# tile reads orders of magnitude above.
GRAD_REL = 5e-3


@rocm
@pytest.mark.parametrize("d", HEAD_DIMS)
def test_backward_matches_sdpa(d):
    """The forward with its log-sum-exp and dq, dk, dv against fp32 SDPA's
    autograd, for segments of 1 token, ending inside and on the dq kernel's
    key tiles (16) and query blocks (64 rows at head_dim 32), the dkdv
    kernel's key blocks (32) and query tiles, and the corpus's longest (692)."""
    rng = random.Random(10 + d)
    lens = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 97, 129, 692] + [rng.randint(141, 692) for _ in range(4)]
    rng.shuffle(lens)
    q, k, v, cu, cu_dev = _segments(lens, 384 // d, d, seed=20 + d)
    g = torch.Generator().manual_seed(30 + d)
    weight = torch.randn(q.shape, generator=g).to(DEV)
    assert hip_varlen_backward_applies(DEV, torch.float16, d)
    errors = attention_errors(q, k, v, cu, weight, lambda *t: packed_attention(*t, _index(cu, cu_dev)))
    for name, e in errors.items():
        assert e["rel"] < GRAD_REL, (name, errors)


@rocm
def test_forward_lse_matches_logsumexp():
    """The forward's log-sum-exp is ln sum_j exp(q.k_j / sqrt(head_dim)) per
    row and head, and its output is the inference kernel's, bit for bit."""
    lens = [1, 17, 300]
    q, k, v, cu, cu_dev = _segments(lens, 12, 32, seed=5)
    out, lse, _ = hip_varlen_attention_lse(q, k, v, cu_dev, max(lens))
    assert torch.equal(out, hip_varlen_attention(q, k, v, cu_dev, max(lens)))
    for lo, hi in zip(cu[:-1], cu[1:]):
        qs, ks = (t[lo:hi].float().transpose(0, 1) for t in (q, k))        # [heads, n, d]
        ref = torch.logsumexp(qs @ ks.transpose(1, 2) / 32 ** 0.5, dim=-1)  # [heads, n]
        assert (lse[:, lo:hi] - ref).abs().max().item() < 1e-3


@rocm
def test_gradient_takes_the_hip_backward():
    """With a gradient wanted, fp16 packed attention takes the HIP kernel
    with its backward; fp32 keeps the per-segment SDPA."""
    q, k, v, cu, cu_dev = _segments([7, 40], 12, 32, seed=4)
    q = q.detach().requires_grad_(True)
    out = packed_attention(q, k, v, _index(cu, cu_dev))
    assert "HipVarlenAttention" in type(out.grad_fn).__name__
    out.float().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    q32 = q.detach().float().requires_grad_(True)
    out32 = packed_attention(q32, k.float(), v.float(), _index(cu, cu_dev))
    assert "HipVarlenAttention" not in type(out32.grad_fn).__name__


@rocm
def test_gradient_rows_past_the_offsets_are_zero():
    """Rows past the offsets' last (the static-shape path's spare rows) get
    zero gradients, whether or not the caller gives the rows on the host."""
    q, k, v, cu, cu_dev = _segments([5, 30, 9], 12, 32, seed=6)      # 44 rows
    used = cu[-1] - 9
    cu_short = torch.tensor(cu[:-1], dtype=torch.int32, device=DEV)   # the last 9 rows are spare
    for rows in (None, used):
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
        out = hip_varlen_attention_trainable(*leaves, cu_short, 30, rows=rows)
        out[:used].float().sum().backward()
        for t in leaves:
            assert torch.isfinite(t.grad[:used]).all()
            assert (t.grad[used:] == 0).all()


@rocm
def test_packed_fp16_trunk_gradients_match_padded_fp32():
    """With a gradient wanted in eval mode (the gradient pass's case), the
    packed fp16 forward skips the fused inference loop and runs
    packed_trunk with the HIP backward; the parameters' gradients of a
    loss over every head match the padded fp32 trunk's."""
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(8)
    model = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=256).to(DEV).eval()
    fused_calls = []
    fused = model._run_hip_fp16_trunk
    model._run_hip_fp16_trunk = lambda x, index: fused_calls.append(1) or fused(x, index)
    rng = random.Random(9)
    sizes = [(rng.randint(1, 30), rng.randint(0, 7), rng.randint(100, 600)) for _ in range(8)]
    g = torch.Generator(device=DEV).manual_seed(10)
    B, d = len(sizes), 128
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))

    def stream(n):
        return torch.randn(B, n, d, generator=g, device=DEV)
    streams = (stream(H_max), stream(U_max), stream(R_max), stream(1), stream(1), sizes)

    def gradients(packed, fp16):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            out = model.forward_streams(*streams, packed=packed)
        loss = 0.0
        for x in out.float32().samples():
            for f in FIELDS:
                t = getattr(x, f)
                loss = loss + t[torch.isfinite(t)].pow(2).mean()
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.encoder.named_parameters() if p.grad is not None}
    ref = gradients(False, False)
    got = gradients(True, True)
    assert not fused_calls, "a forward that wants a gradient took the fused inference loop"
    assert set(got) == set(ref) and ref
    worst = max(((got[n] - ref[n]).abs().max() / ref[n].abs().max()).item() for n in ref)
    assert worst < 0.05, worst


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
