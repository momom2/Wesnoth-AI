"""CUDA-only checks of the packed varlen trunk (wesnoth_ai/
packed_trunk.py, docs/gpu_forward_design_20260904.md section 12):
bookkeeping parity in fp32 on the device (per-segment SDPA, no bf16
rounding), bf16 parity of the flash varlen path against the padded
mem-efficient path with every difference printed, no host
synchronization inside the packed forward, and GPU milliseconds per
batch of 16 for both trunks. Skipped without CUDA; runs on the box with
`pytest -s tests/test_packed_trunk_cuda.py` (the -s shows the numbers).
Random weights at the 15M tier-B shape: parity and timing do not need
a checkpoint."""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CUDA = torch.device("cuda")
FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness")
TIER_B = dict(d_model=384, num_layers=8, num_heads=12, d_ff=1536)   # docs/archive/tier_b_runbook.md:29


def _model(seed=1):
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(seed)
    return WesnothModel(**TIER_B).to(CUDA).eval()


def _sizes(B, seed, homogeneous):
    """(U, R, H) per row. Homogeneous: production-like batches around
    1,270 tokens (padding ratio about 1.1); otherwise tokens anywhere in
    600-2,200 (padding ratio about 1.5)."""
    rng = random.Random(seed)
    sizes = []
    for _ in range(B):
        H = rng.randint(1150, 1300) if homogeneous else rng.randint(560, 2130)
        sizes.append((rng.randint(10, 60), rng.randint(0, 7), H))
    return sizes


def _streams(sizes, seed=0):
    g = torch.Generator(device=CUDA).manual_seed(seed)
    B, d = len(sizes), TIER_B["d_model"]
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))

    def stream(n):
        return torch.randn(B, n, d, generator=g, device=CUDA)
    return stream(H_max), stream(U_max), stream(R_max), stream(1), stream(1), list(sizes)


def _forward(model, streams, packed, bf16):
    with torch.no_grad():
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.forward_streams(*streams, packed=packed).float32()
        return model.forward_streams(*streams, packed=packed)


def _errors(a, b):
    """Per field over the real slots of every sample: max |a - b|, that
    maximum divided by the reference's largest magnitude, the magnitude."""
    out = {}
    for f in FIELDS:
        diff = scale = 0.0
        for x, y in zip(a.samples(), b.samples()):
            xf, yf = getattr(x, f), getattr(y, f)
            if xf.numel():
                diff = max(diff, (xf - yf).abs().max().item())
                scale = max(scale, yf.abs().max().item())
        out[f] = (diff, diff / max(scale, 1e-6), scale)
    return out


def _print_errors(title, errs):
    print(f"\n{title}")
    for f, (d, r, s) in errs.items():
        print(f"  {f:14s} max_abs {d:.3e}   max_abs/scale {r:.3e}   (scale {s:.3g})")


def test_packed_fp32_equals_padded_fp32_on_the_device():
    """fp32 on the device takes the per-segment SDPA fallback: a
    bookkeeping check with no bf16 rounding in it."""
    model = _model()
    streams = _streams(_sizes(6, seed=3, homogeneous=False))
    padded = _forward(model, streams, packed=False, bf16=False)
    packed = _forward(model, streams, packed=True, bf16=False)
    errs = _errors(packed, padded)
    _print_errors("fp32 packed (segment SDPA) vs fp32 padded", errs)
    for f, (d, r, _) in errs.items():
        assert r < 5e-3 and d < 2e-2, (f, d, r)
    for x, y in zip(packed.samples(), padded.samples()):
        assert torch.equal(x.actor_kind, y.actor_kind)


def test_packed_bf16_matches_padded_bf16_within_kernel_rounding():
    """Under bf16 autocast the packed path runs the flash varlen kernel
    and the padded path the mem-efficient kernel with a bias; both
    round differently from the fp32 reference. Expected: a few 1e-2 of
    each field's scale, and the packed path no further from fp32 than
    the padded bf16 path already is (design note section 12). A layout
    bug shows as differences of the order of the scale itself."""
    model = _model()
    streams = _streams(_sizes(8, seed=4, homogeneous=False))
    ref = _forward(model, streams, packed=False, bf16=False)
    padded = _forward(model, streams, packed=False, bf16=True)
    packed = _forward(model, streams, packed=True, bf16=True)
    e_pp, e_pad, e_pk = _errors(packed, padded), _errors(padded, ref), _errors(packed, ref)
    _print_errors("bf16 packed vs bf16 padded", e_pp)
    _print_errors("bf16 padded vs fp32 padded", e_pad)
    _print_errors("bf16 packed vs fp32 padded", e_pk)
    for f in FIELDS:
        _, r_pp, scale = e_pp[f]
        assert r_pp < 0.1, (f, e_pp[f])
        assert e_pk[f][0] <= 2 * e_pad[f][0] + 0.01 * max(scale, 1.0), (f, e_pk[f], e_pad[f])


def test_packed_forward_issues_no_host_sync():
    """torch raises on any implicit sync while the debug mode is
    "error"; the packed forward (index copy, gathers, trunk, heads,
    float32 casts) must issue none. The first call warms the caching
    allocators (a cudaMalloc / cudaHostAlloc on a miss synchronizes
    outside torch's instrumentation)."""
    model = _model()
    streams = _streams(_sizes(4, seed=5, homogeneous=True))
    warm = _forward(model, streams, packed=True, bf16=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = _forward(model, streams, packed=True, bf16=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for f in FIELDS:
        assert torch.allclose(getattr(got, f), getattr(warm, f), rtol=1e-2, atol=1e-2), f


def _gpu_ms(fn, iters=10, warm=3):
    """Median stream wall time between two events around fn: device
    work plus any gap where the stream waited for the host."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@pytest.mark.parametrize("homogeneous", [True, False])
def test_gpu_ms_per_batch_of_16_padded_vs_packed(homogeneous):
    model = _model()
    sizes = _sizes(16, seed=6 + int(homogeneous), homogeneous=homogeneous)
    streams = _streams(sizes)
    total = sum(U + R + H + 2 for U, R, H in sizes)
    L = streams[0].size(1) + streams[1].size(1) + streams[2].size(1) + 2
    padded_ms = _gpu_ms(lambda: _forward(model, streams, packed=False, bf16=True))
    packed_ms = _gpu_ms(lambda: _forward(model, streams, packed=True, bf16=True))
    print(f"\nbatch 16, {'near-homogeneous' if homogeneous else 'mixed'} lengths: "
          f"{total} tokens, padded to {16 * L} (ratio {16 * L / total:.2f}); "
          f"GPU ms per batch padded {padded_ms:.2f}, packed {packed_ms:.2f} "
          f"(padded/packed {padded_ms / packed_ms:.2f})")
    assert padded_ms > 0 and packed_ms > 0
