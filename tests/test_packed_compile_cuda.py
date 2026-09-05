"""CUDA-only checks of the compiled packed trunk (wesnoth_ai/
packed_trunk.py CompiledPackedTrunk, docs/gpu_forward_design_20260904.md
section 13): bf16 parity of the compiled loop over native bf16 weights
against the eager packed trunk under autocast (same flash kernel), no
host synchronization after warmup, warmup seconds, and GPU / wall
milliseconds per batch of 16 for the eager padded, eager packed and
compiled packed paths. Skipped without CUDA; runs on the box with
`pytest -s tests/test_packed_compile_cuda.py` (the -s shows the numbers).
Random weights at the 15M tier-B shape; one compile shared by the
module's tests."""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CUDA = torch.device("cuda")
FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness")
TIER_B = dict(d_model=384, num_layers=8, num_heads=12, d_ff=1536)   # docs/archive/tier_b_runbook.md:29


@pytest.fixture(scope="module")
def compiled_model():
    """One compile for the module (design section 3.4 expects 40-90 s
    for the dynamic graph, 5-10 s with a warm inductor cache)."""
    from wesnoth_ai.model import WesnothModel
    torch._dynamo.reset()
    torch.manual_seed(1)
    model = WesnothModel(**TIER_B).to(CUDA).eval()
    model.infer_packed_trunk = True
    model.configure_packed_compile()
    t0 = time.perf_counter()
    stats = model.warmup_packed_compile()
    print(f"\nwarmup_packed_compile: {time.perf_counter() - t0:.1f} s wall; {stats}")
    assert stats["active"] and stats["recompiles"] == 0, stats
    return model


def _sizes(B, seed, homogeneous):
    """(U, R, H) per row. Homogeneous: production-like batches around
    1,270 tokens; otherwise tokens anywhere in 600-2,200."""
    rng = random.Random(seed)
    sizes = []
    for _ in range(B):
        H = rng.randint(1150, 1300) if homogeneous else rng.randint(560, 2130)
        sizes.append((rng.randint(10, 60), rng.randint(0, 7), H))
    return sizes


def _streams(sizes, seed=0):
    from wesnoth_ai.model import random_padded_streams
    return random_padded_streams(sizes, TIER_B["d_model"], CUDA, seed=seed)


def _forward(model, streams, *, packed, compiled, bf16=True):
    model.infer_compile_packed = compiled
    with torch.no_grad():
        if not bf16:
            return model.forward_streams(*streams, packed=packed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_streams(*streams, packed=packed).float32()


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


def test_compiled_bf16_matches_eager_packed_bf16(compiled_model):
    """Same flash kernel, same bf16 rounding points (the native copy holds
    the values autocast would cast to); only inductor's fusion and
    reduction order differ. Expected well under the padded-vs-packed
    kernel difference of test_packed_trunk_cuda; a weight-copy or
    layout bug shows as differences of the order of the scale."""
    model = compiled_model
    streams = _streams(_sizes(8, seed=4, homogeneous=False))
    ref = _forward(model, streams, packed=False, compiled=False, bf16=False)
    eager = _forward(model, streams, packed=True, compiled=False)
    comp = _forward(model, streams, packed=True, compiled=True)
    e_ce, e_ef, e_cf = _errors(comp, eager), _errors(eager, ref), _errors(comp, ref)
    _print_errors("compiled packed bf16 vs eager packed bf16", e_ce)
    _print_errors("eager packed bf16 vs fp32 padded", e_ef)
    _print_errors("compiled packed bf16 vs fp32 padded", e_cf)
    for f in FIELDS:
        _, r_ce, scale = e_ce[f]
        assert r_ce < 0.05, (f, e_ce[f])
        assert e_cf[f][0] <= 2 * e_ef[f][0] + 0.01 * max(scale, 1.0), (f, e_cf[f], e_ef[f])
    stats = model.packed_compile_stats()
    assert stats["active"] and stats["recompiles"] == 0 and stats["weights_dtype"] == "bfloat16", stats


def test_compiled_forward_issues_no_host_sync(compiled_model):
    """After warmup a compiled forward (index copy, gathers, the
    inductor graph with its extern flash kernel, heads, float32 casts)
    must not synchronize with the host."""
    model = compiled_model
    streams = _streams(_sizes(4, seed=5, homogeneous=True))
    warm = _forward(model, streams, packed=True, compiled=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = _forward(model, streams, packed=True, compiled=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for f in FIELDS:
        assert torch.allclose(getattr(got, f), getattr(warm, f), rtol=1e-2, atol=1e-2), f
    assert model.packed_compile_stats()["recompiles"] == 0


def _gpu_and_wall_ms(fn, iters=10, warm=3):
    """Medians of the stream time between two events around fn (device
    work plus launch gaps) and of the host wall time including the
    final synchronize."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000.0)
        gpu.append(start.elapsed_time(end))
    return statistics.median(gpu), statistics.median(wall)


def test_ms_per_batch_of_16_eager_padded_eager_packed_compiled_packed(compiled_model):
    model = compiled_model
    sizes = _sizes(16, seed=7, homogeneous=True)
    streams = _streams(sizes)
    total = sum(U + R + H + 2 for U, R, H in sizes)
    rows = [("eager padded", dict(packed=False, compiled=False)),
            ("eager packed", dict(packed=True, compiled=False)),
            ("compiled packed", dict(packed=True, compiled=True))]
    print(f"\nbatch 16, near-homogeneous lengths, {total} tokens:")
    print(f"  {'path':16s} {'GPU ms':>8s} {'wall ms':>8s}")
    for name, kw in rows:
        gpu_ms, wall_ms = _gpu_and_wall_ms(lambda: _forward(model, streams, **kw))
        print(f"  {name:16s} {gpu_ms:8.2f} {wall_ms:8.2f}")
        assert gpu_ms > 0 and wall_ms > 0
    stats = model.packed_compile_stats()
    print(f"  packed_compile_stats: {stats}")
    assert stats["active"] and stats["recompiles"] == 0, stats
