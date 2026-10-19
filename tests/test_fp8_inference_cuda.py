"""fp8 inference on CUDA (wesnoth_ai/fp8_inference.py, WesnothModel.infer_fp8):
the products run torch._scaled_mm on the layouts it takes, eager and
compiled by inductor with no recompile over three batch shapes; the served
outputs stay near the bf16 stream's, the compiled loop near the eager one,
and no static scale clips, on random weights at the reference's shape and
on the reference's own weights when its checkpoint is on the machine; the
device milliseconds of a production-shaped batch, fp8 against the bf16
stream, both compiled (printed with -s). Skipped without a GPU of compute
capability 8.9 or more (CI, the laptop); run by scripts/fp8_inference_box.sh
as the gate of its fp8 arms."""
from __future__ import annotations

import random
import statistics
import time

import pytest
import torch

from wesnoth_ai import fp8_inference as fi
from wesnoth_ai.fp8_linear import fp8_problem

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or fp8_problem(torch.device("cuda")) is not None,
                                reason="needs CUDA of compute capability 8.9 or more")
CUDA = torch.device("cuda")
FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits")
REFERENCE = dict(d_model=384, num_layers=8, num_heads=12, d_ff=1536, observation_parity=True, memory_slots=64)


@pytest.fixture(scope="module")
def model():
    from wesnoth_ai.model import WesnothModel
    torch._dynamo.reset()
    torch.manual_seed(1)
    m = WesnothModel(**REFERENCE).to(CUDA).eval()
    m.infer_packed_trunk = True
    return m


def _sizes(B, seed):
    """(U, R, H) per row around production's 400 tokens a leaf."""
    rng = random.Random(seed)
    return [(rng.randint(4, 40), rng.randint(0, 7), rng.randint(250, 650)) for _ in range(B)]


def _inputs(model, sizes, seed):
    """A batch's streams and memory, drawn from `seed`."""
    from wesnoth_ai.model import random_padded_streams
    torch.manual_seed(seed)
    extras = model._warmup_extras(len(sizes), CUDA)
    return random_padded_streams(sizes, model.d_model, CUDA, seed=seed), extras


def _run(model, inputs, **switches):
    streams, extras = inputs
    for k, v in switches.items():
        setattr(model, k, v)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_streams(*streams, packed=True, **extras).float32()
    finally:
        for k in switches:
            setattr(model, k, False)


def _forward(model, sizes, seed, **switches):
    return _run(model, _inputs(model, sizes, seed), **switches)


def _relative(a, b):
    """Per field: max |a - b| over the samples' real slots divided by b's largest magnitude."""
    out = {}
    for f in FIELDS:
        diff = scale = 0.0
        for x, y in zip(a.samples(), b.samples()):
            xf, yf = getattr(x, f), getattr(y, f)
            if xf.numel():
                diff = max(diff, float((xf - yf).abs().max()))
                scale = max(scale, float(yf.abs().max()))
        out[f] = diff / max(scale, 1e-6)
    return out


def test_the_eager_fp8_trunk_runs_scaled_mm_near_the_bf16_stream(model, monkeypatch):
    calls = []
    real = torch._scaled_mm
    monkeypatch.setattr(torch, "_scaled_mm", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    sizes = _sizes(16, seed=3)
    bf16 = _forward(model, sizes, 3, infer_bf16_residual=True)
    fp8 = _forward(model, sizes, 3, infer_fp8=True)
    assert len(calls) == 4 * REFERENCE["num_layers"]
    errors = _relative(fp8, bf16)
    print("\neager fp8 against the eager bf16 stream, max abs / scale:", errors)
    assert max(errors.values()) < 0.1, errors


def test_the_compiled_fp8_loop_runs_near_the_eager_one_with_one_graph(model, monkeypatch):
    seen = []
    real = fi.static_cast

    def spy(t, scale):
        seen.append(float(t.float().abs().max()) * float(scale))
        return real(t, scale)
    shapes = [_sizes(24, seed=s) for s in (4, 5, 6)]
    model.configure_packed_compile()
    compiled = [_forward(model, sizes, 10 + i, infer_fp8=True, infer_compile_packed=True)
                for i, sizes in enumerate(shapes)]
    model.infer_fp8 = True
    stats = model.packed_compile_stats()
    model.infer_fp8 = False
    print("\nfp8 compile:", stats)
    assert stats["active"] and stats["recompiles"] == 0 and stats["fallback_reason"] is None, stats
    monkeypatch.setattr(fi, "static_cast", spy)
    for i, sizes in enumerate(shapes):
        eager = _forward(model, sizes, 10 + i, infer_fp8=True)
        errors = _relative(compiled[i], eager)
        print("compiled fp8 against eager fp8, max abs / scale:", errors)
        assert max(errors.values()) < 0.05, errors
    assert seen and max(seen) <= fi.E4M3_MAX, max(seen)


def _reference_checkpoint():
    from tools import reference_player
    path = reference_player.local_path()
    return path if path.exists() else None


@pytest.mark.skipif(_reference_checkpoint() is None, reason="the reference checkpoint is not on this machine "
                                                           "(tools/reference_player.py --ensure)")
def test_the_reference_checkpoint_runs_fp8_near_the_bf16_stream_without_clipping(monkeypatch):
    """The reference's own weights (the box fetches them before its tests):
    no statically scaled input over 448 before its cast, and eager fp8
    within 0.25 of the eager bf16 stream's largest output on
    production-shaped batches, a crash barrier (a scale off by a factor of
    two moves an output by its own size; rounding moves it by hundredths);
    the errors and the scales printed (with -s) for the records. The play
    check, not this bound, says whether the rounding plays."""
    from tools.eval_players import _load_policy
    torch._dynamo.reset()
    model = _load_policy(_reference_checkpoint(), CUDA, label="reference", infer_bf16=True)._inference_model
    model.eval()
    model.infer_packed_trunk = True
    print("\nreference fp8 scales:", model.fp8_weights(CUDA).summary())
    seen = []
    real = fi.static_cast

    def spy(t, scale):
        seen.append(float(t.float().abs().max()) * float(scale))
        return real(t, scale)
    monkeypatch.setattr(fi, "static_cast", spy)
    for seed in (21, 22):
        sizes = _sizes(32, seed=seed)
        errors = _relative(_forward(model, sizes, seed, infer_fp8=True),
                           _forward(model, sizes, seed, infer_bf16_residual=True))
        print("reference eager fp8 against the eager bf16 stream, max abs / scale:", errors)
        assert max(errors.values()) < 0.25, errors
    print("largest statically scaled input over 448:", max(seen) / fi.E4M3_MAX)
    assert seen and max(seen) <= fi.E4M3_MAX, max(seen)


def _device_ms(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    gpu = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        gpu.append(start.elapsed_time(end))
    return statistics.median(gpu)


def test_the_device_milliseconds_of_a_production_batch(model):
    """56 leaves, about 23,000 tokens (round 5's mean batch): the compiled
    bf16 stream and the compiled fp8 loop, device time per forward."""
    model.configure_packed_compile()
    sizes = _sizes(56, seed=7)
    tokens = sum(U + R + H for U, R, H in sizes)
    rows = {}
    for name, switches in (("bf16 stream", {"infer_bf16_residual": True}), ("fp8", {"infer_fp8": True})):
        t0 = time.perf_counter()
        inputs = _inputs(model, sizes, 7)
        rows[name] = _device_ms(lambda: _run(model, inputs, infer_compile_packed=True, **switches))
        print(f"\n{name}: {rows[name]:.2f} device ms a batch of 56 leaves, {tokens} tokens "
              f"({time.perf_counter() - t0:.1f} s with its compile)")
    print(f"fp8 against the bf16 stream: {rows['fp8'] / rows['bf16 stream']:.3f}")
    assert all(ms > 0 for ms in rows.values())
