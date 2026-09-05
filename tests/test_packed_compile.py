"""Compiled packed trunk (wesnoth_ai/packed_trunk.py CompiledPackedTrunk,
design note docs/gpu_forward_design_20260904.md section 13), CPU tier:
the compiled layer loop equals the eager packed trunk on three batch
compositions and compiles once (dynamic token count, offsets length and
max_len: no recompile on the third shape); the native-dtype weight copy
follows load_state_dict; a failing compile falls back to the eager loop
with one WARNING; a recompile and dynamo giving up are detected; the
switch requires the packed trunk. The inductor variant needs a C++
compiler on PATH (MSVC's cl, gcc or clang) and is skipped without one;
the aot_eager variant covers the dynamo side (tracing, dynamic shapes,
the custom-op boundary, guards) everywhere."""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness")
D = 32
CPU = torch.device("cpu")
LOGGER = "wesnoth_ai.packed_trunk"
# Three batch compositions (U, R, H per row): different row counts,
# token totals and longest segments.
SHAPES = ([(3, 2, 40), (1, 0, 25), (5, 3, 33), (2, 1, 40)],
          [(2, 0, 30), (3, 0, 12), (4, 2, 20)],
          [(2, 1, 18), (1, 2, 44)])


def _has_cpp_compiler() -> bool:
    return any(shutil.which(c) for c in ("cl", "g++", "gcc", "clang++", "clang"))


BACKENDS = ["aot_eager", pytest.param("inductor", marks=pytest.mark.skipif(
    not _has_cpp_compiler(), reason="inductor's CPU backend needs a C++ compiler on PATH"))]


@pytest.fixture(autouse=True)
def _fresh_dynamo_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _model(d_model=D):
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(5)
    model = WesnothModel(d_model=d_model, num_layers=2, num_heads=2, d_ff=64).eval()
    model.infer_packed_trunk = True
    return model


def _streams(sizes, seed=0):
    from wesnoth_ai.model import random_padded_streams
    return random_padded_streams(sizes, D, CPU, seed=seed)


def _eager_and_compiled(model, streams):
    """The eager packed forward and the compiled one on the same streams."""
    with torch.no_grad():
        model.infer_compile_packed = False
        eager = model.forward_streams(*streams, packed=True)
        model.infer_compile_packed = True
        compiled = model.forward_streams(*streams, packed=True)
    return eager, compiled


def _assert_samples_equal(a, b, atol, rtol):
    assert a.sizes == b.sizes
    for x, y in zip(a.samples(), b.samples()):
        assert torch.equal(x.actor_kind, y.actor_kind)
        for f in FIELDS:
            assert getattr(x, f).shape == getattr(y, f).shape, f
            assert torch.allclose(getattr(x, f), getattr(y, f), atol=atol, rtol=rtol), f


def _our_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and r.levelno >= logging.WARNING]


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_packed_matches_eager_and_compiles_once(backend):
    model = _model()
    model.configure_packed_compile(backend=backend)
    tol = dict(atol=1e-4, rtol=1e-3) if backend == "inductor" else dict(atol=1e-5, rtol=1e-4)
    for i, sizes in enumerate(SHAPES):
        eager, compiled = _eager_and_compiled(model, _streams(sizes, seed=i))
        _assert_samples_equal(eager, compiled, **tol)
        assert model.packed_compile_active, model.packed_compile_stats()
    stats = model.packed_compile_stats()
    print(f"\n{backend}: warmup {stats['warmup_seconds']:.1f} s, "
          f"{stats['cache_entries']} cache entry")
    assert stats["recompiles"] == 0 and stats["cache_entries"] == 1, stats
    assert stats["weights_dtype"] == "float32" and stats["fallback_reason"] is None


def test_warmup_helper_compiles_before_serving():
    model = _model()
    model.configure_packed_compile(backend="aot_eager")
    stats = model.warmup_packed_compile()
    assert stats["active"] and stats["recompiles"] == 0 and stats["warmup_seconds"] > 0, stats
    eager, compiled = _eager_and_compiled(model, _streams(SHAPES[2], seed=9))
    _assert_samples_equal(eager, compiled, atol=1e-5, rtol=1e-4)
    assert model.packed_compile_stats()["recompiles"] == 0


def test_compiled_trunk_follows_published_weights():
    model = _model()
    model.configure_packed_compile(backend="aot_eager")
    streams = _streams(SHAPES[0], seed=3)
    _, before = _eager_and_compiled(model, streams)      # compiles, copies the weights
    state = {k: v.clone() for k, v in model.state_dict().items()}
    # Noise, not a constant: linear1 reads a LayerNorm output whose
    # features sum to zero at init, so a constant shift would be inert.
    state["encoder.layers.0.linear1.weight"] += 0.5 * torch.randn_like(
        state["encoder.layers.0.linear1.weight"])
    model.load_state_dict(state)
    eager, after = _eager_and_compiled(model, streams)
    _assert_samples_equal(eager, after, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(before.actor_logits, after.actor_logits)
    assert model.packed_compile_stats()["recompiles"] == 0


def test_failed_compile_falls_back_to_the_eager_loop_with_one_warning(caplog):
    model = _model()

    def broken_backend(gm, example_inputs):
        raise RuntimeError("no backend here")
    model.configure_packed_compile(backend=broken_backend)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        eager, got = _eager_and_compiled(model, _streams(SHAPES[0], seed=4))
        eager2, got2 = _eager_and_compiled(model, _streams(SHAPES[1], seed=5))
    _assert_samples_equal(eager, got, atol=1e-5, rtol=1e-4)
    _assert_samples_equal(eager2, got2, atol=1e-5, rtol=1e-4)
    stats = model.packed_compile_stats()
    assert not model.packed_compile_active and "no backend here" in stats["fallback_reason"]
    assert len(_our_warnings(caplog)) == 1, _our_warnings(caplog)


def test_recompile_and_dynamo_giving_up_are_detected(caplog):
    """The compiled function is shared by every model of the process: a
    second architecture fails the weight-shape guards and recompiles
    (counted, one WARNING); a third one over the cache-size limit makes
    dynamo give up, and the eager loop serves with one more WARNING."""
    from wesnoth_ai.model import ActorKind, TokenKind
    from wesnoth_ai.packed_trunk import (
        CompiledPackedTrunk, PackedTrunkWeights, build_packed_layout,
    )
    trunk = CompiledPackedTrunk(torch.nn.functional.relu, 1e-5, backend="aot_eager")
    sizes = SHAPES[0]
    index = build_packed_layout(sizes, 40, 5, 3, TokenKind, ActorKind).to_device(CPU)
    total = index.cu_host[-1]

    def run(d_model):
        model = _model(d_model)
        weights = PackedTrunkWeights.build(model.encoder, torch.float32, CPU, 0)
        x = torch.randn(total, d_model)
        with torch.no_grad():
            return trunk.run(x, index, weights), trunk._eager(
                x, index.cu_seqlens, index.max_len, weights.layers, weights.final_norm)

    got, ref = run(D)
    assert trunk.active and trunk.recompiles == 0
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        got48, ref48 = run(48)
        assert trunk.active and trunk.recompiles == 1
        with torch._dynamo.config.patch(cache_size_limit=2):
            got16, ref16 = run(16)
    assert not trunk.active and trunk.fallback_reason is not None, trunk.stats()
    for a, b in ((got, ref), (got48, ref48), (got16, ref16)):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    warnings = _our_warnings(caplog)
    assert len(warnings) == 2 and "recompiled" in warnings[0] and "inactive" in warnings[1], warnings


def test_compile_switch_requires_the_packed_trunk():
    model = _model()
    model.infer_packed_trunk = False
    with pytest.raises(ValueError, match="infer_packed_trunk"):
        model.configure_packed_compile(backend="aot_eager")
    model.infer_compile_packed = True
    with pytest.raises(ValueError, match="infer_packed_trunk"), torch.no_grad():
        model.forward_streams(*_streams(SHAPES[1]))
