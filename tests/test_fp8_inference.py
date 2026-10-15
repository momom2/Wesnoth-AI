"""fp8 inference in the packed trunk (wesnoth_ai/fp8_inference.py,
WesnothModel.infer_fp8), on the CPU's emulation of the fp8 products: the
served outputs stay near the bf16 stream's and are not them; the compiled
loop is the eager loop to the bit and compiles once; every static input
scale keeps its input at or under 448, and each bound is attained by an
input built to reach it; a static cast saturates where a bound would fail;
the copy follows published weights; the first layer's inputs take their
batch's scale and the others the weights'; the loop refuses what it cannot
run."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai import fp8_inference as fi

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits")
D = 32
CPU = torch.device("cpu")
SHAPES = ([(3, 2, 40), (1, 0, 25), (5, 3, 33), (2, 1, 40)],
          [(2, 0, 30), (3, 0, 12), (4, 2, 20)],
          [(2, 1, 18), (1, 2, 44)])


@pytest.fixture(autouse=True)
def _fresh_dynamo_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _model(seed=8):
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(seed)
    model = WesnothModel(d_model=D, num_layers=2, num_heads=2, d_ff=64, observation_parity=True,
                         memory_slots=8).eval()
    model.infer_packed_trunk = True
    return model


def _forward(model, sizes, seed, **switches):
    """The packed forward under the CPU's bf16 autocast with `switches` set,
    reset afterwards."""
    from wesnoth_ai.model import random_padded_streams
    torch.manual_seed(seed)                  # the sightings' draw
    extras = model._warmup_extras(len(sizes), CPU)
    streams = random_padded_streams(sizes, D, CPU, seed=seed)
    for k, v in switches.items():
        setattr(model, k, v)
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            return model.forward_streams(*streams, packed=True, **extras).float32()
    finally:
        for k in switches:
            setattr(model, k, False)


def test_the_fp8_trunk_serves_near_the_bf16_stream_and_is_another_rounding():
    model = _model()
    bf16 = _forward(model, SHAPES[0], 31, infer_bf16_residual=True)
    fp8 = _forward(model, SHAPES[0], 31, infer_fp8=True)
    for x, y in zip(bf16.samples(), fp8.samples()):
        for f in FIELDS:
            a, b = getattr(x, f), getattr(y, f)
            assert torch.allclose(a, b, atol=0.1, rtol=0.1), (f, float((a - b).abs().max()))
    assert not torch.equal(bf16.actor_logits, fp8.actor_logits)
    assert torch.allclose(bf16.memory_padded, fp8.memory_padded, atol=0.1)
    assert torch.equal(_forward(model, SHAPES[0], 31, infer_fp8=True).actor_logits, fp8.actor_logits)


def test_the_compiled_fp8_loop_is_the_eager_loop_to_the_bit_and_compiles_once():
    """aot_eager runs the eager kernels, so a compiled loop that changed the
    arithmetic (another cast, another scale) would differ; inductor's own
    kernels are the CUDA tier's (tests/test_fp8_inference_cuda.py)."""
    model = _model()
    eager = [_forward(model, sizes, 40 + i, infer_fp8=True) for i, sizes in enumerate(SHAPES)]
    model.configure_packed_compile(backend="aot_eager")
    compiled = [_forward(model, sizes, 40 + i, infer_fp8=True, infer_compile_packed=True)
                for i, sizes in enumerate(SHAPES)]
    for a, b in zip(eager, compiled):
        for f in FIELDS:
            assert torch.equal(getattr(a, f), getattr(b, f)), f
        assert torch.equal(a.memory_padded, b.memory_padded)
    model.infer_fp8 = True
    stats = model.packed_compile_stats()
    assert stats["active"] and stats["fp8"] and stats["recompiles"] == 0 and stats["fallback_reason"] is None, stats


def test_every_static_scale_keeps_its_input_under_the_top(monkeypatch):
    """On random batches with the norms' weights spread wide: every
    statically scaled input, times its scale, at or under 448 before the
    clamp (the clamp never fires)."""
    model = _model(seed=3)
    with torch.no_grad():
        for layer in model.encoder.layers:
            for norm in (layer.norm1, layer.norm2):
                norm.weight.normal_(0.0, 2.0)
                norm.bias.normal_(0.0, 1.0)
    seen = []
    real = fi.static_cast

    def spy(t, scale):
        seen.append(float(t.float().abs().max()) * float(scale))
        return real(t, scale)
    monkeypatch.setattr(fi, "static_cast", spy)
    for i, sizes in enumerate(SHAPES):
        _forward(model, sizes, 70 + i, infer_fp8=True)
    assert len(seen) == len(SHAPES) * (2 + 4)      # two static inputs in the first layer, four in the second
    assert max(seen) <= fi.E4M3_MAX, max(seen)
    assert min(seen) > fi.E4M3_MAX / 2 ** 12, "the scales are not vacuous"


def test_the_norm_output_bound_is_reached_by_a_spike():
    """A row whose one entry stands out puts sqrt(E-1) in z there: the
    norm's output reaches the bound to its widening."""
    torch.manual_seed(1)
    E = 48
    gamma, beta = torch.randn(E, dtype=torch.float64) * 2, torch.randn(E, dtype=torch.float64)
    bound = fi.norm_output_bound(gamma, beta)
    k = int((gamma.abs() * math.sqrt(E - 1) + beta.abs()).argmax())
    h = torch.zeros(1, E, dtype=torch.float64)
    h[0, k] = 1.0 if float(gamma[k] * beta[k]) >= 0 else -1.0
    x = F.layer_norm(h, (E,), gamma, beta, 0.0)
    reached = float(x.abs().max())
    assert reached <= bound
    assert reached >= bound / ((1 + fi.BF16_ROUNDING) * fi.MARGIN) * (1 - 1e-9)


def test_a_linear_bound_holds_for_rounded_inputs_and_is_reached_by_the_aligned_row():
    """y = W LN(h) + b: on random rows rounded as the loop rounds them every
    output lies under its bound; the row along a_j - mean(a_j) (signed as
    c_j) reaches the bound's exact part, Cauchy-Schwarz's equality."""
    torch.manual_seed(2)
    E, N = 64, 48
    gamma, beta = torch.randn(E) * 1.5, torch.randn(E) * 0.5
    w8, w_inverse = fi._quantized(torch.randn(N, E) * 0.2)
    W = (w8.float() * w_inverse).double()
    b = torch.randn(N).double() * 0.3
    s = fi.scale_for(fi.norm_output_bound(gamma, beta))
    bound = fi.linear_of_norm_bounds(W, b, gamma.double(), beta.double(), s)
    h = torch.randn(4000, E) * torch.rand(4000, 1) * 5
    x = F.layer_norm(h, (E,), gamma, beta, 1e-5).to(torch.bfloat16)
    x8 = fi.static_cast(x, torch.tensor(s)).float() / s
    y = x8.double() @ W.t() + b
    assert bool((y.abs() <= bound).all())
    g, be = gamma.double(), beta.double()
    a = W * g
    centred = a - a.mean(dim=1, keepdim=True)
    c = W @ be + b
    exact = centred.norm(dim=1) * math.sqrt(E) + c.abs()
    for j in range(N):
        row = (centred[j] * (1.0 if float(c[j]) >= 0 else -1.0)).unsqueeze(0)
        reached = float((F.layer_norm(row, (E,), g, be, 0.0) @ W[j] + b[j]).abs())
        assert reached == pytest.approx(float(exact[j]), rel=1e-9)
        assert reached <= float(bound[j])


def test_a_static_cast_saturates_and_a_batch_cast_fills_the_top_binade():
    t = torch.randn(64, 32).to(torch.bfloat16)
    clipped = fi.static_cast(t * 1e4, torch.tensor(1.0)).float()
    assert bool(torch.isfinite(clipped).all()) and float(clipped.abs().max()) == fi.E4M3_MAX
    t8, inverse = fi.batch_cast(t)
    scale = 1.0 / float(inverse)
    assert scale == 2.0 ** round(math.log2(scale))
    assert fi.E4M3_MAX / 2 < float(t.float().abs().max()) * scale <= fi.E4M3_MAX
    assert float(t8.float().abs().max()) <= fi.E4M3_MAX


def test_the_fp8_copy_follows_published_weights():
    model, other = _model(seed=8), _model(seed=9)
    _forward(model, SHAPES[0], 50, infer_fp8=True)
    copy = model._fp8_weights
    model.load_state_dict(other.state_dict())
    followed = _forward(model, SHAPES[0], 50, infer_fp8=True)
    assert model._fp8_weights is copy, "refreshed in place"
    fresh = _forward(other, SHAPES[0], 50, infer_fp8=True)
    for f in FIELDS:
        assert torch.equal(getattr(followed, f), getattr(fresh, f)), f


def test_the_first_layer_takes_its_batch_scale_and_the_others_the_weights():
    model = _model()
    w = model.fp8_weights(CPU)
    exponents = w.summary()["input_scale_exponents"]
    assert exponents[0][:2] == [None, None] and all(isinstance(e, int) for e in exponents[0][2:])
    assert all(isinstance(e, int) for layer in exponents[1:] for e in layer)


def test_the_fp8_loop_refuses_what_it_cannot_run():
    for kwargs in ({"d_model": 24, "nhead": 2, "dim_feedforward": 64},
                   {"d_model": 32, "nhead": 2, "dim_feedforward": 64, "activation": "gelu"},
                   {"d_model": 32, "nhead": 2, "dim_feedforward": 64, "norm_first": True}):
        layer = nn.TransformerEncoderLayer(batch_first=True, **kwargs)
        with pytest.raises(ValueError):
            fi.check_fp8_inference_supported(nn.TransformerEncoder(layer, 1, enable_nested_tensor=False), CPU)
