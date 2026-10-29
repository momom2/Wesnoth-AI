"""The int8 linear kernels (wesnoth_ai/hip_int8.py) and the int8 trunk loop
(wesnoth_ai/int8_trunk.py): the references and the wrappers' refusals on
the CPU, and on a ROCm GPU the GEMM against the exact integer product, the
quantizers against the reference rounding, and the int8 trunk on a small
random network against the padded fp32 trunk. The GPU checks skip without
a ROCm build and a GPU the kernels serve (CI has neither); on the local RX
6600 they run under the ROCm venv (docs/int8_linear_prereg_20261029.md)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.hip_int8 import (  # noqa: E402
    hip_add_layer_norm_q, hip_int8_applies, hip_int8_linear, hip_quantize_rows, int8_linear_reference,
    quantize_rows_reference, quantize_weight,
)
from wesnoth_ai.int8_trunk import LINEAR_CLASSES, check_int8_trunk_supported  # noqa: E402

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits")


def test_dispatch_off_rocm():
    assert not hip_int8_applies(torch.device("cpu"))
    if torch.version.hip is None:
        assert not hip_int8_applies(torch.device("cuda"))


def test_reference_rounding():
    """A row's scale is its largest magnitude over 127, the largest value
    maps to +-127, a zero row to zeros with scale 0, and the dequantized
    row is within half a step of the input."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(6, 128, generator=g) * 4
    x[2] = 0
    x[4, 7] = -50.0
    q, s = quantize_rows_reference(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert s[2] == 0 and (q[2] == 0).all()
    assert q[4, 7] == -127 and torch.isclose(s[4], torch.tensor(50.0 / 127))
    rows = [0, 1, 3, 4, 5]
    err = (q[rows].float() * s[rows, None] - x[rows]).abs()
    assert (err <= s[rows, None] * 0.5 * (1 + 1e-5)).all()


def test_weight_quantization_and_product_reference():
    """quantize_weight folds a smoothing factor out of the weight exactly
    as the input carries it in, and the reference product of the
    quantized operands stays near the float product."""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(40, 64, generator=g)
    w = torch.randn(24, 64, generator=g) * 0.1
    b = torch.randn(24, generator=g)
    sm = torch.rand(64, generator=g) + 0.5
    for smooth in (None, sm):
        xq, xs = quantize_rows_reference(x, smooth)
        wq, ws = quantize_weight(w, smooth)
        got = int8_linear_reference(xq, xs, wq, ws, b).float()
        full = x @ w.t() + b
        assert (got - full).abs().max().item() < 0.03 * full.abs().max().item()
    relu = int8_linear_reference(xq, xs, wq, ws, b, relu=True)
    assert (relu >= 0).all()


def test_wrappers_refuse_malformed_inputs():
    """Wrong dtypes, shapes, K or widths are refused before any launch (CPU
    tensors: nothing here needs a GPU)."""
    xq = torch.zeros(5, 64, dtype=torch.int8)
    wq = torch.zeros(8, 64, dtype=torch.int8)
    xs, ws, b = torch.ones(5), torch.ones(8), torch.zeros(8)
    with pytest.raises(TypeError, match="int8"):
        hip_int8_linear(xq.float(), xs, wq, ws, b)
    with pytest.raises(ValueError, match="multiple of 32"):
        hip_int8_linear(xq[:, :48], xs, wq[:, :48], ws, b)
    with pytest.raises(ValueError, match="multiple of 4"):
        hip_int8_linear(xq, xs, wq[:6], ws[:6], b[:6])
    with pytest.raises(TypeError, match="bias must be float32"):
        hip_int8_linear(xq, xs, wq, ws, b.half())
    with pytest.raises(TypeError, match="float16"):
        hip_quantize_rows(torch.zeros(3, 128))
    with pytest.raises(ValueError, match="width 96"):
        hip_quantize_rows(torch.zeros(3, 96, dtype=torch.float16))
    with pytest.raises(TypeError, match="fp16 y"):
        hip_add_layer_norm_q(torch.zeros(3, 128), torch.zeros(3, 128), torch.ones(128), torch.zeros(128), 1e-5)


def test_trunk_support_check():
    """The int8 loop takes post-norm ReLU layers only."""
    ok = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, batch_first=True), 1,
                               enable_nested_tensor=False)
    check_int8_trunk_supported(ok)
    gelu = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, activation="gelu", batch_first=True),
                                 1, enable_nested_tensor=False)
    with pytest.raises(ValueError, match="ReLU"):
        check_int8_trunk_supported(gelu)
    pre = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, norm_first=True, batch_first=True), 1,
                                enable_nested_tensor=False)
    with pytest.raises(ValueError, match="post-norm"):
        check_int8_trunk_supported(pre)


def _rocm_int8() -> bool:
    return torch.version.hip is not None and torch.cuda.is_available() and hip_int8_applies(torch.device("cuda"))


rocm = pytest.mark.skipif(not _rocm_int8(), reason="needs a ROCm GPU the int8 kernels serve")
DEV = torch.device("cuda")


@rocm
@pytest.mark.parametrize("M,N,K", [(1, 4, 32), (129, 388, 96), (300, 1152, 384), (77, 384, 1536)])
def test_gemm_matches_exact_integer_product(M, N, K):
    """M and N ending inside a tile and spanning several, K over one to 48
    tiles, both epilogues: within one fp16 rounding of the exact product."""
    g = torch.Generator().manual_seed(M + N + K)
    xq = torch.randint(-127, 128, (M, K), generator=g, dtype=torch.int8)
    wq = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
    xs, ws = torch.rand(M, generator=g) * 1e-3, torch.rand(N, generator=g) * 1e-3
    b = torch.randn(N, generator=g)
    for relu in (False, True):
        ref = int8_linear_reference(xq, xs, wq, ws, b, relu).float()
        got = hip_int8_linear(*(t.to(DEV) for t in (xq, xs, wq, ws, b)), relu=relu).float().cpu()
        assert ((got - ref).abs() <= 2 ** -10 * ref.abs() + 1e-4).all()


@rocm
@pytest.mark.parametrize("d", [384, 1536])
def test_quantizers_match_reference(d):
    """The row quantizer is bit-exact against the reference, with and
    without smoothing; add_ln_q's residual is F.layer_norm's and its int8
    rows the reference's quantization of that residual."""
    g = torch.Generator().manual_seed(d)
    x = (torch.randn(1001, d, generator=g) * 3).half()
    sm = torch.rand(d, generator=g) + 0.5
    for smooth in (None, sm):
        q_ref, s_ref = quantize_rows_reference(x, smooth)
        q, s = hip_quantize_rows(x.to(DEV), None if smooth is None else smooth.to(DEV))
        assert torch.equal(q.cpu(), q_ref)
        assert torch.allclose(s.cpu(), s_ref, rtol=1e-6, atol=0)
    r = torch.randn(513, d, generator=g) * 3
    y = torch.randn(513, d, generator=g).half()
    w, b = torch.rand(d, generator=g) + 0.5, torch.randn(d, generator=g)
    o32, q, s = hip_add_layer_norm_q(*(t.to(DEV) for t in (r, y, w, b)), 1e-5)
    assert (o32.cpu() - F.layer_norm(r + y.float(), (d,), w, b, 1e-5)).abs().max().item() < 1e-4
    q_ref, s_ref = quantize_rows_reference(o32.cpu())
    assert (q.cpu().int() - q_ref.int()).abs().max().item() <= 1
    assert torch.allclose(s.cpu(), s_ref, rtol=1e-6, atol=0)


@rocm
def test_int8_trunk_matches_padded_fp32():
    """A small random network (head_dim 32, width 128, d_ff 256): the packed
    fp16 forward with the int8 loop in place of the fused fp16 loop, every
    class on int8 and each alone, against the padded fp32 trunk over every
    sample's real slots. int8's rounding is coarser than fp16's, so the
    bound is looser than the fp16 loop's 0.02 of the head's scale."""
    from wesnoth_ai.int8_trunk import Int8TrunkWeights, run_int8_trunk
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(5)
    model = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=256).to(DEV).eval()
    weights = Int8TrunkWeights.build(model.encoder, DEV, model._weights_version)
    fp16_layers = model._weight_copy(torch.float16, torch.float32, DEV).layers
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
    for classes in [LINEAR_CLASSES] + [(c,) for c in LINEAR_CLASSES]:
        calls = []
        model._run_hip_fp16_trunk = (lambda x, index, c=classes: calls.append(1) or run_int8_trunk(
            x, index, weights, int8_classes=c, fp16_layers=fp16_layers))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            got = model.forward_streams(*streams, packed=True).float32()
        assert calls, "the fp16 forward did not take the int8 loop"
        for f in FIELDS:
            diff = scale = 0.0
            for x, y in zip(got.samples(), ref.samples()):
                xf, yf = getattr(x, f), getattr(y, f)
                fin = torch.isfinite(yf)
                assert torch.equal(fin, torch.isfinite(xf)), f
                if fin.any():
                    diff = max(diff, (xf - yf).abs()[fin].max().item())
                    scale = max(scale, yf.abs()[fin].max().item())
            assert diff <= 0.1 * max(scale, 1.0), (classes, f, diff, scale)
