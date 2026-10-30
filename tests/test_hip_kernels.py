"""The HIP kernels (wesnoth_ai/hip_kernels.py, wesnoth_ai/hip_elementwise.py,
wesnoth_ai/hip_attention_grad.py): their dispatch off ROCm, and on a ROCm
GPU the varlen attention against F.scaled_dot_product_attention per
segment (with and without the in-projection's bias, and without it bit for
bit as before the bias variant), its backward against SDPA's fp32 autograd
and its dispatch when a gradient is wanted, the fused residual add and
layer norm against F.layer_norm (with and without the sublayer's bias), the
bias and ReLU against F.relu, and the packed fp16 trunk (the model's fused
loop, its biases folded into those kernels or not; packed_trunk with the
backward when a gradient is wanted) against the padded fp32 trunk. On the
CPU, the self-tests run with the caller's autocast off and their
references stay float32 inside an fp16 autocast. The GPU checks skip
without a ROCm build and a GPU the kernel serves (CI has neither); on the
local RX 6600 they run under the ROCm venv
(training/metrics/local_gpu_20261028/README.md)."""
from __future__ import annotations

import hashlib
import math
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
from wesnoth_ai.hip_attention_grad import closed_form_gradients as _closed_form  # noqa: E402 (unpatched)
from wesnoth_ai.hip_elementwise import (  # noqa: E402
    hip_add_layer_norm, hip_add_layer_norm_applies, hip_bias_relu_,
)
from wesnoth_ai.hip_kernels import (  # noqa: E402
    HEAD_DIMS, device_arch, hip_varlen_applies, hip_varlen_attention, hip_varlen_attention_lse,
    hip_varlen_bias_applies,
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
    assert not hip_varlen_bias_applies(torch.device("cpu"), 32)
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


@rocm
@pytest.mark.parametrize("d", HEAD_DIMS)
def test_bias_variant_is_the_kernel_on_biased_inputs(d):
    """The in-projection's bias added as the kernel loads q, k and v: bit for
    bit the kernel without one on q + bias[0], k + bias[1], v + bias[2]
    rounded to fp16 by torch (both adds are correctly rounded), and against
    fp32 SDPA of the fp32 sums, on the segment lengths of
    test_kernel_matches_sdpa."""
    assert hip_varlen_bias_applies(DEV, d)
    rng = random.Random(40 + d)
    lens = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 692] + [rng.randint(141, 692) for _ in range(4)]
    rng.shuffle(lens)
    q, k, v, cu, cu_dev = _segments(lens, 384 // d, d, seed=50 + d)
    bias = torch.randn(3, 384 // d, d, generator=torch.Generator().manual_seed(60 + d)).to(DEV, torch.float16)
    out = hip_varlen_attention(q, k, v, cu_dev, max(lens), bias=bias)
    assert torch.equal(out, hip_varlen_attention(q + bias[0], k + bias[1], v + bias[2], cu_dev, max(lens)))
    assert torch.equal(out, hip_varlen_attention(q, k, v, cu_dev, max(lens), bias=bias.view(-1)))
    ref = _reference(*(t.float() + b.float() for t, b in zip((q, k, v), bias)), cu)
    assert (out.float() - ref).abs().max().item() < 2e-2


# SHA-256 of the kernels' outputs at 041bd7ca, before the bias variants,
# on gfx1032 (the RX 6600), for the inputs of the test below: the kernels
# without a bias are the same code since. Another architecture compiles
# other code, so the digests hold on gfx1032 only.
_DIGESTS_041BD7CA_GFX1032 = {
    "attn32": "d7138ac17453015875015d6036a3828d2de51b96195d3844dc1db2c912bdc8be",
    "lse32": "b91454223954f797eab91de8c7ff9f26bce0b0dec890d943ddb0be38f8b8e44a",
    "attn64": "59fd5dedb7b8081753c9f5a74d0c14bb81ea09897d341f7116e355c0ee4509c8",
    "lse64": "ee53ac82fad89ca499c5a2090d244bcc2d97d542ace37a2b5706734d24b19cff",
    "ln32_128": "1bd365d00ed7ffa32ac6b9855515a7f1b1952060fff3db36e999fd99a788e212",
    "ln16_128": "12a049f66cc99f2bfe0f1d16ed2cad306060e27d2b2249c3dd53d2bd7b8e7d8c",
    "ln32_384": "6b455b4433705b6d226d834717ce5c2b2875206e76a9ee060421ffe7f8611478",
    "ln16_384": "4134d2212afc72e14b44bc784aaf2e4d2d84e3969d87ebf88bf4628af14c4ee7",
}


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()


@rocm
def test_kernels_without_a_bias_are_unchanged_bit_for_bit():
    """The attention, its log-sum-exp variant (the training path's forward)
    and the add and norm without a bias give 041bd7ca's outputs bit for bit
    on fixed inputs (the digests above)."""
    if device_arch(DEV) != "gfx1032":
        pytest.skip("the digests are of gfx1032's code")
    got = {}
    lens = [1, 15, 16, 17, 95, 96, 97, 141, 300, 692]
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=DEV)
    for d in HEAD_DIMS:
        g = torch.Generator().manual_seed(100 + d)
        q, k, v = (torch.randn(cu[-1], 3, 384 // d, d, generator=g) * 2).to(DEV, torch.float16).unbind(1)
        got[f"attn{d}"] = _digest(hip_varlen_attention(q, k, v, cu_dev, max(lens)))
        out, lse, _ = hip_varlen_attention_lse(q, k, v, cu_dev, max(lens))
        assert _digest(out) == got[f"attn{d}"]
        got[f"lse{d}"] = _digest(lse)
    for d in (128, 384):
        g = torch.Generator().manual_seed(200 + d)
        x = (torch.randn(1001, d, generator=g) * 3).to(DEV)
        y = torch.randn(1001, d, generator=g).to(DEV, torch.float16)
        w, b = (torch.rand(d, generator=g) + 0.5).to(DEV), torch.randn(d, generator=g).to(DEV)
        o32, o16 = hip_add_layer_norm(x, y, w, b, 1e-5)
        got[f"ln32_{d}"], got[f"ln16_{d}"] = _digest(o32), _digest(o16)
    assert got == _DIGESTS_041BD7CA_GFX1032


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


def test_closed_form_gradients_match_sdpa_autograd():
    """The backward's self-test reference (no autograd: it can run inside a
    checkpointed forward) against SDPA's autograd, in float32 on the CPU."""
    from wesnoth_ai.hip_attention_grad import closed_form_gradients, reference_attention
    g = torch.Generator().manual_seed(12)
    cu = [0, 1, 6, 23]
    q, k, v = (torch.randn(cu[-1], 3, 8, generator=g).requires_grad_(True) for _ in range(3))
    up = torch.randn(cu[-1], 3, 8, generator=g)
    out = reference_attention(q, k, v, cu)
    (out * up).sum().backward()
    want = {"out": out.detach(), "dq": q.grad, "dk": k.grad, "dv": v.grad}
    got = closed_form_gradients(q.detach(), k.detach(), v.detach(), up, cu)
    for name in want:
        assert torch.allclose(got[name], want[name], atol=1e-5), name


def _autocast_casts_on_the_cpu() -> bool:
    with torch.autocast("cpu", dtype=torch.float16):
        return (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float16


def test_the_self_tests_references_are_float32_inside_an_fp16_autocast():
    """The kernels' first use can come inside the gradient pass's fp16
    autocast (tools/pg_pass.py `_step`) or the server's, and no_grad does
    not turn an autocast off. The self-tests' references (SDPA per segment,
    the closed-form gradients) stay float32 there, equal to those built
    outside it, on the CPU's fp16 autocast, which casts matmul and SDPA to
    fp16 as CUDA's does."""
    from wesnoth_ai.hip_attention_grad import closed_form_gradients, reference_attention
    if not _autocast_casts_on_the_cpu():
        pytest.skip("this torch's CPU autocast does not cast matmul to fp16")
    g = torch.Generator().manual_seed(13)
    cu = [0, 3, 20]
    q, k, v, up = (torch.randn(cu[-1], 2, 32, generator=g).half() for _ in range(4))
    want_out, want = reference_attention(q, k, v, cu), closed_form_gradients(q, k, v, up, cu)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        got_out, got = reference_attention(q, k, v, cu), closed_form_gradients(q, k, v, up, cu)
    assert got_out.dtype == torch.float32 and torch.equal(got_out, want_out)
    for name in want:
        assert got[name].dtype == torch.float32 and torch.equal(got[name], want[name]), name


class _CpuKernels:
    """The kernels' entry points computed by torch on the CPU, recording
    whether an autocast was on at each launch: the self-tests run on them."""

    def __init__(self):
        from wesnoth_ai.hip_elementwise import _LN_NI
        self.add_ln = {128 * ni: None for ni in _LN_NI}
        self.autocast_at_launch = []

    def _seen(self):
        self.autocast_at_launch.append(torch.is_autocast_enabled("cpu"))

    def launch(self, q, k, v, out, cu_seqlens, max_len, lse=None, bias=None):
        self._seen()
        cu = cu_seqlens.tolist()
        with torch.autocast("cpu", enabled=False):
            if bias is not None:
                q, k, v = (t + b for t, b in zip((q, k, v), bias))
            out[:cu[-1]] = _closed_form(q, k, v, torch.zeros_like(q), cu)["out"].to(out.dtype)
            if lse is not None:
                s = 1.0 / math.sqrt(q.shape[-1])
                for lo, hi in zip(cu[:-1], cu[1:]):
                    qs, ks = q[lo:hi].float().transpose(0, 1), k[lo:hi].float().transpose(0, 1)
                    lse[:, lo:hi] = torch.logsumexp(s * qs @ ks.transpose(1, 2), dim=-1)

    def launch_add_ln(self, x, y, w, b, o32, o16, eps, yb=None):
        self._seen()
        with torch.autocast("cpu", enabled=False):
            sub = y.float() if yb is None else y.float() + yb.float()
            o32.copy_(F.layer_norm(x + sub, (x.shape[1],), w, b, eps))
            o16.copy_(o32.half())

    def launch_bias_relu(self, y, bias):
        self._seen()
        with torch.autocast("cpu", enabled=False):
            y.copy_(F.relu(y.float() + bias.float()).half())


class _CpuBackward:
    def __init__(self):
        self.autocast_at_launch = []

    def launch(self, q, k, v, o, g, lse, delta, dq, dk, dv, cu_seqlens, max_len):
        self.autocast_at_launch.append(torch.is_autocast_enabled("cpu"))
        with torch.autocast("cpu", enabled=False):
            ref = _closed_form(q, k, v, g, cu_seqlens.tolist())
        for name, t in (("dq", dq), ("dk", dk), ("dv", dv)):
            t.copy_(ref[name].to(t.dtype))


def test_the_self_tests_run_with_the_callers_autocast_off(monkeypatch):
    """The self-tests (the attention, its bias variant, the backward, the
    row-wise kernels) called inside an fp16 autocast on their device (the
    CPU here, with the kernels computed by torch): every launch, and every
    reference built between them, sees no autocast."""
    import wesnoth_ai.hip_attention_grad as hg
    import wesnoth_ai.hip_elementwise as he
    import wesnoth_ai.hip_kernels as hk
    if not _autocast_casts_on_the_cpu():
        pytest.skip("this torch's CPU autocast does not cast matmul to fp16")
    forward, backward = _CpuKernels(), _CpuBackward()
    monkeypatch.setattr(hk, "_kernels_for", lambda device: forward)
    references = {"attention": [], "gradients": []}

    def recorded(name, function):
        return lambda *a: references[name].append(torch.is_autocast_enabled("cpu")) or function(*a)
    monkeypatch.setattr(hk, "reference_attention", recorded("attention", hk.reference_attention))
    monkeypatch.setattr(hg, "closed_form_gradients", recorded("gradients", hg.closed_form_gradients))
    cpu = torch.device("cpu")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        hk._self_test(forward, cpu)
        hk._self_test(forward, cpu, bias=True)
        he._self_test(forward, cpu, with_bias=False)
        he._self_test(forward, cpu, with_bias=True)
        hg._self_test(backward, cpu)
    assert forward.autocast_at_launch and not any(forward.autocast_at_launch)
    assert backward.autocast_at_launch and not any(backward.autocast_at_launch)
    for name, seen in references.items():
        assert seen and not any(seen), name


@rocm
def test_the_backward_builds_inside_a_checkpointed_forward():
    """The gradient pass's case (tools/pg_pass.py): the kernels' first use
    comes inside a forward under non-reentrant activation checkpointing.
    Building and self-testing them there finishes, and the gradients are
    right (2026-10-29: a self-test through autograd deadlocked there, its
    graph taken by the checkpoint's hooks). Run in a thread with a timeout,
    so a deadlock fails the test instead of hanging the run; last in the
    file, since a deadlocked build keeps its lock."""
    import threading

    from torch.utils.checkpoint import checkpoint

    import wesnoth_ai.hip_attention_grad as hg
    from wesnoth_ai.hip_attention_grad import closed_form_gradients
    q, k, v, cu, cu_dev = _segments([7, 40, 129], 12, 32, seed=11)
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
    built = dict(hg._backward)
    hg._backward.clear()
    outcome = []

    def run():
        try:
            out = checkpoint(lambda a, b, c: packed_attention(a, b, c, _index(cu, cu_dev)), *leaves,
                             use_reentrant=False)
            out.float().sum().backward()
            outcome.append(None)
        except Exception as e:  # noqa: BLE001 -- reported by the assertion below
            outcome.append(e)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(120)
    if not hg._backward:
        hg._backward.update(built)
    assert outcome, "the first use inside a checkpointed forward did not finish within 120 s (a deadlock)"
    assert outcome[0] is None, outcome[0]
    assert hg._backward and all(kern is not None for kern in hg._backward.values())
    ref = closed_form_gradients(q, k, v, torch.ones(q.shape, device=DEV), cu)
    for name, leaf in zip(("dq", "dk", "dv"), leaves):
        rel = ((leaf.grad.float() - ref[name]).abs().max() / ref[name].abs().max()).item()
        assert rel < GRAD_REL, (name, rel)


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
@pytest.mark.parametrize("d", [128, 384])
def test_add_layer_norm_with_a_bias_matches_torch(d):
    """The sublayer's fp16 bias added to its output in float32 before the
    residual add: F.layer_norm(x + (y + bias)) to float32 rounding."""
    g = torch.Generator().manual_seed(70 + d)
    x = (torch.randn(1001, d, generator=g) * 3).to(DEV)
    y = torch.randn(1001, d, generator=g).to(DEV, torch.float16)
    w, b = (torch.rand(d, generator=g) + 0.5).to(DEV), torch.randn(d, generator=g).to(DEV)
    y_bias = torch.randn(d, generator=g).to(DEV, torch.float16)
    o32, o16 = hip_add_layer_norm(x, y, w, b, 1e-5, y_bias=y_bias)
    ref = F.layer_norm(x + (y.float() + y_bias.float()), (d,), w, b, 1e-5)
    assert (o32 - ref).abs().max().item() < 1e-4
    assert (o16.float() - ref.half().float()).abs().max().item() <= 2 ** -10 * ref.abs().max().item()
    assert (o32 - hip_add_layer_norm(x, y, w, b, 1e-5)[0]).abs().max().item() > 1e-2   # the bias is read


@rocm
def test_bias_relu_matches_torch():
    """relu(y + bias) in fp16: F.relu of the fp32 sum rounded to fp16, bit
    for bit (the kernel's fp16 add is correctly rounded); a NaN stays NaN
    as F.relu keeps it; written in place on a contiguous aligned y, into a
    copy otherwise."""
    g = torch.Generator().manual_seed(80)
    y = (torch.randn(1003, 1536, generator=g) * 2).to(DEV, torch.float16)
    bias = torch.randn(1536, generator=g).to(DEV, torch.float16)
    y[3, 5] = float("nan")
    ref = F.relu(y.float() + bias.float()).half()
    out = hip_bias_relu_(y, bias)
    assert out.data_ptr() == y.data_ptr()
    assert torch.equal(out.isnan(), ref.isnan()) and out.isnan().sum().item() == 1
    assert torch.equal(torch.nan_to_num(out), torch.nan_to_num(ref))
    wide = (torch.randn(24, 64, generator=g)).to(DEV, torch.float16)
    view = wide[:, :32]                                   # not contiguous
    want = F.relu(view.float() + bias[:32].float()).half()
    got = hip_bias_relu_(view, bias[:32].clone())
    assert got.data_ptr() != wide.data_ptr() and torch.equal(got, want)


def _worst(got, ref, field):
    """The largest difference of `field` over every sample's real slots,
    and the reference's largest magnitude there."""
    diff = scale = 0.0
    for x, y in zip(got.samples(), ref.samples()):
        xf, yf = getattr(x, field), getattr(y, field)
        fin = torch.isfinite(yf)
        assert torch.equal(fin, torch.isfinite(xf)), field
        if fin.any():
            diff = max(diff, (xf - yf).abs()[fin].max().item())
            scale = max(scale, yf.abs()[fin].max().item())
    return diff, scale


@rocm
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_fused_loop_folds_the_biases(monkeypatch, activation):
    """Every bias of a small random network set nonzero (the attention's are
    zero at init). The fused loop runs its linears without them, adding the
    in-projection's in the attention, the out-projection's and the second
    feed-forward linear's in the add and norm, the first's in the bias and
    ReLU (with a ReLU; a GELU keeps its linear's bias): against the padded
    fp32 trunk as test_packed_fp16_trunk_matches_padded_fp32, and close to
    the loop whose linears take their biases, which runs where the
    attention's bias variant is not served."""
    import wesnoth_ai.model as wm
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(15)
    model = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=256).to(DEV).eval()
    g = torch.Generator().manual_seed(16)
    with torch.no_grad():
        for layer in model.encoder.layers:
            for p in (layer.self_attn.in_proj_bias, layer.self_attn.out_proj.bias, layer.linear1.bias,
                      layer.linear2.bias):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            if activation == "gelu":
                layer.activation, layer.activation_relu_or_gelu = F.gelu, 2
    fused, bias_relu = [], []
    trunk = model._run_hip_fp16_trunk
    model._run_hip_fp16_trunk = lambda x, index: fused.append(1) or trunk(x, index)
    monkeypatch.setattr(wm, "hip_bias_relu_", lambda y, b: bias_relu.append(1) or hip_bias_relu_(y, b))
    rng = random.Random(17)
    sizes = [(rng.randint(1, 30), rng.randint(0, 7), rng.randint(100, 600)) for _ in range(12)]
    gd = torch.Generator(device=DEV).manual_seed(18)
    B, d = len(sizes), 128
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))
    streams = tuple(torch.randn(B, n, d, generator=gd, device=DEV) for n in (H_max, U_max, R_max, 1, 1)) + (sizes,)

    def fp16():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return model.forward_streams(*streams, packed=True).float32()
    with torch.no_grad():
        ref = model.forward_streams(*streams, packed=False)
    folded = fp16()
    assert fused and len(bias_relu) == (2 if activation == "relu" else 0)
    model.fold_biases = False                  # the policy-gradient tools' setting (tools/pg_pass.py load_network)
    switched = fp16()
    assert len(fused) == 2 and len(bias_relu) == (2 if activation == "relu" else 0)    # no bias kernel ran
    model.fold_biases = True
    monkeypatch.setattr(wm, "hip_varlen_bias_applies", lambda device, head_dim: False)
    unfolded = fp16()
    assert len(fused) == 3 and len(bias_relu) == (2 if activation == "relu" else 0)
    for f in FIELDS:
        diff, scale = _worst(folded, ref, f)
        assert diff <= 0.02 * max(scale, 1.0), (f, diff, scale)
        diff, scale = _worst(folded, unfolded, f)
        assert diff <= 0.01 * max(scale, 1.0), (f, diff, scale)
        assert _worst(switched, unfolded, f)[0] == 0.0, f        # the switch is the unfolded loop, bit for bit


@rocm
def test_a_failed_bias_self_test_keeps_the_fused_loop(monkeypatch):
    """The row-wise kernels' bias variants fail their self-test (a code
    generation that rounds one ulp apart, say): the plain add and norm
    still serves, the fold is off, and the fused loop runs with its
    linears taking their biases, close to the padded fp32 trunk."""
    import wesnoth_ai.hip_elementwise as he
    import wesnoth_ai.model as wm
    from wesnoth_ai.model import WesnothModel
    real = he._self_test

    def failing(kern, device, *, with_bias):
        if with_bias:
            raise RuntimeError("bias variants fail here")
        real(kern, device, with_bias=False)
    monkeypatch.setattr(he, "_kernels", {})
    monkeypatch.setattr(he, "_self_test", failing)
    assert he.hip_add_layer_norm_applies(DEV, 128)
    assert not he.hip_elementwise_bias_applies(DEV, 128)
    torch.manual_seed(15)
    model = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=256).to(DEV).eval()
    g = torch.Generator().manual_seed(16)
    with torch.no_grad():
        for layer in model.encoder.layers:
            for p in (layer.self_attn.in_proj_bias, layer.self_attn.out_proj.bias, layer.linear1.bias,
                      layer.linear2.bias):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    fused, bias_relu = [], []
    trunk = model._run_hip_fp16_trunk
    model._run_hip_fp16_trunk = lambda x, index: fused.append(1) or trunk(x, index)
    monkeypatch.setattr(wm, "hip_bias_relu_", lambda y, b: bias_relu.append(1) or hip_bias_relu_(y, b))
    rng = random.Random(17)
    sizes = [(rng.randint(1, 30), rng.randint(0, 7), rng.randint(100, 600)) for _ in range(12)]
    gd = torch.Generator(device=DEV).manual_seed(18)
    B, d = len(sizes), 128
    H_max, U_max, R_max = (max(s[i] for s in sizes) for i in (2, 0, 1))
    streams = tuple(torch.randn(B, n, d, generator=gd, device=DEV) for n in (H_max, U_max, R_max, 1, 1)) + (sizes,)
    with torch.no_grad():
        ref = model.forward_streams(*streams, packed=False)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.forward_streams(*streams, packed=True).float32()
    assert fused and not bias_relu
    for f in FIELDS:
        diff, scale = _worst(out, ref, f)
        assert diff <= 0.02 * max(scale, 1.0), (f, diff, scale)


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


def test_misaligned_and_malformed_inputs_never_reach_a_kernel():
    """The kernels read raw pointers with 16-byte loads: a contiguous view
    starting off a 16-byte boundary is copied (contiguous() alone returns
    it unchanged), and a wrong dtype, shape, head size or device is refused
    before any launch (CPU tensors: nothing here needs a GPU)."""
    from wesnoth_ai.hip_kernels import _aligned, _check_attention_args, _check_bias
    base = torch.zeros(3 * 12 * 32 + 4, dtype=torch.float16)
    view = base[4:].view(3, 12, 32)                       # 8 bytes past base's start
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    copy = _aligned(view)
    assert copy.data_ptr() % 16 == 0 and torch.equal(copy, view)
    aligned = torch.zeros(3, 12, 32, dtype=torch.float16)
    assert _aligned(aligned) is aligned
    cu = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(TypeError, match="fp16"):
        _check_attention_args(aligned.float(), aligned, aligned, cu)
    with pytest.raises(ValueError, match="shape"):
        _check_attention_args(aligned, aligned[:2], aligned, cu)
    odd = torch.zeros(3, 12, 48, dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim 48"):
        _check_attention_args(odd, odd, odd, cu)
    x = torch.zeros(5, 128)
    with pytest.raises(TypeError, match="float32 x and fp16 y"):
        hip_add_layer_norm(x.half(), x.half(), torch.ones(128), torch.zeros(128), 1e-5)
    with pytest.raises(TypeError, match="weight and bias"):
        hip_add_layer_norm(x, x.half(), torch.ones(128).half(), torch.zeros(128), 1e-5)
    with pytest.raises(TypeError, match="y_bias"):
        hip_add_layer_norm(x, x.half(), torch.ones(128), torch.zeros(128), 1e-5, y_bias=torch.zeros(128))
    with pytest.raises(TypeError, match="bias"):
        _check_bias(torch.zeros(3, 12, 32), aligned)            # float32
    with pytest.raises(TypeError, match="bias"):
        _check_bias(torch.zeros(3, 12, 16, dtype=torch.float16), aligned)
    assert _check_bias(torch.zeros(3 * 12 * 32, dtype=torch.float16), aligned).shape == (3, 12, 32)
    with pytest.raises(TypeError, match="multiple of 8"):
        hip_bias_relu_(torch.zeros(4, 12, dtype=torch.float16), torch.zeros(12, dtype=torch.float16))
    with pytest.raises(TypeError, match="fp16"):
        hip_bias_relu_(torch.zeros(4, 16), torch.zeros(16, dtype=torch.float16))
