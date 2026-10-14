"""The packed trunk's linear layers with fp8 matrix products (`--fp8-trunk`,
wesnoth_ai/trunk_modes.py; docs/trainer_speed_20261013.md, round 4).

Each product runs on float8 operands with one float32 scale per tensor,
recomputed from the tensor at every call (dynamic per-tensor scaling, the
tensorwise recipe of torchao's float8 training): the forward's x @ W.T on
float8_e4m3fn x and W, the backward's grad @ W and grad.T @ x on a
float8_e5m2 gradient (two more exponent bits for the gradient's range, one
fewer mantissa bit) and the forward's e4m3 operands, saved for it in one
byte an element. A tensor's scale is the power of two that puts its largest
magnitude in the format's top binade: multiplying by it is exact, so the
cast to float8 is the only rounding. The products accumulate in float32 and
come out in the compute dtype (bfloat16 under the autocast), the bias added
in the product's epilogue; the bias's gradient is summed from the gradient
before its quantization.

On CUDA the products are `torch._scaled_mm`, at torch 2.5.1:
aten/src/ATen/native/native_functions.yaml:7024 declares it; Blas.cpp:875
allows compute capability 8.9 and 9.x (the 4090 is 8.9); two float32
scalars are per-tensor scales (Blas.cpp:914); the first operand must be
row-major and the second column-major (Blas.cpp:1069, with the layout rules
of cublasCommonArgs, Blas.cpp:96-119); the inner dimension and both of the
second operand's divisible by 16 (Blas.cpp:1012-1021); never two e5m2
operands (Blas.cpp:1027). It has no derivative (tools/autograd/derivatives.yaml
holds no entry), so the backward pass is written here. The weight
gradient's inner dimension is the token count, padded with zero rows to a
multiple of 16.

On the CPU the products are emulated: the float8 operands' exact values
multiplied in float32, scaled, the bias added, cast once. Both paths check
the layouts `_scaled_mm` requires, so a CPU test fails where the CUDA kernel
would refuse.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
FP8_DTYPES = (E4M3, E5M2)
# _scaled_mm's divisibility rule for the inner and the second operand's dimensions.
GEMM_MULTIPLE = 16
# The scale's exponent stays where both the scale and its inverse are
# normal float32 numbers.
MAX_SCALE_EXPONENT = 126


def quantize(t: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """(t in `dtype`, contiguous, after multiplying it by the power of two
    that puts its largest magnitude in the format's top binade; the inverse
    of that power, a float32 scalar on t's device). A zero tensor stays zero."""
    top = float(torch.finfo(dtype).max)
    amax = torch.linalg.vector_norm(t.detach(), ord=float("inf")).float()
    tiny = torch.finfo(torch.float32).tiny
    exponent = torch.floor(torch.log2(top / amax.clamp(min=tiny))).clamp(-MAX_SCALE_EXPONENT, MAX_SCALE_EXPONENT)
    t8 = (t.detach() * torch.exp2(exponent)).to(dtype).contiguous()
    return t8, torch.exp2(-exponent)


def check_operands(a8: torch.Tensor, b8: torch.Tensor) -> None:
    """ValueError unless `_scaled_mm(a8, b8, ...)` takes the operands
    (module docstring): float8, not both e5m2, a8 [M, K] row-major, b8
    [K, N] column-major, K and N multiples of 16."""
    if a8.dim() != 2 or b8.dim() != 2 or a8.shape[1] != b8.shape[0]:
        raise ValueError(f"operands {tuple(a8.shape)} and {tuple(b8.shape)} do not multiply")
    if a8.dtype not in FP8_DTYPES or b8.dtype not in FP8_DTYPES:
        raise ValueError(f"float8 operands, not {a8.dtype} and {b8.dtype}")
    if a8.dtype == E5M2 and b8.dtype == E5M2:
        raise ValueError("two e5m2 operands")
    if not a8.is_contiguous():
        raise ValueError(f"the first operand must be row-major, its strides are {a8.stride()}")
    if b8.is_contiguous() or not b8.t().is_contiguous():
        raise ValueError(f"the second operand must be column-major, its strides are {b8.stride()}")
    k, n = b8.shape
    if k % GEMM_MULTIPLE or n % GEMM_MULTIPLE:
        raise ValueError(f"the second operand's dimensions {k} x {n} must be multiples of {GEMM_MULTIPLE}")


def scaled_mm(a8: torch.Tensor, b8: torch.Tensor, a_inverse: torch.Tensor, b_inverse: torch.Tensor,
              out_dtype: torch.dtype, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(a8 * a_inverse) @ (b8 * b_inverse) + bias in out_dtype, the product
    accumulated in float32: `_scaled_mm` on CUDA, emulated on the CPU."""
    check_operands(a8, b8)
    if a8.is_cuda:
        if bias is not None and (out_dtype == torch.float32 or bias.dtype != out_dtype):
            # _scaled_mm takes a bias only in the 16-bit output's dtype (Blas.cpp:1033-1040).
            out = torch._scaled_mm(a8, b8, scale_a=a_inverse, scale_b=b_inverse, out_dtype=out_dtype)
            return out + bias.to(out_dtype)
        return torch._scaled_mm(a8, b8, scale_a=a_inverse, scale_b=b_inverse, bias=bias, out_dtype=out_dtype)
    out = torch.mm(a8.float(), b8.float()) * (a_inverse * b_inverse)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _bytes_transposed(t8: torch.Tensor) -> torch.Tensor:
    """t8 [R, C] transposed into a contiguous [C, R], copied as bytes."""
    return t8.view(torch.uint8).t().contiguous().view(t8.dtype)


def _column_major(t8: torch.Tensor) -> torch.Tensor:
    """t8 [R, C] as a column-major [R, C]."""
    return _bytes_transposed(t8).t()


def _pad_rows(t8: torch.Tensor) -> torch.Tensor:
    """t8 [R, C] with zero rows up to a multiple of GEMM_MULTIPLE rows (a
    zero byte is +0 in both float8 formats)."""
    pad = -t8.shape[0] % GEMM_MULTIPLE
    if not pad:
        return t8
    raw = t8.view(torch.uint8)
    return torch.cat([raw, raw.new_zeros(pad, raw.shape[1])]).view(t8.dtype)


class _Fp8Linear(torch.autograd.Function):
    """x @ weight.T + bias with fp8 products, forward and backward (module docstring)."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        x8, x_inverse = quantize(x, E4M3)
        w8, w_inverse = quantize(weight, E4M3)
        out = scaled_mm(x8, w8.t(), x_inverse, w_inverse, x.dtype, bias)
        ctx.save_for_backward(x8, x_inverse, w8, w_inverse)
        ctx.dtypes = (x.dtype, weight.dtype, None if bias is None else bias.dtype)
        return out

    @staticmethod
    def backward(ctx, grad):
        x8, x_inverse, w8, w_inverse = ctx.saved_tensors
        x_dtype, w_dtype, b_dtype = ctx.dtypes
        g8, g_inverse = quantize(grad, E5M2)
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gx = scaled_mm(g8, _column_major(w8), g_inverse, w_inverse, x_dtype)
        if ctx.needs_input_grad[1]:
            gw = scaled_mm(_bytes_transposed(_pad_rows(g8)), _column_major(_pad_rows(x8)), g_inverse, x_inverse,
                           w_dtype)
        if ctx.needs_input_grad[2]:
            gb = grad.sum(0).to(b_dtype)
        return gx, gw, gb


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """F.linear(x, weight, bias) for 2-D x with fp8 products (module docstring)."""
    return _Fp8Linear.apply(x, weight, bias)


def fp8_problem(device: torch.device) -> Optional[str]:
    """Why fp8 products cannot run on `device`, or None: the CPU emulates
    them, CUDA needs compute capability 8.9 or more (Blas.cpp:875)."""
    if device.type == "cpu":
        return None
    if device.type != "cuda":
        return f"fp8 products on {device.type}: CUDA or the CPU's emulation only"
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        return f"fp8 products need compute capability 8.9 or more; this GPU has {capability[0]}.{capability[1]}"
    return None
