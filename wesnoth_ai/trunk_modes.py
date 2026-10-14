"""The packed trunk's training variants (docs/trainer_speed_20261013.md,
round 3): each a flag of tools/sequence_train.py, each its own arithmetic.

With every variant off the trainer runs `packed_trunk.packed_trunk` under
the bfloat16 autocast, as before. With any on, `run_trunk` runs the same
post-norm layers with every cast written out and the autocast disabled
inside the trunk: the compute dtype is the autocast's (bfloat16 on CUDA),
the weights are cast to it at each use as the autocast casts them, and

- bf16 residual (`--trunk-bf16`): the residual stream is held in the
  compute dtype instead of float32. No cast before the in-projection or the
  first feed-forward linear; the residual sums and the LayerNorms read and
  write bfloat16, the norms' statistics in float32 and their weights cast
  to bfloat16, which CUDA's LayerNorm requires of a bfloat16 input (torch
  2.5.1 aten/src/ATen/native/cuda/layer_norm_kernel.cu:771-773 reads gamma
  with the input's type). On the CPU the variant runs in bfloat16 too.
- fused ReLU (`--fused-relu`): the first feed-forward linear, its bias and
  the ReLU in one call, `torch._addmm_activation`, which on CUDA is one
  cuBLASLt matmul with the RELU_BIAS epilogue where the plain linear runs
  the BIAS epilogue (aten/src/ATen/native/cuda/Blas.cpp:286-300 and
  375-412); the backward pass is ReLU's and addmm's as autograd writes them
  (tools/autograd/derivatives.yaml, FunctionsManual.cpp mm_mat1_backward
  and mm_mat2_backward). On the CPU it is the unfused layer to the bit
  (addmm, then relu in place); on CUDA cuBLASLt may pick another algorithm
  for the other epilogue.
- compiled (`--compiled-trunk`): everything after the attention in a layer
  (the out-projection, both residual sums and norms, the feed-forward)
  compiled by torch.compile, the token count dynamic, the attention left
  eager between the compiled blocks: inductor fuses the casts, the bias
  additions, the ReLU, the residual sums and the norms around the matrix
  products, forward and backward. A compile failure raises (fullgraph):
  a run never falls back to other arithmetic silently.
- fp8 (`--fp8-trunk`): the four linear layers of every layer (the in- and
  out-projections, both feed-forward linears) multiply float8 operands with
  per-tensor scales, forward and backward (wesnoth_ai/fp8_linear.py); the
  attention, the norms and the residual sums are untouched. Not with the
  fused ReLU, whose epilogue belongs to a bfloat16 product; inside the
  compiled block, inductor compiles the quantization around the products.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.fp8_linear import GEMM_MULTIPLE, fp8_linear, fp8_problem
from wesnoth_ai.packed_trunk import PackedIndex, _dynamo_cache_entries, packed_attention

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrunkMode:
    """The variants a trainer runs (module docstring); the default runs none."""
    bf16_residual: bool = False
    fused_relu: bool = False
    compile_backend: Optional[str] = None       # torch.compile's backend; None: eager
    fp8: bool = False

    @property
    def active(self) -> bool:
        return self.bf16_residual or self.fused_relu or self.compile_backend is not None or self.fp8

    def describe(self) -> str:
        """The variants as one comparable string ("none" for the default)."""
        parts = (["bf16_residual"] if self.bf16_residual else []) + (["fused_relu"] if self.fused_relu else [])
        if self.compile_backend is not None:
            parts.append(f"compiled:{self.compile_backend}")
        if self.fp8:
            parts.append("fp8")
        return "+".join(parts) or "none"


def check_trunk_mode_supported(encoder: nn.TransformerEncoder, mode: TrunkMode) -> None:
    """The variants rewrite the packed trunk's post-norm ReLU layers; they
    refuse anything else, and the fused ReLU inside a compiled block."""
    if mode.fused_relu and mode.compile_backend is not None:
        raise ValueError("the fused ReLU is an eager call: the compiled trunk fuses its own")
    if mode.fp8:
        _check_fp8_supported(encoder, mode)
    for layer in encoder.layers:
        if layer.norm_first:
            raise NotImplementedError("trunk variants: norm_first layers")
        if layer.activation is not F.relu and not isinstance(layer.activation, nn.ReLU):
            raise NotImplementedError(f"trunk variants: the ReLU layer only, not {layer.activation}")
        mha = layer.self_attn
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn or not mha._qkv_same_embed_dim:
            raise NotImplementedError("trunk variants: one in-projection, no bias_kv or zero attention")


def _check_fp8_supported(encoder: nn.TransformerEncoder, mode: TrunkMode) -> None:
    """fp8 products: not with the fused ReLU, on a device that runs them,
    every linear's dimensions multiples of GEMM_MULTIPLE (fp8_linear)."""
    if mode.fused_relu:
        raise ValueError("the fused ReLU's epilogue belongs to a bfloat16 product; fp8 runs that product: not both")
    problem = fp8_problem(next(encoder.parameters()).device)
    if problem is not None:
        raise ValueError(problem)
    for layer in encoder.layers:
        for weight in (layer.self_attn.in_proj_weight, layer.self_attn.out_proj.weight, layer.linear1.weight,
                       layer.linear2.weight):
            if any(n % GEMM_MULTIPLE for n in weight.shape):
                raise ValueError(f"fp8 products need dimensions in multiples of {GEMM_MULTIPLE}, "
                                 f"not a {tuple(weight.shape)} weight")


def _cast(t: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    return None if t is None else t.to(dtype)


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], fp8: bool) -> torch.Tensor:
    """F.linear, with fp8 products when `fp8` (wesnoth_ai/fp8_linear.py)."""
    return fp8_linear(x, weight, bias) if fp8 else F.linear(x, weight, bias)


class _LinearReLU(torch.autograd.Function):
    """relu(x @ weight.T + bias) as one `_addmm_activation` call, its
    backward pass written as autograd writes relu's and addmm's: the ReLU
    mask from the saved output, then mm_mat1_backward's general branch for
    x (grad @ weight) and mm_mat2_backward's column-major branch for the
    weight (grad.T @ x, which the transpose's backward returns unchanged),
    and the bias's gradient summed to its size as the engine reduces a
    broadcast input's."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        out = torch._addmm_activation(bias, x, weight.t())
        ctx.save_for_backward(x, weight, out)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, out = ctx.saved_tensors
        g = torch.ops.aten.threshold_backward(grad, out, 0)
        gx = g.mm(weight) if ctx.needs_input_grad[0] else None
        gw = g.t().mm(x) if ctx.needs_input_grad[1] else None
        gb = g.sum_to_size(weight.shape[0]) if ctx.needs_input_grad[2] else None
        return gx, gw, gb


def linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """relu(F.linear(x, weight, bias)) for 2-D x, in one call (module docstring)."""
    return _LinearReLU.apply(x, weight, bias)


def layer_block(h: torch.Tensor, ctx: torch.Tensor,
                out_w: torch.Tensor, out_b: Optional[torch.Tensor],
                n1_w: Optional[torch.Tensor], n1_b: Optional[torch.Tensor],
                f1_w: torch.Tensor, f1_b: Optional[torch.Tensor],
                f2_w: torch.Tensor, f2_b: Optional[torch.Tensor],
                n2_w: Optional[torch.Tensor], n2_b: Optional[torch.Tensor],
                eps: float, fused_relu: bool, fp8: bool = False) -> torch.Tensor:
    """A post-norm layer after its attention (nn.TransformerEncoderLayer at
    torch 2.5.1 torch/nn/modules/transformer.py:902-906, without dropout):
    the out-projection of the attention's output `ctx` (compute dtype),
    then norm1(h + sa), then norm2(h + ff(h)). `h` is the residual stream
    in its dtype; the linears run in ctx's dtype on weights cast to it, the
    norms in h's dtype. A float32 h with a bfloat16 ctx is the autocast's
    arithmetic: the sum promotes to float32, the norm runs in float32, its
    output is cast for the linear. `fp8`: the linears' products in fp8."""
    cd, rd = ctx.dtype, h.dtype
    E = h.shape[-1]
    sa = _linear(ctx, out_w.to(cd), _cast(out_b, cd), fp8)
    h = F.layer_norm(h + sa, (E,), _cast(n1_w, rd), _cast(n1_b, rd), eps)
    hc = h.to(cd)
    w1, b1 = f1_w.to(cd), _cast(f1_b, cd)
    r = linear_relu(hc, w1, b1) if fused_relu and b1 is not None else F.relu(_linear(hc, w1, b1, fp8))
    ff = _linear(r, f2_w.to(cd), _cast(f2_b, cd), fp8)
    return F.layer_norm(h + ff, (E,), _cast(n2_w, rd), _cast(n2_b, rd), eps)


def _block_weights(layer: nn.TransformerEncoderLayer) -> List[Optional[torch.Tensor]]:
    """layer_block's weight arguments, in its order."""
    return [layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias,
            layer.norm1.weight, layer.norm1.bias, layer.linear1.weight, layer.linear1.bias,
            layer.linear2.weight, layer.linear2.bias, layer.norm2.weight, layer.norm2.bias]


class CompiledBlock:
    """`torch.compile(layer_block, fullgraph=True)` with the token count
    marked dynamic on every call: one graph for every step's token count
    (a graph per dtype, grad mode and backend). Compile errors, graph
    breaks and the cache-size limit raise (fullgraph). `stats` reads
    dynamo's cache entries of layer_block's code, which every
    CompiledBlock of a process shares, and the first call's wall time
    (the forward graph's compile, the backward's comes at the first
    backward pass)."""

    def __init__(self, backend: str):
        self.backend = backend
        # The telemetry's backward passes retain the graph, which a compiled
        # backward with donated buffers refuses: off, as torch 2.5.1 has it
        # (torch/_functorch/config.py:185); later versions turn it on.
        import torch._functorch.config as functorch_config
        if hasattr(functorch_config, "donated_buffer"):
            functorch_config.donated_buffer = False
        self._compiled = torch.compile(layer_block, backend=backend, fullgraph=True, dynamic=False)
        self.calls = 0
        self.first_call_seconds: Optional[float] = None

    def __call__(self, h, ctx, *weights, eps: float, fused_relu: bool, fp8: bool = False):
        if h.shape[0] < 2:                     # dynamo specializes sizes 0 and 1
            return layer_block(h, ctx, *weights, eps, fused_relu, fp8)
        torch._dynamo.mark_dynamic(h, 0)
        torch._dynamo.mark_dynamic(ctx, 0)
        t0 = time.perf_counter() if self.first_call_seconds is None else None
        out = self._compiled(h, ctx, *weights, eps, fused_relu, fp8)
        if t0 is not None:
            self.first_call_seconds = time.perf_counter() - t0
            log.info("trunk block compiled (%s): first call %.1f s, %d dynamo cache entries", self.backend,
                     self.first_call_seconds, _dynamo_cache_entries(layer_block))
        self.calls += 1
        return out

    def stats(self) -> dict:
        return {"backend": self.backend, "calls": self.calls, "first_call_seconds": self.first_call_seconds,
                "cache_entries": _dynamo_cache_entries(layer_block)}


def _attention(mha: nn.MultiheadAttention, hc: torch.Tensor, index: PackedIndex, fp8: bool = False) -> torch.Tensor:
    """The in-projection and the packed attention of `hc` (compute dtype),
    as packed_trunk._self_attention with the weights cast to hc's dtype;
    [total, E]. `fp8`: the in-projection's product in fp8."""
    total, E = hc.shape
    qkv = _linear(hc, mha.in_proj_weight.to(hc.dtype), _cast(mha.in_proj_bias, hc.dtype), fp8)
    q, k, v = qkv.view(total, 3, mha.num_heads, E // mha.num_heads).unbind(1)
    return packed_attention(q, k, v, index).reshape(total, E)


def trunk_dtypes(x: torch.Tensor, mode: TrunkMode):
    """(compute, residual) dtypes: the autocast's dtype where one is active,
    else bfloat16 under the bf16 residual and x's dtype otherwise; the
    residual stream in the compute dtype under the bf16 residual, else in
    x's dtype (float32)."""
    device = x.device.type
    if torch.is_autocast_enabled(device):
        compute = torch.get_autocast_dtype(device)
    else:
        compute = torch.bfloat16 if mode.bf16_residual else x.dtype
    return compute, (compute if mode.bf16_residual else x.dtype)


def run_trunk(encoder: nn.TransformerEncoder, x: torch.Tensor, index: PackedIndex, mode: TrunkMode,
              compiled: Optional[CompiledBlock] = None) -> torch.Tensor:
    """packed_trunk.packed_trunk under `mode` (module docstring), x the
    packed tokens [total, d]. The output is the residual stream's dtype
    under an autocast (the heads cast to the compute dtype anyway), x's
    dtype otherwise."""
    if mode.compile_backend is not None and compiled is None:
        raise ValueError("a compiled trunk mode needs its CompiledBlock")
    device = x.device.type
    under_autocast = torch.is_autocast_enabled(device)
    compute, residual = trunk_dtypes(x, mode)
    block = compiled if mode.compile_backend is not None else None
    with torch.autocast(device, enabled=False):
        h = x.to(residual)
        for layer in encoder.layers:
            ctx = _attention(layer.self_attn, h.to(compute), index, mode.fp8)
            weights = _block_weights(layer)
            if block is not None:
                h = block(h, ctx, *weights, eps=layer.norm1.eps, fused_relu=False, fp8=mode.fp8)
            else:
                h = layer_block(h, ctx, *weights, layer.norm1.eps, mode.fused_relu, mode.fp8)
        if encoder.norm is not None:
            norm = encoder.norm
            h = F.layer_norm(h, (h.shape[-1],), _cast(norm.weight, residual), _cast(norm.bias, residual), norm.eps)
    return h if under_autocast else h.to(x.dtype)
