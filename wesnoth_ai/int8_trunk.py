"""The packed trunk's inference loop with int8 linear layers on the HIP
kernels of wesnoth_ai/hip_int8.py (the RX 6600; docs/int8_linear_prereg_20261029.md).

The loop is WesnothModel._run_hip_fp16_trunk's with the four linears of
every layer on int8 operands:
  qkv   the layer's input, quantized by the add-and-norm that wrote it
        (hip_add_layer_norm_q; the first layer's by hip_quantize_rows);
  out   the attention's fp16 output, quantized by hip_quantize_rows;
  ff1   norm1's output, quantized by its add-and-norm; the ReLU in the
        GEMM's epilogue;
  ff2   the ReLU's fp16 output, quantized by hip_quantize_rows.
Each weight is int8 with a scale per output channel, quantized once from
the float32 parameters (`Int8TrunkWeights.build`); the biases stay
float32 and are added in the epilogue. The residual stream stays float32,
the attention and the heads fp16, as on the fp16 path.

`int8_classes` keeps a subset of the four on int8 and runs the others as
the fp16 path does (F.linear on the model's fp16 weight copy): the
pre-registered diagnostic of where the error comes from.

`smooth` (the prereg's arm S): per layer and linear, a float32 factor per
input channel that multiplies the input before its quantization while the
weight is quantized divided by it (SmoothQuant's migration, exact before
rounding). `record_activation_ranges` reads each linear input's largest
magnitude per channel from a float32 pass over calibration positions, and
`smoothing_factors` turns those ranges into the factors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.hip_int8 import (
    hip_add_layer_norm_q, hip_int8_linear, hip_quantize_rows, int8_width_served, quantize_weight,
)
from wesnoth_ai.hip_kernels import HEAD_DIMS, hip_add_layer_norm, hip_varlen_attention
from wesnoth_ai import packed_trunk  # noqa: F401 -- registers the packed attention op the calibration calls

LINEAR_CLASSES = ("qkv", "out", "ff1", "ff2")
_K_MULTIPLE = 32            # the GEMM's K tile (hip_int8._BK)


@dataclass
class Int8Linear:
    """One linear on int8: the weight [N, K] and its scale per output
    channel, the float32 bias, and the input's smoothing factor [K] or None."""
    wq: torch.Tensor
    ws: torch.Tensor
    bias: torch.Tensor
    smooth: Optional[torch.Tensor]

    @classmethod
    def build(cls, weight: torch.Tensor, bias: Optional[torch.Tensor], device: torch.device,
              smooth: Optional[torch.Tensor] = None) -> "Int8Linear":
        w = weight.detach().float().cpu()
        m = None if smooth is None else smooth.detach().float().cpu()
        wq, ws = quantize_weight(w, m)
        b = torch.zeros(w.shape[0]) if bias is None else bias.detach().float().cpu()
        return cls(wq.to(device), ws.to(device), b.to(device), None if m is None else m.to(device))

    def __call__(self, inp: Tuple[torch.Tensor, torch.Tensor], relu: bool = False) -> torch.Tensor:
        return hip_int8_linear(inp[0], inp[1], self.wq, self.ws, self.bias, relu=relu)


@dataclass
class Int8Layer:
    qkv: Int8Linear
    out: Int8Linear
    ff1: Int8Linear
    ff2: Int8Linear
    norm1: Tuple[torch.Tensor, torch.Tensor, float]
    norm2: Tuple[torch.Tensor, torch.Tensor, float]
    heads: int


def check_int8_trunk_supported(encoder: nn.TransformerEncoder) -> None:
    """ValueError unless the int8 loop runs this trunk: post-norm ReLU
    layers with one in-projection and affine norms, widths the quantizers
    take, K a multiple of the GEMM's K tile, a head size the attention
    kernel takes."""
    for layer in encoder.layers:
        mha = layer.self_attn
        if layer.norm_first:
            raise ValueError("int8 trunk: post-norm layers only")
        if layer.activation is not F.relu and not isinstance(layer.activation, nn.ReLU):
            raise ValueError(f"int8 trunk: the ReLU layer only, not {layer.activation}")
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn or not mha._qkv_same_embed_dim:
            raise ValueError("int8 trunk: one in-projection, no bias_kv or zero attention")
        if any(n.weight is None or n.bias is None for n in (layer.norm1, layer.norm2)):
            raise ValueError("int8 trunk: affine layer norms only")
        E, d_ff = mha.embed_dim, layer.linear1.out_features
        if not (int8_width_served(E) and int8_width_served(d_ff)) or E % _K_MULTIPLE or d_ff % _K_MULTIPLE:
            raise ValueError(f"int8 trunk: widths {E} and {d_ff} are not ones the kernels take")
        if E // mha.num_heads not in HEAD_DIMS:
            raise ValueError(f"int8 trunk: head size {E // mha.num_heads} is not one the attention takes")


@dataclass
class Int8TrunkWeights:
    """The trunk's linears on int8 and its norms in float32, on `device`,
    built from the encoder's parameters at weight `version`."""
    layers: List[Int8Layer]
    final_norm: Optional[Tuple[torch.Tensor, torch.Tensor, float]]
    device: torch.device
    version: int

    @classmethod
    def build(cls, encoder: nn.TransformerEncoder, device: torch.device, version: int,
              smooth: Optional[Sequence[Dict[str, torch.Tensor]]] = None) -> "Int8TrunkWeights":
        check_int8_trunk_supported(encoder)

        def norm(n):
            return (n.weight.detach().float().to(device), n.bias.detach().float().to(device), float(n.eps))
        layers = []
        for i, layer in enumerate(encoder.layers):
            mha = layer.self_attn
            sm = smooth[i] if smooth is not None else {}
            layers.append(Int8Layer(
                qkv=Int8Linear.build(mha.in_proj_weight, mha.in_proj_bias, device, sm.get("qkv")),
                out=Int8Linear.build(mha.out_proj.weight, mha.out_proj.bias, device, sm.get("out")),
                ff1=Int8Linear.build(layer.linear1.weight, layer.linear1.bias, device, sm.get("ff1")),
                ff2=Int8Linear.build(layer.linear2.weight, layer.linear2.bias, device, sm.get("ff2")),
                norm1=norm(layer.norm1), norm2=norm(layer.norm2), heads=mha.num_heads))
        final = None if encoder.norm is None else norm(encoder.norm)
        return cls(layers, final, device, version)


def run_int8_trunk(x: torch.Tensor, index, weights: Int8TrunkWeights,
                   int8_classes: Sequence[str] = LINEAR_CLASSES, fp16_layers=None) -> torch.Tensor:
    """The packed trunk's layers on x [total, d] (module docstring); float32
    rows out. `index`: the packed index (cu_seqlens, max_len). With a class
    left out of `int8_classes`, that linear runs on `fp16_layers` (the
    model's PackedTrunkWeights in fp16, packed_trunk.LayerWeights order)."""
    on8 = set(int8_classes)
    if on8 - set(LINEAR_CLASSES):
        raise ValueError(f"unknown linear classes {sorted(on8 - set(LINEAR_CLASSES))}")
    if on8 != set(LINEAR_CLASSES) and fp16_layers is None:
        raise ValueError("a linear class off int8 needs the fp16 weight copy")
    layers = weights.layers
    total, E = x.shape

    def feed(t16: torch.Tensor, cls: str, lin: Int8Linear):
        """An fp16 tensor as class `cls`'s input: quantized for int8."""
        return hip_quantize_rows(t16, lin.smooth) if cls in on8 else t16

    def add_norm(h32, y16, norm, cls: Optional[str], lin: Optional[Int8Linear]):
        """The add and norm, its output formed as the next linear's input."""
        w, b, eps = norm
        if cls is not None and cls in on8:
            o32, q, s = hip_add_layer_norm_q(h32, y16, w, b, eps, lin.smooth)
            return o32, (q, s)
        return hip_add_layer_norm(h32, y16, w, b, eps)

    def linear(cls: str, lin: Int8Linear, inp, i: int, relu: bool = False) -> torch.Tensor:
        if cls in on8:
            return lin(inp, relu=relu)
        f = fp16_layers[i]
        w, b = {"qkv": (f[0], f[1]), "out": (f[2], f[3]), "ff1": (f[6], f[7]), "ff2": (f[8], f[9])}[cls]
        if cls == "qkv":
            w, b = w.view(3 * E, E), None if b is None else b.view(3 * E)
        y = F.linear(inp, w, b)
        return F.relu(y) if relu else y

    with torch.autocast("cuda", enabled=False):
        h32 = x.float()
        inp = feed(h32.half(), "qkv", layers[0].qkv)
        for i, lw in enumerate(layers):
            head_dim = E // lw.heads
            qkv = linear("qkv", lw.qkv, inp, i)
            q, k, v = qkv.view(total, 3, lw.heads, head_dim).unbind(1)
            ctx = hip_varlen_attention(q, k, v, index.cu_seqlens, index.max_len)
            sa = linear("out", lw.out, feed(ctx.view(total, E), "out", lw.out), i)
            h32, inp = add_norm(h32, sa, lw.norm1, "ff1", lw.ff1)
            f = linear("ff1", lw.ff1, inp, i, relu=True)
            ff = linear("ff2", lw.ff2, feed(f, "ff2", lw.ff2), i)
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            h32, inp = add_norm(h32, ff, lw.norm2, "qkv" if nxt is not None else None,
                                nxt.qkv if nxt is not None else None)
        if weights.final_norm is not None:
            w, b, eps = weights.final_norm
            h32 = F.layer_norm(h32, (E,), w, b, eps)
    return h32


# ---------------------------------------------------------------------
# Arm S: SmoothQuant's per-channel migration
# ---------------------------------------------------------------------

def record_activation_ranges(x: torch.Tensor, index, encoder: nn.TransformerEncoder,
                             ranges: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """The packed trunk's layers in float32 on x [total, d] (F.linear on the
    encoder's parameters, the attention per segment in float32), folding
    each linear input's largest magnitude per channel into `ranges` (per
    layer a dict by linear class, filled on first use). Returns the
    float32 rows, so a forward's heads can run on them."""
    total, E = x.shape
    while len(ranges) < len(encoder.layers):
        ranges.append({})

    def seen(i: int, cls: str, t: torch.Tensor) -> None:
        m = t.detach().abs().amax(dim=0).float()
        ranges[i][cls] = m if cls not in ranges[i] else torch.maximum(ranges[i][cls], m)

    with torch.autocast(x.device.type, enabled=False):
        h = x.float()
        for i, layer in enumerate(encoder.layers):
            mha = layer.self_attn
            heads = mha.num_heads
            seen(i, "qkv", h)
            qkv = F.linear(h, mha.in_proj_weight.float(), mha.in_proj_bias.float())
            q, k, v = qkv.view(total, 3, heads, E // heads).unbind(1)
            ctx = torch.ops.wesnoth_ai.packed_attention.default(q, k, v, index.cu_seqlens,
                                                                index.max_len).reshape(total, E)
            seen(i, "out", ctx)
            sa = F.linear(ctx, mha.out_proj.weight.float(), mha.out_proj.bias.float())
            h = F.layer_norm(h + sa, (E,), layer.norm1.weight.float(), layer.norm1.bias.float(), layer.norm1.eps)
            seen(i, "ff1", h)
            f = F.relu(F.linear(h, layer.linear1.weight.float(), layer.linear1.bias.float()))
            seen(i, "ff2", f)
            ff = F.linear(f, layer.linear2.weight.float(), layer.linear2.bias.float())
            h = F.layer_norm(h + ff, (E,), layer.norm2.weight.float(), layer.norm2.bias.float(), layer.norm2.eps)
        if encoder.norm is not None:
            n = encoder.norm
            h = F.layer_norm(h, (E,), n.weight.float(), n.bias.float(), n.eps)
    return h


# SmoothQuant's reference code clamps the weight ranges and the factors at
# 1e-5 (an all-zero weight column; a channel never seen nonzero in
# calibration); `smoothing_factors` reports where the factors' floor binds.
_SMOOTH_FLOOR = 1e-5


def smoothing_factors(encoder: nn.TransformerEncoder, ranges: Sequence[Dict[str, torch.Tensor]],
                      alpha: float = 0.5) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    """(per layer and linear class, the factor m = 1 / s that multiplies the
    input, s_k = max|X_k|^alpha / max_j |W_jk|^(1 - alpha) (SmoothQuant,
    Xiao et al. 2023); the count of channels where the floor bound s)."""
    out, floored = [], 0
    for layer, rng in zip(encoder.layers, ranges):
        weights = {"qkv": layer.self_attn.in_proj_weight, "out": layer.self_attn.out_proj.weight,
                   "ff1": layer.linear1.weight, "ff2": layer.linear2.weight}
        factors = {}
        for cls, w in weights.items():
            w_max = w.detach().float().abs().amax(dim=0).cpu().clamp_min(_SMOOTH_FLOOR)
            s = rng[cls].cpu().pow(alpha) / w_max.pow(1.0 - alpha)
            floored += int((s < _SMOOTH_FLOOR).sum())
            factors[cls] = 1.0 / s.clamp_min(_SMOOTH_FLOOR)
        out.append(factors)
    return out, floored
