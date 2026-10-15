"""fp8 matrix products in the inference server's packed trunk
(tools/eval_inference_server.py --fp8-inference, WesnothModel.infer_fp8).

The four linear layers of every layer (the in- and out-projections, both
feed-forward linears) multiply float8_e4m3fn operands with one float32 scale
per tensor: `torch._scaled_mm` on CUDA (compute capability 8.9, the 4090),
its emulation on the CPU (wesnoth_ai/fp8_linear.py `scaled_mm`, which also
gives the operand layouts `_scaled_mm` requires). The products accumulate in
float32 and come out in bfloat16, the bias added in the product's epilogue.
Everything else is the bf16 residual stream's arithmetic
(WesnothModel.infer_bf16_residual, wesnoth_ai/trunk_modes.py): the residual
sums and layer norms in bfloat16, the attention, the ReLU. A rounding change:
for self-play generation only, never a strength match.

Per-tensor scales only: at torch 2.5.1 a row-wise scale is a CUTLASS kernel
built for sm90 alone (aten/src/ATen/native/cuda/RowwiseScaledMM.cu: ArchTag
Sm90), so on the 4090 `_scaled_mm` takes two scalars (Blas.cpp:912-915).

Weights. Each weight's scale is the power of two that puts its largest
magnitude in e4m3's top binade (fp8_linear.quantize), computed once from the
frozen weight when the copy is built, and again when the weights change
(load_state_dict bumps the model's weight version).

Activations. e4m3 is a floating format: its three mantissa bits give every
normal value the same relative precision (a rounding error of 2^-4 at most)
whatever the scale, so a scale only has to keep the largest value at or
under 448 and the bulk of the values above 2^-6, the smallest normal
number, 14.8 binades below 448. Every input of a layer past the first is
bounded by the weights alone (`bounds`):

  - a layer norm's output x = gamma * z + beta, z of mean 0 with
    sum(z^2) <= E (the norm divides by sqrt(var + eps)), so
    |z_k| <= sqrt(E - 1) and |x_k| <= |gamma_k| sqrt(E - 1) + |beta_k|;
  - a linear of a layer norm's output, y_j = sum_k W_jk x_k + b_j
    = sum_k a_jk z_k + c_j with a_j = W_j * gamma and c_j = W_j . beta + b_j:
    z has mean 0, so sum_k a_jk z_k = sum_k (a_jk - mean(a_j)) z_k, and by
    Cauchy-Schwarz |y_j| <= ||a_j - mean(a_j)||_2 sqrt(E) + |c_j|, plus the
    error of the input's roundings to bfloat16 and to e4m3 (relative 2^-8
    and 2^-4, and half e4m3's smallest subnormal step absolute), bounded by
    ||W_j||_2 ||x||_2 and ||W_j||_1;
  - the attention's output, a convex combination of the value rows: the
    bound of the in-projection's value rows;
  - the ReLU of the first feed-forward linear: that linear's bound.

Each bound is widened by the output's rounding to bfloat16 and by MARGIN.
The scale of such an input is the power of two that puts its bound at or
under 448, fixed with the weights: no pass over the input reads its largest
value, the cast fuses into the kernel that writes the input when the loop is
compiled, and a position's rounding does not depend on its batch. The first
layer's two inputs (the embedding, and its attention's output, whose value
rows read the embedding) have no bound from the weights: their scale is the
batch's, the power of two that puts the input's largest magnitude in the top
binade, read at every call. Every cast saturates at 448 (a clamp), so a
bound that failed would clip, never make a NaN.

Rejected: a per-batch scale for every input, as the trainer's fp8 recipe has
it, because reading each input's largest value is one more pass over it
before its cast can start (about 1 GB of reads a 23,000-token batch over the
eight layers, about a millisecond on the 4090, against the products' saving
of one to two); and int8 products (`torch._int_mm`), because they return
int32 with no scale or bias epilogue, so every product's output is written
and read back at twice bfloat16's bytes (about 2.5 GB more traffic a batch),
and an integer grid holds the heavy-tailed activations less well than e4m3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesnoth_ai.fp8_linear import E4M3, GEMM_MULTIPLE, MAX_SCALE_EXPONENT, fp8_problem, quantize, scaled_mm
from wesnoth_ai import packed_trunk  # noqa: F401 -- registers the packed attention op the loop calls

E4M3_MAX = 448.0
# e4m3's largest relative rounding error on a normal value (three mantissa
# bits, round to nearest), bfloat16's (seven), and e4m3's largest absolute
# one below its smallest normal number (half its subnormal step 2^-9).
E4M3_ROUNDING = 2.0 ** -4
BF16_ROUNDING = 2.0 ** -8
E4M3_SUBNORMAL_ERROR = 2.0 ** -10
# The input's rounding to bfloat16, then to e4m3, as one relative error.
INPUT_ROUNDING = (1.0 + BF16_ROUNDING) * (1.0 + E4M3_ROUNDING) - 1.0
# Every bound's widening besides the roundings it counts (the float32
# accumulation's error is under K * 2^-24 of the bound, 2^-13 at K = 1536).
MARGIN = 1.0 + 2.0 ** -6
_TINY = torch.finfo(torch.float32).tiny

# One product's operands: the weight in e4m3 [N, K] (row-major, so its
# transpose is the column-major second operand), the weight's inverse scale,
# the bias in bfloat16, and the input's scale and inverse (float32 scalars),
# both None for an input scaled by its batch.
Product = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
# One layer: the in-projection (its bias kept [3, heads, head_dim], from
# which the loop reads the head count), the out-projection, norm1's weight
# and bias, the two feed-forward products, norm2's weight and bias.
Fp8Layer = Tuple[Product, Product, torch.Tensor, torch.Tensor, Product, Product, torch.Tensor, torch.Tensor]


def check_fp8_inference_supported(encoder: nn.TransformerEncoder, device: torch.device) -> None:
    """ValueError unless the fp8 loop runs this trunk on `device`: post-norm
    ReLU layers with one in-projection, every linear's dimensions multiples
    of GEMM_MULTIPLE, a device that runs fp8 products."""
    problem = fp8_problem(device)
    if problem is not None:
        raise ValueError(problem)
    for layer in encoder.layers:
        mha = layer.self_attn
        if layer.norm_first:
            raise ValueError("fp8 inference: post-norm layers only")
        if layer.activation is not F.relu and not isinstance(layer.activation, nn.ReLU):
            raise ValueError(f"fp8 inference: the ReLU layer only, not {layer.activation}")
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn or not mha._qkv_same_embed_dim:
            raise ValueError("fp8 inference: one in-projection, no bias_kv or zero attention")
        for weight in (mha.in_proj_weight, mha.out_proj.weight, layer.linear1.weight, layer.linear2.weight):
            if any(n % GEMM_MULTIPLE for n in weight.shape):
                raise ValueError(f"fp8 products need dimensions in multiples of {GEMM_MULTIPLE}, "
                                 f"not a {tuple(weight.shape)} weight")


# ---- the bounds --------------------------------------------------------------

def norm_output_bound(gamma: torch.Tensor, beta: torch.Tensor) -> float:
    """The largest magnitude a layer norm's output can take (module
    docstring), its rounding to bfloat16 and MARGIN included."""
    E = gamma.numel()
    g, b = gamma.double().abs(), beta.double().abs()
    return float((g * math.sqrt(E - 1) + b).max()) * (1.0 + BF16_ROUNDING) * MARGIN


def linear_of_norm_bounds(weight: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                          input_scale: float) -> torch.Tensor:
    """float64 [N]: the largest magnitude each output of `weight` (its
    dequantized e4m3 values) and `bias` can take on a layer norm's output
    (gamma, beta) rounded to bfloat16 and then to e4m3 under `input_scale`
    (module docstring), the output's rounding to bfloat16 and MARGIN
    included."""
    W, b = weight.double(), bias.double()
    g, be = gamma.double(), beta.double()
    E = g.numel()
    a = W * g
    core = (a - a.mean(dim=1, keepdim=True)).norm(dim=1) * math.sqrt(E)
    constant = (W @ be + b).abs()
    input_norm = float(g.abs().max()) * math.sqrt(E) + float(be.norm())
    rounding = INPUT_ROUNDING * W.norm(dim=1) * input_norm + W.abs().sum(dim=1) * E4M3_SUBNORMAL_ERROR / input_scale
    return (core + constant + rounding) * (1.0 + BF16_ROUNDING) * MARGIN


def scale_for(bound: float) -> float:
    """The power of two that puts `bound` at or under E4M3_MAX."""
    exponent = math.floor(math.log2(E4M3_MAX / max(bound, _TINY)))
    return 2.0 ** max(-MAX_SCALE_EXPONENT, min(MAX_SCALE_EXPONENT, exponent))


@dataclass(frozen=True)
class LayerBounds:
    """A layer's static input bounds; None where its batch scales the input."""
    in_proj: Optional[float]
    out_proj: Optional[float]
    linear1: float
    linear2: float


def bounds(layers: List[dict]) -> List[LayerBounds]:
    """Each layer's input bounds from its weights (module docstring).
    `layers`: per layer the dequantized in-projection weight [3E, E] and its
    bias [3E], linear1's weight and bias, norm1's and norm2's weight and
    bias, all on the CPU."""
    out = []
    previous_norm2 = None
    for layer in layers:
        E = layer["norm1_w"].numel()
        if previous_norm2 is None:
            in_bound = out_bound = None
        else:
            g2, b2 = previous_norm2
            in_bound = norm_output_bound(g2, b2)
            values = linear_of_norm_bounds(layer["in_w"][2 * E:], layer["in_b"][2 * E:], g2, b2, scale_for(in_bound))
            out_bound = float(values.max()) * (1.0 + BF16_ROUNDING)        # the attention's rounding
        ff_in = norm_output_bound(layer["norm1_w"], layer["norm1_b"])
        ff_mid = float(linear_of_norm_bounds(layer["linear1_w"], layer["linear1_b"], layer["norm1_w"],
                                             layer["norm1_b"], scale_for(ff_in)).max())
        out.append(LayerBounds(in_bound, out_bound, ff_in, ff_mid))
        previous_norm2 = (layer["norm2_w"], layer["norm2_b"])
    return out


# ---- the weight copy -----------------------------------------------------------

def _frozen(t: torch.Tensor) -> nn.Parameter:
    """A frozen copy held as a Parameter, so dynamo keeps its shape static
    under dynamic=True (as packed_trunk.PackedTrunkWeights' copies)."""
    return nn.Parameter(t.detach().clone(), requires_grad=False)


def _scalar(value: float, device: torch.device) -> nn.Parameter:
    return _frozen(torch.tensor(value, dtype=torch.float32, device=device))


def _layer_tensors(layer: nn.TransformerEncoderLayer) -> dict:
    """A layer's parameters (biases as zeros where absent), on the CPU."""
    mha = layer.self_attn

    def bias(b, n):
        return torch.zeros(n) if b is None else b.detach().float().cpu()

    def norm(n):
        E = n.normalized_shape[0]
        w = torch.ones(E) if n.weight is None else n.weight.detach().float().cpu()
        return w, bias(n.bias, E)
    E = mha.embed_dim
    n1_w, n1_b = norm(layer.norm1)
    n2_w, n2_b = norm(layer.norm2)
    return {"in_w": mha.in_proj_weight.detach().float().cpu(), "in_b": bias(mha.in_proj_bias, 3 * E),
            "out_w": mha.out_proj.weight.detach().float().cpu(), "out_b": bias(mha.out_proj.bias, E),
            "linear1_w": layer.linear1.weight.detach().float().cpu(),
            "linear1_b": bias(layer.linear1.bias, layer.linear1.out_features),
            "linear2_w": layer.linear2.weight.detach().float().cpu(),
            "linear2_b": bias(layer.linear2.bias, E),
            "norm1_w": n1_w, "norm1_b": n1_b, "norm2_w": n2_w, "norm2_b": n2_b,
            "heads": mha.num_heads}


def _quantized(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """(w in e4m3, its inverse scale) by fp8_linear.quantize's rule."""
    w8, inverse = quantize(w, E4M3)
    return w8, float(inverse)


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """t's values as bfloat16 holds them, in float32 (what the bounds read)."""
    return t.to(torch.bfloat16).float()


def build_values(encoder: nn.TransformerEncoder):
    """(per layer the tuple of tensors the loop reads, on the CPU, in
    Fp8Layer's order; the final norm's weight and bias in bfloat16 or None;
    the layers' bounds), all from the encoder's current parameters."""
    raw = [_layer_tensors(layer) for layer in encoder.layers]
    quantized, for_bounds = [], []
    for t in raw:
        q = {name: _quantized(t[f"{name}_w"]) for name in ("in", "out", "linear1", "linear2")}
        quantized.append(q)
        for_bounds.append({"in_w": q["in"][0].float() * q["in"][1], "in_b": _bf16(t["in_b"]),
                           "linear1_w": q["linear1"][0].float() * q["linear1"][1],
                           "linear1_b": _bf16(t["linear1_b"]),
                           **{k: _bf16(t[k]) for k in ("norm1_w", "norm1_b", "norm2_w", "norm2_b")}})
    layer_bounds = bounds(for_bounds)
    layers = []
    for t, q, lb in zip(raw, quantized, layer_bounds):
        E, heads = t["out_w"].shape[0], t["heads"]

        def product(name, bound, bias_shape=None):
            w8, inverse = q[name]
            b = t[f"{name}_b"].to(torch.bfloat16)
            if bias_shape is not None:
                b = b.reshape(bias_shape)
            if bound is None:
                return (w8, torch.tensor(inverse), b, None, None)
            s = scale_for(bound)
            return (w8, torch.tensor(inverse), b, torch.tensor(s), torch.tensor(1.0 / s))
        layers.append((product("in", lb.in_proj, (3, heads, E // heads)), product("out", lb.out_proj),
                       t["norm1_w"].to(torch.bfloat16), t["norm1_b"].to(torch.bfloat16),
                       product("linear1", lb.linear1), product("linear2", lb.linear2),
                       t["norm2_w"].to(torch.bfloat16), t["norm2_b"].to(torch.bfloat16)))
    final = None
    if encoder.norm is not None:
        final = (encoder.norm.weight.detach().to(torch.bfloat16).cpu(),
                 encoder.norm.bias.detach().to(torch.bfloat16).cpu())
    return layers, final, layer_bounds


def _flatten(layers) -> List[Optional[torch.Tensor]]:
    out = []
    for layer in layers:
        for entry in layer:
            out.extend(entry if isinstance(entry, tuple) else (entry,))
    return out


@dataclass
class Fp8TrunkWeights:
    """The trunk's weights as the fp8 loop reads them (`layers`, Fp8Layer
    per layer; `final_norm`), frozen copies on `device`, with the static
    input bounds they were scaled from. `refresh` recomputes them from the
    encoder's current parameters in place, so a compiled loop's guards see
    the same tensors."""
    layers: List[Fp8Layer]
    final_norm: Optional[Tuple[torch.Tensor, torch.Tensor]]
    bounds: List[LayerBounds]
    device: torch.device
    version: int

    @classmethod
    def build(cls, encoder: nn.TransformerEncoder, device: torch.device, version: int) -> "Fp8TrunkWeights":
        values, final, layer_bounds = build_values(encoder)

        def frozen(entry):
            if isinstance(entry, tuple):
                return tuple(None if t is None else _frozen(t.to(device)) for t in entry)
            return _frozen(entry.to(device))
        layers = [tuple(frozen(entry) for entry in layer) for layer in values]
        final_norm = None if final is None else tuple(_frozen(t.to(device)) for t in final)
        return cls(layers, final_norm, layer_bounds, device, version)

    def refresh(self, encoder: nn.TransformerEncoder, version: int) -> None:
        values, final, layer_bounds = build_values(encoder)
        mine, theirs = _flatten(self.layers), _flatten(values)
        if [t is None for t in mine] != [t is None for t in theirs]:
            raise ValueError("the refreshed weights scale another input by its batch: rebuild the copy")
        with torch.no_grad():
            for dst, src in zip(mine, theirs):
                if dst is not None:
                    dst.copy_(src)
            if self.final_norm is not None:
                for dst, src in zip(self.final_norm, final):
                    dst.copy_(src)
        self.bounds, self.version = layer_bounds, version

    def summary(self) -> dict:
        """The input scales as powers of two, per layer (None: the batch's)."""
        def exponent(p):
            return None if p[3] is None else int(round(math.log2(float(p[3]))))
        return {"input_scale_exponents": [[exponent(p) for p in (layer[0], layer[1], layer[4], layer[5])]
                                          for layer in self.layers],
                "bounds": [[b.in_proj, b.out_proj, b.linear1, b.linear2] for b in self.bounds]}


# ---- the loop ------------------------------------------------------------------

def batch_cast(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(t in e4m3 scaled by the power of two that puts its largest magnitude
    in the top binade, the inverse of that power as a float32 scalar)."""
    amax = t.abs().amax().float()
    exponent = torch.floor(torch.log2(E4M3_MAX / amax.clamp(min=_TINY))).clamp(-MAX_SCALE_EXPONENT,
                                                                                MAX_SCALE_EXPONENT)
    return (t * torch.exp2(exponent)).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3), torch.exp2(-exponent)


def static_cast(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """t times its static scale (a power of two: exact), saturated, in e4m3."""
    return (t * scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def fp8_product(x: torch.Tensor, product: Product, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x [M, K] (bfloat16) @ W.T + b with e4m3 operands, in bfloat16.
    `bias`: the product's bias reshaped (the in-projection's)."""
    w8, w_inverse, b, scale, inverse = product
    if scale is None:
        x8, inverse = batch_cast(x)
    else:
        x8 = static_cast(x, scale)
    return scaled_mm(x8, w8.t(), inverse, w_inverse, torch.bfloat16, b if bias is None else bias)


def make_fp8_trunk_layers(eps: float):
    """The fp8 layer loop over Fp8TrunkWeights (`layers`, `final_norm`),
    with packed_trunk.make_packed_trunk_layers' signature, so
    packed_trunk.CompiledPackedTrunk compiles it as it compiles that loop
    (the head count read from the in-projection bias's static shape)."""
    def fp8_trunk_layers(x, cu_seqlens, max_len, layers, final_norm):
        for in_p, out_p, ln1_w, ln1_b, ff1_p, ff2_p, ln2_w, ln2_b in layers:
            _, heads, head_dim = in_p[2].shape
            total, E = x.shape
            qkv = fp8_product(x, in_p, in_p[2].view(3 * E))
            q, k, v = qkv.view(total, 3, heads, head_dim).unbind(1)
            ctx = torch.ops.wesnoth_ai.packed_attention.default(q, k, v, cu_seqlens, max_len)
            x = F.layer_norm(x + fp8_product(ctx.view(total, E), out_p), (E,), ln1_w, ln1_b, eps)
            ff = fp8_product(F.relu(fp8_product(x, ff1_p)), ff2_p)
            x = F.layer_norm(x + ff, (E,), ln2_w, ln2_b, eps)
        if final_norm is not None:
            x = F.layer_norm(x, (x.shape[1],), final_norm[0], final_norm[1], eps)
        return x
    return fp8_trunk_layers
