"""Packed (variable-length) transformer trunk for batched inference
(docs/gpu_forward_design_20260904.md sections 4.3 and 5.3).

A padded batch [B, L_max, d] carries three interior pad runs per row
(hex, unit, recruit). The packed trunk gathers the real tokens of every
row into one [total, d] tensor -- row b becomes the segment
hex(H_b) | unit(U_b) | recruit(R_b) | global | end_turn -- runs the
per-token layers (in/out projections, LayerNorms, MLP) on it, and runs
attention per segment through the flash varlen kernel with cumulative
sequence lengths. No pad token is computed and no attention mask
exists. The heads read the padded layout back through index_select, so
model.PaddedOutput and its consumers are unchanged.

Kernel (torch 2.5.1, aten/src/ATen/native/native_functions.yaml:14814):

    _flash_attention_forward(Tensor query, Tensor key, Tensor value,
        Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k,
        float dropout_p, bool is_causal, bool return_debug_mask, *,
        float? scale=None, SymInt? window_size_left=None,
        SymInt? window_size_right=None, Tensor? seqused_k=None,
        Tensor? alibi_slopes=None)
      -> (output, softmax_logsumexp, philox_seed, philox_offset, debug_attn_mask)

With cum_seq_* given, aten/src/ATen/native/transformers/cuda/attention.cu
:935-964 calls pytorch_flash::mha_varlen_fwd (.../cuda/flash_attn/
flash_api.cpp:543), whose checks fix the calling convention: q, k, v are
[total, heads, head_dim] with a unit-stride last dim (595-597, 640-644);
fp16 or bf16 only (572); cu_seqlens int32, contiguous, on the device,
shape [B+1] (579-584, 598-599, 650-651); head_dim a multiple of 8 and at
most 256 (633-635); sm80 or newer (567). The output is
[total, heads, head_dim] (678). The softmax scale defaults to
1/sqrt(head_dim) (attention.cu:915-916), the default
F.scaled_dot_product_attention applies as well. torch 2.10 keeps the same
positional signature (checked on the local wheel).

Outside the kernel's domain (CPU, fp32) the same packed bookkeeping runs
with F.scaled_dot_product_attention per segment, so the layout is tested
without a GPU; production callers get the packed trunk only where the
flash kernel applies (WesnothModel._packed_trunk_applies).

Compiled variant (design note section 13): `CompiledPackedTrunk` runs
`torch.compile(dynamic=True, fullgraph=True)` over `packed_trunk_layers`,
the same layer loop written as a pure function of tensors with the
attention behind the custom op `wesnoth_ai::packed_attention`. The op is
opaque to dynamo and inductor (an extern kernel in the graph); its eager
kernel picks the flash varlen op or the per-segment SDPA at run time, so
neither the private flash schema nor the host-side segment loop is ever
traced. The weights come from `PackedTrunkWeights`, a native-dtype copy
of the encoder's parameters (bf16 on CUDA) held as nn.Parameters so
their shapes stay static; only `total`, the offsets' length and
`max_len` are symbolic.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_FLASH_DTYPES = (torch.float16, torch.bfloat16)


@dataclass
class PackedLayout:
    """Host-built (numpy int64) index arrays of one batch. Padded rows are
    addressed flat (b * L_max + position); packed rows by their offset in
    [total, d]. Padded actor, hex and unit slots point at their row's
    end_turn token: a finite value no consumer reads (the priors masks
    exclude padded slots, as they do for the padded trunk's outputs)."""
    src: np.ndarray          # [total] padded flat row of each packed token
    kind: np.ndarray         # [total] TokenKind of each packed token
    actor: np.ndarray        # [B * A_max] packed row of each actor slot
    hex: np.ndarray          # [B * H_max] packed row of each hex slot
    unit: np.ndarray         # [B * U_max] packed row of each unit slot
    glob: np.ndarray         # [B] packed row of each global token
    cu_seqlens: np.ndarray   # [B + 1] segment offsets
    actor_kind: np.ndarray   # [B, A_max] ActorKind per slot
    max_len: int

    def to_device(self, device: torch.device) -> "PackedIndex":
        """One host->device copy of every array (pinned and non-blocking
        on CUDA: no host synchronization, torch 2.5.1
        aten/src/ATen/native/cuda/Copy.cu:362-381), split into views."""
        parts = [self.src, self.kind, self.actor, self.hex, self.unit, self.glob, self.cu_seqlens]
        flat = torch.from_numpy(np.concatenate(parts))
        if device.type == "cuda":
            flat = flat.pin_memory().to(device, non_blocking=True)
        elif device.type != "cpu":
            flat = flat.to(device)
        src, kind, actor, hexes, unit, glob, cu = torch.split(flat, [len(p) for p in parts])
        return PackedIndex(src=src, kind=kind, actor=actor, hex=hexes, unit=unit, glob=glob,
                           cu_seqlens=cu.to(torch.int32), cu_host=self.cu_seqlens.tolist(),
                           max_len=self.max_len)


@dataclass
class PackedIndex:
    """A PackedLayout on the compute device (int64 views of one buffer;
    cu_seqlens int32 as the kernel requires)."""
    src: torch.Tensor
    kind: torch.Tensor
    actor: torch.Tensor
    hex: torch.Tensor
    unit: torch.Tensor
    glob: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_host: List[int]       # the same offsets on the host, for the SDPA fallback
    max_len: int


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """0 .. c-1 for every count c, concatenated: [sum(counts)]."""
    starts = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    return np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(starts, counts)


def _exclusive_cumsum(a: np.ndarray) -> np.ndarray:
    out = np.zeros(a.size + 1, dtype=np.int64)
    np.cumsum(a, out=out[1:])
    return out


def _sizes_array(sizes: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """[B, 3] int64 of (U_b, R_b, H_b)."""
    return np.asarray(sizes, dtype=np.int64).reshape(len(sizes), 3)


def _segment_sources(sizes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Source row of every packed token given, per sample, the source
    row of its first hex, unit, recruit, global and end_turn token
    (`starts` [B, 5]); the packed segment of a sample lists them in
    that order."""
    Us, Rs, Hs = sizes[:, 0], sizes[:, 1], sizes[:, 2]
    ones = np.ones(len(sizes), dtype=np.int64)
    counts = np.stack([Hs, Us, Rs, ones, ones], axis=1).reshape(-1)
    return np.repeat(starts.reshape(-1), counts) + _ragged_arange(counts)


def _padded_starts(sizes: np.ndarray, H_max: int, U_max: int, R_max: int) -> np.ndarray:
    """First-token rows in the flat padded layout (row b at b * L_max)."""
    L = H_max + U_max + R_max + 2
    row = np.arange(len(sizes), dtype=np.int64) * L
    return np.stack([row, row + H_max, row + H_max + U_max, row + H_max + U_max + R_max,
                     row + H_max + U_max + R_max + 1], axis=1)


def _stream_starts(sizes: np.ndarray) -> np.ndarray:
    """First-token rows in the stream-concatenated layout of
    EmbeddedStreams: every sample's hexes, then units, then recruits,
    one global row per sample, one shared end_turn row."""
    Us, Rs, Hs = sizes[:, 0], sizes[:, 1], sizes[:, 2]
    B = len(sizes)
    hex_off, unit_off, rec_off = _exclusive_cumsum(Hs), _exclusive_cumsum(Us), _exclusive_cumsum(Rs)
    th, tu, tr = int(hex_off[-1]), int(unit_off[-1]), int(rec_off[-1])
    return np.stack([hex_off[:-1], th + unit_off[:-1], th + tu + rec_off[:-1],
                     th + tu + tr + np.arange(B, dtype=np.int64),
                     np.full(B, th + tu + tr + B, dtype=np.int64)], axis=1)


def build_packed_layout(sizes: Sequence[Tuple[int, int, int]], H_max: int, U_max: int,
                        R_max: int, token_kind, actor_kind,
                        source: str = "padded") -> PackedLayout:
    """Index arrays for per-sample sizes (U_b, R_b, H_b). `src` addresses
    the tokens of `source`: "padded", the flat padded layout
    hex(H_max) | unit(U_max) | recruit(R_max) | global | end_turn of
    model.forward_streams (row b at b * L_max); "streams", the
    stream-concatenated tokens of EmbeddedStreams. The head arrays
    (actor, hex, unit, glob) and the offsets address packed rows either
    way. `token_kind` and `actor_kind` are the model's TokenKind and
    ActorKind tables, passed in so this module imports nothing from the
    model. Vectorized numpy; no per-sample loop."""
    B = len(sizes)
    sz = _sizes_array(sizes)
    Us, Rs, Hs = sz[:, 0], sz[:, 1], sz[:, 2]
    A_max = U_max + R_max + 1
    lengths = Us + Rs + Hs + 2
    cu = _exclusive_cumsum(lengths)
    if source == "padded":
        starts = _padded_starts(sz, H_max, U_max, R_max)
    elif source == "streams":
        starts = _stream_starts(sz)
    else:
        raise ValueError(f"build_packed_layout: unknown source {source!r}")
    src = _segment_sources(sz, starts)
    ones = np.ones(B, dtype=np.int64)
    counts = np.stack([Hs, Us, Rs, ones, ones], axis=1).reshape(-1)
    kind = np.repeat(np.tile(np.array([token_kind.HEX, token_kind.UNIT, token_kind.RECRUIT,
                                       token_kind.GLOBAL, token_kind.END_TURN], dtype=np.int64),
                             B), counts)
    seg0, end = cu[:-1, None], (cu[1:] - 1)[:, None]      # [B, 1] each
    a = np.arange(A_max, dtype=np.int64)[None, :]
    actor = np.where(a < (Us + Rs)[:, None], seg0 + Hs[:, None] + a, end)
    h = np.arange(H_max, dtype=np.int64)[None, :]
    hexes = np.where(h < Hs[:, None], seg0 + h, end)
    u = np.arange(U_max, dtype=np.int64)[None, :]
    unit = np.where(u < Us[:, None], seg0 + Hs[:, None] + u, end)
    kinds = np.full((B, A_max), actor_kind.END_TURN, dtype=np.int64)
    kinds[a < (Us + Rs)[:, None]] = actor_kind.RECRUIT
    kinds[a < Us[:, None]] = actor_kind.UNIT
    return PackedLayout(src=src, kind=kind, actor=actor.reshape(-1), hex=hexes.reshape(-1),
                        unit=unit.reshape(-1), glob=(end - 1).reshape(-1), cu_seqlens=cu,
                        actor_kind=kinds, max_len=int(lengths.max()) if B else 0)


def padded_gather_index(sizes: Sequence[Tuple[int, int, int]], H_max: int, U_max: int,
                        R_max: int) -> np.ndarray:
    """Stream row (EmbeddedStreams order) of every slot of the flat
    padded layout ([B * L_max]); pad slots point one past the last
    stream row, so gathering from the stream tokens with a zero row
    appended yields the padded streams with zeros at the pads, as
    pad_sequence builds them."""
    sz = _sizes_array(sizes)
    B = len(sizes)
    L = H_max + U_max + R_max + 2
    padded_rows = _segment_sources(sz, _padded_starts(sz, H_max, U_max, R_max))
    stream_rows = _segment_sources(sz, _stream_starts(sz))
    out = np.full(B * L, int(sz.sum()) + B + 1, dtype=np.int64)
    out[padded_rows] = stream_rows
    return out


@dataclass
class EmbeddedStreams:
    """A batch's token embeddings before the token-kind term, concatenated
    per stream: the hex tokens of every sample, then the unit tokens of
    every sample, the recruit tokens, one global row per sample and one
    end_turn row: [N, d], N = sum(H) + sum(U) + sum(R) + B + 1. Built by
    GameStateEncoder.encode_from_raw_embedded from one pinned host
    buffer; WesnothModel.forward_embedded orders the rows into the
    packed layout (build_packed_layout(source="streams")) or, for the
    padded trunk, into the padded streams (padded_gather_index)."""
    tokens: torch.Tensor
    sizes: List[Tuple[int, int, int]]   # (U_b, R_b, H_b)


# ---------------------------------------------------------------------
# One flat byte buffer for several typed arrays
# ---------------------------------------------------------------------

_NP_DTYPE = {torch.float64: np.float64, torch.float32: np.float32,
             torch.int64: np.int64, torch.int32: np.int32,
             torch.int8: np.int8, torch.uint8: np.uint8}


class FlatLayout:
    """Byte layout of several typed arrays in one flat uint8 buffer, so
    a batch's arrays cross to the device in one copy. Fields are placed
    in decreasing element size, so every offset is a multiple of its
    field's element size (what `Tensor.view(dtype)` requires) with no
    padding bytes."""
    __slots__ = ("fields", "nbytes")

    def __init__(self, fields: Sequence[Tuple[str, torch.dtype, Tuple[int, ...]]]):
        self.fields: List[Tuple[str, torch.dtype, Tuple[int, ...], int, int]] = []
        off = 0
        for name, dt, shape in sorted(fields, key=lambda f: -f[1].itemsize):
            n = math.prod(shape) * dt.itemsize
            self.fields.append((name, dt, tuple(shape), off, n))
            off += n
        self.nbytes = off

    def torch_views(self, buf: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: buf[off:off + n].view(dt).view(shape)
                for name, dt, shape, off, n in self.fields}

    def numpy_views(self, buf: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: buf[off:off + n].view(_NP_DTYPE[dt]).reshape(shape)
                for name, dt, shape, off, n in self.fields}


# ---------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------

def flash_varlen_applies(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the flash varlen kernel serves this device and dtype
    (flash_api.cpp:567-575: CUDA, fp16 or bf16)."""
    return device.type == "cuda" and dtype in _FLASH_DTYPES


def packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     index: PackedIndex) -> torch.Tensor:
    """Self-attention within each segment. q, k, v are
    [total, heads, head_dim]; the result has the same shape."""
    if flash_varlen_applies(q.device, q.dtype):
        return _flash_varlen(q, k, v, index.cu_seqlens, index.max_len)
    return _segment_sdpa(q, k, v, index.cu_host)


def _flash_varlen(q, k, v, cu_seqlens: torch.Tensor, max_len: int) -> torch.Tensor:
    """Positional arguments in the order of the 2.5.1 schema quoted in
    the module docstring: dropout 0, not causal, no debug mask, default
    scale."""
    return torch.ops.aten._flash_attention_forward(
        q, k, v, cu_seqlens, cu_seqlens, max_len, max_len, 0.0, False, False)[0]


def _segment_sdpa(q, k, v, cu_host: List[int]) -> torch.Tensor:
    """Reference attention for the packed layout off the flash kernel's
    domain: one F.scaled_dot_product_attention call per segment."""
    outs = []
    for lo, hi in zip(cu_host[:-1], cu_host[1:]):
        seg = [t[lo:hi].transpose(0, 1).unsqueeze(0) for t in (q, k, v)]   # [1, heads, n, hd]
        outs.append(F.scaled_dot_product_attention(*seg)[0].transpose(0, 1))
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------
# The layer loop
# ---------------------------------------------------------------------

def check_packed_trunk_supported(encoder: nn.TransformerEncoder) -> None:
    """The packed loop re-implements nn.TransformerEncoderLayer's
    post-norm forward (torch 2.5.1 torch/nn/modules/transformer.py:902-906,
    _sa_block 911-927, _ff_block 930-932) over the layer's own
    parameters; it refuses the layer options it does not reproduce."""
    first = encoder.layers[0]
    for layer in encoder.layers:
        mha = layer.self_attn
        if layer.norm_first:
            raise NotImplementedError("packed trunk: norm_first layers")
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
            raise NotImplementedError("packed trunk: add_bias_kv / add_zero_attn")
        if not mha._qkv_same_embed_dim:
            raise NotImplementedError("packed trunk: separate q/k/v projection weights")
        # The compiled loop closes over one activation and one eps.
        if not _same_activation(layer.activation, first.activation) or \
                layer.norm1.eps != first.norm1.eps or layer.norm2.eps != first.norm1.eps:
            raise NotImplementedError("packed trunk: layers with different activations or eps")
    if encoder.norm is not None and encoder.norm.eps != first.norm1.eps:
        raise NotImplementedError("packed trunk: final norm with a different eps")


def _same_activation(a, b) -> bool:
    """Functions by identity; modules (nn.TransformerEncoder deep-copies
    its layer) by type."""
    return a is b or (isinstance(a, nn.Module) and type(a) is type(b))


def packed_trunk(encoder: nn.TransformerEncoder, x: torch.Tensor,
                 index: PackedIndex) -> torch.Tensor:
    """The encoder stack on the packed [total, d] tensor. Inference
    only: the layers' dropouts are identities in eval mode and are
    omitted."""
    for layer in encoder.layers:
        x = layer.norm1(x + _self_attention(layer.self_attn, x, index))
        x = layer.norm2(x + layer.linear2(layer.activation(layer.linear1(x))))
    return x if encoder.norm is None else encoder.norm(x)


def _self_attention(mha: nn.MultiheadAttention, x: torch.Tensor,
                    index: PackedIndex) -> torch.Tensor:
    """F.multi_head_attention_forward's self-attention path on packed
    tokens (torch 2.5.1 torch/nn/functional.py: _in_projection_packed
    5501-5510, head split 6274-6276, scaled_dot_product_attention 6278,
    out projection 6285). Under bf16 autocast the two linears run in
    bf16, so the kernel receives bf16 q, k, v as the padded path's
    scaled_dot_product_attention does."""
    total, E = x.shape
    heads = mha.num_heads
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).view(total, 3, heads, E // heads)
    q, k, v = qkv.unbind(1)                    # [total, heads, head_dim], unit-stride last dim
    ctx = packed_attention(q, k, v, index)
    return F.linear(ctx.reshape(total, E), mha.out_proj.weight, mha.out_proj.bias)


# ---------------------------------------------------------------------
# The attention as a custom op: the compile boundary
# ---------------------------------------------------------------------
#
# torch 2.5.1 torch/library.py: define (line 421) takes a schema string
# and tags; impl (502) with "CompositeExplicitAutograd" registers one
# kernel for every device; register_fake (684) gives the shape function
# used while tracing. `SymInt max_len` lets a symbolic length flow
# through under dynamic shapes (an `int` schema would specialize the
# graph on every value). The needs_fixed_stride_order tag makes inductor
# hand the kernel the strides the eager code has (torch/_inductor/
# lowering.py:106-127; the 2.5.1 default for custom ops is
# flexible_layout), i.e. the unbind views of the in-projection output,
# which the flash kernel reads directly.

PACKED_ATTENTION_OP = "wesnoth_ai::packed_attention"


def _packed_attention_impl(q, k, v, cu_seqlens, max_len: int) -> torch.Tensor:
    """Eager kernel of the op. Contiguous [total, heads, head_dim] output,
    as the fake kernel promises (flash_api.cpp:678 allocates the output
    with empty_like on the strided q view, which yields a contiguous
    tensor; .contiguous() is then a no-op)."""
    if flash_varlen_applies(q.device, q.dtype):
        q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
        return _flash_varlen(q, k, v, cu_seqlens, int(max_len)).contiguous()
    return _segment_sdpa(q, k, v, cu_seqlens.tolist())


def _packed_attention_fake(q, k, v, cu_seqlens, max_len):
    return q.new_empty(q.shape)


def _register_packed_attention_op() -> None:
    """Defines the op once per process (torch refuses a second
    definition; the namespace lookup raises AttributeError before the
    first, torch/_ops.py:1207-1234 at v2.5.1)."""
    if hasattr(torch.ops.wesnoth_ai, "packed_attention"):
        return
    torch.library.define(
        PACKED_ATTENTION_OP,
        "(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, SymInt max_len) -> Tensor",
        tags=(torch._C.Tag.needs_fixed_stride_order,))
    torch.library.impl(PACKED_ATTENTION_OP, "CompositeExplicitAutograd", _packed_attention_impl)
    torch.library.register_fake(PACKED_ATTENTION_OP, _packed_attention_fake)


_register_packed_attention_op()


# ---------------------------------------------------------------------
# The layer loop as a pure function of tensors
# ---------------------------------------------------------------------

# One layer's weights in the order packed_trunk_layers unpacks them:
# in-projection weight [3, heads, head_dim, E] and bias [3, heads,
# head_dim], out-projection weight and bias, norm1 weight and bias,
# linear1 weight and bias, linear2 weight and bias, norm2 weight and
# bias. Absent biases / affine weights are None.
LayerWeights = Tuple[Optional[torch.Tensor], ...]


def make_packed_trunk_layers(activation: Callable, eps: float):
    """Builds the loop closed over the two layer constants that are not
    tensors. Under `dynamic=True` dynamo turns every int reaching the
    function through a local, a closure cell or an attribute into a
    symbol (torch 2.5.1 torch/_dynamo/variables/builder.py:1416-1445 and
    wrap_symint 1710-1800), so `heads` and `head_dim` are read from the
    static shape of the in-projection parameter instead of being passed;
    floats are specialized (config.specialize_float, config.py:64) and
    functions are constants."""
    def packed_trunk_layers(x, cu_seqlens, max_len, layers, final_norm):
        for (qkv_w, qkv_b, out_w, out_b, ln1_w, ln1_b,
             ff1_w, ff1_b, ff2_w, ff2_b, ln2_w, ln2_b) in layers:
            _, heads, head_dim, E = qkv_w.shape
            total = x.shape[0]
            qkv = F.linear(x.to(qkv_w.dtype), qkv_w.view(3 * E, E),
                           None if qkv_b is None else qkv_b.view(3 * E))
            q, k, v = qkv.view(total, 3, heads, head_dim).unbind(1)
            ctx = torch.ops.wesnoth_ai.packed_attention.default(q, k, v, cu_seqlens, max_len)
            sa = F.linear(ctx.view(total, E), out_w, out_b)
            x = F.layer_norm(x + sa.to(x.dtype), (E,), ln1_w, ln1_b, eps)
            ff = F.linear(activation(F.linear(x.to(ff1_w.dtype), ff1_w, ff1_b)), ff2_w, ff2_b)
            x = F.layer_norm(x + ff.to(x.dtype), (E,), ln2_w, ln2_b, eps)
        if final_norm is not None:
            x = F.layer_norm(x, (x.shape[1],), final_norm[0], final_norm[1], eps)
        return x
    return packed_trunk_layers


def _layer_sources(layer: nn.TransformerEncoderLayer, compute_dtype: torch.dtype):
    """(source parameter or None, dtype of the copy, shape of the copy or
    None for the source's) per entry of LayerWeights."""
    mha = layer.self_attn
    heads, E = mha.num_heads, mha.embed_dim
    c, f = compute_dtype, torch.float32
    return [
        (mha.in_proj_weight, c, (3, heads, E // heads, E)),
        (mha.in_proj_bias, c, (3, heads, E // heads)),
        (mha.out_proj.weight, c, None), (mha.out_proj.bias, c, None),
        (layer.norm1.weight, f, None), (layer.norm1.bias, f, None),
        (layer.linear1.weight, c, None), (layer.linear1.bias, c, None),
        (layer.linear2.weight, c, None), (layer.linear2.bias, c, None),
        (layer.norm2.weight, f, None), (layer.norm2.bias, f, None),
    ]


@dataclass
class PackedTrunkWeights:
    """The encoder's parameters as packed_trunk_layers reads them: linear
    weights in the compute dtype (bf16 on CUDA), LayerNorm weights fp32,
    the in-projection shaped [3, heads, head_dim, E]. Distinct storage
    from the encoder's parameters, held as nn.Parameters so dynamo keeps
    their shapes static under dynamic=True (torch 2.5.1
    torch/_dynamo/utils.py:2307-2311, config.force_parameter_static_shapes).
    `refresh` copies the encoder's current values in place, so the
    compiled graph's guards see the same tensors."""
    layers: List[LayerWeights]
    final_norm: Optional[Tuple[torch.Tensor, torch.Tensor]]
    dtype: torch.dtype
    device: torch.device
    version: int

    @classmethod
    def build(cls, encoder: nn.TransformerEncoder, dtype: torch.dtype, device: torch.device,
              version: int) -> "PackedTrunkWeights":
        def copy(src, dt, shape):
            if src is None:
                return None
            t = src.detach().to(device=device, dtype=dt, copy=True)
            return nn.Parameter(t.reshape(shape) if shape else t, requires_grad=False)
        layers = [tuple(copy(*entry) for entry in _layer_sources(layer, dtype))
                  for layer in encoder.layers]
        final = None if encoder.norm is None else (
            copy(encoder.norm.weight, torch.float32, None),
            copy(encoder.norm.bias, torch.float32, None))
        return cls(layers, final, dtype, device, version)

    def refresh(self, encoder: nn.TransformerEncoder, version: int) -> None:
        with torch.no_grad():
            for dst_layer, layer in zip(self.layers, encoder.layers):
                for dst, (src, _, _) in zip(dst_layer, _layer_sources(layer, self.dtype)):
                    if dst is not None:
                        dst.copy_(src.detach().reshape(dst.shape))
            if self.final_norm is not None:
                self.final_norm[0].copy_(encoder.norm.weight)
                self.final_norm[1].copy_(encoder.norm.bias)
        self.version = version


# ---------------------------------------------------------------------
# torch.compile of the loop, with recompile and fallback accounting
# ---------------------------------------------------------------------

def _dynamo_cache_entries(fn) -> int:
    """Compiled variants dynamo holds for fn's code object (torch 2.5.1
    torch/_dynamo/eval_frame.py:130-139); grows by one per recompile."""
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    return len(_debug_get_cache_entry_list(fn.__code__))


def _dynamo_guard_failures(fn) -> List[str]:
    """Reasons of every guard failure on fn's code object, appended
    before dynamo decides between recompiling and giving up (torch 2.5.1
    torch/_dynamo/guards.py:2754 from convert_frame.py:828-833)."""
    from torch._dynamo.utils import guard_failures
    return guard_failures.get(fn.__code__, [])


class CompiledPackedTrunk:
    """`torch.compile(packed_trunk_layers, dynamic=True, fullgraph=True)`
    in the default inductor mode (no CUDA graphs: cudagraph trees would
    record one graph per distinct shape, design note section 3.4), or
    `mode="max-autotune-no-cudagraphs"` for the GEMM templates.

    Warmup compiles on the first batch and runs a second, distinct shape
    so that a graph specialized on the first shows up as a second cache
    entry (`recompiles`). Afterwards every call is checked against the
    per-function guard-failure list: a failure followed by a new cache
    entry is a recompile (logged, counted), a failure without one means
    dynamo gave up on the frame (the design's silent eager fallback:
    logged once, `active` drops and this object's own eager loop serves).
    With fullgraph=True, graph breaks, the cache-size limit and backend
    errors raise instead of falling back (torch 2.5.1 optimize_assert,
    eval_frame.py:1602, runs convert_frame_assert, which bypasses the
    suppress_errors swallow at convert_frame.py:1111); those raise into
    the same fallback. Calls run under no_grad with autocast disabled,
    the two global states dynamo guards on, so the caller's context
    cannot trigger a recompile."""

    def __init__(self, activation: Callable, eps: float, *, backend="inductor",
                 mode: Optional[str] = None):
        self.backend, self.mode = backend, mode
        self._eager = make_packed_trunk_layers(activation, eps)
        self._compiled = torch.compile(self._eager, dynamic=True, fullgraph=True,
                                       backend=backend, mode=mode)
        self._lock = threading.Lock()
        self.warmed = False
        self.active = False
        self.warmup_seconds: Optional[float] = None
        self.recompiles = 0
        self.fallback_reason: Optional[str] = None
        self._entries = 0
        self._failures = 0

    def run(self, x: torch.Tensor, index: PackedIndex, weights: PackedTrunkWeights) -> torch.Tensor:
        if not self.warmed:
            out = self._warmup(x, index, weights)
            if out is not None:
                return out
        if self.active:
            try:
                out = self._call(x, index.cu_seqlens, index.max_len, weights)
            except Exception as e:                       # noqa: BLE001 -- any compile or run error
                self._fall_back(f"compiled call raised {type(e).__name__}: {e}")
            else:
                self._check_after_call()
                return out
        return self._eager(x, index.cu_seqlens, index.max_len, weights.layers, weights.final_norm)

    def stats(self) -> dict:
        return {"active": self.active, "backend": str(self.backend), "mode": self.mode,
                "warmup_seconds": self.warmup_seconds, "recompiles": self.recompiles,
                "fallback_reason": self.fallback_reason,
                "cache_entries": _dynamo_cache_entries(self._eager)}

    def _call(self, x, cu_seqlens, max_len, weights):
        # The activation's token dim and the offsets' length are the
        # dynamic dims; marking them makes dynamo raise at compile time
        # if anything specializes them (RelaxedUnspecConstraint,
        # builder.py:2600-2611). The model dim is static.
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_static(x, 1)
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            return self._compiled(x, cu_seqlens, max_len, weights.layers, weights.final_norm)

    def _warmup(self, x, index, weights):
        with self._lock:
            if self.warmed:
                return None
            t0 = time.perf_counter()
            try:
                before = _dynamo_cache_entries(self._eager)
                out = self._call(x, index.cu_seqlens, index.max_len, weights)
                first = _dynamo_cache_entries(self._eager)
                second = self._second_shape(x, index)
                if second is not None:
                    self._call(*second, weights)
                after = _dynamo_cache_entries(self._eager)
            except Exception as e:                       # noqa: BLE001 -- eager serves from here on
                self.warmed = True
                self._fall_back(f"warmup failed: {type(e).__name__}: {e}")
                return None
            self.warmup_seconds = time.perf_counter() - t0
            self.recompiles = after - first
            self._entries, self._failures = after, len(_dynamo_guard_failures(self._eager))
            self.warmed = self.active = True
            log.info("packed trunk compiled (%s, mode %s): warmup %.1f s, %d cache entries "
                     "(%d new), second shape %s",
                     self.backend, self.mode or "default", self.warmup_seconds, after,
                     after - before, "recompiled" if self.recompiles else "reused the graph")
            if self.recompiles:
                log.warning("packed trunk recompiled on the second warmup shape: the graph is "
                            "specialized (last guard failure: %s)",
                            (_dynamo_guard_failures(self._eager) or ["?"])[-1])
            return out

    @staticmethod
    def _second_shape(x, index):
        """One segment cut from the batch: another total, offsets of
        length 2, another max_len. None if the batch is too small."""
        n = index.cu_host[1]
        if len(index.cu_host) == 2 or n == index.max_len:
            n -= 1
        if n < 2:
            return None
        return x[:n], torch.tensor([0, n], dtype=torch.int32, device=x.device), n

    def _check_after_call(self):
        failures = _dynamo_guard_failures(self._eager)
        if len(failures) == self._failures:
            return
        self._failures = len(failures)
        entries = _dynamo_cache_entries(self._eager)
        if entries > self._entries:
            self.recompiles += entries - self._entries
            self._entries = entries
            log.warning("packed trunk recompiled (%d cache entries; guard failure: %s)",
                        entries, failures[-1])
        else:
            self._fall_back(f"dynamo guard failed without a new compile ({failures[-1]})")

    def _fall_back(self, reason: str):
        self.active = False
        if self.fallback_reason is None:
            self.fallback_reason = reason
            log.warning("packed trunk compile inactive, eager loop serves: %s", reason)
