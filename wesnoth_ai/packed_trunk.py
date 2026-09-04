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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def build_packed_layout(sizes: Sequence[Tuple[int, int, int]], H_max: int, U_max: int,
                        R_max: int, token_kind, actor_kind) -> PackedLayout:
    """Index arrays for per-sample sizes (U_b, R_b, H_b) in the padded
    layout hex(H_max) | unit(U_max) | recruit(R_max) | global | end_turn
    of model.forward_streams. `token_kind` and `actor_kind` are the
    model's TokenKind and ActorKind tables, passed in so this module
    imports nothing from the model."""
    B = len(sizes)
    L = H_max + U_max + R_max + 2
    A_max = U_max + R_max + 1
    lengths = np.array([U + R + H + 2 for U, R, H in sizes], dtype=np.int64)
    cu = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(lengths, out=cu[1:])
    src = np.empty(int(cu[-1]), dtype=np.int64)
    kind = np.empty(int(cu[-1]), dtype=np.int64)
    actor = np.empty((B, A_max), dtype=np.int64)
    hexes = np.empty((B, H_max), dtype=np.int64)
    unit = np.empty((B, U_max), dtype=np.int64)
    glob = np.empty(B, dtype=np.int64)
    kinds = np.full((B, A_max), actor_kind.END_TURN, dtype=np.int64)
    for b, (U, R, H) in enumerate(sizes):
        o, row, n = int(cu[b]), b * L, int(lengths[b])
        seg = src[o:o + n]
        seg[:H] = row + np.arange(H)
        seg[H:H + U] = row + H_max + np.arange(U)
        seg[H + U:H + U + R] = row + H_max + U_max + np.arange(R)
        seg[n - 2] = row + H_max + U_max + R_max
        seg[n - 1] = row + H_max + U_max + R_max + 1
        kseg = kind[o:o + n]
        kseg[:H] = token_kind.HEX
        kseg[H:H + U] = token_kind.UNIT
        kseg[H + U:H + U + R] = token_kind.RECRUIT
        kseg[n - 2] = token_kind.GLOBAL
        kseg[n - 1] = token_kind.END_TURN
        end = o + n - 1
        actor[b] = end
        actor[b, :U + R] = o + H + np.arange(U + R)    # units then recruits, contiguous
        hexes[b] = end
        hexes[b, :H] = o + np.arange(H)
        unit[b] = end
        unit[b, :U] = o + H + np.arange(U)
        glob[b] = end - 1
        kinds[b, :U] = actor_kind.UNIT
        kinds[b, U:U + R] = actor_kind.RECRUIT
    return PackedLayout(src=src, kind=kind, actor=actor.reshape(-1), hex=hexes.reshape(-1),
                        unit=unit.reshape(-1), glob=glob, cu_seqlens=cu, actor_kind=kinds,
                        max_len=int(lengths.max()) if B else 0)


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
        # Positional arguments in the order of the 2.5.1 schema quoted in
        # the module docstring: dropout 0, not causal, no debug mask,
        # default scale.
        return torch.ops.aten._flash_attention_forward(
            q, k, v, index.cu_seqlens, index.cu_seqlens, index.max_len, index.max_len,
            0.0, False, False)[0]
    return _segment_sdpa(q, k, v, index.cu_host)


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
    for layer in encoder.layers:
        mha = layer.self_attn
        if layer.norm_first:
            raise NotImplementedError("packed trunk: norm_first layers")
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
            raise NotImplementedError("packed trunk: add_bias_kv / add_zero_attn")
        if not mha._qkv_same_embed_dim:
            raise NotImplementedError("packed trunk: separate q/k/v projection weights")


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
