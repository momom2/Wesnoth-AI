"""The backward of the HIP varlen attention (wesnoth_ai/hip_kernels.py), so
the packed trunk trains on GPUs PyTorch's fused attention does not serve
(the local RX 6600, gfx1032, ROCm on Windows; 2026-10-29,
training/metrics/hip_backward_20261029/README.md).

FlashAttention-2's backward (Dao, "FlashAttention-2: Faster Attention with
Better Parallelism and Work Partitioning", 2023, Algorithm 2), in the
forward kernel's arithmetic: the forward (`hip_varlen_attention_lse`, the
inference kernel's arithmetic) also writes each row's log-sum-exp L_i of
its scaled scores, and the backward recomputes the probabilities
P_ij = exp(s q_i.k_j - L_i) from q, k and L, s = 1/sqrt(head_dim):

    D_i  = dO_i . O_i
    dV_j = sum_i P_ij dO_i
    dS_ij = P_ij (dO_i . v_j - D_i)
    dQ_i = s sum_j dS_ij k_j
    dK_j = s sum_i dS_ij q_i

in two kernels, so no row is written by two blocks (no atomics):
  dq    one thread owns a few query rows (_DQ_QPT) of one head of one
        segment and walks the segment's keys in tiles held in shared
        memory, as the forward does; it also computes D_i from O and dO
        and writes it, float32 [heads, total];
  dkdv  one thread owns a key row (_KV_KPT) and walks the segment's
        queries in tiles (q, dO, L, D), after the dq kernel on the same
        stream.
The products are v_dot2_f32_f16 into fp32 accumulators, as in the
forward: the dot products read rows, and the sums over a tile's tokens
(dV, dQ, dK) read the tile stored as token pairs per dimension, so P and
dS enter as fp16 pairs (as FlashAttention's matrix products take them).
Scores, P before its rounding, dS before its rounding, and D stay fp32.
The gradients are fp16 [total, heads, head_dim], zero past the offsets'
last row.

Each block's loop runs over one segment's tokens and the grid is the
forward's, so no launch runs long (a launch past 2 s resets the display
driver on Windows, the RX 6600 driving the screen).

Built with hiprtc beside the forward's module, and checked on first use
per device against F.scaled_dot_product_attention's fp32 gradients on a
fixed problem; a device where either fails is reported once in the log,
and the callers keep the per-segment SDPA (`hip_varlen_backward_applies`
is False).
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from wesnoth_ai.hip_kernels import (
    HEAD_DIMS, compile_source, device_arch, function_resources, hip_runtime, hip_varlen_applies,
    hip_varlen_attention_lse, launch, load_module, module_function,
)
from wesnoth_ai.hip_kernels import _kernel_view

log = logging.getLogger(__name__)

_NT = 32                        # threads a block, one wave32
# dq: query rows a thread and keys a tile; three rows a thread (the
# forward's choice) do not fit beside dO's rows and dQ's accumulators.
_DQ_QPT = {32: 2, 64: 1}
_DQ_BK = 16
# dkdv: key rows a thread and queries a tile (dK and dV's accumulators are
# twice the forward's).
_KV_KPT = {32: 1, 64: 1}
_KV_BQ = {32: 16, 64: 8}
_LOG2E = 1.4426950408889634
# The self-test's bound on each gradient's largest error, relative to that
# gradient's largest magnitude: fp16 operands and fp32 sums read about
# 7e-4 (2026-10-29, training/metrics/hip_backward_20261029/README.md),
# so a wrong index or a missed tile reads far above it.
_SELF_TEST_REL = 1e-2

_SOURCE = r'''
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ h2 pair_of(h8 t, int e) { return h2{t[2 * e], t[2 * e + 1]}; }

// Dimensions [8c, 8c + 8) of tokens a and b as pairs per dimension:
// dst[e] = (a[e], b[e]).
__device__ __forceinline__ void store_pairs(h2* dst, h8 a, h8 b)
{
    *(h8*)dst = __builtin_shufflevector(a, b, 0, 8, 1, 9, 2, 10, 3, 11);
    *(h8*)(dst + 4) = __builtin_shufflevector(a, b, 4, 12, 5, 13, 6, 14, 7, 15);
}

template <int D>
__device__ __forceinline__ void load_row(h2* dst, const _Float16* src)
{
    #pragma unroll
    for (int c = 0; c < D / 8; ++c) {
        const h8 t = ((const h8*)src)[c];
        #pragma unroll
        for (int e = 0; e < 4; ++e) dst[c * 4 + e] = pair_of(t, e);
    }
}

template <int D>
__device__ __forceinline__ void store_row(_Float16* dst, const float* acc, float mul)
{
    #pragma unroll
    for (int c = 0; c < D / 8; ++c) {
        h8 t;
        #pragma unroll
        for (int e = 0; e < 8; ++e) t[e] = (_Float16)(acc[c * 8 + e] * mul);
        ((h8*)dst)[c] = t;
    }
}

// dQ and D. Grid (query blocks, heads, segments); NT threads, QPT query
// rows a thread (rows q0 + tid, q0 + tid + NT, ...); BK keys a tile.
template <int NT, int QPT, int BK, int D>
__device__ __forceinline__ void attn_bwd_dq(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k, const _Float16* __restrict__ v,
    const _Float16* __restrict__ o, const _Float16* __restrict__ g, const float* __restrict__ lse,
    float* __restrict__ delta, _Float16* __restrict__ dq, const int* __restrict__ cu,
    int sq, int sk, int sv, int so, int sg, int sdq, int ls, float scale_log2, float scale)
{
    constexpr int C = D / 2, CH = D / 8, BQ = NT * QPT;
    __shared__ __attribute__((aligned(16))) h2 Ks[BK][C];
    __shared__ __attribute__((aligned(16))) h2 Vs[BK][C];
    __shared__ __attribute__((aligned(16))) h2 Kp[BK / 2][D];
    const int seg = blockIdx.z, hd = blockIdx.y, tid = threadIdx.x;
    const int start = cu[seg], n = cu[seg + 1] - start;
    const int q0 = blockIdx.x * BQ;
    if (q0 >= n) return;
    h2 qr[QPT][C], gr[QPT][C];
    float lr[QPT], dl[QPT];
    bool active[QPT];
    #pragma unroll
    for (int r = 0; r < QPT; ++r) {
        const int qi = q0 + r * NT + tid;
        active[r] = qi < n;
        lr[r] = 0.f;
        dl[r] = 0.f;
        #pragma unroll
        for (int c = 0; c < C; ++c) { qr[r][c] = (h2)(0); gr[r][c] = (h2)(0); }
        if (active[r]) {
            const long row = start + qi;
            load_row<D>(qr[r], q + row * sq + hd * D);
            load_row<D>(gr[r], g + row * sg + hd * D);
            const h8* os = (const h8*)(o + row * so + hd * D);
            float dsum = 0.f;
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                const h8 t = os[c];
                #pragma unroll
                for (int e = 0; e < 4; ++e) dsum = __builtin_amdgcn_fdot2(gr[r][c * 4 + e], pair_of(t, e), dsum, false);
            }
            dl[r] = dsum;
            delta[(long)hd * ls + row] = dsum;
            lr[r] = lse[(long)hd * ls + row] * LOG2E;
        }
    }
    float acc[QPT][D];
    #pragma unroll
    for (int r = 0; r < QPT; ++r) {
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[r][d] = 0.f;
    }
    for (int kb = 0; kb < n; kb += BK) {
        for (int i = tid; i < BK * CH; i += NT) {
            const int j = i / CH, c = i % CH;
            h8 tk = (h8)(0), tv = (h8)(0);
            if (kb + j < n) {
                tk = ((const h8*)(k + (long)(start + kb + j) * sk + hd * D))[c];
                tv = ((const h8*)(v + (long)(start + kb + j) * sv + hd * D))[c];
            }
            *(h8*)&Ks[j][c * 4] = tk;
            *(h8*)&Vs[j][c * 4] = tv;
        }
        for (int i = tid; i < (BK / 2) * CH; i += NT) {
            const int jj = i / CH, c = i % CH;
            const int j0 = kb + 2 * jj;
            h8 a = (h8)(0), b = (h8)(0);
            if (j0 < n) a = ((const h8*)(k + (long)(start + j0) * sk + hd * D))[c];
            if (j0 + 1 < n) b = ((const h8*)(k + (long)(start + j0 + 1) * sk + hd * D))[c];
            store_pairs(&Kp[jj][c * 8], a, b);
        }
        __syncthreads();
        h2 dsp[QPT][BK / 2];
        #pragma unroll
        for (int jj = 0; jj < BK / 2; ++jj) {
            float ds[2][QPT];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * jj + h;
                float sd[QPT], pd[QPT];
                #pragma unroll
                for (int r = 0; r < QPT; ++r) { sd[r] = 0.f; pd[r] = 0.f; }
                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    const h8 kk = *(const h8*)&Ks[j][c * 4];
                    const h8 vv = *(const h8*)&Vs[j][c * 4];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const h2 ke = pair_of(kk, e), ve = pair_of(vv, e);
                        #pragma unroll
                        for (int r = 0; r < QPT; ++r) {
                            sd[r] = __builtin_amdgcn_fdot2(qr[r][c * 4 + e], ke, sd[r], false);
                            pd[r] = __builtin_amdgcn_fdot2(gr[r][c * 4 + e], ve, pd[r], false);
                        }
                    }
                }
                const bool valid = kb + j < n;
                #pragma unroll
                for (int r = 0; r < QPT; ++r) {
                    const float p = valid ? __builtin_amdgcn_exp2f(sd[r] * scale_log2 - lr[r]) : 0.f;
                    ds[h][r] = p * (pd[r] - dl[r]);
                }
            }
            #pragma unroll
            for (int r = 0; r < QPT; ++r) dsp[r][jj] = h2{(_Float16)ds[0][r], (_Float16)ds[1][r]};
        }
        #pragma unroll
        for (int jj = 0; jj < BK / 2; ++jj) {
            #pragma unroll
            for (int c = 0; c < D / 4; ++c) {
                const h8 kk = *(const h8*)&Kp[jj][c * 4];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const h2 ke = pair_of(kk, e);
                    #pragma unroll
                    for (int r = 0; r < QPT; ++r)
                        acc[r][c * 4 + e] = __builtin_amdgcn_fdot2(dsp[r][jj], ke, acc[r][c * 4 + e], false);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < QPT; ++r)
        if (active[r]) store_row<D>(dq + (long)(start + q0 + r * NT + tid) * sdq + hd * D, acc[r], scale);
}

// dK and dV. Grid (key blocks, heads, segments); NT threads, KPT key rows
// a thread; BQ queries a tile. Reads the D the dq kernel wrote.
template <int NT, int KPT, int BQ, int D>
__device__ __forceinline__ void attn_bwd_dkdv(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k, const _Float16* __restrict__ v,
    const _Float16* __restrict__ g, const float* __restrict__ lse, const float* __restrict__ delta,
    _Float16* __restrict__ dk, _Float16* __restrict__ dv, const int* __restrict__ cu,
    int sq, int sk, int sv, int sg, int sdk, int sdv, int ls, float scale_log2, float scale)
{
    constexpr int C = D / 2, CH = D / 8, BKV = NT * KPT;
    __shared__ __attribute__((aligned(16))) h2 Qs[BQ][C];
    __shared__ __attribute__((aligned(16))) h2 Gs[BQ][C];
    __shared__ __attribute__((aligned(16))) h2 Qp[BQ / 2][D];
    __shared__ __attribute__((aligned(16))) h2 Gp[BQ / 2][D];
    __shared__ float Ls[BQ], Dl[BQ];
    const int seg = blockIdx.z, hd = blockIdx.y, tid = threadIdx.x;
    const int start = cu[seg], n = cu[seg + 1] - start;
    const int k0 = blockIdx.x * BKV;
    if (k0 >= n) return;
    h2 kr[KPT][C], vr[KPT][C];
    bool active[KPT];
    #pragma unroll
    for (int r = 0; r < KPT; ++r) {
        const int kj = k0 + r * NT + tid;
        active[r] = kj < n;
        #pragma unroll
        for (int c = 0; c < C; ++c) { kr[r][c] = (h2)(0); vr[r][c] = (h2)(0); }
        if (active[r]) {
            load_row<D>(kr[r], k + (long)(start + kj) * sk + hd * D);
            load_row<D>(vr[r], v + (long)(start + kj) * sv + hd * D);
        }
    }
    float gk[KPT][D], gv[KPT][D];
    #pragma unroll
    for (int r = 0; r < KPT; ++r) {
        #pragma unroll
        for (int d = 0; d < D; ++d) { gk[r][d] = 0.f; gv[r][d] = 0.f; }
    }
    for (int qb = 0; qb < n; qb += BQ) {
        for (int i = tid; i < BQ * CH; i += NT) {
            const int j = i / CH, c = i % CH;
            h8 tq = (h8)(0), tg = (h8)(0);
            if (qb + j < n) {
                tq = ((const h8*)(q + (long)(start + qb + j) * sq + hd * D))[c];
                tg = ((const h8*)(g + (long)(start + qb + j) * sg + hd * D))[c];
            }
            *(h8*)&Qs[j][c * 4] = tq;
            *(h8*)&Gs[j][c * 4] = tg;
        }
        for (int i = tid; i < (BQ / 2) * CH; i += NT) {
            const int jj = i / CH, c = i % CH;
            const int i0 = qb + 2 * jj;
            h8 a = (h8)(0), b = (h8)(0), e = (h8)(0), f = (h8)(0);
            if (i0 < n) {
                a = ((const h8*)(q + (long)(start + i0) * sq + hd * D))[c];
                e = ((const h8*)(g + (long)(start + i0) * sg + hd * D))[c];
            }
            if (i0 + 1 < n) {
                b = ((const h8*)(q + (long)(start + i0 + 1) * sq + hd * D))[c];
                f = ((const h8*)(g + (long)(start + i0 + 1) * sg + hd * D))[c];
            }
            store_pairs(&Qp[jj][c * 8], a, b);
            store_pairs(&Gp[jj][c * 8], e, f);
        }
        for (int i = tid; i < BQ; i += NT) {
            const bool ok = qb + i < n;
            // A query past the segment: P = exp2(score - inf) = 0.
            Ls[i] = ok ? lse[(long)hd * ls + start + qb + i] * LOG2E : __builtin_inff();
            Dl[i] = ok ? delta[(long)hd * ls + start + qb + i] : 0.f;
        }
        __syncthreads();
        h2 pp[KPT][BQ / 2], dsp[KPT][BQ / 2];
        #pragma unroll
        for (int ii = 0; ii < BQ / 2; ++ii) {
            float pv[2][KPT], ds[2][KPT];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int i = 2 * ii + h;
                float sd[KPT], pd[KPT];
                #pragma unroll
                for (int r = 0; r < KPT; ++r) { sd[r] = 0.f; pd[r] = 0.f; }
                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    const h8 qq = *(const h8*)&Qs[i][c * 4];
                    const h8 gg = *(const h8*)&Gs[i][c * 4];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const h2 qe = pair_of(qq, e), ge = pair_of(gg, e);
                        #pragma unroll
                        for (int r = 0; r < KPT; ++r) {
                            sd[r] = __builtin_amdgcn_fdot2(qe, kr[r][c * 4 + e], sd[r], false);
                            pd[r] = __builtin_amdgcn_fdot2(ge, vr[r][c * 4 + e], pd[r], false);
                        }
                    }
                }
                const float li = Ls[i], di = Dl[i];
                #pragma unroll
                for (int r = 0; r < KPT; ++r) {
                    const float p = __builtin_amdgcn_exp2f(sd[r] * scale_log2 - li);
                    pv[h][r] = p;
                    ds[h][r] = p * (pd[r] - di);
                }
            }
            #pragma unroll
            for (int r = 0; r < KPT; ++r) {
                pp[r][ii] = h2{(_Float16)pv[0][r], (_Float16)pv[1][r]};
                dsp[r][ii] = h2{(_Float16)ds[0][r], (_Float16)ds[1][r]};
            }
        }
        #pragma unroll
        for (int ii = 0; ii < BQ / 2; ++ii) {
            #pragma unroll
            for (int c = 0; c < D / 4; ++c) {
                const h8 gg = *(const h8*)&Gp[ii][c * 4];
                const h8 qq = *(const h8*)&Qp[ii][c * 4];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const h2 ge = pair_of(gg, e), qe = pair_of(qq, e);
                    #pragma unroll
                    for (int r = 0; r < KPT; ++r) {
                        gv[r][c * 4 + e] = __builtin_amdgcn_fdot2(pp[r][ii], ge, gv[r][c * 4 + e], false);
                        gk[r][c * 4 + e] = __builtin_amdgcn_fdot2(dsp[r][ii], qe, gk[r][c * 4 + e], false);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < KPT; ++r) {
        if (active[r]) {
            const long row = start + k0 + r * NT + tid;
            store_row<D>(dk + row * sdk + hd * D, gk[r], scale);
            store_row<D>(dv + row * sdv + hd * D, gv[r], 1.f);
        }
    }
}

#define BWD_INSTANCE(D, QPT, KPT, BQ) \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
varlen_attn_bwd_dq_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, const _Float16* o, \
                            const _Float16* g, const float* lse, float* delta, _Float16* dq, const int* cu, \
                            int sq, int sk, int sv, int so, int sg, int sdq, int ls, float scale_log2, \
                            float scale) \
{ attn_bwd_dq<NT_VALUE, QPT, BK_VALUE, D>(q, k, v, o, g, lse, delta, dq, cu, sq, sk, sv, so, sg, sdq, ls, \
                                          scale_log2, scale); } \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
varlen_attn_bwd_dkdv_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, const _Float16* g, \
                              const float* lse, const float* delta, _Float16* dk, _Float16* dv, \
                              const int* cu, int sq, int sk, int sv, int sg, int sdk, int sdv, int ls, \
                              float scale_log2, float scale) \
{ attn_bwd_dkdv<NT_VALUE, KPT, BQ, D>(q, k, v, g, lse, delta, dk, dv, cu, sq, sk, sv, sg, sdk, sdv, ls, \
                                      scale_log2, scale); }
BWD_INSTANCE(32, DQ_QPT32, KV_KPT32, KV_BQ32)
BWD_INSTANCE(64, DQ_QPT64, KV_KPT64, KV_BQ64)
'''


def _source() -> str:
    s = _SOURCE.replace("NT_VALUE", str(_NT)).replace("BK_VALUE", str(_DQ_BK))
    for d in HEAD_DIMS:
        s = (s.replace(f"DQ_QPT{d}", str(_DQ_QPT[d])).replace(f"KV_KPT{d}", str(_KV_KPT[d]))
             .replace(f"KV_BQ{d}", str(_KV_BQ[d])))
    return s


class _BackwardKernels:
    """The backward's compiled module on one device and its entry points."""

    def __init__(self, device: torch.device):
        self.hip = hip_runtime()
        self.arch = device_arch(device)
        self.module = load_module(self.hip, compile_source(self.arch, _source(), "wesnoth_attn_bwd.hip"),
                                  device)
        self.dq = {d: module_function(self.hip, self.module, f"varlen_attn_bwd_dq_f16_d{d}") for d in HEAD_DIMS}
        self.dkdv = {d: module_function(self.hip, self.module, f"varlen_attn_bwd_dkdv_f16_d{d}")
                     for d in HEAD_DIMS}

    def resources(self) -> Dict[str, Dict[str, int]]:
        """Registers and scratch bytes a thread per kernel (scratch > 0: the
        compiler spilled)."""
        out = {}
        for d in HEAD_DIMS:
            out[f"dq_d{d}"] = function_resources(self.hip, self.dq[d])
            out[f"dkdv_d{d}"] = function_resources(self.hip, self.dkdv[d])
        return out

    def launch(self, q, k, v, o, g, lse, delta, dq, dk, dv, cu_seqlens, max_len: int) -> None:
        """dq (writing delta), then dk and dv, on torch's current stream."""
        total, heads, d = q.shape
        segments = cu_seqlens.numel() - 1
        scale = 1.0 / math.sqrt(d)
        ls = lse.stride(0)
        bq = _NT * _DQ_QPT[d]
        launch(self.hip, self.dq[d], ((max_len + bq - 1) // bq, heads, segments), _NT,
               [q, k, v, o, g, lse, delta, dq, cu_seqlens, q.stride(0), k.stride(0), v.stride(0), o.stride(0),
                g.stride(0), dq.stride(0), ls, _LOG2E * scale, scale], q.device)
        bk = _NT * _KV_KPT[d]
        launch(self.hip, self.dkdv[d], ((max_len + bk - 1) // bk, heads, segments), _NT,
               [q, k, v, g, lse, delta, dk, dv, cu_seqlens, q.stride(0), k.stride(0), v.stride(0), g.stride(0),
                dk.stride(0), dv.stride(0), ls, _LOG2E * scale, scale], q.device)


_lock = threading.Lock()
_backward: Dict[int, Optional[_BackwardKernels]] = {}


def _backward_for(device: torch.device) -> Optional[_BackwardKernels]:
    """The device's backward kernels, built and self-tested on first use;
    None (reported once) where either fails."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        if index not in _backward:
            try:
                kern = _BackwardKernels(torch.device("cuda", index))
                _self_test(kern, torch.device("cuda", index))
                log.info("hip attention backward: built for %s on cuda:%d", kern.arch, index)
            except Exception as e:  # noqa: BLE001 -- any failure means: not served here
                log.warning("hip attention backward unavailable on cuda:%d: %s", index, e)
                kern = None
            _backward[index] = kern
        return _backward[index]


def reference_attention(q, k, v, cu_host):
    """F.scaled_dot_product_attention per segment in float32, differentiable:
    the reference the kernels are checked against."""
    outs = []
    for lo, hi in zip(cu_host[:-1], cu_host[1:]):
        seg = [t[lo:hi].float().transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
        outs.append(F.scaled_dot_product_attention(*seg)[0].transpose(0, 1))
    return torch.cat(outs)


def attention_errors(q, k, v, cu_host, weight, function) -> Dict[str, Dict[str, float]]:
    """The output and the three gradients of sum(weight * attention) by
    `function(q, k, v)` against `reference_attention`: per tensor, the
    largest absolute error and that error over the reference's largest
    magnitude."""
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
    out = function(*leaves)
    (out.float() * weight).sum().backward()
    ref_leaves = [t.detach().float().requires_grad_(True) for t in (q, k, v)]
    ref = reference_attention(*ref_leaves, cu_host)
    (ref * weight).sum().backward()
    rows = cu_host[-1]
    pairs = {"out": (out[:rows].detach().float(), ref.detach())}
    for name, a, b in zip(("dq", "dk", "dv"), leaves, ref_leaves):
        pairs[name] = (a.grad[:rows].float(), b.grad[:rows])
    errors = {}
    for name, (got, want) in pairs.items():
        err = (got - want).abs().max().item()
        errors[name] = {"max_abs": err, "rel": err / max(want.abs().max().item(), 1e-30)}
    return errors


def _self_test(kern: _BackwardKernels, device: torch.device) -> None:
    """The forward with its log-sum-exp and the three gradients against the
    fp32 reference on a fixed problem per instantiated head size: segments
    that end inside, on and past a tile and a block, and one longer than
    a block."""
    gen = torch.Generator().manual_seed(1)
    lens = [1, 2, _DQ_BK - 1, _DQ_BK, _NT + 1, 2 * _NT * max(_DQ_QPT.values()) + 5]
    cu_host = [0]
    for n in lens:
        cu_host.append(cu_host[-1] + n)
    cu = torch.tensor(cu_host, dtype=torch.int32, device=device)
    for d in HEAD_DIMS:
        heads = 2
        qkv = torch.randn(cu_host[-1], 3, heads, d, generator=gen).to(device, torch.float16)
        weight = torch.randn(cu_host[-1], heads, d, generator=gen).to(device)
        errors = attention_errors(*qkv.unbind(1), cu_host, weight,
                                  lambda q, k, v: _HipVarlenAttention.apply(q, k, v, cu, max(lens), None, kern))
        worst = max(e["rel"] for e in errors.values())
        if not worst <= _SELF_TEST_REL:
            raise RuntimeError(f"self-test at head_dim {d}: relative errors {errors} above {_SELF_TEST_REL}")


class _HipVarlenAttention(torch.autograd.Function):
    """The HIP varlen attention with its backward. `rows`: the offsets' last
    value when the caller knows it on the host (gradient rows past it are
    zeroed only when there are any), else None (all zeroed first). `kern`:
    the device's backward kernels."""

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_len: int, rows: Optional[int], kern: _BackwardKernels):
        out, lse, (qk, kk, vk) = hip_varlen_attention_lse(q, k, v, cu_seqlens, max_len)
        cu = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
        ctx.save_for_backward(qk, kk, vk, out, lse, cu.contiguous())
        ctx.max_len, ctx.rows, ctx.kern = int(max_len), rows, kern
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, out, lse, cu = ctx.saved_tensors
        g = _kernel_view(grad.to(torch.float16))
        alloc = torch.empty if ctx.rows is not None and ctx.rows >= q.shape[0] else torch.zeros
        dq, dk, dv = (alloc(q.shape, dtype=torch.float16, device=q.device) for _ in range(3))
        delta = torch.empty_like(lse)
        ctx.kern.launch(q, k, v, out, g, lse, delta, dq, dk, dv, cu, ctx.max_len)
        return dq, dk, dv, None, None, None, None


def hip_varlen_backward_applies(device: torch.device, dtype: torch.dtype,
                                head_dim: Optional[int] = None) -> bool:
    """Whether `hip_varlen_attention_trainable` serves this call: the forward
    kernel applies (hip_kernels.hip_varlen_applies), and the backward is
    built and self-tested on the device (built here on first ask)."""
    if not hip_varlen_applies(device, dtype, head_dim):
        return False
    return _backward_for(device) is not None


def hip_varlen_attention_trainable(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   cu_seqlens: torch.Tensor, max_len: int,
                                   rows: Optional[int] = None) -> torch.Tensor:
    """hip_varlen_attention (the same forward arithmetic) with a backward:
    differentiable in q, k and v. The caller has checked
    `hip_varlen_backward_applies`."""
    return _HipVarlenAttention.apply(q, k, v, cu_seqlens, int(max_len), rows, _backward_for(q.device))
