"""int8 linear layers in HIP for GPUs without matrix units (2026-10-29: the
local RX 6600, gfx1032, where rocBLAS fp16 reaches 9.5-12.1 TFLOP/s on the
trunk's linears and torch._int_mm has no kernel; hipBLASLt ships none for
the architecture). docs/int8_linear_prereg_20261029.md pre-registers the
accuracy gate; training/metrics/int8_linear_20261029/README.md records the
rates and the reading.

The product: out = half((x_q . w_q^T) * row_scale[m] * col_scale[n] + bias[n])
(and its ReLU), x_q int8 [M, K] with a scale per row (token), w_q int8
[N, K] with a scale per output channel, the dot products exact in int32 by
v_dot4_i32_i8 (`__builtin_amdgcn_sdot4`: four int8 products summed into an
int32 per lane per instruction, RDNA2's dot1 instructions).

The GEMM: a block of NT threads computes a BM x BN tile of the output; each
thread holds TM x TN int32 accumulators (register blocking). The block
walks K in tiles of BK bytes: each thread loads 16-byte chunks of A's and
B's tile rows from global memory into registers for the next tile while
the current one is computed, and stores them to shared memory transposed,
as [BK / 4][BM] dwords (a dword is four consecutive K values of one row),
so a thread reads its rows' dwords at one K step with 16-byte loads. Its TN
columns are TN / 4 groups of four, the groups TX * 4 columns apart, so a
16-byte read of B's tile by consecutive threads covers consecutive dwords
(no bank conflict) and the output rows are written in coalesced 8-byte
stores. One K step of four values costs TM * TN dot4 instructions for
(TM + TN) / 4 16-byte shared-memory reads.

The quantizers write int8 rows and their scales, one wave32 a row:
  quant_rows   fp16 rows (the attention's output, the ReLU's) -> int8, scale;
  add_ln_q     the residual add and layer norm of hip_kernels.add_ln, its
               float32 output kept as the residual stream and quantized
               for the next linear (instead of add_ln's fp16 copy).
A row's scale is its largest magnitude over 127; each value is rounded to
the nearest integer (ties to even) of value * 127 / largest. The SMOOTH
variants first multiply each column by a per-channel factor (SmoothQuant's
migration, the prereg's arm S; the weight then carries its inverse).

Build, launch and dispatch as in wesnoth_ai/hip_kernels.py: hiprtc at first
use, ctypes launches on torch's stream, a self-test per device against
exact integer products and torch's rounding on fixed problems, and
`hip_int8_applies` False (reported once) where either fails. Forward only.
Every launch's grid follows the problem's size and every loop is bounded
by K, so no launch runs long (a launch past 2 s resets the display driver
on Windows, the RX 6600 driving the screen).
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from wesnoth_ai.hip_kernels import (
    _aligned, compile_source, device_arch, function_resources, hip_runtime, launch, load_module,
    module_function,
)

log = logging.getLogger(__name__)

# The GEMM's tiling: NT threads a block, a BM x BN output tile, K tiles of
# BK bytes, TM x TN accumulators a thread.
_NT = 256
_BM, _BN, _BK = 128, 128, 32
_TM, _TN = 8, 8
# The quantizers: a wave32 a row, LN_THREADS threads a block; the widths are
# 128 times NI (own2: 384 = 128 x 3, its d_ff 1536 = 128 x 12).
_Q_THREADS = 128
_Q_NI = (1, 2, 3, 4, 6, 8, 12)
_MAX_K = 8192               # the widest K a launch takes (bounds the K loop)
INT8_MAX = 127

_SOURCE = r'''
typedef int i4v __attribute__((ext_vector_type(4)));
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

// C[m, n] = half((A[m, :] . B[n, :]) * rs[m] * cs[n] + bias[n]), ReLU'd if
// RELU. A int8 [M, K], B int8 [N, K], both K-contiguous with K a multiple
// of BK and 16-byte aligned rows; N a multiple of 4. Grid (ceil(N / BN),
// ceil(M / BM)).
template <int NT, int BM, int BN, int BK, int TM, int TN, bool RELU>
__device__ __forceinline__ void gemm_i8(
    const signed char* __restrict__ A, const signed char* __restrict__ B,
    const float* __restrict__ rs, const float* __restrict__ cs, const float* __restrict__ bias,
    _Float16* __restrict__ C, int M, int N, int K)
{
    constexpr int KQ = BK / 4, TX = BN / TN, TY = BM / TM, G = TN / 4;
    constexpr int CPR = BK / 16;                    // 16-byte chunks a tile row
    constexpr int LA = BM * CPR / NT, LB = BN * CPR / NT;
    static_assert(TX * TY == NT, "the thread grid covers the tile");
    static_assert(LA * NT == BM * CPR && LB * NT == BN * CPR, "the loads cover the tiles");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "16-byte shared reads");
    __shared__ __attribute__((aligned(16))) int As[KQ][BM];
    __shared__ __attribute__((aligned(16))) int Bs[KQ][BN];
    const int tid = threadIdx.x, tx = tid % TX, ty = tid / TX;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    // Chunk c of a tile: row c % BM, 16-byte part c / BM, so consecutive
    // threads store consecutive dwords of a shared row. Rows past M or N
    // read the last row (in bounds; their outputs are not written).
    const signed char* pa[LA];
    const signed char* pb[LB];
    #pragma unroll
    for (int l = 0; l < LA; ++l) {
        const int c = tid + l * NT;
        pa[l] = A + (long long)min(m0 + c % BM, M - 1) * K + (c / BM) * 16;
    }
    #pragma unroll
    for (int l = 0; l < LB; ++l) {
        const int c = tid + l * NT;
        pb[l] = B + (long long)min(n0 + c % BN, N - 1) * K + (c / BN) * 16;
    }
    i4v ra[LA], rb[LB];
    #pragma unroll
    for (int l = 0; l < LA; ++l) ra[l] = *(const i4v*)pa[l];
    #pragma unroll
    for (int l = 0; l < LB; ++l) rb[l] = *(const i4v*)pb[l];
    int acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;
    }
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int l = 0; l < LA; ++l) {
            const int c = tid + l * NT, r = c % BM, p = c / BM;
            #pragma unroll
            for (int e = 0; e < 4; ++e) As[p * 4 + e][r] = ra[l][e];
        }
        #pragma unroll
        for (int l = 0; l < LB; ++l) {
            const int c = tid + l * NT, r = c % BN, p = c / BN;
            #pragma unroll
            for (int e = 0; e < 4; ++e) Bs[p * 4 + e][r] = rb[l][e];
        }
        __syncthreads();
        if (k0 + BK < K) {
            #pragma unroll
            for (int l = 0; l < LA; ++l) ra[l] = *(const i4v*)(pa[l] + k0 + BK);
            #pragma unroll
            for (int l = 0; l < LB; ++l) rb[l] = *(const i4v*)(pb[l] + k0 + BK);
        }
        #pragma unroll
        for (int kq = 0; kq < KQ; ++kq) {
            int a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM / 4; ++i) {
                const i4v t = *(const i4v*)&As[kq][ty * TM + 4 * i];
                a[4 * i] = t[0]; a[4 * i + 1] = t[1]; a[4 * i + 2] = t[2]; a[4 * i + 3] = t[3];
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const i4v t = *(const i4v*)&Bs[kq][g * TX * 4 + tx * 4];
                b[4 * g] = t[0]; b[4 * g + 1] = t[1]; b[4 * g + 2] = t[2]; b[4 * g + 3] = t[3];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_sdot4(a[i], b[j], acc[i][j], false);
            }
        }
        __syncthreads();
    }
    f4 cv[G], bv[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int n = min(n0 + g * TX * 4 + tx * 4, N - 4);
        cv[g] = *(const f4*)(cs + n);
        bv[g] = *(const f4*)(bias + n);
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + ty * TM + i;
        if (m < M) {
            const float r = rs[m];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const int n = n0 + g * TX * 4 + tx * 4;
                if (n < N) {
                    h4 o;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float v = (float)acc[i][g * 4 + e] * r * cv[g][e] + bv[g][e];
                        if (RELU) v = fmaxf(v, 0.f);
                        o[e] = (_Float16)v;
                    }
                    *(h4*)(C + (long long)m * N + n) = o;
                }
            }
        }
    }
}

#define GEMM_INSTANCE(NAME, RELU) \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
NAME(const signed char* A, const signed char* B, const float* rs, const float* cs, const float* bias, \
     _Float16* C, int M, int N, int K) \
{ gemm_i8<NT_VALUE, BM_VALUE, BN_VALUE, BK_VALUE, TM_VALUE, TN_VALUE, RELU>(A, B, rs, cs, bias, C, M, N, K); }
GEMM_INSTANCE(gemm_i8_f16, false)
GEMM_INSTANCE(gemm_i8_relu_f16, true)

// v (NI float4 a lane, the row's 4-column groups lane * 4 + 128 i) as int8
// at q's row and the row's scale at s[row]: scale = largest / 127, each
// value rounded to nearest (ties to even) of v * 127 / largest.
template <int NI>
__device__ __forceinline__ void quantize_row(f4* v, signed char* __restrict__ q, float* __restrict__ s,
                                             long long base, int row, int lane)
{
    float mx = 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i)
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[i][0]), fabsf(v[i][1])), fmaxf(fabsf(v[i][2]), fabsf(v[i][3]))));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
    const float inv = mx > 0.f ? 127.f / mx : 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        unsigned int word = 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = (int)fminf(fmaxf(__builtin_rintf(v[i][e] * inv), -127.f), 127.f);
            word |= ((unsigned int)t & 0xffu) << (8 * e);
        }
        *(unsigned int*)(q + base + i * 128 + lane * 4) = word;
    }
    if (lane == 0) s[row] = mx / 127.f;
}

template <int NI, bool SMOOTH>
__device__ __forceinline__ void quant_rows(const _Float16* __restrict__ x, const float* __restrict__ sm,
                                           signed char* __restrict__ q, float* __restrict__ s, int rows)
{
    constexpr int D = 128 * NI;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;        // a whole wave: no lane of it reaches the shuffles
    const long long base = (long long)row * D;
    f4 v[NI];
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int c = i * 128 + lane * 4;
        v[i] = __builtin_convertvector(*(const h4*)(x + base + c), f4);
        if (SMOOTH) v[i] *= *(const f4*)(sm + c);
    }
    quantize_row<NI>(v, q, s, base, row, lane);
}

// hip_kernels.add_ln's residual add and layer norm (o32 = LN(x + float(y))),
// o32 quantized for the next linear instead of add_ln's fp16 copy.
template <int NI, bool SMOOTH>
__device__ __forceinline__ void add_ln_q(const float* __restrict__ x, const _Float16* __restrict__ y,
                                         const float* __restrict__ w, const float* __restrict__ b,
                                         const float* __restrict__ sm, float* __restrict__ o32,
                                         signed char* __restrict__ q, float* __restrict__ s,
                                         int rows, float eps)
{
    constexpr int D = 128 * NI;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long long base = (long long)row * D;
    f4 v[NI];
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int c = i * 128 + lane * 4;
        v[i] = *(const f4*)(x + base + c) + __builtin_convertvector(*(const h4*)(y + base + c), f4);
        sum += v[i][0] + v[i][1] + v[i][2] + v[i][3];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
    const float mean = sum * (1.f / D);
    float sq = 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const f4 dl = v[i] - mean;
        sq += dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2] + dl[3] * dl[3];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
    const float rstd = 1.f / sqrtf(sq * (1.f / D) + eps);
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int c = i * 128 + lane * 4;
        const f4 r = (v[i] - mean) * rstd * *(const f4*)(w + c) + *(const f4*)(b + c);
        *(f4*)(o32 + base + c) = r;
        v[i] = SMOOTH ? r * *(const f4*)(sm + c) : r;
    }
    quantize_row<NI>(v, q, s, base, row, lane);
}

#define Q_INSTANCE(NI) \
extern "C" __global__ void __launch_bounds__(Q_THREADS_VALUE) \
quant_rows_d##NI(const _Float16* x, signed char* q, float* s, int rows) \
{ quant_rows<NI, false>(x, nullptr, q, s, rows); } \
extern "C" __global__ void __launch_bounds__(Q_THREADS_VALUE) \
quant_rows_sm_d##NI(const _Float16* x, const float* sm, signed char* q, float* s, int rows) \
{ quant_rows<NI, true>(x, sm, q, s, rows); } \
extern "C" __global__ void __launch_bounds__(Q_THREADS_VALUE) \
add_ln_q_d##NI(const float* x, const _Float16* y, const float* w, const float* b, float* o32, \
               signed char* q, float* s, int rows, float eps) \
{ add_ln_q<NI, false>(x, y, w, b, nullptr, o32, q, s, rows, eps); } \
extern "C" __global__ void __launch_bounds__(Q_THREADS_VALUE) \
add_ln_q_sm_d##NI(const float* x, const _Float16* y, const float* w, const float* b, const float* sm, \
                  float* o32, signed char* q, float* s, int rows, float eps) \
{ add_ln_q<NI, true>(x, y, w, b, sm, o32, q, s, rows, eps); }
Q_INSTANCES
'''


def _source() -> str:
    s = (_SOURCE.replace("NT_VALUE", str(_NT)).replace("BM_VALUE", str(_BM)).replace("BN_VALUE", str(_BN))
         .replace("BK_VALUE", str(_BK)).replace("TM_VALUE", str(_TM)).replace("TN_VALUE", str(_TN))
         .replace("Q_THREADS_VALUE", str(_Q_THREADS)))
    return s.replace("Q_INSTANCES", "\n".join(f"Q_INSTANCE({ni})" for ni in _Q_NI))


class _Int8Kernels:
    """The compiled module on one device and its entry points."""

    def __init__(self, device: torch.device):
        self.hip = hip_runtime()
        self.arch = device_arch(device)
        self.module = load_module(self.hip, compile_source(self.arch, _source(), "wesnoth_int8.hip"), device)
        fn = self._function
        self.gemm = {False: fn("gemm_i8_f16"), True: fn("gemm_i8_relu_f16")}
        self.quant = {(128 * ni, sm): fn(f"quant_rows{'_sm' if sm else ''}_d{ni}")
                      for ni in _Q_NI for sm in (False, True)}
        self.add_ln_q = {(128 * ni, sm): fn(f"add_ln_q{'_sm' if sm else ''}_d{ni}")
                         for ni in _Q_NI for sm in (False, True)}

    def _function(self, name: str):
        return module_function(self.hip, self.module, name)

    def resources(self) -> Dict[str, Dict[str, int]]:
        """Registers and scratch bytes a thread of the GEMMs and of the
        widest quantizers (scratch > 0: the compiler spilled)."""
        out = {f"gemm_relu_{r}": function_resources(self.hip, self.gemm[r]) for r in (False, True)}
        wide = 128 * max(_Q_NI)
        out[f"quant_d{wide}"] = function_resources(self.hip, self.quant[(wide, False)])
        out[f"add_ln_q_d{wide}"] = function_resources(self.hip, self.add_ln_q[(wide, True)])
        return out

    def launch_gemm(self, xq, xs, wq, ws, bias, out, relu: bool) -> None:
        M, K = xq.shape
        N = wq.shape[0]
        grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)
        launch(self.hip, self.gemm[bool(relu)], grid, _NT, [xq, wq, xs, ws, bias, out, M, N, K], xq.device)

    def launch_quant(self, x, smooth, q, s) -> None:
        rows, d = x.shape
        per_block = _Q_THREADS // 32
        grid = ((rows + per_block - 1) // per_block, 1, 1)
        args = [x] + ([smooth] if smooth is not None else []) + [q, s, rows]
        launch(self.hip, self.quant[(d, smooth is not None)], grid, _Q_THREADS, args, x.device)

    def launch_add_ln_q(self, x, y, w, b, smooth, o32, q, s, eps: float) -> None:
        rows, d = x.shape
        per_block = _Q_THREADS // 32
        grid = ((rows + per_block - 1) // per_block, 1, 1)
        args = [x, y, w, b] + ([smooth] if smooth is not None else []) + [o32, q, s, rows, float(eps)]
        launch(self.hip, self.add_ln_q[(d, smooth is not None)], grid, _Q_THREADS, args, x.device)


_lock = threading.Lock()
_kernels: Dict[int, Optional[_Int8Kernels]] = {}


def _kernels_for(device: torch.device) -> Optional[_Int8Kernels]:
    """The device's kernels, built and self-tested on first use; None
    (reported once) where either fails."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        if index not in _kernels:
            try:
                kern = _Int8Kernels(torch.device("cuda", index))
                _self_test(kern, torch.device("cuda", index))
                log.info("hip int8 kernels: built for %s on cuda:%d", kern.arch, index)
            except Exception as e:  # noqa: BLE001 -- any failure means: not served here
                log.warning("hip int8 kernels unavailable on cuda:%d: %s", index, e)
                kern = None
            _kernels[index] = kern
        return _kernels[index]


# ---------------------------------------------------------------------
# References (torch, any device): what the kernels compute
# ---------------------------------------------------------------------

def quantize_rows_reference(x: torch.Tensor, smooth: Optional[torch.Tensor] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(int8 rows, float32 scale per row) as the kernels round them: the
    row's largest magnitude L, scale L / 127, each value rounded to nearest
    (ties to even) of value * (127 / L); a zero row has scale 0."""
    v = x.float()
    if smooth is not None:
        v = v * smooth.float()
    mx = v.abs().amax(dim=1)
    # A tensor quotient, correctly rounded as the kernels' 127.f / L is:
    # torch computes a scalar over a tensor as the scalar times the
    # reciprocal, which rounds twice and turns some exact ties (a value at
    # half of L) the other way.
    inv = torch.where(mx > 0, torch.full_like(mx, 127.0) / mx, torch.zeros_like(mx))
    q = torch.round(v * inv[:, None]).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, mx / 127.0


def quantize_weight(w: torch.Tensor, smooth: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """A weight [N, K] (times the inverse of a per-input-channel `smooth`
    factor, arm S) as int8 with one float32 scale per output channel, by
    quantize_rows_reference's rule."""
    v = w.detach().float()
    if smooth is not None:
        v = v / smooth.float()
    return quantize_rows_reference(v)


def int8_linear_reference(xq, xs, wq, ws, bias, relu: bool = False) -> torch.Tensor:
    """The GEMM's output computed by torch: the integer products exact in
    float64 (|sum| < 2^53), scaled in float32, rounded to fp16."""
    acc = (xq.double() @ wq.double().t()).float()
    out = acc * xs.float()[:, None] * ws.float()[None, :] + bias.float()[None, :]
    if relu:
        out = out.clamp_min(0)
    return out.half()


def _self_test(kern: _Int8Kernels, device: torch.device) -> None:
    """The GEMM (both epilogues) against int8_linear_reference on problems
    whose M and N end inside a tile and span several, and K over several
    tiles; the quantizers against quantize_rows_reference, every width,
    with and without smoothing (int8 values equal, scales to float32's
    rounding), and add_ln_q's residual against F.layer_norm."""
    gen = torch.Generator().manual_seed(0)
    for M, N, K in ((37, 132, 2 * _BK), (_BM + 3, _BN + 4, 3 * _BK), (5, 8, _BK)):
        xq = torch.randint(-127, 128, (M, K), generator=gen, dtype=torch.int8)
        wq = torch.randint(-127, 128, (N, K), generator=gen, dtype=torch.int8)
        xs = torch.rand(M, generator=gen) * 1e-2
        ws = torch.rand(N, generator=gen) * 1e-2
        bias = torch.randn(N, generator=gen)
        for relu in (False, True):
            ref = int8_linear_reference(xq, xs, wq, ws, bias, relu).float()
            got = hip_int8_linear(*(t.to(device) for t in (xq, xs, wq, ws, bias)), relu=relu, _kern=kern)
            err = ((got.float().cpu() - ref).abs() / (ref.abs() + 1e-3)).max().item()
            if not err <= 2e-3:
                raise RuntimeError(f"self-test of the int8 GEMM at {M}x{N}x{K} relu={relu}: relative error {err}")
    for ni in _Q_NI:
        d = 128 * ni
        x = (torch.randn(33, d, generator=gen) * 3).half()
        x[3] = 0
        sm = torch.rand(d, generator=gen) + 0.5
        for smooth in (None, sm):
            q_ref, s_ref = quantize_rows_reference(x, smooth)
            q, s = hip_quantize_rows(x.to(device), None if smooth is None else smooth.to(device), _kern=kern)
            if not torch.equal(q.cpu(), q_ref) or not torch.allclose(s.cpu(), s_ref, rtol=1e-6, atol=0):
                raise RuntimeError(f"self-test of the row quantizer at d {d} (smooth {smooth is not None})")
        r = (torch.randn(29, d, generator=gen) * 3)
        y = torch.randn(29, d, generator=gen).half()
        w, b = torch.rand(d, generator=gen) + 0.5, torch.randn(d, generator=gen)
        for smooth in (None, sm):
            o32, q, s = hip_add_layer_norm_q(*(t.to(device) for t in (r, y, w, b)), 1e-5,
                                             smooth=None if smooth is None else smooth.to(device), _kern=kern)
            ref = F.layer_norm(r + y.float(), (d,), w, b, 1e-5)
            if not (o32.cpu() - ref).abs().max().item() <= 1e-4:
                raise RuntimeError(f"self-test of the add, layer norm and quantizer at d {d}: the residual")
            q_ref, s_ref = quantize_rows_reference(o32.cpu(), smooth)
            # o32 is the kernel's own float32 output, so the quantization
            # reads the same values; the products by 127 / L may round
            # differently only where the compiler contracts them.
            if (q.cpu().int() - q_ref.int()).abs().max().item() > 1 \
                    or not torch.allclose(s.cpu(), s_ref, rtol=1e-6, atol=0):
                raise RuntimeError(f"self-test of the add, layer norm and quantizer at d {d}: the int8 rows")


# ---------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------

def hip_int8_applies(device: torch.device) -> bool:
    """Whether the int8 kernels serve `device`: a ROCm build of torch, a
    GPU device, and kernels built and self-tested there (built here on
    first ask)."""
    if torch.version.hip is None or device.type != "cuda":
        return False
    return _kernels_for(device) is not None


def int8_width_served(d: int) -> bool:
    """Whether the quantizers take rows of width `d`."""
    return d in tuple(128 * ni for ni in _Q_NI)


def _check_rows(x: torch.Tensor, dtype: torch.dtype, what: str) -> None:
    if x.dtype != dtype or x.dim() != 2:
        raise TypeError(f"{what} takes {dtype} [rows, d], not {x.dtype} {tuple(x.shape)}")
    if not int8_width_served(x.shape[1]):
        raise ValueError(f"{what}: width {x.shape[1]} is not one the source instantiates "
                         f"{tuple(128 * ni for ni in _Q_NI)}")


def _check_smooth(smooth: Optional[torch.Tensor], d: int, device: torch.device) -> Optional[torch.Tensor]:
    if smooth is None:
        return None
    if smooth.dtype != torch.float32 or smooth.shape != (d,) or smooth.device != device:
        raise TypeError(f"the smoothing factors are float32 [{d}] on the rows' device")
    return _aligned(smooth)


def hip_quantize_rows(x: torch.Tensor, smooth: Optional[torch.Tensor] = None, _kern=None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp16 rows [M, d] (times `smooth` per column, when given) as int8 with
    a float32 scale per row (quantize_rows_reference's rule). The caller
    has checked `hip_int8_applies`."""
    _check_rows(x, torch.float16, "hip_quantize_rows")
    kern = _kern or _kernels_for(x.device)
    x = _aligned(x)
    smooth = _check_smooth(smooth, x.shape[1], x.device)
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    if x.shape[0]:
        kern.launch_quant(x, smooth, q, s)
    return q, s


def hip_add_layer_norm_q(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         eps: float, smooth: Optional[torch.Tensor] = None, _kern=None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hip_kernels.hip_add_layer_norm's float32 output (the residual
    stream), with that output (times `smooth`, when given) quantized as
    hip_quantize_rows does instead of its fp16 copy: (o32, int8 rows,
    float32 scale per row). The caller has checked `hip_int8_applies`."""
    _check_rows(x, torch.float32, "hip_add_layer_norm_q")
    if y.dtype != torch.float16 or y.shape != x.shape:
        raise TypeError(f"hip_add_layer_norm_q takes fp16 y of x's shape: {y.dtype} {tuple(y.shape)}")
    d = x.shape[1]
    if weight.dtype != torch.float32 or bias.dtype != torch.float32 or weight.shape != (d,) or bias.shape != (d,):
        raise TypeError(f"hip_add_layer_norm_q takes float32 weight and bias of [{d}]")
    if not (x.device == y.device == weight.device == bias.device):
        raise ValueError("hip_add_layer_norm_q's tensors must be on one device")
    kern = _kern or _kernels_for(x.device)
    x, y = _aligned(x), _aligned(y)
    smooth = _check_smooth(smooth, d, x.device)
    o32 = torch.empty_like(x)
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    if x.shape[0]:
        kern.launch_add_ln_q(x, y, _aligned(weight), _aligned(bias), smooth, o32, q, s, eps)
    return o32, q, s


def hip_int8_linear(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
                    bias: torch.Tensor, relu: bool = False, _kern=None) -> torch.Tensor:
    """half((xq . wq^T) * xs[:, None] * ws[None, :] + bias), ReLU'd when
    `relu`: xq int8 [M, K], xs float32 [M], wq int8 [N, K], ws and bias
    float32 [N]; K a multiple of the K tile (32) up to 8192, N a multiple of
    4. The caller has checked `hip_int8_applies`; forward only."""
    if xq.dtype != torch.int8 or wq.dtype != torch.int8 or xq.dim() != 2 or wq.dim() != 2:
        raise TypeError(f"hip_int8_linear takes int8 [M, K] and [N, K]: {xq.dtype} {tuple(xq.shape)}, "
                        f"{wq.dtype} {tuple(wq.shape)}")
    M, K = xq.shape
    N = wq.shape[0]
    if wq.shape[1] != K or K % _BK or not 0 < K <= _MAX_K or N % 4 or N == 0:
        raise ValueError(f"hip_int8_linear: K {K} (weight {wq.shape[1]}) must be a multiple of {_BK} up to "
                         f"{_MAX_K}, N {N} a positive multiple of 4")
    for name, t, n in (("xs", xs, M), ("ws", ws, N), ("bias", bias, N)):
        if t.dtype != torch.float32 or t.shape != (n,):
            raise TypeError(f"hip_int8_linear: {name} must be float32 [{n}], not {t.dtype} {tuple(t.shape)}")
    if len({t.device for t in (xq, xs, wq, ws, bias)}) != 1:
        raise ValueError("hip_int8_linear's tensors must be on one device")
    out = torch.empty((M, N), dtype=torch.float16, device=xq.device)
    if M == 0:
        return out
    kern = _kern or _kernels_for(xq.device)
    kern.launch_gemm(_aligned(xq), _aligned(xs), _aligned(wq), _aligned(ws), _aligned(bias), out, relu)
    return out
