"""HIP kernels for the packed trunk's fp16 inference on AMD GPUs that
PyTorch's fused attention does not serve (2026-10-28: the local RX 6600,
gfx1032, ROCm on Windows, where flash and memory-efficient SDPA both
report "No available kernel" and the math path took 65% of a forward;
training/metrics/local_gpu_20261028/README.md): the variable-length
self-attention, and the residual add fused with the layer norm after it
(WesnothModel._run_hip_fp16_trunk).

The attention is a flash-attention forward written in HIP for GPUs without
matrix units (RDNA2): one thread owns a few query rows (_QPT) of one head
of one segment and walks the segment's keys in tiles held in shared
memory, with an online softmax. The products are v_dot2_f32_f16 (two fp16
multiplies summed into an fp32 accumulator): Q.K^T reads a key row per
step, and P.V reads the value tile stored as key pairs per dimension, so
the probabilities enter as fp16 pairs (as flash attention's P.V does).
Scores, the running max and sum, and the accumulator stay fp32.

Calling convention: the flash varlen kernel's (packed_trunk.py module
docstring): q, k, v [total, heads, head_dim] fp16 with a unit-stride
last dim and the heads adjacent, cu_seqlens int32 [B+1] on the device,
softmax scale 1/sqrt(head_dim), no mask, not causal; the output is a
new contiguous [total, heads, head_dim] whose rows past cu_seqlens[-1]
are left unwritten, as flash leaves them. Forward only: no backward.

The add and norm reads the float32 residual rows and the fp16 sublayer
output, and writes the float32 layer norm of their sum (the autocast's
residual stream) and its fp16 rounding (the next linear's input): an
add, a layer norm and a cast in one pass, one wave32 a row.

Build: hiprtc (shipped with the ROCm runtime) compiles the source for
the device's architecture on first use, and the HIP module API loads
it; both are reached through ctypes, so no host compiler is needed (the
local machine has none). The launch goes on torch's current stream. On
first use per device the kernels are checked against
F.scaled_dot_product_attention and F.layer_norm on fixed problems; a
device where the build or a check fails is reported once in the log and
served by the callers' fallbacks (the `*_applies` functions are False).
"""
from __future__ import annotations

import ctypes
import glob
import logging
import math
import os
import sys
import threading
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

HEAD_DIMS = (32, 64)        # the head sizes the source instantiates
# The tiling (measured 2026-10-28 on the RX 6600, 48 segments of 141-692
# tokens, 12 heads of 32): NT threads a block, QPT query rows a thread
# (rows tid, tid + NT, ...), so a key or value read from shared memory
# serves QPT rows; BK keys a tile. One row a thread ran 4.3 TFLOP/s, two
# 7.8, three 9.5 at NT 32 and BK 16, four spilled registers (2.8-7.9).
_NT = 32
_QPT = {32: 3, 64: 1}       # head_dim 64's accumulators allow one row a thread
_BK = 16
# The layer norm: LN_THREADS threads a block, a wave32 a row; the model
# widths it serves are 128 times LN_NI (own2's 384: 3).
_LN_THREADS = 128
_LN_NI = (1, 2, 3, 4, 6)
_LOG2E = 1.4426950408889634
_SELF_TEST_TOL = 2e-2       # fp16 inputs and output against an fp32 reference

_SOURCE = r'''
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));

template <int NT, int QPT, int BK, int D>
__device__ __forceinline__ void attn_fwd(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k, const _Float16* __restrict__ v,
    _Float16* __restrict__ o, const int* __restrict__ cu,
    int sq, int sk, int sv, int so, float scale_log2)
{
    constexpr int C = D / 2, CH = D / 8, BQ = NT * QPT;
    __shared__ __attribute__((aligned(16))) h2 Ks[BK][C];
    __shared__ __attribute__((aligned(16))) h2 Vp[BK / 2][D];
    const int seg = blockIdx.z, hd = blockIdx.y, tid = threadIdx.x;
    const int start = cu[seg], n = cu[seg + 1] - start;
    const int q0 = blockIdx.x * BQ;
    if (q0 >= n) return;
    h2 qr[QPT][C];
    bool active[QPT];
    #pragma unroll
    for (int r = 0; r < QPT; ++r) {
        const int qi = q0 + r * NT + tid;
        active[r] = qi < n;
        if (active[r]) {
            const h8* src = (const h8*)(q + (long)(start + qi) * sq + hd * D);
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                h8 t = src[c];
                qr[r][c * 4 + 0] = __builtin_shufflevector(t, t, 0, 1);
                qr[r][c * 4 + 1] = __builtin_shufflevector(t, t, 2, 3);
                qr[r][c * 4 + 2] = __builtin_shufflevector(t, t, 4, 5);
                qr[r][c * 4 + 3] = __builtin_shufflevector(t, t, 6, 7);
            }
        } else {
            #pragma unroll
            for (int c = 0; c < C; ++c) qr[r][c] = (h2)(0);
        }
    }
    float m[QPT], l[QPT], acc[QPT][D];
    #pragma unroll
    for (int r = 0; r < QPT; ++r) {
        m[r] = -__builtin_inff(); l[r] = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[r][d] = 0.f;
    }
    for (int kb = 0; kb < n; kb += BK) {
        for (int i = tid; i < BK * CH; i += NT) {
            const int j = i / CH, c = i % CH;
            h8 t = (h8)(0);
            if (kb + j < n) t = ((const h8*)(k + (long)(start + kb + j) * sk + hd * D))[c];
            *(h8*)&Ks[j][c * 4] = t;
        }
        for (int i = tid; i < (BK / 2) * CH; i += NT) {
            const int jj = i / CH, c = i % CH;
            const int j0 = kb + 2 * jj;
            h8 a = (h8)(0), b = (h8)(0);
            if (j0 < n) a = ((const h8*)(v + (long)(start + j0) * sv + hd * D))[c];
            if (j0 + 1 < n) b = ((const h8*)(v + (long)(start + j0 + 1) * sv + hd * D))[c];
            *(h8*)&Vp[jj][c * 8] = __builtin_shufflevector(a, b, 0, 8, 1, 9, 2, 10, 3, 11);
            *(h8*)&Vp[jj][c * 8 + 4] = __builtin_shufflevector(a, b, 4, 12, 5, 13, 6, 14, 7, 15);
        }
        __syncthreads();
        float s[QPT][BK];
        float mt[QPT];
        #pragma unroll
        for (int r = 0; r < QPT; ++r) mt[r] = -__builtin_inff();
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            float dot[QPT];
            #pragma unroll
            for (int r = 0; r < QPT; ++r) dot[r] = 0.f;
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                h8 kk = *(const h8*)&Ks[j][c * 4];
                h2 k0 = __builtin_shufflevector(kk, kk, 0, 1), k1 = __builtin_shufflevector(kk, kk, 2, 3);
                h2 k2 = __builtin_shufflevector(kk, kk, 4, 5), k3 = __builtin_shufflevector(kk, kk, 6, 7);
                #pragma unroll
                for (int r = 0; r < QPT; ++r) {
                    dot[r] = __builtin_amdgcn_fdot2(qr[r][c * 4 + 0], k0, dot[r], false);
                    dot[r] = __builtin_amdgcn_fdot2(qr[r][c * 4 + 1], k1, dot[r], false);
                    dot[r] = __builtin_amdgcn_fdot2(qr[r][c * 4 + 2], k2, dot[r], false);
                    dot[r] = __builtin_amdgcn_fdot2(qr[r][c * 4 + 3], k3, dot[r], false);
                }
            }
            const bool valid = kb + j < n;
            #pragma unroll
            for (int r = 0; r < QPT; ++r) {
                float x = valid ? dot[r] * scale_log2 : -__builtin_inff();
                s[r][j] = x;
                mt[r] = fmaxf(mt[r], x);
            }
        }
        h2 pp[QPT][BK / 2];
        #pragma unroll
        for (int r = 0; r < QPT; ++r) {
            const float mn = fmaxf(m[r], mt[r]);
            const float alpha = __builtin_amdgcn_exp2f(m[r] - mn);
            l[r] *= alpha;
            #pragma unroll
            for (int d = 0; d < D; ++d) acc[r][d] *= alpha;
            #pragma unroll
            for (int jj = 0; jj < BK / 2; ++jj) {
                const float p0 = __builtin_amdgcn_exp2f(s[r][2 * jj] - mn);
                const float p1 = __builtin_amdgcn_exp2f(s[r][2 * jj + 1] - mn);
                l[r] += p0 + p1;
                pp[r][jj] = h2{(_Float16)p0, (_Float16)p1};
            }
            m[r] = mn;
        }
        #pragma unroll
        for (int jj = 0; jj < BK / 2; ++jj) {
            #pragma unroll
            for (int c = 0; c < D / 4; ++c) {
                h8 vv = *(const h8*)&Vp[jj][c * 4];
                h2 v0 = __builtin_shufflevector(vv, vv, 0, 1), v1 = __builtin_shufflevector(vv, vv, 2, 3);
                h2 v2 = __builtin_shufflevector(vv, vv, 4, 5), v3 = __builtin_shufflevector(vv, vv, 6, 7);
                #pragma unroll
                for (int r = 0; r < QPT; ++r) {
                    acc[r][c * 4 + 0] = __builtin_amdgcn_fdot2(pp[r][jj], v0, acc[r][c * 4 + 0], false);
                    acc[r][c * 4 + 1] = __builtin_amdgcn_fdot2(pp[r][jj], v1, acc[r][c * 4 + 1], false);
                    acc[r][c * 4 + 2] = __builtin_amdgcn_fdot2(pp[r][jj], v2, acc[r][c * 4 + 2], false);
                    acc[r][c * 4 + 3] = __builtin_amdgcn_fdot2(pp[r][jj], v3, acc[r][c * 4 + 3], false);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < QPT; ++r) {
        if (active[r]) {
            const float inv = 1.f / l[r];
            h8* dst = (h8*)(o + (long)(start + q0 + r * NT + tid) * so + hd * D);
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                h8 t;
                #pragma unroll
                for (int e = 0; e < 8; ++e) t[e] = (_Float16)(acc[r][c * 8 + e] * inv);
                dst[c] = t;
            }
        }
    }
}

#define ATTN_INSTANCE(D, QPT) \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
varlen_attn_fwd_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, _Float16* o, \
                         const int* cu, int sq, int sk, int sv, int so, float scale_log2) \
{ attn_fwd<NT_VALUE, QPT, BK_VALUE, D>(q, k, v, o, cu, sq, sk, sv, so, scale_log2); }
ATTN_INSTANCE(32, QPT32_VALUE)
ATTN_INSTANCE(64, QPT64_VALUE)

// The residual add and the layer norm after it: o32 = LN(x + float(y)),
// o16 = half(o32), the next linear's input. One wave32 a row, four
// elements a lane a step (d = 128 NI); the mean and the variance by two
// passes over the row held in registers.
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

template <int NI>
__device__ __forceinline__ void add_ln(const float* __restrict__ x, const _Float16* __restrict__ y,
                                       const float* __restrict__ w, const float* __restrict__ b,
                                       float* __restrict__ o32, _Float16* __restrict__ o16,
                                       int rows, float eps)
{
    constexpr int D = 128 * NI;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;        // a whole wave: no lane of it reaches the shuffles
    const long base = (long)row * D;
    f4 v[NI];
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int c = i * 128 + lane * 4;
        v[i] = *(const f4*)(x + base + c) + __builtin_convertvector(*(const h4*)(y + base + c), f4);
        s += v[i][0] + v[i][1] + v[i][2] + v[i][3];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor(s, off, 32);
    const float mean = s * (1.f / D);
    float q = 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const f4 dl = v[i] - mean;
        q += dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2] + dl[3] * dl[3];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) q += __shfl_xor(q, off, 32);
    const float rstd = 1.f / sqrtf(q * (1.f / D) + eps);
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int c = i * 128 + lane * 4;
        const f4 r = (v[i] - mean) * rstd * *(const f4*)(w + c) + *(const f4*)(b + c);
        *(f4*)(o32 + base + c) = r;
        *(h4*)(o16 + base + c) = __builtin_convertvector(r, h4);
    }
}

#define LN_INSTANCE(NI) \
extern "C" __global__ void __launch_bounds__(LN_THREADS_VALUE) \
add_ln_d##NI(const float* x, const _Float16* y, const float* w, const float* b, float* o32, \
             _Float16* o16, int rows, float eps) \
{ add_ln<NI>(x, y, w, b, o32, o16, rows, eps); }
LN_INSTANCE(1)
LN_INSTANCE(2)
LN_INSTANCE(3)
LN_INSTANCE(4)
LN_INSTANCE(6)
'''


def _source() -> str:
    return (_SOURCE.replace("NT_VALUE", str(_NT)).replace("BK_VALUE", str(_BK))
            .replace("QPT32_VALUE", str(_QPT[32])).replace("QPT64_VALUE", str(_QPT[64]))
            .replace("LN_THREADS_VALUE", str(_LN_THREADS)))


# ---------------------------------------------------------------------
# The HIP runtime and hiprtc through ctypes
# ---------------------------------------------------------------------

def _rocm_library(names) -> Optional[ctypes.CDLL]:
    """The first of `names` that loads: by name (the copy torch already
    loaded), then from the ROCm SDK beside torch (the Windows wheels'
    _rocm_sdk_core/bin, Linux's torch/lib)."""
    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    site = os.path.dirname(os.path.dirname(torch.__file__))
    dirs = [os.path.join(site, "_rocm_sdk_core", "bin"), os.path.join(os.path.dirname(torch.__file__), "lib")]
    for pattern in names:
        for candidate in [pattern] + [p for d in dirs for p in sorted(glob.glob(os.path.join(d, pattern)))]:
            if "*" in candidate:
                continue
            try:
                return loader(candidate)
            except OSError:
                continue
    return None


_HIP_NAMES = ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64*.dll", "libamdhip64.so")
_RTC_NAMES = ("hiprtc0*.dll", "libhiprtc.so")


def _compile(arch: str) -> bytes:
    rtc = _rocm_library(_RTC_NAMES)
    if rtc is None:
        raise RuntimeError("hiprtc not found beside torch")
    prog = ctypes.c_void_p()
    _check_rtc(rtc.hiprtcCreateProgram(ctypes.byref(prog), _source().encode(), b"wesnoth_kernels.hip",
                                       0, None, None), "hiprtcCreateProgram")
    try:
        opts = [f"--offload-arch={arch}".encode(), b"-O3"]
        r = rtc.hiprtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        if r != 0:
            size = ctypes.c_size_t()
            rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value + 1)
            rtc.hiprtcGetProgramLog(prog, buf)
            raise RuntimeError(f"hiprtc compile for {arch} failed ({r}): "
                               f"{buf.value.decode(errors='replace')[:2000]}")
        size = ctypes.c_size_t()
        _check_rtc(rtc.hiprtcGetCodeSize(prog, ctypes.byref(size)), "hiprtcGetCodeSize")
        code = ctypes.create_string_buffer(size.value)
        _check_rtc(rtc.hiprtcGetCode(prog, code), "hiprtcGetCode")
        return code.raw
    finally:
        rtc.hiprtcDestroyProgram(ctypes.byref(prog))


def _check_rtc(r: int, what: str) -> None:
    if r != 0:
        raise RuntimeError(f"{what} returned {r}")


class _DeviceKernels:
    """The compiled module on one device and its entry points."""

    def __init__(self, device: torch.device):
        self.hip = _rocm_library(_HIP_NAMES)
        if self.hip is None:
            raise RuntimeError("the HIP runtime library was not found")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        self.arch = arch
        code = _compile(arch)
        self.module = ctypes.c_void_p()
        with torch.cuda.device(device):
            self._check(self.hip.hipModuleLoadData(ctypes.byref(self.module), code), "hipModuleLoadData")
        self.functions = {d: self._function(f"varlen_attn_fwd_f16_d{d}") for d in HEAD_DIMS}
        self.add_ln = {128 * ni: self._function(f"add_ln_d{ni}") for ni in _LN_NI}

    def _function(self, name: str) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        self._check(self.hip.hipModuleGetFunction(ctypes.byref(fn), self.module, name.encode()),
                    "hipModuleGetFunction")
        return fn

    @staticmethod
    def _check(r: int, what: str) -> None:
        if r != 0:
            raise RuntimeError(f"{what} returned hipError {r}")

    def _launch(self, fn, grid, threads: int, args, device: torch.device) -> None:
        """One launch on torch's current stream; `args` are tensors (their
        data pointers), ints and floats, in the kernel's order."""
        held = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor)
                else ctypes.c_float(a) if isinstance(a, float) else ctypes.c_int(a) for a in args]
        params = (ctypes.c_void_p * len(held))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in held])
        stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        self._check(self.hip.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], threads, 1, 1, 0,
                                                   stream, params, None),
                    "hipModuleLaunchKernel")

    def launch(self, q, k, v, out, cu_seqlens, max_len: int) -> None:
        total, heads, d = q.shape
        bq = _NT * _QPT[d]
        self._launch(self.functions[d], ((max_len + bq - 1) // bq, heads, cu_seqlens.numel() - 1), _NT,
                     [q, k, v, out, cu_seqlens, q.stride(0), k.stride(0), v.stride(0), out.stride(0),
                      _LOG2E / math.sqrt(d)], q.device)

    def launch_add_ln(self, x, y, w, b, o32, o16, eps: float) -> None:
        rows, d = x.shape
        per_block = _LN_THREADS // 32
        self._launch(self.add_ln[d], ((rows + per_block - 1) // per_block, 1, 1), _LN_THREADS,
                     [x, y, w, b, o32, o16, rows, float(eps)], x.device)


_lock = threading.Lock()
_kernels: Dict[int, Optional[_DeviceKernels]] = {}


def _kernels_for(device: torch.device) -> Optional[_DeviceKernels]:
    """The device's kernels, built and self-tested on first use; None
    (reported once) where either fails."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        if index not in _kernels:
            try:
                kern = _DeviceKernels(torch.device("cuda", index))
                _self_test(kern, torch.device("cuda", index))
                log.info("hip kernels: built for %s on cuda:%d", kern.arch, index)
            except Exception as e:  # noqa: BLE001 -- any failure means: not served here
                log.warning("hip kernels unavailable on cuda:%d: %s", index, e)
                kern = None
            _kernels[index] = kern
        return _kernels[index]


def _self_test(kern: _DeviceKernels, device: torch.device) -> None:
    """The attention against F.scaled_dot_product_attention (fp32) on a
    fixed problem per instantiated head size (segment lengths that end
    inside, on and past a tile, and a segment longer than a query block),
    and the add and norm against F.layer_norm per instantiated width."""
    gen = torch.Generator().manual_seed(0)
    bq = _NT * _QPT[32]
    lens = [1, _BK - 1, _BK, _NT + 1, bq - 1, bq + 3, 2 * bq + _BK + 5]
    cu_host = [0]
    for n in lens:
        cu_host.append(cu_host[-1] + n)
    for d in HEAD_DIMS:
        qkv = torch.randn(cu_host[-1], 3, 3, d, generator=gen).to(device, torch.float16)
        q, k, v = qkv.unbind(1)
        out = torch.empty(q.shape, dtype=q.dtype, device=device)
        kern.launch(q, k, v, out, torch.tensor(cu_host, dtype=torch.int32, device=device), max(lens))
        worst = 0.0
        for lo, hi in zip(cu_host[:-1], cu_host[1:]):
            seg = [t[lo:hi].float().transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
            ref = F.scaled_dot_product_attention(*seg)[0].transpose(0, 1)
            worst = max(worst, (out[lo:hi].float() - ref).abs().max().item())
        if not worst <= _SELF_TEST_TOL:
            raise RuntimeError(f"self-test at head_dim {d}: max abs error {worst} > {_SELF_TEST_TOL}")
    for d in sorted(kern.add_ln):
        x = (torch.randn(37, d, generator=gen) * 3).to(device)
        y = torch.randn(37, d, generator=gen).to(device, torch.float16)
        w, b = (torch.rand(d, generator=gen) + 0.5).to(device), torch.randn(d, generator=gen).to(device)
        o32, o16 = torch.empty_like(x), torch.empty_like(y)
        kern.launch_add_ln(x, y, w, b, o32, o16, 1e-5)
        ref = F.layer_norm(x + y.float(), (d,), w, b, 1e-5)
        worst = (o32 - ref).abs().max().item()
        if not worst <= 1e-4 or not (o16.float() - ref).abs().max().item() <= _SELF_TEST_TOL:
            raise RuntimeError(f"self-test of the add and layer norm at d {d}: max abs error {worst}")


# ---------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------

def hip_varlen_applies(device: torch.device, dtype: torch.dtype, head_dim: Optional[int] = None) -> bool:
    """Whether the kernel serves this call: a ROCm build of torch, a GPU
    device, fp16, an instantiated head size (when given), and a kernel
    built and self-tested on the device (built here on first ask)."""
    if torch.version.hip is None or device.type != "cuda" or dtype != torch.float16:
        return False
    if head_dim is not None and head_dim not in HEAD_DIMS:
        return False
    return _kernels_for(device) is not None


def _kernel_view(t: torch.Tensor) -> torch.Tensor:
    """t, or a contiguous copy where its layout is not the kernel's:
    unit-stride head dim, heads adjacent, 16-byte rows and token strides."""
    d = t.shape[-1]
    if (t.stride(2) == 1 and t.stride(1) == d and t.stride(0) % 8 == 0
            and t.data_ptr() % 16 == 0):
        return t
    return t.contiguous()


def hip_add_layer_norm_applies(device: torch.device, d: int) -> bool:
    """Whether hip_add_layer_norm serves rows of width `d` on `device`."""
    if torch.version.hip is None or device.type != "cuda" or d not in tuple(128 * ni for ni in _LN_NI):
        return False
    return _kernels_for(device) is not None


def hip_add_layer_norm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """F.layer_norm(x + y) for the float32 residual rows x [rows, d] and the
    fp16 sublayer output y, in float32 (as an autocast's layer norm
    computes it, up to the order of the sums), and its fp16 rounding, the
    next linear's input, in one pass. weight and bias float32 [d]. The
    caller has checked `hip_add_layer_norm_applies`; forward only."""
    kern = _kernels_for(x.device)
    x, y = x.contiguous(), y.contiguous()
    o32 = torch.empty_like(x)
    o16 = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    kern.launch_add_ln(x, y, weight.contiguous(), bias.contiguous(), o32, o16, eps)
    return o32, o16


def hip_varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         cu_seqlens: torch.Tensor, max_len: int) -> torch.Tensor:
    """Self-attention within each segment of the packed [total, heads,
    head_dim] q, k, v (module docstring). The caller has checked
    `hip_varlen_applies`."""
    if q.requires_grad or k.requires_grad or v.requires_grad:
        if torch.is_grad_enabled():
            raise RuntimeError("hip_varlen_attention has no backward")
    kern = _kernels_for(q.device)
    q, k, v = (_kernel_view(t) for t in (q, k, v))
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    cu = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    kern.launch(q, k, v, out, cu.contiguous(), int(max_len))
    return out
