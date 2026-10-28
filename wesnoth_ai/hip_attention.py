"""Variable-length self-attention for the packed trunk on AMD GPUs that
PyTorch's fused attention does not serve (2026-10-28: the local RX 6600,
gfx1032, ROCm on Windows, where flash and memory-efficient SDPA both
report "No available kernel" and the math path took 65% of a forward;
training/metrics/local_gpu_20261028/README.md).

The kernel is a flash-attention forward written in HIP for GPUs without
matrix units (RDNA2): one thread owns one query row of one head of one
segment and walks the segment's keys in tiles held in shared memory,
with an online softmax. The products are v_dot2_f32_f16 (two fp16
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

Build: hiprtc (shipped with the ROCm runtime) compiles the source for
the device's architecture on first use, and the HIP module API loads
it; both are reached through ctypes, so no host compiler is needed (the
local machine has none). The launch goes on torch's current stream. On
first use per device the kernel is checked against
F.scaled_dot_product_attention on a fixed problem; a device where the
build or the check fails is reported once in the log and served by the
caller's fallback (`available` is False).
"""
from __future__ import annotations

import ctypes
import glob
import logging
import math
import os
import sys
import threading
from typing import Dict, Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

HEAD_DIMS = (32, 64)        # the head sizes the source instantiates
_BQ = 64                    # queries a block: one per thread, two wave32s
_BK = 32                    # keys a shared-memory tile (BK 64 measured slower, 2026-10-28)
_LOG2E = 1.4426950408889634
_SELF_TEST_TOL = 2e-2       # fp16 inputs and output against an fp32 reference

_SOURCE = r'''
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));

template <int BQ, int BK, int D>
__device__ __forceinline__ void attn_fwd(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k, const _Float16* __restrict__ v,
    _Float16* __restrict__ o, const int* __restrict__ cu,
    int sq, int sk, int sv, int so, float scale_log2)
{
    constexpr int C = D / 2;        // fp16 pairs a row
    constexpr int CH = D / 8;       // 16-byte chunks a row
    __shared__ __attribute__((aligned(16))) h2 Ks[BK][C];       // key rows
    __shared__ __attribute__((aligned(16))) h2 Vp[BK / 2][D];   // (v[2j][d], v[2j+1][d])
    const int seg = blockIdx.z, hd = blockIdx.y, tid = threadIdx.x;
    const int start = cu[seg], n = cu[seg + 1] - start;
    const int q0 = blockIdx.x * BQ;
    if (q0 >= n) return;            // uniform over the block: before any barrier
    const int qi = q0 + tid;
    const bool active = qi < n;
    h2 qr[C];
    if (active) {
        const h8* src = (const h8*)(q + (long)(start + qi) * sq + hd * D);
        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            h8 t = src[c];
            qr[c * 4 + 0] = __builtin_shufflevector(t, t, 0, 1);
            qr[c * 4 + 1] = __builtin_shufflevector(t, t, 2, 3);
            qr[c * 4 + 2] = __builtin_shufflevector(t, t, 4, 5);
            qr[c * 4 + 3] = __builtin_shufflevector(t, t, 6, 7);
        }
    } else {
        #pragma unroll
        for (int c = 0; c < C; ++c) qr[c] = (h2)(0);
    }
    float m = -__builtin_inff(), l = 0.f;
    float acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0.f;

    for (int kb = 0; kb < n; kb += BK) {
        for (int i = tid; i < BK * CH; i += BQ) {
            const int j = i / CH, c = i % CH;
            h8 t = (h8)(0);
            if (kb + j < n) t = ((const h8*)(k + (long)(start + kb + j) * sk + hd * D))[c];
            *(h8*)&Ks[j][c * 4] = t;
        }
        for (int i = tid; i < (BK / 2) * CH; i += BQ) {
            const int jj = i / CH, c = i % CH;
            const int j0 = kb + 2 * jj;
            h8 a = (h8)(0), b = (h8)(0);
            if (j0 < n) a = ((const h8*)(v + (long)(start + j0) * sv + hd * D))[c];
            if (j0 + 1 < n) b = ((const h8*)(v + (long)(start + j0 + 1) * sv + hd * D))[c];
            *(h8*)&Vp[jj][c * 8] = __builtin_shufflevector(a, b, 0, 8, 1, 9, 2, 10, 3, 11);
            *(h8*)&Vp[jj][c * 8 + 4] = __builtin_shufflevector(a, b, 4, 12, 5, 13, 6, 14, 7, 15);
        }
        __syncthreads();
        float s[BK];
        float mt = -__builtin_inff();
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            float dot = 0.f;
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                h8 kk = *(const h8*)&Ks[j][c * 4];
                dot = __builtin_amdgcn_fdot2(qr[c * 4 + 0], __builtin_shufflevector(kk, kk, 0, 1), dot, false);
                dot = __builtin_amdgcn_fdot2(qr[c * 4 + 1], __builtin_shufflevector(kk, kk, 2, 3), dot, false);
                dot = __builtin_amdgcn_fdot2(qr[c * 4 + 2], __builtin_shufflevector(kk, kk, 4, 5), dot, false);
                dot = __builtin_amdgcn_fdot2(qr[c * 4 + 3], __builtin_shufflevector(kk, kk, 6, 7), dot, false);
            }
            dot *= scale_log2;
            if (kb + j >= n) dot = -__builtin_inff();
            s[j] = dot;
            mt = fmaxf(mt, dot);
        }
        // Every tile holds a key (kb < n), so mn is finite and the first
        // tile's alpha is exp2(-inf) = 0.
        const float mn = fmaxf(m, mt);
        const float alpha = __builtin_amdgcn_exp2f(m - mn);
        l *= alpha;
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[d] *= alpha;
        #pragma unroll
        for (int jj = 0; jj < BK / 2; ++jj) {
            const float p0 = __builtin_amdgcn_exp2f(s[2 * jj] - mn);
            const float p1 = __builtin_amdgcn_exp2f(s[2 * jj + 1] - mn);
            l += p0 + p1;
            const h2 pp = {(_Float16)p0, (_Float16)p1};
            #pragma unroll
            for (int c = 0; c < D / 4; ++c) {
                h8 vv = *(const h8*)&Vp[jj][c * 4];
                acc[c * 4 + 0] = __builtin_amdgcn_fdot2(pp, __builtin_shufflevector(vv, vv, 0, 1), acc[c * 4 + 0], false);
                acc[c * 4 + 1] = __builtin_amdgcn_fdot2(pp, __builtin_shufflevector(vv, vv, 2, 3), acc[c * 4 + 1], false);
                acc[c * 4 + 2] = __builtin_amdgcn_fdot2(pp, __builtin_shufflevector(vv, vv, 4, 5), acc[c * 4 + 2], false);
                acc[c * 4 + 3] = __builtin_amdgcn_fdot2(pp, __builtin_shufflevector(vv, vv, 6, 7), acc[c * 4 + 3], false);
            }
        }
        m = mn;
        __syncthreads();
    }
    if (active) {
        const float inv = 1.f / l;
        h8* dst = (h8*)(o + (long)(start + qi) * so + hd * D);
        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            h8 t;
            #pragma unroll
            for (int e = 0; e < 8; ++e) t[e] = (_Float16)(acc[c * 8 + e] * inv);
            dst[c] = t;
        }
    }
}

#define INSTANCE(D) \
extern "C" __global__ void __launch_bounds__(BQ_VALUE) \
varlen_attn_fwd_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, _Float16* o, \
                         const int* cu, int sq, int sk, int sv, int so, float scale_log2) \
{ attn_fwd<BQ_VALUE, BK_VALUE, D>(q, k, v, o, cu, sq, sk, sv, so, scale_log2); }
INSTANCE(32)
INSTANCE(64)
'''


def _source() -> str:
    return _SOURCE.replace("BQ_VALUE", str(_BQ)).replace("BK_VALUE", str(_BK))


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
    _check_rtc(rtc.hiprtcCreateProgram(ctypes.byref(prog), _source().encode(), b"varlen_attention.hip",
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
        self.functions: Dict[int, ctypes.c_void_p] = {}
        for d in HEAD_DIMS:
            fn = ctypes.c_void_p()
            self._check(self.hip.hipModuleGetFunction(ctypes.byref(fn), self.module,
                                                      f"varlen_attn_fwd_f16_d{d}".encode()),
                        "hipModuleGetFunction")
            self.functions[d] = fn

    @staticmethod
    def _check(r: int, what: str) -> None:
        if r != 0:
            raise RuntimeError(f"{what} returned hipError {r}")

    def launch(self, q, k, v, out, cu_seqlens, max_len: int) -> None:
        total, heads, d = q.shape
        args = [ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(cu_seqlens.data_ptr()),
                ctypes.c_int(q.stride(0)), ctypes.c_int(k.stride(0)), ctypes.c_int(v.stride(0)),
                ctypes.c_int(out.stride(0)), ctypes.c_float(_LOG2E / math.sqrt(d))]
        params = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
        grid = ((max_len + _BQ - 1) // _BQ, heads, cu_seqlens.numel() - 1)
        self._check(self.hip.hipModuleLaunchKernel(self.functions[d], grid[0], grid[1], grid[2],
                                                   _BQ, 1, 1, 0, stream, params, None),
                    "hipModuleLaunchKernel")


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
                log.info("hip varlen attention: built for %s on cuda:%d", kern.arch, index)
            except Exception as e:  # noqa: BLE001 -- any failure means: not served here
                log.warning("hip varlen attention unavailable on cuda:%d: %s", index, e)
                kern = None
            _kernels[index] = kern
        return _kernels[index]


def _self_test(kern: _DeviceKernels, device: torch.device) -> None:
    """The kernel against F.scaled_dot_product_attention (fp32) on a fixed
    problem per instantiated head size: segment lengths that end inside,
    on and past a tile, and a segment longer than a query block."""
    gen = torch.Generator().manual_seed(0)
    lens = [1, _BK - 1, _BK, _BQ + 3, 2 * _BQ + _BK + 5]
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
