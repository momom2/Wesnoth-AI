"""HIP kernels for the packed trunk's fp16 inference on AMD GPUs that
PyTorch's fused attention does not serve (2026-10-28: the local RX 6600,
gfx1032, ROCm on Windows, where flash and memory-efficient SDPA both
report "No available kernel" and the math path took 65% of a forward;
training/metrics/local_gpu_20261028/README.md): the variable-length
self-attention (WesnothModel._run_hip_fp16_trunk, packed_trunk), and the
loader the other HIP modules share. The row-wise kernels of the fused
loop (the residual add and layer norm, the bias and ReLU) are in
wesnoth_ai/hip_elementwise.py.

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
are left unwritten, as flash leaves them. A variant also writes each
row's log-sum-exp, which the backward (wesnoth_ai/hip_attention_grad.py)
recomputes the probabilities from; this module has no autograd. Another
(inference only) adds the in-projection's bias as it loads q, k and v,
so the projection's GEMM runs without one (ROCm's addmm adds a bias in a
pass of its own, training/metrics/hip_bias_fold_20261029/README.md): an
fp16 add, correctly rounded as torch's is, of the bias [3, heads,
head_dim]. The kernels without it compile to the instructions they had
before it (that record's same_code.py).

Build: hiprtc (shipped with the ROCm runtime) compiles the source for
the device's architecture on first use, and the HIP module API loads
it; both are reached through ctypes, so no host compiler is needed (the
local machine has none). The launch goes on torch's current stream. On
first use per device the kernels are checked against
F.scaled_dot_product_attention in float32 on fixed problems, with the
caller's autocast off (the first use can come inside an fp16 autocast,
which no_grad leaves on); a device where the build or a check fails is
reported once in the log and served by the callers' fallbacks (the
`*_applies` functions are False). The bias variant has its own check,
on its first use, so its failure leaves the others serving.
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
_LOG2E = 1.4426950408889634
_SELF_TEST_TOL = 2e-2       # fp16 inputs and output against an fp32 reference

_SOURCE = r'''
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));

// LSE: also write each row's log-sum-exp of its scaled scores (natural
// log, float32 [heads, ls]) for the backward (hip_attention_grad.py).
// BIAS: add bias [3, heads, D] (q's, k's, v's; fp16) to each q, k and v
// row as it is loaded, an fp16 add (correctly rounded). A thread loads
// the same 8-wide chunk of every key and value row it loads (NT a
// multiple of CH), so it holds that chunk of k's and v's bias.
template <int NT, int QPT, int BK, int D, bool LSE, bool BIAS>
__device__ __forceinline__ void attn_fwd(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k, const _Float16* __restrict__ v,
    _Float16* __restrict__ o, const int* __restrict__ cu,
    int sq, int sk, int sv, int so, float scale_log2, float* __restrict__ lse, int ls,
    const _Float16* __restrict__ bias)
{
    constexpr int C = D / 2, CH = D / 8, BQ = NT * QPT;
    static_assert(NT % CH == 0, "a thread's key and value chunk must not vary over the loads");
    __shared__ __attribute__((aligned(16))) h2 Ks[BK][C];
    __shared__ __attribute__((aligned(16))) h2 Vp[BK / 2][D];
    const int seg = blockIdx.z, hd = blockIdx.y, tid = threadIdx.x;
    const int start = cu[seg], n = cu[seg + 1] - start;
    const int q0 = blockIdx.x * BQ;
    if (q0 >= n) return;
    h8 kbias = (h8)(0), vbias = (h8)(0);
    if (BIAS) {
        kbias = ((const h8*)(bias + (gridDim.y + hd) * D))[tid % CH];
        vbias = ((const h8*)(bias + (2 * gridDim.y + hd) * D))[tid % CH];
    }
    h2 qr[QPT][C];
    bool active[QPT];
    #pragma unroll
    for (int r = 0; r < QPT; ++r) {
        const int qi = q0 + r * NT + tid;
        active[r] = qi < n;
        if (active[r]) {
            const h8* src = (const h8*)(q + (long long)(start + qi) * sq + hd * D);
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                h8 t = src[c];
                if (BIAS) t += ((const h8*)(bias + hd * D))[c];
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
            if (kb + j < n) {
                t = ((const h8*)(k + (long long)(start + kb + j) * sk + hd * D))[c];
                if (BIAS) t += kbias;
            }
            *(h8*)&Ks[j][c * 4] = t;
        }
        for (int i = tid; i < (BK / 2) * CH; i += NT) {
            const int jj = i / CH, c = i % CH;
            const int j0 = kb + 2 * jj;
            h8 a = (h8)(0), b = (h8)(0);
            if (j0 < n) {
                a = ((const h8*)(v + (long long)(start + j0) * sv + hd * D))[c];
                if (BIAS) a += vbias;
            }
            if (j0 + 1 < n) {
                b = ((const h8*)(v + (long long)(start + j0 + 1) * sv + hd * D))[c];
                if (BIAS) b += vbias;
            }
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
            h8* dst = (h8*)(o + (long long)(start + q0 + r * NT + tid) * so + hd * D);
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                h8 t;
                #pragma unroll
                for (int e = 0; e < 8; ++e) t[e] = (_Float16)(acc[r][c * 8 + e] * inv);
                dst[c] = t;
            }
            // m and log2(l) are base 2 of the scaled scores: times ln 2, the natural log.
            if (LSE) lse[(long long)hd * ls + start + q0 + r * NT + tid] = (m[r] + log2f(l[r])) * 0.6931471805599453f;
        }
    }
}

#define ATTN_INSTANCE(D, QPT) \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
varlen_attn_fwd_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, _Float16* o, \
                         const int* cu, int sq, int sk, int sv, int so, float scale_log2) \
{ attn_fwd<NT_VALUE, QPT, BK_VALUE, D, false, false>(q, k, v, o, cu, sq, sk, sv, so, scale_log2, \
                                                     nullptr, 0, nullptr); } \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
varlen_attn_fwd_lse_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, _Float16* o, \
                             const int* cu, int sq, int sk, int sv, int so, float scale_log2, \
                             float* lse, int ls) \
{ attn_fwd<NT_VALUE, QPT, BK_VALUE, D, true, false>(q, k, v, o, cu, sq, sk, sv, so, scale_log2, \
                                                    lse, ls, nullptr); } \
extern "C" __global__ void __launch_bounds__(NT_VALUE) \
varlen_attn_fwd_bias_f16_d##D(const _Float16* q, const _Float16* k, const _Float16* v, _Float16* o, \
                              const int* cu, int sq, int sk, int sv, int so, float scale_log2, \
                              const _Float16* bias) \
{ attn_fwd<NT_VALUE, QPT, BK_VALUE, D, false, true>(q, k, v, o, cu, sq, sk, sv, so, scale_log2, \
                                                    nullptr, 0, bias); }
ATTN_INSTANCE(32, QPT32_VALUE)
ATTN_INSTANCE(64, QPT64_VALUE)
'''


def _source() -> str:
    return (_SOURCE.replace("NT_VALUE", str(_NT)).replace("BK_VALUE", str(_BK))
            .replace("QPT32_VALUE", str(_QPT[32])).replace("QPT64_VALUE", str(_QPT[64])))


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


def compile_source(arch: str, source: str, name: str = "wesnoth_kernels.hip") -> bytes:
    """`source` compiled by hiprtc for `arch`: the code object's bytes."""
    rtc = _rocm_library(_RTC_NAMES)
    if rtc is None:
        raise RuntimeError("hiprtc not found beside torch")
    prog = ctypes.c_void_p()
    _check_rtc(rtc.hiprtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(),
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


def check_hip(r: int, what: str) -> None:
    if r != 0:
        raise RuntimeError(f"{what} returned hipError {r}")


def hip_runtime() -> ctypes.CDLL:
    hip = _rocm_library(_HIP_NAMES)
    if hip is None:
        raise RuntimeError("the HIP runtime library was not found")
    return hip


def device_arch(device: torch.device) -> str:
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


def load_module(hip: ctypes.CDLL, code: bytes, device: torch.device) -> ctypes.c_void_p:
    """A compiled code object loaded on `device`."""
    module = ctypes.c_void_p()
    with torch.cuda.device(device):
        check_hip(hip.hipModuleLoadData(ctypes.byref(module), code), "hipModuleLoadData")
    return module


def module_function(hip: ctypes.CDLL, module: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
    fn = ctypes.c_void_p()
    check_hip(hip.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()), "hipModuleGetFunction")
    return fn


def launch(hip: ctypes.CDLL, fn, grid, threads: int, args, device: torch.device) -> None:
    """One launch of `fn` on torch's current stream; `args` are tensors
    (their data pointers), ints and floats, in the kernel's order."""
    held = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor)
            else ctypes.c_float(a) if isinstance(a, float) else ctypes.c_int(a) for a in args]
    params = (ctypes.c_void_p * len(held))(
        *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in held])
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    check_hip(hip.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], threads, 1, 1, 0,
                                        stream, params, None),
              "hipModuleLaunchKernel")


# hipFunction_attribute (hip/driver_types.h, an enum from 0): the
# per-thread scratch ("local") bytes, nonzero where the compiler spilled
# registers, and the registers a thread uses.
_ATTR_LOCAL_SIZE_BYTES, _ATTR_NUM_REGS = 3, 4


def function_resources(hip: ctypes.CDLL, fn) -> Dict[str, int]:
    """A compiled kernel's registers and scratch bytes per thread."""
    out = {}
    for key, attr in (("registers", _ATTR_NUM_REGS), ("scratch_bytes", _ATTR_LOCAL_SIZE_BYTES)):
        value = ctypes.c_int()
        check_hip(hip.hipFuncGetAttribute(ctypes.byref(value), attr, fn), "hipFuncGetAttribute")
        out[key] = int(value.value)
    return out


class _DeviceKernels:
    """The compiled module on one device and its entry points."""

    def __init__(self, device: torch.device):
        self.hip = hip_runtime()
        self.arch = device_arch(device)
        self.module = load_module(self.hip, compile_source(self.arch, _source()), device)
        self.functions = {d: self._function(f"varlen_attn_fwd_f16_d{d}") for d in HEAD_DIMS}
        self.functions_lse = {d: self._function(f"varlen_attn_fwd_lse_f16_d{d}") for d in HEAD_DIMS}
        self.functions_bias = {d: self._function(f"varlen_attn_fwd_bias_f16_d{d}") for d in HEAD_DIMS}

    def _function(self, name: str) -> ctypes.c_void_p:
        return module_function(self.hip, self.module, name)

    def _launch(self, fn, grid, threads: int, args, device: torch.device) -> None:
        launch(self.hip, fn, grid, threads, args, device)

    def launch(self, q, k, v, out, cu_seqlens, max_len: int, lse: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None) -> None:
        """The attention; with `lse` (float32 [heads, rows], rows at least
        the offsets' last), the variant that also writes each row's
        log-sum-exp, the arithmetic otherwise the same; with `bias` (fp16
        [3, heads, head_dim], contiguous), the variant that adds it to q,
        k and v as it loads them. Not both."""
        total, heads, d = q.shape
        bq = _NT * _QPT[d]
        grid = ((max_len + bq - 1) // bq, heads, cu_seqlens.numel() - 1)
        args = [q, k, v, out, cu_seqlens, q.stride(0), k.stride(0), v.stride(0), out.stride(0),
                _LOG2E / math.sqrt(d)]
        if lse is not None and bias is not None:
            raise ValueError("the attention has no variant with both a bias and the log-sum-exp")
        if lse is not None:
            self._launch(self.functions_lse[d], grid, _NT, args + [lse, lse.stride(0)], q.device)
        elif bias is not None:
            self._launch(self.functions_bias[d], grid, _NT, args + [bias], q.device)
        else:
            self._launch(self.functions[d], grid, _NT, args, q.device)


_lock = threading.Lock()
_kernels: Dict[int, Optional[_DeviceKernels]] = {}
_bias_checked: Dict[int, bool] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _kernels_for(device: torch.device) -> Optional[_DeviceKernels]:
    """The device's kernels, built and self-tested on first use; None
    (reported once) where either fails."""
    index = _device_index(device)
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


def _bias_served(device: torch.device) -> bool:
    """Whether the bias variant passed its self-test on `device` (run on
    the first ask, reported once on failure); the device's kernels built."""
    kern = _kernels_for(device)
    if kern is None:
        return False
    index = _device_index(device)
    with _lock:
        if index not in _bias_checked:
            try:
                _self_test(kern, torch.device("cuda", index), bias=True)
                _bias_checked[index] = True
            except Exception as e:  # noqa: BLE001 -- any failure means: not served here
                log.warning("hip attention's bias variant unavailable on cuda:%d: %s", index, e)
                _bias_checked[index] = False
        return _bias_checked[index]


def reference_attention(q, k, v, cu_host):
    """F.scaled_dot_product_attention per segment in float32, differentiable:
    the reference the kernels are checked against. Any autocast on the
    inputs' device is off here: under one, SDPA runs at its reduced
    precision, no_grad or not, and the kernels' first use can come inside
    the gradient pass's fp16 autocast or the server's."""
    with torch.autocast(q.device.type, enabled=False):
        outs = []
        for lo, hi in zip(cu_host[:-1], cu_host[1:]):
            seg = [t[lo:hi].float().transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
            outs.append(F.scaled_dot_product_attention(*seg)[0].transpose(0, 1))
        return torch.cat(outs)


def _self_test(kern: _DeviceKernels, device: torch.device, bias: bool = False) -> None:
    """The attention against F.scaled_dot_product_attention (fp32) on a
    fixed problem per instantiated head size (segment lengths that end
    inside, on and past a tile, and a segment longer than a query block);
    with `bias`, its bias variant against SDPA of the biased q, k, v. Run
    with the caller's autocast off (`reference_attention`), so the
    references are the fp32 they are checked as."""
    gen = torch.Generator().manual_seed(int(bias))
    bq = _NT * _QPT[32]
    lens = [1, _BK - 1, _BK, _NT + 1, bq - 1, bq + 3, 2 * bq + _BK + 5]
    cu_host = [0]
    for n in lens:
        cu_host.append(cu_host[-1] + n)
    with torch.autocast(device.type, enabled=False):
        for d in HEAD_DIMS:
            qkv = torch.randn(cu_host[-1], 3, 3, d, generator=gen).to(device, torch.float16)
            b = torch.randn(3, 3, d, generator=gen).to(device, torch.float16) if bias else None
            q, k, v = qkv.unbind(1)
            out = torch.empty(q.shape, dtype=q.dtype, device=device)
            kern.launch(q, k, v, out, torch.tensor(cu_host, dtype=torch.int32, device=device), max(lens),
                        bias=b)
            ins = (q, k, v) if b is None else [t.float() + c.float() for t, c in zip((q, k, v), b)]
            worst = (out.float() - reference_attention(*ins, cu_host)).abs().max().item()
            if not worst <= _SELF_TEST_TOL:
                raise RuntimeError(f"self-test{' of the bias variant' if bias else ''} at head_dim {d}: "
                                   f"max abs error {worst} > {_SELF_TEST_TOL}")


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


def hip_varlen_bias_applies(device: torch.device, head_dim: int) -> bool:
    """Whether hip_varlen_attention takes a `bias` here: the kernel serves
    fp16 at this head size, and the bias variant passed its self-test
    (run here on first ask)."""
    return hip_varlen_applies(device, torch.float16, head_dim) and _bias_served(device)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """A contiguous t whose data starts on a 16-byte boundary, for the
    kernels' 16-byte loads and stores: t itself when it is one, else a
    copy (contiguous() returns a contiguous view unchanged, however its
    start is aligned)."""
    c = t.contiguous()
    return c if c.data_ptr() % 16 == 0 else c.clone()


def _kernel_view(t: torch.Tensor) -> torch.Tensor:
    """t, or an aligned contiguous copy where its layout is not the
    kernel's: unit-stride head dim, heads adjacent, 16-byte rows and token
    strides."""
    d = t.shape[-1]
    if (t.stride(2) == 1 and t.stride(1) == d and t.stride(0) % 8 == 0
            and t.data_ptr() % 16 == 0):
        return t
    return _aligned(t)


def _check_attention_args(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    """The kernels read raw pointers: refuse what they would misread."""
    if not (q.dtype == k.dtype == v.dtype == torch.float16):
        raise TypeError(f"hip varlen attention takes fp16 q, k, v, not {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dim() != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must share a [total, heads, head_dim] shape: {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"head_dim {q.shape[-1]} is not one the source instantiates {HEAD_DIMS}")
    if not (q.device == k.device == v.device == cu_seqlens.device):
        raise ValueError("q, k, v and cu_seqlens must be on one device")


def _check_bias(bias: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The in-projection's bias as the bias variant reads it: fp16 [3,
    heads, head_dim] (any shape of that many elements), aligned."""
    _, heads, d = q.shape
    if bias.dtype != torch.float16 or bias.numel() != 3 * heads * d or bias.device != q.device:
        raise TypeError(f"hip_varlen_attention's bias is fp16 with 3 x {heads} x {d} elements on q's device: "
                        f"{bias.dtype} {tuple(bias.shape)} on {bias.device}")
    return _aligned(bias.reshape(3, heads, d))


def hip_varlen_attention_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             cu_seqlens: torch.Tensor, max_len: int
                             ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
    """hip_varlen_attention's output (the same arithmetic), each row's
    log-sum-exp of its scaled scores, ln sum_j exp(q.k_j / sqrt(head_dim))
    over its segment's keys (float32 [heads, total]; rows past the
    offsets' last unwritten), and the q, k, v the kernel read (kernel
    views). The backward's forward (wesnoth_ai/hip_attention_grad.py);
    no autograd here. The caller has checked `hip_varlen_applies`."""
    _check_attention_args(q, k, v, cu_seqlens)
    kern = _kernels_for(q.device)
    q, k, v = (_kernel_view(t) for t in (q, k, v))
    total, heads, _ = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((heads, total), dtype=torch.float32, device=q.device)
    cu = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    kern.launch(q, k, v, out, cu.contiguous(), int(max_len), lse=lse)
    return out, lse, (q, k, v)


def hip_varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         cu_seqlens: torch.Tensor, max_len: int,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Self-attention within each segment of the packed [total, heads,
    head_dim] q, k, v (module docstring); with `bias` (fp16, [3, heads,
    head_dim]: q's, k's, v's), of q + bias[0], k + bias[1], v + bias[2],
    each sum rounded to fp16. The caller has checked `hip_varlen_applies`,
    and `hip_varlen_bias_applies` for a bias."""
    if any(t is not None and t.requires_grad for t in (q, k, v, bias)):
        if torch.is_grad_enabled():
            raise RuntimeError("hip_varlen_attention has no backward")
    _check_attention_args(q, k, v, cu_seqlens)
    kern = _kernels_for(q.device)
    q, k, v = (_kernel_view(t) for t in (q, k, v))
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    cu = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    kern.launch(q, k, v, out, cu.contiguous(), int(max_len),
                bias=None if bias is None else _check_bias(bias, q))
    return out
