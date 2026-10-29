"""The row-wise HIP kernels of the packed trunk's fp16 inference loop on
ROCm (WesnothModel._run_hip_fp16_trunk), built and launched by the loader
of wesnoth_ai/hip_kernels.py, which holds the attention.

The add and norm reads the float32 residual rows and the fp16 sublayer
output, and writes the float32 layer norm of their sum (the autocast's
residual stream) and its fp16 rounding (the next linear's input): an
add, a layer norm and a cast in one pass, one wave32 a row. A variant
also adds the sublayer's bias (fp16 [d]) to its output in float32 before
the residual add, so the sublayer's GEMM runs without one: ROCm's addmm
adds a bias in a pass of its own
(training/metrics/hip_bias_fold_20261029/README.md). The kernels without
the bias compile to the instructions they had in hip_kernels.py before
it (that record's same_code.py).

The bias and ReLU writes relu(y + bias) over the fp16 rows y in place:
the first feed-forward linear's bias and activation in one pass, the add
in fp16 (correctly rounded, as torch's fp16 add), NaN kept as F.relu
keeps it.

On first use per device the kernels are checked against F.layer_norm and
F.relu in float32 with the caller's autocast off (hip_kernels' module
docstring); a device where the build or a check fails is reported once
in the log, and `hip_add_layer_norm_applies` is False there.
"""
from __future__ import annotations

import ctypes
import logging
import threading
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from wesnoth_ai.hip_kernels import (
    compile_source, device_arch, hip_runtime, launch, load_module, module_function,
)
from wesnoth_ai.hip_kernels import _aligned

log = logging.getLogger(__name__)

# The layer norm: LN_THREADS threads a block, a wave32 a row; the model
# widths it serves are 128 times LN_NI (own2's 384: 3).
_LN_THREADS = 128
_LN_NI = (1, 2, 3, 4, 6)
_RELU_THREADS = 256         # threads a block, one 8-wide fp16 chunk a thread
_SELF_TEST_TOL = 2e-2       # an fp16 output against an fp32 reference

_SOURCE = r'''
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));

// The residual add and the layer norm after it: o32 = LN(x + float(y)),
// with BIAS LN(x + (float(y) + float(yb))), o16 = half(o32), the next
// linear's input. One wave32 a row, four elements a lane a step (d = 128
// NI); the mean and the variance by two passes over the row held in
// registers.
template <int NI, bool BIAS>
__device__ __forceinline__ void add_ln(const float* __restrict__ x, const _Float16* __restrict__ y,
                                       const _Float16* __restrict__ yb,
                                       const float* __restrict__ w, const float* __restrict__ b,
                                       float* __restrict__ o32, _Float16* __restrict__ o16,
                                       int rows, float eps)
{
    constexpr int D = 128 * NI;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;        // a whole wave: no lane of it reaches the shuffles
    const long long base = (long long)row * D;
    f4 v[NI];
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int c = i * 128 + lane * 4;
        if (BIAS) {
            const f4 yv = __builtin_convertvector(*(const h4*)(y + base + c), f4)
                          + __builtin_convertvector(*(const h4*)(yb + c), f4);
            v[i] = *(const f4*)(x + base + c) + yv;
        } else {   // the expression before the bias variant, so the same code
            v[i] = *(const f4*)(x + base + c) + __builtin_convertvector(*(const h4*)(y + base + c), f4);
        }
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
{ add_ln<NI, false>(x, y, nullptr, w, b, o32, o16, rows, eps); } \
extern "C" __global__ void __launch_bounds__(LN_THREADS_VALUE) \
add_ln_bias_d##NI(const float* x, const _Float16* y, const _Float16* yb, const float* w, \
                  const float* b, float* o32, _Float16* o16, int rows, float eps) \
{ add_ln<NI, true>(x, y, yb, w, b, o32, o16, rows, eps); }
LN_INSTANCE(1)
LN_INSTANCE(2)
LN_INSTANCE(3)
LN_INSTANCE(4)
LN_INSTANCE(6)

// y = relu(y + bias) in place over the contiguous fp16 [rows, n], one
// 8-wide chunk a thread (chunks = rows n / 8, cpr = n / 8 a row). x < 0 ?
// 0 : x keeps a NaN, as F.relu does.
extern "C" __global__ void __launch_bounds__(RELU_THREADS_VALUE)
bias_relu_f16(_Float16* __restrict__ y, const _Float16* __restrict__ bias, int chunks, int cpr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunks) return;
    h8 t = ((h8*)y)[i] + ((const h8*)bias)[i % cpr];
    #pragma unroll
    for (int e = 0; e < 8; ++e) t[e] = t[e] < (_Float16)0 ? (_Float16)0 : t[e];
    ((h8*)y)[i] = t;
}
'''


def _source() -> str:
    return (_SOURCE.replace("LN_THREADS_VALUE", str(_LN_THREADS))
            .replace("RELU_THREADS_VALUE", str(_RELU_THREADS)))


class _DeviceKernels:
    """The compiled module on one device and its entry points."""

    def __init__(self, device: torch.device):
        self.hip = hip_runtime()
        self.arch = device_arch(device)
        self.module = load_module(self.hip, compile_source(self.arch, _source(), "wesnoth_elementwise.hip"),
                                  device)
        self.add_ln = {128 * ni: self._function(f"add_ln_d{ni}") for ni in _LN_NI}
        self.add_ln_bias = {128 * ni: self._function(f"add_ln_bias_d{ni}") for ni in _LN_NI}
        self.bias_relu = self._function("bias_relu_f16")

    def _function(self, name: str) -> ctypes.c_void_p:
        return module_function(self.hip, self.module, name)

    def launch_add_ln(self, x, y, w, b, o32, o16, eps: float, yb: Optional[torch.Tensor] = None) -> None:
        """The add and norm; with `yb` (fp16 [d]), the variant that adds it
        to y first."""
        rows, d = x.shape
        per_block = _LN_THREADS // 32
        grid = ((rows + per_block - 1) // per_block, 1, 1)
        if yb is None:
            launch(self.hip, self.add_ln[d], grid, _LN_THREADS, [x, y, w, b, o32, o16, rows, float(eps)],
                   x.device)
        else:
            launch(self.hip, self.add_ln_bias[d], grid, _LN_THREADS,
                   [x, y, yb, w, b, o32, o16, rows, float(eps)], x.device)

    def launch_bias_relu(self, y, bias) -> None:
        chunks = y.numel() // 8
        launch(self.hip, self.bias_relu, ((chunks + _RELU_THREADS - 1) // _RELU_THREADS, 1, 1), _RELU_THREADS,
               [y, bias, chunks, y.shape[-1] // 8], y.device)


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
                log.info("hip elementwise kernels: built for %s on cuda:%d", kern.arch, index)
            except Exception as e:  # noqa: BLE001 -- any failure means: not served here
                log.warning("hip elementwise kernels unavailable on cuda:%d: %s", index, e)
                kern = None
            _kernels[index] = kern
        return _kernels[index]


def _self_test(kern: _DeviceKernels, device: torch.device) -> None:
    """The add and norm, without and with the bias, against F.layer_norm per
    instantiated width, and the bias and ReLU against F.relu of the fp32
    sum rounded to fp16 (the same value: the kernel's fp16 add is
    correctly rounded), on fixed problems, with the caller's autocast off."""
    gen = torch.Generator().manual_seed(0)
    with torch.autocast(device.type, enabled=False):
        for d in sorted(kern.add_ln):
            x = (torch.randn(37, d, generator=gen) * 3).to(device)
            y = torch.randn(37, d, generator=gen).to(device, torch.float16)
            w, b = (torch.rand(d, generator=gen) + 0.5).to(device), torch.randn(d, generator=gen).to(device)
            yb = torch.randn(d, generator=gen).to(device, torch.float16)
            for bias in (None, yb):
                o32, o16 = torch.empty_like(x), torch.empty_like(y)
                kern.launch_add_ln(x, y, w, b, o32, o16, 1e-5, yb=bias)
                sub = y.float() if bias is None else y.float() + bias.float()
                ref = F.layer_norm(x + sub, (d,), w, b, 1e-5)
                worst = (o32 - ref).abs().max().item()
                if not worst <= 1e-4 or not (o16.float() - ref).abs().max().item() <= _SELF_TEST_TOL:
                    raise RuntimeError(f"self-test of the add and layer norm at d {d} "
                                       f"({'with' if bias is not None else 'without'} a bias): "
                                       f"max abs error {worst}")
        y = (torch.randn(29, 24, generator=gen) * 2).to(device, torch.float16)
        bias = torch.randn(24, generator=gen).to(device, torch.float16)
        ref = F.relu(y.float() + bias.float()).half()
        kern.launch_bias_relu(y, bias)
        if not torch.equal(y, ref):
            raise RuntimeError(f"self-test of the bias and ReLU: max abs error "
                               f"{(y.float() - ref.float()).abs().max().item()}")


# ---------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------

def hip_add_layer_norm_applies(device: torch.device, d: int) -> bool:
    """Whether hip_add_layer_norm serves rows of width `d` on `device`, and
    hip_bias_relu_ any rows there."""
    if torch.version.hip is None or device.type != "cuda" or d not in tuple(128 * ni for ni in _LN_NI):
        return False
    return _kernels_for(device) is not None


def hip_add_layer_norm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       eps: float, y_bias: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """F.layer_norm(x + y) for the float32 residual rows x [rows, d] and the
    fp16 sublayer output y, in float32 (as an autocast's layer norm
    computes it, up to the order of the sums), and its fp16 rounding, the
    next linear's input, in one pass. weight and bias float32 [d]. With
    `y_bias` (fp16 [d], the sublayer's bias), F.layer_norm(x + (y +
    y_bias)), the sum in float32. The caller has checked
    `hip_add_layer_norm_applies`; forward only."""
    if x.dtype != torch.float32 or y.dtype != torch.float16 or x.dim() != 2 or y.shape != x.shape:
        raise TypeError(f"hip_add_layer_norm takes float32 x and fp16 y of one [rows, d] shape: "
                        f"{x.dtype} {tuple(x.shape)}, {y.dtype} {tuple(y.shape)}")
    d = x.shape[1]
    if weight.dtype != torch.float32 or bias.dtype != torch.float32 or weight.shape != (d,) or bias.shape != (d,):
        raise TypeError(f"hip_add_layer_norm takes float32 weight and bias of [{d}]")
    if y_bias is not None and (y_bias.dtype != torch.float16 or y_bias.shape != (d,)):
        raise TypeError(f"hip_add_layer_norm takes an fp16 y_bias of [{d}]: {y_bias.dtype} {tuple(y_bias.shape)}")
    if not all(t.device == x.device for t in (y, weight, bias, y_bias) if t is not None):
        raise ValueError("hip_add_layer_norm's tensors must be on one device")
    kern = _kernels_for(x.device)
    x, y = _aligned(x), _aligned(y)
    o32 = torch.empty_like(x)
    o16 = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    kern.launch_add_ln(x, y, _aligned(weight), _aligned(bias), o32, o16, eps,
                       yb=None if y_bias is None else _aligned(y_bias))
    return o32, o16


def hip_bias_relu_(y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """relu(y + bias) for the fp16 rows y [..., n] and the fp16 bias [n],
    written into y when y is contiguous and 16-byte aligned (a fresh
    F.linear output is), else into an aligned copy; returns the tensor
    written. n a multiple of 8. The caller has checked
    `hip_add_layer_norm_applies`; forward only."""
    n = y.shape[-1]
    if y.dtype != torch.float16 or bias.dtype != torch.float16 or bias.shape != (n,) or n % 8:
        raise TypeError(f"hip_bias_relu_ takes fp16 y [..., n] and bias [n], n a multiple of 8: "
                        f"{y.dtype} {tuple(y.shape)}, {bias.dtype} {tuple(bias.shape)}")
    if bias.device != y.device:
        raise ValueError("hip_bias_relu_'s tensors must be on one device")
    if y.numel() // 8 >= 2 ** 31:
        raise ValueError(f"hip_bias_relu_ indexes chunks in int32: {y.numel()} elements")
    if y.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("hip_bias_relu_ has no backward")
    kern = _kernels_for(y.device)
    y = _aligned(y)
    if y.numel():
        kern.launch_bias_relu(y, _aligned(bias))
    return y
