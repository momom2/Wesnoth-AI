"""The int8 GEMM (wesnoth_ai/hip_int8.py) against F.linear in fp16 on own2's
four linear shapes at M tokens (20,693: a 48-leaf batch), interleaved call
by call in one process (the order reversed every other round), each call
timed by CUDA events; plus the quantizers, the fp16 ReLU and the two
add-and-norm kernels. Random operands (weights 0.05 N(0, 1)): the rates do
not depend on the values. Writes JSON: per call the median and the
fastest of the rounds, per shape the rates and the ratios (of the medians,
of the fastest calls, and the median of the per-round ratios).

    set HIP_VISIBLE_DEVICES=1
    python bench_int8_gemm.py --root <repo> --out gemm_rates.json [--m 20693] [--rounds 60]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--m", type=int, default=20693)
ap.add_argument("--rounds", type=int, default=60)
args = ap.parse_args()
sys.path.insert(0, str(args.root))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from wesnoth_ai import hip_int8  # noqa: E402
from wesnoth_ai.hip_kernels import hip_add_layer_norm  # noqa: E402

M, ROUNDS = args.m, args.rounds
dev = torch.device("cuda")
assert hip_int8.hip_int8_applies(dev)
g = torch.Generator(device=dev).manual_seed(0)
SHAPES = {"qkv": (384, 1152), "out": (384, 384), "ff1": (384, 1536), "ff2": (1536, 384)}
data = {}
for name, (K, N) in SHAPES.items():
    x16 = torch.randn(M, K, generator=g, device=dev).half()
    w16 = (torch.randn(N, K, generator=g, device=dev) * 0.05).half()
    b16 = (torch.randn(N, generator=g, device=dev) * 0.1).half()
    xq, xs = hip_int8.hip_quantize_rows(x16)
    wq, ws = hip_int8.quantize_weight(w16.float())
    data[name] = (x16, w16, b16, xq, xs, wq.to(dev), ws.to(dev), b16.float())
r32 = torch.randn(M, 384, generator=g, device=dev)
y16 = torch.randn(M, 384, generator=g, device=dev).half()
lw = torch.rand(384, generator=g, device=dev) + 0.5
lb = torch.randn(384, generator=g, device=dev)
f16 = torch.randn(M, 1536, generator=g, device=dev).half()


def timed(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


calls = {}
for name in SHAPES:
    x16, w16, b16, xq, xs, wq, ws, b32 = data[name]
    relu = name == "ff1"
    calls[f"{name}_fp16"] = (lambda x16=x16, w16=w16, b16=b16: F.linear(x16, w16, b16))
    calls[f"{name}_int8"] = (lambda xq=xq, xs=xs, wq=wq, ws=ws, b32=b32, relu=relu:
                             hip_int8.hip_int8_linear(xq, xs, wq, ws, b32, relu=relu))
calls["relu_fp16_1536"] = lambda: F.relu(f16)
calls["quant_384"] = lambda: hip_int8.hip_quantize_rows(data["out"][0])
calls["quant_1536"] = lambda: hip_int8.hip_quantize_rows(f16)
calls["add_ln_fp16"] = lambda: hip_add_layer_norm(r32, y16, lw, lb, 1e-5)
calls["add_ln_q"] = lambda: hip_int8.hip_add_layer_norm_q(r32, y16, lw, lb, 1e-5)
for fn in calls.values():          # warm-up
    fn()
    fn()
torch.cuda.synchronize()
times = {k: [] for k in calls}
t0 = time.perf_counter()
names = list(calls)
for r in range(ROUNDS):
    order = names if r % 2 == 0 else names[::-1]
    for k in order:
        times[k].append(timed(calls[k]))
wall = time.perf_counter() - t0
med = {k: statistics.median(v) for k, v in times.items()}
mins = {k: min(v) for k, v in times.items()}
out = {"M": M, "rounds": ROUNDS, "wall_s": round(wall, 2),
       "median_ms": {k: round(v, 4) for k, v in med.items()},
       "min_ms": {k: round(v, 4) for k, v in mins.items()}}
rates = {}
for name, (K, N) in SHAPES.items():
    flops = 2.0 * M * K * N
    paired = [a / b for a, b in zip(times[f"{name}_fp16"], times[f"{name}_int8"])]
    rates[name] = {"fp16_tflops_min": round(flops / mins[f"{name}_fp16"] / 1e9, 2),
                   "int8_tops_min": round(flops / mins[f"{name}_int8"] / 1e9, 2),
                   "ratio_of_medians": round(med[f"{name}_fp16"] / med[f"{name}_int8"], 3),
                   "ratio_of_mins": round(mins[f"{name}_fp16"] / mins[f"{name}_int8"], 3),
                   "median_paired_ratio": round(statistics.median(paired), 3)}
out["rates"] = rates
for label, src in (("median", med), ("min", mins)):
    sum16 = sum(src[f"{n}_fp16"] for n in SHAPES)
    sum8 = sum(src[f"{n}_int8"] for n in SHAPES)
    out[f"sum_ms_{label}"] = {"fp16": round(sum16, 3), "int8": round(sum8, 3), "ratio": round(sum16 / sum8, 3)}
kern = hip_int8._kernels_for(dev)
out.update({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "arch": kern.arch,
            "resources": kern.resources(),
            "tiling": {"threads": hip_int8._NT, "BM": hip_int8._BM, "BN": hip_int8._BN, "BK": hip_int8._BK,
                       "TM": hip_int8._TM, "TN": hip_int8._TN}})
args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
