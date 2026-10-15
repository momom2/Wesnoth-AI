"""The inference server's device work, profiled in production: torch.profiler
over `batches` batches of a live match, starting after `after` (so the
batches are the match's steady state), the summary in the server's stats
under "profile" (tools/eval_inference_server.py --profile-batches).

The summary gives, per batch:
- the kernels' device time by class (attention, GEMM, layer norm, softmax,
  scan and sort, index and gather, cast and copy, reduction, other
  elementwise, memory copies) and by the server's named region that
  launched them (wesnoth_ai/profile_regions.py: the embedding, the order of
  the rows, the trunk, the gathers, the memory's write, the heads, the
  priors, the memory's gather and scatter; the innermost region wins);
- the heaviest kernels by name;
- the device's idle fraction over the profiled wall time;
- the batches' shapes (leaves, tokens, the sum of squared segment
  lengths, the padded widths), and from them the trunk's arithmetic
  (GEMM and attention FLOPs) against the classes' time: the achieved
  TFLOP/s of the GEMMs and of the attention.

Profiling slows the profiled batches; an arm that profiles is not a timing
arm.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger("server_profile")

# (class, substrings of the kernel name, lowercase), first match wins.
KERNEL_CLASSES = (
    ("attention", ("flash", "fmha", "attention")),
    ("gemm", ("gemm", "gemv", "cutlass", "xmma", "cublas", "splitk")),
    ("layer_norm", ("layer_norm", "layernorm")),
    ("softmax", ("softmax",)),
    ("scan_sort", ("scan", "cumsum", "searchsorted", "sort", "radix")),
    ("index", ("index", "gather", "scatter", "catarray", "cat_")),
    ("cast_copy", ("copy_kernel", "direct_copy")),
    ("reduce", ("reduce",)),
    ("elementwise", ("elementwise", "gelu", "relu", "tanh", "sigmoid")),
)


def kernel_class(name: str, category: str) -> str:
    low = name.lower()
    if category in ("gpu_memcpy", "gpu_memset") or "memcpy" in low or "memset" in low:
        return "memcpy"
    for cls, keys in KERNEL_CLASSES:
        if any(k in low for k in keys):
            return cls
    return "other"


def _launch_regions(events: List[dict]) -> Dict[int, str]:
    """The innermost user region around each launch, by the launch's
    correlation id: a sweep over each thread's region spans and launches."""
    by_tid: Dict[object, List[tuple]] = defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        ts, dur = float(e.get("ts", 0.0)), float(e.get("dur", 0.0))
        if cat == "user_annotation":
            by_tid[e.get("tid")].append((ts, 0, ts + dur, e.get("name", "")))
        elif cat == "cuda_runtime" and "correlation" in (e.get("args") or {}):
            by_tid[e.get("tid")].append((ts, 1, ts, int(e["args"]["correlation"])))
    out: Dict[int, str] = {}
    for items in by_tid.values():
        items.sort(key=lambda x: (x[0], x[1]))
        stack: List[tuple] = []                      # (end, name)
        for ts, kind, end, what in items:
            while stack and stack[-1][0] < ts:
                stack.pop()
            if kind == 0:
                stack.append((end, what))
            elif stack:
                out[what] = stack[-1][1]
    return out


def summarize_trace(trace: dict, wall_s: float, batches: int, shapes: Optional[List[Dict[str, int]]] = None,
                    dims: Optional[Dict[str, int]] = None, top: int = 25) -> Dict:
    """The profile's summary (module docstring) from a chrome trace."""
    events = trace.get("traceEvents", [])
    regions = _launch_regions(events)
    spans, by_class, by_region = [], Counter(), Counter()
    by_name: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
    for e in events:
        cat = e.get("cat", "")
        if e.get("ph") != "X" or cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        start, dur = float(e["ts"]), float(e.get("dur", 0.0))
        spans.append((start, start + dur))
        cls = kernel_class(e.get("name", ""), cat)
        by_class[cls] += dur
        corr = (e.get("args") or {}).get("correlation")
        by_region[regions.get(int(corr), "outside") if corr is not None else "outside"] += dur
        row = by_name[e.get("name", "")[:120]]
        row[0] += dur
        row[1] += 1
    busy, end = 0.0, -1.0
    for lo, hi in sorted(spans):
        if hi <= end:
            continue
        busy += hi - max(lo, end)
        end = hi
    total = sum(by_class.values())
    n = max(1, batches)

    def per_batch(counter):
        return {k: round(v / 1e3 / n, 3) for k, v in sorted(counter.items(), key=lambda kv: -kv[1])}
    out: Dict[str, object] = {
        "batches": batches, "wall_s": round(wall_s, 3), "kernels_per_batch": round(len(spans) / n, 1),
        "device_ms_per_batch": round(total / 1e3 / n, 3),
        "ms_per_batch": per_batch(by_class),
        "share": {k: round(v / total, 4) for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])} if total else {},
        "region_ms_per_batch": per_batch(by_region),
        "top_kernels": [{"name": k, "ms_per_batch": round(v[0] / 1e3 / n, 3), "per_batch": round(v[1] / n, 1),
                         "class": kernel_class(k, "kernel")}
                        for k, v in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]],
        "idle_fraction": round(max(0.0, 1.0 - busy / 1e6 / wall_s), 4) if wall_s > 0 else None,
    }
    if shapes:
        keys = ("leaves", "tokens", "tokens_sq", "a_max", "h_max", "k_max")
        mean = {k: round(sum(s[k] for s in shapes) / len(shapes), 1) for k in keys}
        out["batch_shape_mean"] = mean
        if dims:
            out["arithmetic"] = trunk_arithmetic(shapes, dims, by_class, n)
    return out


def trunk_arithmetic(shapes: List[Dict[str, int]], dims: Dict[str, int], by_class: Counter,
                     n: int) -> Dict[str, float]:
    """The trunk's FLOPs per batch (the layers' four projections and two
    feed-forward GEMMs over every token, the attention's two products over
    each segment's squared length) and the TFLOP/s the GEMM and attention
    classes achieve on them (the heads' GEMMs, small beside the trunk's,
    count in the GEMM class's time and not in its FLOPs)."""
    d, layers, ff = dims["d_model"], dims["num_layers"], dims["d_ff"]
    tokens = sum(s["tokens"] for s in shapes) / len(shapes)
    tokens_sq = sum(s["tokens_sq"] for s in shapes) / len(shapes)
    gemm = 2.0 * tokens * layers * (4 * d * d + 2 * d * ff)
    attention = 4.0 * layers * d * tokens_sq
    out = {"gemm_gflop_per_batch": round(gemm / 1e9, 2), "attention_gflop_per_batch": round(attention / 1e9, 2),
           "gflop_per_leaf": round((gemm + attention) / 1e9 / max(1.0, sum(s["leaves"] for s in shapes) / len(shapes)), 3)}
    for cls, flops in (("gemm", gemm), ("attention", attention)):
        ms = by_class.get(cls, 0.0) / 1e3 / n
        if ms > 0:
            out[f"{cls}_tflops"] = round(flops / (ms / 1e3) / 1e12, 1)
    return out


class BatchProfiler:
    """`infer_fn` with batches `after` to `after + batches` (counted from
    1) served under torch.profiler; `summary()` is the profile's summary,
    or the error that stopped it. One serve thread calls it."""

    def __init__(self, infer_fn: Callable, *, after: int, batches: int, workdir: Path, device,
                 dims: Optional[Dict[str, int]] = None):
        self._infer = infer_fn
        self.after, self.batches = int(after), int(batches)
        self.workdir, self.device, self.dims = Path(workdir), device, dims
        self._seen = 0
        self._prof = None
        self._t0 = 0.0
        self._result: Optional[Dict] = None

    def __call__(self, requests, stats=None):
        self._seen += 1
        if self._seen == self.after + 1 and self._result is None:
            self._start()
        try:
            return self._infer(requests, stats=stats)
        finally:
            if self._prof is not None and self._seen == self.after + self.batches:
                self._finish()

    def _start(self) -> None:
        from torch.profiler import ProfilerActivity, profile
        from wesnoth_ai import profile_regions
        try:
            self._sync()
            activities = [ProfilerActivity.CPU]
            if getattr(self.device, "type", "cpu") == "cuda":
                activities.append(ProfilerActivity.CUDA)
            self._prof = profile(activities=activities)
            self._prof.start()
            profile_regions.start()
            self._t0 = time.perf_counter()
        except Exception as e:                           # noqa: BLE001 -- informational, the serving goes on
            self._prof = None
            self._result = {"error": type(e).__name__, "message": str(e)[:500]}
            log.exception("the profile did not start")

    def _finish(self) -> None:
        from wesnoth_ai import profile_regions
        prof, self._prof = self._prof, None
        shapes = profile_regions.stop()
        batches = self._seen - self.after
        try:
            self._sync()
            wall = time.perf_counter() - self._t0
            prof.stop()
            path = self.workdir / f".server_profile_{id(self)}.json"
            prof.export_chrome_trace(str(path))
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            finally:
                path.unlink(missing_ok=True)
            self._result = summarize_trace(trace, wall, batches, shapes, self.dims)
            log.info("profile of %d batches: %s", batches, json.dumps(self._result)[:2000])
        except Exception as e:                           # noqa: BLE001 -- informational, the serving goes on
            self._result = {"error": type(e).__name__, "message": str(e)[:500]}
            log.exception("the profile failed")

    def _sync(self) -> None:
        if getattr(self.device, "type", "cpu") == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def summary(self) -> Optional[Dict]:
        """The summary; a profile the match ended inside is finished on
        the batches it holds (call it once the serve thread has stopped)."""
        if self._prof is not None:
            self._finish()
        if self._result is None:
            return {"error": "too_few_batches", "served": self._seen, "after": self.after}
        return self._result
