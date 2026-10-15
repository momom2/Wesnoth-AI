"""Time the sequence trainer's step (tools/sequence_train.py) under the
production arguments, and check that a change to it leaves its numbers
where they were (docs/trainer_speed_20261013.md, "Benchmark").

    python tools/bench_sequence_step.py run --out arm.jsonl [--repo DIR] [--preload] -- TRAIN_ARGS
    python tools/bench_sequence_step.py compare A.jsonl B.jsonl [--allow FIELD ...] [--out ratio.json]
    python tools/bench_sequence_step.py trace [--repo DIR] [--windows 3] [--dropout 0.3] -- TRAIN_ARGS
    python tools/bench_sequence_step.py gradcheck --out check.json [--windows 3] [--reference=FLAGS]
        [--candidate=FLAGS ...] -- TRAIN_ARGS
    python tools/bench_sequence_step.py interleave --out rows.jsonl --arm NAME=FLAGS ... -- TRAIN_ARGS
    python tools/bench_sequence_step.py interleave-summary rows.jsonl [--out speed.json]
    python tools/bench_sequence_step.py memory --out memory.json --flags=FLAGS ... -- TRAIN_ARGS

TRAIN_ARGS is the training command line, verbatim (the round preset's of
scripts/memory_own_trajectories_box.sh). The trainer is built from it as
`sequence_train.main` builds it, and `train_window` is called directly with
every probe, telemetry row and checkpoint moved past the bench. `--repo DIR`
times or traces the code of another tree (a code stage of the base commit):
its `tools` and `wesnoth_ai` are imported instead of this tree's.

run: a first record read from the live objects (the code: its commit, its
code stage, its version and the git blob id of every source the process
loaded from the tree's tools and wesnoth_ai; the versions,
the streams, the autocast, every dropout probability, the trunk, the
activation budget, the attention kernels allowed, the learning rate and
clip, the init checkpoint's SHA-256, the records' fingerprints, the
allocator's configuration and the GC thresholds), refused unless the
production values hold (32 streams, window 16, 16 own streams at 64 slots,
bfloat16 autocast; `--not-production` lifts it); `--warmup` untimed
windows; `--windows` timed ones, each row its wall time and the trainer's
timing fields, the total between two device synchronizations; nvidia-smi
sampled every `--smi-ms` during them; `--profile-windows` more under
torch.profiler, the kernel time by class and the device's idle fraction
(a profile that fails leaves a row naming its error; the timed windows
stand). `--preload` reads every record the bench's windows need before the first
one (the loader's cost, measured as the difference from a streaming run).
Rows are appended as they come.

compare: per-window ratio of wall times (the schedule depends only on the
seed, so window i holds the same positions in both runs), its median with a
bootstrap 95% interval, and positions a second from each run's
synchronized total. Refuses two runs whose first records differ in a field
other than the code and the `--allow`ed ones, and two runs whose windows
ran other trunk tokens (real or padded) when both count them; a tree that
does not count them (the base) relies on `trace` for that.

trace: the SHA-256 of N windows on the CPU, every dropout probability set to
`--dropout`: the returned losses and sums, every gradient before it is
zeroed, the parameters after each step and the carried memories. Two trees
whose change only moves bytes print the same digest.

gradcheck: candidate configurations (`--candidate`, repeated, the training
flags each adds; default the packed trunk) against one reference
(`--reference`; default none, the padded trunk), each candidate on its own
(tools/bench_gradcheck.py).

interleave, interleave-summary: several configurations timed in one
process, window by window in turn (tools/bench_interleave.py).

memory: the activation budget calibrated on the pass's largest windows
(tools/bench_memory.py).
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import os
import platform
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PRODUCTION = {"streams": 32, "window": 16, "own_streams": 16, "own_slots": 64, "autocast": "torch.bfloat16"}
# Fields two runs may differ in whatever is under test.
ALWAYS_ALLOWED = ("kind", "code", "argv", "started")
# Two arms of the same flags agree when their median window-time ratio is
# within this of 1 (tools/bench_interleave.py).
INTERLEAVE_SPREAD = 0.03
FAR = 10 ** 18                          # a position no bench reaches
# A window's trunk tokens, counted from its records whatever the trunk
# (tools/sequence_timing.trunk_tokens): equal in two runs of one schedule.
TOKEN_FIELDS = ("tokens_real", "tokens_padded")


# ---- the trainer ------------------------------------------------------------

def use_repo(repo: Optional[str]) -> Path:
    """Put the code tree `repo` (default: this file's) first on the import
    path, before any of its modules is imported."""
    root = Path(repo).resolve() if repo else Path(__file__).resolve().parent.parent
    for name in ("tools", "wesnoth_ai"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__path__[0]).resolve().parent != root:
            raise SystemExit(f"{name} is already imported from {loaded.__path__[0]}, not from {root}")
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != here]
    for p in (root / "tools", root):
        sys.path.insert(0, str(p))
    return root


def build(train_argv: Sequence[str]):
    """(sequence_train module, its arguments, the trainer) as `main` builds
    them; a tree without `build_trainer` gets main's steps of its day."""
    from tools import sequence_train as st
    args = st.parse_args(list(train_argv))
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if hasattr(st, "build_trainer"):
        return st, args, st.build_trainer(args)
    torch.manual_seed(args.seed)
    trainer = st.Trainer(args, torch.device(args.device))
    if args.resume:
        trainer.resume(args.out)
    elif args.init_from is not None:
        trainer.init_from(args.init_from)
    return st, args, trainer


def quiet(trainer) -> None:
    """No probe, telemetry row or checkpoint during the bench."""
    for key in ("next_probe", "next_signal", "next_checkpoint"):
        trainer.state[key] = FAR


def set_dropout(trainer, p: float) -> None:
    """Every dropout probability of the network set to `p`."""
    for module in list(trainer.model.modules()) + list(trainer.encoder.modules()):
        if isinstance(module, torch.nn.Dropout):
            module.p = p
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = p


def dropout_probabilities(trainer) -> Dict[str, int]:
    found: Counter = Counter()
    for module in list(trainer.model.modules()) + list(trainer.encoder.modules()):
        if isinstance(module, torch.nn.Dropout):
            found[f"dropout {module.p:g}"] += 1
        if isinstance(module, torch.nn.MultiheadAttention):
            found[f"attention {module.dropout:g}"] += 1
    return dict(sorted(found.items()))


def file_sha256(path) -> Optional[str]:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_id(path: Path) -> str:
    """The id git gives a file's bytes (`git hash-object`, `git ls-tree`)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def loaded_sources(root: Path) -> Dict[str, str]:
    """The git blob id of every module this process loaded from the tree's
    tools and wesnoth_ai, by its path in the tree: a code stage carries no
    commit, and these name the code that ran (`git ls-tree -r <commit>
    -- tools wesnoth_ai` lists a commit's)."""
    found = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        try:
            rel = Path(path).resolve().relative_to(root)
        except ValueError:
            continue
        if rel.parts[0] in ("tools", "wesnoth_ai"):
            found[rel.as_posix()] = git_blob_id(Path(path))
    return dict(sorted(found.items()))


def code_identity(root: Path) -> Dict:
    """The tree's commit (git, when it is a checkout), the code stage it was
    unpacked from (boxlib's .staged_from), its version and the sources the
    process loaded from it."""
    commit = None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    staged = root / ".staged_from"
    import wesnoth_ai
    return {"commit": commit, "staged_from": staged.read_text().strip() if staged.exists() else None,
            "version": getattr(wesnoth_ai, "__version__", None), "root": str(root),
            "sources": loaded_sources(root.resolve())}


def config_record(root: Path, args, trainer, train_argv, bench: Dict) -> Dict:
    """The run's first record, read from the live objects."""
    cuda = trainer.device.type == "cuda"
    own = trainer.own
    budget = getattr(args, "activation_budget_gb", 0.0) or 0.0           # a size, or "auto"
    keeps = budget == "auto" or float(budget) > 0
    return {
        "kind": "config", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": code_identity(root), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(trainer.device) if cuda else None,
        "python": platform.python_version(), "streams": args.streams, "window": args.window,
        "own_streams": own.schedule.n_streams if own is not None else 0,
        "own_slots": own.slots if own is not None else None,
        "autocast": str(trainer.autocast), "dropout": dropout_probabilities(trainer),
        "trunk": getattr(trainer, "trunk", "padded"), "activation_budget_gb": budget,
        "columns": bool(getattr(args, "columns", False)),
        "window_batched": bool(getattr(args, "window_batched", False)),
        "trunk_mode": trunk_mode_name(trainer),
        "activation_bytes_per_token": getattr(trainer, "bytes_per_token", None) if keeps else None,
        "memory_margin_gb": getattr(args, "memory_margin_gb", None) if budget == "auto" else None,
        "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                 "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                 "math": torch.backends.cuda.math_sdp_enabled()},
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "lr": args.lr, "grad_clip": args.grad_clip, "seed": args.seed, "warmup_steps": args.warmup_steps,
        "decay_from": args.decay_from, "pass_positions": args.pass_positions,
        "init_sha256": file_sha256(args.init_from),
        "fingerprints": {"corpus": trainer.manifest["fingerprint"],
                         "own": own.manifest["fingerprint"] if own is not None else None},
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "gc_thresholds": list(gc.get_threshold()),
        "argv": [a for a in train_argv], "bench": bench}


def trunk_mode_name(trainer) -> str:
    """The trainer's trunk variants (wesnoth_ai/trunk_modes.py); "none" for
    a tree that has none."""
    mode = getattr(trainer, "trunk_mode", None)
    return "none" if mode is None else mode.describe()


def production_problems(record: Dict) -> List[str]:
    return [f"{k}={record.get(k)!r}, production has {v!r}" for k, v in PRODUCTION.items() if record.get(k) != v]


def bench_files(trainer, n_windows: int) -> set:
    """The records the next `n_windows` windows read, from copies of the
    schedules."""
    T = trainer.args.window
    schedules = [copy.deepcopy(trainer.schedule)]
    if trainer.own is not None:
        schedules.append(copy.deepcopy(trainer.own.schedule))
    files = set()
    for _ in range(n_windows):
        for schedule in schedules:
            for steps in schedule.window(T):
                files.update(s.game_side.file for s in steps)
    return files


class Rows:
    """The run's records, appended and flushed as they come."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise SystemExit(f"{self.path} exists: a run writes a file of its own")

    def write(self, row: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_window(trainer) -> Tuple[float, Dict]:
    t0 = time.perf_counter()
    result = trainer.train_window()
    seconds = time.perf_counter() - t0
    if not result:
        raise SystemExit("the pass ran out of positions before the bench's last window")
    return seconds, result


def window_row(i: int, seconds: float, result: Dict, trainer) -> Dict:
    row = {"kind": "window", "i": i, "seconds": round(seconds, 4), "positions": result["positions"],
           "loss": result["loss"]}
    for key, value in (getattr(trainer, "last_timing", None) or {}).items():
        row["trainer_seconds" if key == "seconds" else key] = value
    return row


# ---- nvidia-smi and the profiler ---------------------------------------------

class GpuSampler:
    """nvidia-smi's utilization, SM clock and power every `ms` milliseconds."""

    QUERY = "utilization.gpu,clocks.sm,power.draw"

    def __init__(self, ms: int, path: Path):
        self.path = path
        self.proc = None
        try:
            self._out = open(path, "w", encoding="utf-8")
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits",
                                          f"-lms={ms}"], stdout=self._out, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None

    def stop(self) -> Dict:
        if self.proc is None:
            return {"samples": 0}
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._out.close()
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                continue
        if not rows:
            return {"samples": 0}
        a = np.asarray(rows)
        return {"samples": len(a), "utilization_mean": round(float(a[:, 0].mean()), 1),
                "utilization_median": float(np.median(a[:, 0])), "sm_clock_mean": round(float(a[:, 1].mean())),
                "power_mean": round(float(a[:, 2].mean()), 1)}


KERNEL_CLASSES = (("attention", ("flash", "fmha", "attention", "attn")),
                  ("gemm", ("gemm", "gemv", "cutlass", "xmma", "cublas")))


def kernel_class(name: str, category: str) -> str:
    if category in ("gpu_memcpy", "gpu_memset") or "memcpy" in name.lower() or "memset" in name.lower():
        return "memcpy"
    low = name.lower()
    for cls, keys in KERNEL_CLASSES:
        if any(k in low for k in keys):
            return cls
    return "elementwise_norm"


def trace_summary(trace_path: Path, wall: float, n_windows: int) -> Dict:
    """Kernel time by class per window and the device's idle fraction over
    the profiled windows' wall time, from a chrome trace."""
    events = json.loads(Path(trace_path).read_text(encoding="utf-8")).get("traceEvents", [])
    spans, by_class = [], Counter()
    for e in events:
        cat = e.get("cat", "")
        if e.get("ph") != "X" or cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        start, dur = float(e["ts"]), float(e.get("dur", 0.0))
        spans.append((start, start + dur))
        by_class[kernel_class(e.get("name", ""), cat)] += dur
    busy, end = 0.0, -1.0
    for lo, hi in sorted(spans):
        if hi <= end:
            continue
        busy += hi - max(lo, end)
        end = hi
    total = sum(by_class.values())
    return {"windows": n_windows, "wall_seconds": round(wall, 3), "kernels": len(spans),
            "ms_per_window": {k: round(v / 1e3 / n_windows, 2) for k, v in sorted(by_class.items())},
            "share": {k: round(v / total, 4) for k, v in sorted(by_class.items())} if total else {},
            "idle_fraction": round(max(0.0, 1.0 - busy / 1e6 / wall), 4) if wall > 0 else None}


def profile_windows(trainer, n: int, workdir: Path) -> Dict:
    from torch.profiler import ProfilerActivity, profile
    synchronize(trainer.device)
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            timed_window(trainer)
        synchronize(trainer.device)
    wall = time.perf_counter() - t0
    path = workdir / "profile_trace.json"
    prof.export_chrome_trace(str(path))
    try:
        return trace_summary(path, wall, n)
    finally:
        path.unlink(missing_ok=True)


def profile_row(trainer, n: int, workdir: Path) -> Dict:
    """The profile's row; one naming the error when profiling, exporting or
    reading the trace fails (the timed windows' records stand)."""
    try:
        return {"kind": "profile", **profile_windows(trainer, n, workdir)}
    except Exception as e:                  # any failure of an informational step
        logging.getLogger(__name__).exception("the profile failed")
        return {"kind": "profile", "error": type(e).__name__, "message": str(e)[:500]}


# ---- run ----------------------------------------------------------------------

def run(opts, train_argv) -> int:
    root = use_repo(opts.repo)
    _, args, trainer = build(train_argv)
    quiet(trainer)
    bench = {"warmup": opts.warmup, "windows": opts.windows, "profile_windows": opts.profile_windows,
             "preload": bool(opts.preload)}
    record = config_record(root, args, trainer, train_argv, bench)
    problems = production_problems(record)
    if problems and not opts.not_production:
        raise SystemExit("not production's arguments (--not-production to run anyway): " + "; ".join(problems))
    rows = Rows(opts.out)
    rows.write(record)
    if opts.preload:
        if not hasattr(trainer.loader, "hold"):
            raise SystemExit("--preload: this tree's loader cannot hold records")
        t0 = time.perf_counter()
        files = bench_files(trainer, opts.warmup + opts.windows + opts.profile_windows)
        trainer.loader.hold(sorted(files))
        rows.write({"kind": "preload", "records": len(files), "seconds": round(time.perf_counter() - t0, 1)})
    for _ in range(opts.warmup):
        timed_window(trainer)
    device = trainer.device
    sampler = GpuSampler(opts.smi_ms, Path(opts.out).with_suffix(".smi.csv")) \
        if device.type == "cuda" and opts.smi_ms > 0 else None
    synchronize(device)
    t0 = time.perf_counter()
    positions = 0
    for i in range(opts.windows):
        seconds, result = timed_window(trainer)
        positions += result["positions"]
        rows.write(window_row(i, seconds, result, trainer))
    synchronize(device)
    wall = time.perf_counter() - t0
    rows.write({"kind": "total", "windows": opts.windows, "positions": positions, "seconds": round(wall, 3),
                "positions_per_second": round(positions / wall, 2)})
    if sampler is not None:
        rows.write({"kind": "gpu", **sampler.stop()})
    if opts.profile_windows and device.type == "cuda":
        rows.write(profile_row(trainer, opts.profile_windows, Path(opts.out).parent))
    return 0


# ---- compare --------------------------------------------------------------------

def read_run(path) -> Tuple[Dict, List[Dict], Dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    config = next(r for r in rows if r["kind"] == "config")
    windows = sorted((r for r in rows if r["kind"] == "window"), key=lambda r: r["i"])
    total = next((r for r in rows if r["kind"] == "total"), None)
    if total is None:
        raise SystemExit(f"{path}: no total row (the run did not finish)")
    return config, windows, total


def config_differences(a: Dict, b: Dict, allowed: Sequence[str]) -> List[str]:
    skip = set(ALWAYS_ALLOWED) | set(allowed)
    return sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))


def median_ratio(a: Sequence[float], b: Sequence[float], resamples: int = 2000,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Median of the per-window ratios b/a and its bootstrap 95% interval."""
    r = np.asarray(b, dtype=float) / np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(r[rng.integers(0, len(r), size=(resamples, len(r)))], axis=1)
    return float(np.median(r)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def compare(opts) -> int:
    ca, wa, ta = read_run(opts.a)
    cb, wb, tb = read_run(opts.b)
    differ = config_differences(ca, cb, opts.allow)
    if differ:
        raise SystemExit(f"the runs differ in {', '.join(differ)}: not a comparison of one change")
    if [w["positions"] for w in wa] != [w["positions"] for w in wb]:
        raise SystemExit("the runs' windows hold other positions: not the same schedule")
    for key in TOKEN_FIELDS:
        if all(key in w for w in wa + wb) and [w[key] for w in wa] != [w[key] for w in wb]:
            raise SystemExit(f"the runs' windows ran other tokens ({key}): not the same positions")
    med, lo, hi = median_ratio([w["seconds"] for w in wa], [w["seconds"] for w in wb])
    out = {"a": str(opts.a), "b": str(opts.b), "windows": len(wa),
           "a_positions_per_second": ta["positions_per_second"],
           "b_positions_per_second": tb["positions_per_second"],
           "speedup_total": round(tb["positions_per_second"] / ta["positions_per_second"], 4),
           "window_time_ratio_median": round(med, 4), "window_time_ratio_ci95": [round(lo, 4), round(hi, 4)],
           "speedup_median": round(1.0 / med, 4), "speedup_ci95": [round(1.0 / hi, 4), round(1.0 / lo, 4)],
           "differ": {k: [ca.get(k), cb.get(k)] for k in opts.allow if ca.get(k) != cb.get(k)}}
    text = json.dumps(out, indent=1)
    print(text)
    if opts.out:
        Path(opts.out).write_text(text + "\n", encoding="utf-8")
    return 0


# ---- trace ----------------------------------------------------------------------

def _tensor_bytes(t: Optional[torch.Tensor]) -> bytes:
    return b"none" if t is None else t.detach().cpu().contiguous().numpy().tobytes()


def _value_bytes(result: Dict) -> bytes:
    parts = []
    for key in sorted(result):
        v = result[key]
        parts.append(key.encode() + (struct.pack("<d", float(v)) if isinstance(v, (int, float)) else repr(v).encode()))
    return b"|".join(parts)


def trace(opts, train_argv) -> int:
    use_repo(opts.repo)
    torch.set_num_threads(opts.threads)
    st, args, trainer = build(train_argv)
    if trainer.device.type != "cpu":
        raise SystemExit("trace runs on the CPU (--device cpu): the GPU's kernels are not bit-reproducible")
    quiet(trainer)
    set_dropout(trainer, opts.dropout)
    grads: List[bytes] = []
    zero_grad = trainer.opt.zero_grad

    def hooked(*a, **kw):
        grads.append(b"".join(_tensor_bytes(p.grad) for p in trainer.params))
        return zero_grad(*a, **kw)

    trainer.opt.zero_grad = hooked
    total = hashlib.sha256()
    windows = []
    for i in range(opts.windows):
        result = trainer.train_window()
        if not result:
            raise SystemExit(f"the pass ran out after {i} windows")
        parts = {"values": _value_bytes(result), "gradients": grads[-1],
                 "parameters": b"".join(_tensor_bytes(p) for p in trainer.params),
                 "memories": b"".join(struct.pack("<q", s) + _tensor_bytes(trainer.memories[s])
                                      for s in sorted(trainer.memories))}
        digests = {k: hashlib.sha256(v).hexdigest() for k, v in parts.items()}
        for k in sorted(parts):
            total.update(parts[k])
        windows.append({"i": i, "loss": result["loss"], **digests})
    print(json.dumps({"sha256": total.hexdigest(), "trainer": str(Path(st.__file__).resolve()), "windows": windows},
                     indent=1))
    return 0


# ---- the command line -------------------------------------------------------------

def parse(argv: Sequence[str]):
    import argparse
    argv = list(argv)
    train = []
    if "--" in argv:
        i = argv.index("--")
        argv, train = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="time the step")
    r.add_argument("--out", type=Path, required=True, help="the run's records (JSONL, a new file)")
    r.add_argument("--repo", default=None, help="time this code tree's trainer instead of this one's")
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--windows", type=int, default=80)
    r.add_argument("--profile-windows", type=int, default=3)
    r.add_argument("--smi-ms", type=int, default=100, help="nvidia-smi's sampling period; 0: none")
    r.add_argument("--preload", action="store_true", help="read every record the bench needs before it starts")
    r.add_argument("--not-production", action="store_true", help="run under other than production's arguments")
    c = sub.add_parser("compare", help="the speedup of run B over run A")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--allow", nargs="*", default=[], help="first-record fields the runs may differ in")
    c.add_argument("--out", type=Path, default=None)
    t = sub.add_parser("trace", help="the SHA-256 of a few windows on the CPU")
    t.add_argument("--repo", default=None)
    t.add_argument("--windows", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--threads", type=int, default=4)
    g = sub.add_parser("gradcheck", help="each candidate's gradients against a reference's (default: packed against padded)")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--windows", type=int, default=3)
    g.add_argument("--reference", default="", help="the reference's training flags (write --reference=FLAGS)")
    g.add_argument("--candidate", action="append", default=None,
                   help="a candidate's training flags (write --candidate=FLAGS; repeat for several; default "
                        "--packed-trunk)")
    m = sub.add_parser("memory", help="calibrate the activation budget on the pass's largest windows")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--flags", default="", help="the configuration under calibration (write --flags=FLAGS)")
    m.add_argument("--warmup", type=int, default=5)
    m.add_argument("--windows", type=int, default=40)
    m.add_argument("--worst", type=int, default=6)
    m.add_argument("--scan-positions", type=int, default=None, help="the scan's positions (default: the pass)")
    i = sub.add_parser("interleave", help="time several configurations window by window in one process")
    i.add_argument("--out", type=Path, required=True, help="the rows (JSONL, a new file)")
    i.add_argument("--arm", action="append", required=True, help="NAME=FLAGS: an arm's training flags")
    i.add_argument("--reference", default=None, help="the arm the others are timed against (default the first)")
    i.add_argument("--warmup", type=int, default=5)
    i.add_argument("--rounds", type=int, default=40)
    i.add_argument("--profile-windows", type=int, default=3)
    i.add_argument("--spread", type=float, default=INTERLEAVE_SPREAD)
    i.add_argument("--summary", type=Path, default=None, help="the summary also written here (JSON)")
    i.add_argument("--not-production", action="store_true", help="run under other than production's arguments")
    s = sub.add_parser("interleave-summary", help="the summary of an interleaved run's rows")
    s.add_argument("rows", type=Path)
    s.add_argument("--reference", default=None)
    s.add_argument("--spread", type=float, default=INTERLEAVE_SPREAD)
    s.add_argument("--out", type=Path, default=None)
    opts = ap.parse_args(argv)
    if opts.command not in ("compare", "interleave-summary") and not train:
        ap.error("the training arguments follow `--`")
    return opts, train


def main(argv=None) -> int:
    opts, train = parse(sys.argv[1:] if argv is None else argv)
    if opts.command == "run":
        return run(opts, train)
    if opts.command == "compare":
        return compare(opts)
    if opts.command == "trace":
        return trace(opts, train)
    use_repo(None)
    if opts.command in ("interleave", "interleave-summary"):
        from tools import bench_interleave
        return (bench_interleave.interleave(opts, train) if opts.command == "interleave"
                else bench_interleave.summary_command(opts))
    if opts.command == "memory":
        from tools import bench_memory
        return bench_memory.calibrate(opts, train)
    from tools import bench_gradcheck
    return bench_gradcheck.gradcheck(opts, train)


if __name__ == "__main__":
    raise SystemExit(main())
