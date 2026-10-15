"""The activation budget's calibration on the box (tools/bench_sequence_step.py
memory; docs/trainer_speed_20261013.md "The activation budget under the
compiled block").

    python tools/bench_sequence_step.py memory --out memory.json --flags=FLAGS [--warmup 5] [--windows 40]
        [--worst 6] [--scan-positions N] -- TRAIN_ARGS

FLAGS is the configuration under calibration (its trunk, its budget:
`--activation-budget-gb auto` or a size), TRAIN_ARGS the pass's other
arguments. The trainer is built as `sequence_train.main` builds it, probes
and telemetry moved past the bench, and:

1. the scan: the pass's window order replayed on copies of its schedules
   (the first `--scan-positions`, default the whole pass), each window's real
   trunk tokens counted from its records' sizes, nothing trained;
2. `--warmup` windows from the pass's start (the first holds any compile),
   then `--windows` more timed between two device synchronizations: the
   pass's positions a second at this budget, and each window's peak;
3. the `--worst` windows with the most tokens among those after the timed
   ones, each run where
   the pass would meet it (its schedules replayed to it, each slot that
   continues a game-side given a fresh memory of its size): the activation
   estimate's worst case, the planner's figures learned from the windows
   before. A window that runs out of memory is recorded and the others run.

The JSON gives the device, every run window's tokens, kept steps, budget,
peak allocated and reserved, the planner's figures at the end, the headroom
(the device's capacity less the memory outside PyTorch and the largest peak
reserved), whether every window fit (none out of memory), and the flags a
real run should use: FLAGS when every window fit, else none.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from tools.bench_gradcheck import failure
from tools.bench_sequence_step import build, quiet, synchronize, use_repo
from tools.sequence_timing import GIB, trunk_tokens

log = logging.getLogger(__name__)


def _schedules(trainer):
    return (trainer.schedule, None if trainer.own is None else trainer.own.schedule)


def _set_schedules(trainer, schedules) -> None:
    trainer.schedule = schedules[0]
    if trainer.own is not None:
        trainer.own.schedule = schedules[1]


def _copies(schedules):
    return tuple(None if s is None else copy.deepcopy(s) for s in schedules)


def window_tokens(trainer, window) -> int:
    """A window's real trunk tokens, from its records' sizes."""
    total = 0
    for steps in window:
        sizes, sightings = trainer.inputs.sizes(steps)
        total += trunk_tokens(sizes, sightings, [s.k for s in steps])[0]
    return total


def scan(trainer, max_positions: int) -> List[int]:
    """Each window's real tokens, in the pass's order, from copies of its
    schedules (the trainer's own are left where they were)."""
    start = _schedules(trainer)
    _set_schedules(trainer, _copies(start))
    tokens, positions = [], 0
    try:
        while positions < max_positions:
            window = trainer._window()
            if not window:
                break
            trainer.loader.prefetch(trainer._upcoming())
            tokens.append(window_tokens(trainer, window))
            positions += sum(len(steps) for steps in window)
            trainer.loader.keep_only(trainer._upcoming())
    finally:
        _set_schedules(trainer, start)
    return tokens


def schedules_at(trainer, indices: Sequence[int]) -> Dict[int, tuple]:
    """Copies of the schedules as they stand before each window of `indices`."""
    start = _schedules(trainer)
    _set_schedules(trainer, _copies(start))
    found, i = {}, 0
    try:
        for target in sorted(indices):
            while i < target:
                trainer._window()
                i += 1
            found[target] = _copies(_schedules(trainer))
    finally:
        _set_schedules(trainer, start)
    return found


def _peak_reserved(device: torch.device) -> float:
    return torch.cuda.max_memory_reserved(device) / GIB if device.type == "cuda" else 0.0


def _run_window(trainer) -> Dict:
    """One timed window and what it measured; an out-of-memory error is recorded."""
    synchronize(trainer.device)
    t0 = time.perf_counter()
    try:
        result = trainer.train_window()
        synchronize(trainer.device)
    except torch.OutOfMemoryError as e:
        trainer.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"oom": failure(e), "seconds": round(time.perf_counter() - t0, 4)}
    timing = dict(trainer.last_timing or {})
    return {"seconds": round(time.perf_counter() - t0, 4), "positions": result.get("positions", 0),
            "peak_reserved_gb": round(_peak_reserved(trainer.device), 3), **timing}


def _fresh_memories(trainer, window) -> None:
    """Each slot that continues a game-side into `window` given a fresh memory
    of its size (the memory's values do not move the memory's footprint)."""
    trainer.memories = {s.slot: trainer.model.initial_memory(s.k).detach().to(trainer.device)
                        for s in window[0] if not s.starts}


def worst_windows(tokens: Sequence[int], first: int, n: int) -> List[int]:
    """The `n` windows from index `first` on with the most tokens, in order."""
    later = sorted(range(first, len(tokens)), key=lambda i: (-tokens[i], i))
    return sorted(later[:n])


def calibrate(opts, train_argv) -> int:
    import shlex
    use_repo(None)
    _, args, trainer = build(list(train_argv) + shlex.split(opts.flags))
    quiet(trainer)
    positions = int(opts.scan_positions or trainer.pass_total)
    t0 = time.perf_counter()
    tokens = scan(trainer, positions)
    scan_seconds = time.perf_counter() - t0
    worst = worst_windows(tokens, opts.warmup + opts.windows, opts.worst)
    states = schedules_at(trainer, worst)
    warm = [_run_window(trainer) for _ in range(opts.warmup)]
    timed = [_run_window(trainer) for _ in range(opts.windows)]
    rows = []
    for i in worst:
        _set_schedules(trainer, _copies(states[i]))
        _fresh_memories(trainer, _peek(trainer))
        rows.append({"i": i, "scan_tokens": tokens[i], **_run_window(trainer)})
    result = _summary(trainer, args, opts.flags, tokens, scan_seconds, warm, timed, rows)
    text = json.dumps(result, indent=1)
    print(text)
    Path(opts.out).write_text(text + "\n", encoding="utf-8")
    return 0


def _peek(trainer):
    """The next window, the schedules left where they were."""
    start = _schedules(trainer)
    _set_schedules(trainer, _copies(start))
    try:
        return trainer._window()
    finally:
        _set_schedules(trainer, start)


def _summary(trainer, args, flags: str, tokens, scan_seconds, warm, timed, worst) -> Dict:
    device = trainer.device
    ran = [r for r in warm + timed + worst if "oom" not in r]
    capacity = outside = None
    if trainer.memory_planner is not None:
        capacity, outside = trainer.memory_planner.capacity / GIB, trainer.memory_planner.outside / GIB
    elif device.type == "cuda":
        capacity = torch.cuda.get_device_properties(device).total_memory / GIB
        free, total = torch.cuda.mem_get_info(device)
        outside = (total - free - torch.cuda.memory_reserved(device)) / GIB
    peak_reserved = max((r.get("peak_reserved_gb", 0.0) for r in ran), default=0.0)
    ooms = [r for r in warm + timed + worst if "oom" in r]
    fits = not ooms
    seconds = sum(r["seconds"] for r in timed if "oom" not in r)
    flags = " ".join(flags.split())
    covered = max([r.get("scan_tokens", 0) for r in worst] + [r.get("tokens_real", 0) for r in warm + timed],
                  default=0)
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "capacity_gb": None if capacity is None else round(capacity, 3),
            "outside_gb": None if outside is None else round(outside, 3),
            "activation_budget_gb": args.activation_budget_gb, "memory_margin_gb": args.memory_margin_gb,
            "cuda_allocator": trainer.meta.get("cuda_allocator"),
            "scan": {"windows": len(tokens), "seconds": round(scan_seconds, 1),
                     "tokens_max": max(tokens, default=0),
                     "tokens_p99": int(np.percentile(tokens, 99)) if tokens else 0,
                     "tokens_median": int(np.median(tokens)) if tokens else 0},
            "timed": {"windows": len(timed), "positions_per_second":
                      round(sum(r.get("positions", 0) for r in timed) / seconds, 1) if seconds else None,
                      "peak_gb_max": max((r.get("peak_gb") or 0.0 for r in timed), default=None)},
            "warmup": warm, "worst": worst,
            "planner": None if trainer.memory_planner is None else trainer.memory_planner.state(),
            "peak_reserved_gb_max": round(peak_reserved, 3),
            "headroom_gb": None if capacity is None else round(capacity - (outside or 0.0) - peak_reserved, 3),
            "largest_window_run": covered >= max(tokens, default=0),
            "ooms": len(ooms), "fits": fits, "flags": flags if fits else None}
