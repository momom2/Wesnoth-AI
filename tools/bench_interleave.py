"""Several trainer configurations timed in one process, window by window in
turn (tools/bench_sequence_step.py interleave; docs/trainer_speed_20261013.md,
round 3).

    python tools/bench_sequence_step.py interleave --out rows.jsonl --arm NAME=FLAGS [--arm ...]
        [--reference NAME] [--warmup 5] [--rounds 40] [--profile-windows 3] [--spread 0.03] -- TRAIN_ARGS
    python tools/bench_sequence_step.py interleave-summary rows.jsonl [--reference NAME] [--spread 0.03]
        [--out speed.json]

Two runs of the same code timed one after the other on a shared host have
drifted 7 to 14% apart (round 2's this_1 against this_3), more than most
changes being measured. Here every arm is a trainer of its own, built from
the training arguments plus its flags, in one process; every record its
windows read is read before the first (each arm's loader holds them), and
the arms take turns: round r runs window `warmup + r` of every arm, the
order rotated by one arm each round, each window timed between two device
synchronizations (so its optimizer step is in it). The schedule depends
only on the seed, so the arms of a round run the same positions, seconds
apart: whatever the host does to one arm's window it does to its
neighbours'.

An arm against the reference arm: the median over rounds of the ratio of
their window times, with a bootstrap 95% interval, and positions a second
from the summed window times. Arms of identical flags form A/A pairs, which
measure what the method resolves on the day: the session is `valid` when
every pair's median ratio is within `--spread` of 1. A round whose arms ran
other positions or trunk tokens than the reference marks the arm
`same_positions: false`.

An arm that fails to build (a refusal of its arguments, out of memory) or
whose window raises (out of memory, a compile error, a pass out of
positions) is recorded (`kind: error`, with `stage` build or window) and
left out of the later rounds, its trainer released; the others go on, the
reference's failure included (the others then have no ratio). The
rows: a config per arm (bench_sequence_step.config_record with `arm` and
`flags`), a preload row per arm, a warm-up row per arm (each window's time,
the first holding any compile, and the compiled trunk's dynamo counts), a
row per window, a profile per arm (kernel time by class), an `after` row per
arm (the compiled trunk's counts again: a recompile shows as more cache
entries), and the summary.
"""
from __future__ import annotations

import gc
import json
import re
import shlex
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tools.bench_sequence_step import (INTERLEAVE_SPREAD, TOKEN_FIELDS, Rows, bench_files, build, config_record,
                                       median_ratio, production_problems, profile_row, quiet, synchronize, use_repo,
                                       window_row)

ARM_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def parse_arms(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """NAME=FLAGS per arm, names unique."""
    arms = []
    for spec in specs:
        name, sep, flags = spec.partition("=")
        if not sep or not ARM_NAME.match(name):
            raise SystemExit(f"--arm {spec!r}: NAME=FLAGS with a name of letters, digits and _")
        arms.append((name, " ".join(shlex.split(flags))))
    names = [n for n, _ in arms]
    if len(set(names)) != len(names) or len(names) < 2:
        raise SystemExit(f"the arms {names}: at least two, each named once")
    return arms


def timed_window(trainer) -> Tuple[float, Dict]:
    """One window between two device synchronizations."""
    synchronize(trainer.device)
    t0 = time.perf_counter()
    result = trainer.train_window()
    synchronize(trainer.device)
    seconds = time.perf_counter() - t0
    if not result:
        raise SystemExit("the pass ran out of positions before the bench's last window")
    return seconds, result


def block_stats(trainer) -> Optional[Dict]:
    stats = getattr(trainer.model, "trunk_block_stats", None)
    return None if stats is None else stats()


def _build_arm(opts, train_argv, rows: Rows, root: Path, name: str, flags: str, bench: Dict):
    """One arm's trainer, its config and preload rows written."""
    n_windows = opts.warmup + opts.rounds + opts.profile_windows
    argv = list(train_argv) + shlex.split(flags) + ["--out", str(Path(opts.out).parent / f"arm_{name}.pt")]
    _, args, trainer = build(argv)
    quiet(trainer)
    record = config_record(root, args, trainer, argv, bench)
    problems = production_problems(record)
    if problems and not opts.not_production:
        raise SystemExit(f"arm {name}: not production's arguments (--not-production to run anyway): "
                         + "; ".join(problems))
    rows.write({**record, "arm": name, "flags": flags})
    t0 = time.perf_counter()
    files = bench_files(trainer, n_windows)
    trainer.loader.hold(sorted(files))
    rows.write({"kind": "preload", "arm": name, "records": len(files),
                "seconds": round(time.perf_counter() - t0, 1)})
    return trainer


def _build_arms(opts, train_argv, rows: Rows, root: Path) -> Dict[str, object]:
    """Every arm's trainer; an arm that fails to build is recorded and left out."""
    trainers = {}
    bench = {"warmup": opts.warmup, "rounds": opts.rounds, "profile_windows": opts.profile_windows,
             "preload": True, "interleaved": True}
    for name, flags in parse_arms(opts.arm):
        try:
            trainers[name] = _build_arm(opts, train_argv, rows, root, name, flags, bench)
        except (Exception, SystemExit) as e:     # one arm's failure leaves the others built
            rows.write(_error_row(name, None, e, "build", flags))
    return trainers


def _error_row(name: str, i: Optional[int], e: BaseException, stage: str, flags: Optional[str] = None) -> Dict:
    """The failure of an arm, its memory released."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    row = {"kind": "error", "arm": name, "i": i, "stage": stage, "error": type(e).__name__,
           "message": str(e)[:500]}
    return row if flags is None else {**row, "flags": flags}


def interleave(opts, train_argv) -> int:
    root = use_repo(None)
    names = [name for name, _ in parse_arms(opts.arm)]
    reference = opts.reference or names[0]
    if reference not in names:
        raise SystemExit(f"--reference {reference}: not an arm")
    rows = Rows(opts.out)
    trainers = _build_arms(opts, train_argv, rows, root)
    alive = list(trainers)

    def drop(name: str, i: Optional[int], e: BaseException) -> None:
        alive.remove(name)
        del trainers[name]
        rows.write(_error_row(name, i, e, "window"))

    for name in list(alive):
        try:
            seconds = [round(timed_window(trainers[name])[0], 4) for _ in range(opts.warmup)]
        except (Exception, SystemExit) as e:     # one arm's failure leaves the others timed
            drop(name, None, e)
            continue
        rows.write({"kind": "warmup", "arm": name, "seconds": seconds, "trunk_block": block_stats(trainers[name])})
    for r in range(opts.rounds):
        k = r % max(1, len(alive))
        for slot, name in enumerate(alive[k:] + alive[:k]):
            try:
                seconds, result = timed_window(trainers[name])
            except (Exception, SystemExit) as e:     # one arm's failure leaves the others timed
                drop(name, r, e)
                continue
            rows.write({**window_row(r, seconds, result, trainers[name]), "arm": name, "slot": slot})
    for name in alive:
        if opts.profile_windows and trainers[name].device.type == "cuda":
            rows.write({**profile_row(trainers[name], opts.profile_windows, Path(opts.out).parent), "arm": name})
        rows.write({"kind": "after", "arm": name, "trunk_block": block_stats(trainers[name])})
    summary = summarize(read_rows(opts.out), reference, opts.spread)
    rows.write({"kind": "summary", **summary})
    text = json.dumps(summary, indent=1)
    print(text)
    if opts.summary:
        Path(opts.summary).write_text(text + "\n", encoding="utf-8")
    return 0


def read_rows(path) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _same_positions(a: Dict, b: Dict) -> bool:
    return a["positions"] == b["positions"] and all(a[k] == b[k] for k in TOKEN_FIELDS if k in a and k in b)


def _ratio(ref: Dict[int, Dict], arm: Dict[int, Dict]) -> Optional[Dict]:
    """The arm's window times over the reference's, round by round."""
    rounds = sorted(set(ref) & set(arm))
    if len(rounds) < 2:
        return None
    med, lo, hi = median_ratio([ref[i]["seconds"] for i in rounds], [arm[i]["seconds"] for i in rounds])
    return {"rounds": len(rounds), "window_time_ratio_median": round(med, 4),
            "window_time_ratio_ci95": [round(lo, 4), round(hi, 4)], "speedup_median": round(1.0 / med, 4),
            "speedup_ci95": [round(1.0 / hi, 4), round(1.0 / lo, 4)],
            "same_positions": all(_same_positions(ref[i], arm[i]) for i in rounds)}


def _arm_entry(flags: str, windows: Dict[int, Dict]) -> Dict:
    w = list(windows.values())
    seconds = sum(r["seconds"] for r in w)
    peaks = [r["peak_gb"] for r in w if r.get("peak_gb") is not None]
    return {"flags": flags, "windows": len(w),
            "positions_per_second": round(sum(r["positions"] for r in w) / seconds, 2) if seconds else None,
            "window_seconds_median": round(float(np.median([r["seconds"] for r in w])), 4) if w else None,
            "peak_gb_max": max(peaks) if peaks else None,
            "kept_steps_mean": round(sum(r.get("kept_steps") or 0 for r in w) / len(w), 3) if w else None}


def summarize(rows: Sequence[Dict], reference: Optional[str] = None, spread: float = INTERLEAVE_SPREAD) -> Dict:
    """Each arm's speed against the reference and the A/A pairs (module docstring)."""
    configs = {r["arm"]: r for r in rows if r["kind"] == "config"}
    for r in rows:                                # an arm that failed to build has its error row only
        if r["kind"] == "error" and r.get("stage") == "build" and r["arm"] not in configs:
            configs[r["arm"]] = {"arm": r["arm"], "flags": r.get("flags", "")}
    if not configs:
        raise SystemExit("no arm's config row")
    reference = reference or next(iter(configs))
    windows: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for r in rows:
        if r["kind"] == "window":
            windows[r["arm"]][r["i"]] = r
    arms = {}
    for name, config in configs.items():
        entry = _arm_entry(config.get("flags", ""), windows.get(name, {}))
        if name != reference:
            entry["against_reference"] = _ratio(windows.get(reference, {}), windows.get(name, {}))
        arms[name] = entry
    for r in rows:
        if r["kind"] in ("warmup", "profile", "after", "error", "preload") and r.get("arm") in arms:
            arms[r["arm"]].setdefault(r["kind"], {k: v for k, v in r.items() if k not in ("kind", "arm")})
    for name, entry in arms.items():
        warm = (entry.get("warmup") or {}).get("seconds") or []
        if warm and entry["window_seconds_median"]:
            entry["first_window_extra_seconds"] = round(warm[0] - entry["window_seconds_median"], 2)
    by_flags: Dict[str, List[str]] = defaultdict(list)
    for name in configs:
        by_flags[arms[name]["flags"]].append(name)
    pairs = []
    for names in by_flags.values():
        for a, b in combinations(names, 2):
            ratio = _ratio(windows.get(a, {}), windows.get(b, {}))
            if ratio is not None:
                pairs.append({"a": a, "b": b, **ratio,
                              "agree": abs(ratio["window_time_ratio_median"] - 1.0) <= spread})
    return {"reference": reference, "spread": spread, "valid": all(p["agree"] for p in pairs) if pairs else None,
            "aa": pairs, "arms": arms}


def summary_command(opts) -> int:
    summary = summarize(read_rows(opts.rows), opts.reference, opts.spread)
    text = json.dumps(summary, indent=1)
    print(text)
    if opts.out:
        Path(opts.out).write_text(text + "\n", encoding="utf-8")
    return 0
