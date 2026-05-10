"""Measure simulator throughput on real replays.

Sequential reconstruction of a replay (what diff_replay does) is the
closest analog to a self-play rollout's inner loop: pick action,
apply, repeat. We time that path on a length-stratified sample of
30 replays from the corpus and report:

  - replays / second
  - commands / second
  - mean ms / command, broken down by command kind
  - cProfile breakdown of where wall-clock goes

cProfile output lands at `benchmarks/sim_profile.prof` so it can be
opened with snakeviz for future optimization work.

Dependencies: tools.diff_replay, tools.replay_dataset
"""
from __future__ import annotations

import cProfile
import gzip
import json
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from tools.diff_replay import (  # noqa: E402
    _build_initial_gamestate,
    _setup_scenario_events,
    _apply_command,
)


def _pick_sample(n_short=10, n_mid=10, n_long=10):
    files = sorted(Path("replays_dataset_full").glob("*.json.gz"))
    sizes = []
    for p in files[:1500]:
        try:
            with gzip.open(p, "rt") as f:
                sizes.append((p, len(json.load(f).get("commands", []))))
        except Exception:
            pass
    sizes.sort(key=lambda x: x[1])
    nz = [(p, s) for p, s in sizes if s > 30]  # drop turn-0 abandons
    n = len(nz)
    short = nz[:n_short]
    mid = nz[n // 2 - n_mid // 2 : n // 2 + n_mid // 2]
    long_ = nz[-n_long:]
    return short + mid + long_


def _run_one(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    cmds = data.get("commands", [])
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    per_kind_ns = defaultdict(int)
    per_kind_n = defaultdict(int)
    t0 = time.perf_counter_ns()
    for c in cmds:
        kind = c[0] if c else "?"
        ts = time.perf_counter_ns()
        _apply_command(gs, c)
        dt = time.perf_counter_ns() - ts
        per_kind_ns[kind] += dt
        per_kind_n[kind] += 1
    elapsed_s = (time.perf_counter_ns() - t0) / 1e9
    return elapsed_s, len(cmds), per_kind_ns, per_kind_n


def main():
    sample = _pick_sample()
    n_short, n_mid, n_long = 10, 10, 10
    print(f"sample: {len(sample)} replays, "
          f"{sum(s for _, s in sample)} total commands")
    print(f"  short: {sample[0][1]}–{sample[n_short-1][1]} cmds")
    print(f"  mid:   {sample[n_short][1]}–{sample[n_short+n_mid-1][1]} cmds")
    print(f"  long:  {sample[-n_long][1]}–{sample[-1][1]} cmds")
    print()

    # Profile + time everything together.
    profiler = cProfile.Profile()
    profiler.enable()

    total_s = 0.0
    total_cmds = 0
    per_kind_ns = defaultdict(int)
    per_kind_n = defaultdict(int)
    per_replay = []

    for p, n_cmds in sample:
        elapsed_s, ncmds, k_ns, k_n = _run_one(p)
        total_s += elapsed_s
        total_cmds += ncmds
        per_replay.append((p.name, ncmds, elapsed_s))
        for k, v in k_ns.items():
            per_kind_ns[k] += v
        for k, v in k_n.items():
            per_kind_n[k] += v

    profiler.disable()

    out_prof = Path("benchmarks/sim_profile.prof")
    out_prof.parent.mkdir(exist_ok=True)
    profiler.dump_stats(str(out_prof))
    print(f"profile dumped: {out_prof}\n")

    # Aggregate.
    cmds_per_s = total_cmds / total_s
    replays_per_s = len(sample) / total_s

    print(f"=== Sim throughput on {len(sample)} replays ===")
    print(f"  wall:           {total_s:8.3f} s")
    print(f"  commands:       {total_cmds:8d}")
    print(f"  commands/sec:   {cmds_per_s:8.0f}")
    print(f"  replays/sec:    {replays_per_s:8.2f}")
    print(f"  mean us/cmd:    {1e6 * total_s / total_cmds:8.1f}")
    print()

    print("=== Per-command-kind cost ===")
    print(f"  {'kind':14s} {'count':>8s} {'mean_us':>10s} {'%total':>8s}")
    for k in sorted(per_kind_ns, key=lambda x: -per_kind_ns[x]):
        n = per_kind_n[k]
        ns = per_kind_ns[k]
        mean_us = ns / n / 1e3 if n else 0
        pct = 100 * ns / sum(per_kind_ns.values())
        print(f"  {k:14s} {n:8d} {mean_us:10.1f} {pct:7.1f}%")
    print()

    print("=== Per-replay (length vs wall) ===")
    for name, ncmds, s in sorted(per_replay, key=lambda x: x[1]):
        print(f"  {name:35s} cmds={ncmds:6d}  wall={s*1000:8.1f} ms")
    print()

    # Top hot functions from cProfile.
    print("=== Top 20 functions by cumulative time ===")
    s = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    s.print_stats(20)


if __name__ == "__main__":
    main()
