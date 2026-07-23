"""Tiny in-process profiler used by game_manager to track where time
goes during a training run.

Not intended to replace a real profiler (cProfile, py-spy). Just a
dead-simple "time these named stages" utility that produces readable
log lines like::

    read_state:     mean=220ms  p95=1500ms  n=812
    parse:          mean= 12ms  p95=  18ms  n=812
    policy.select:  mean= 85ms  p95= 130ms  n=812
    send_action:    mean=  2ms  p95=   4ms  n=812

The call sites are inline ``with profiler.time("name"): ...`` blocks.
Keep it narrow so adding and removing instrumentation costs one line.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque


@dataclass
class _Stage:
    count:  int   = 0
    total:  float = 0.0        # running sum of all samples (seconds)
    recent: Deque = field(default_factory=lambda: deque(maxlen=200))
    # recent holds the last N samples; used for percentiles. Small
    # window so the profiler tracks *current* behavior, not ancient.

    def add(self, dt: float) -> None:
        self.count += 1
        self.total += dt
        self.recent.append(dt)

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, q: float) -> float:
        """Approximate percentile (0..1) over the recent window."""
        if not self.recent:
            return 0.0
        ordered = sorted(self.recent)
        idx = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
        return ordered[idx]


class Profiler:
    """Aggregates named-stage timings."""

    def __init__(self):
        self._stages: Dict[str, _Stage] = {}
        self._start_wall = time.perf_counter()

    def time(self, label: str) -> "_TimerCM":
        return _TimerCM(self, label)

    def record(self, label: str, dt: float) -> None:
        s = self._stages.get(label)
        if s is None:
            s = _Stage()
            self._stages[label] = s
        s.add(dt)

    def report(self) -> str:
        """Multi-line table. Sorted by mean time, descending."""
        if not self._stages:
            return "(no timings)"
        items = sorted(
            self._stages.items(),
            key=lambda kv: kv[1].mean(),
            reverse=True,
        )
        # Fixed-width labels so the numbers line up.
        width = max(len(k) for k, _ in items)
        lines = []
        for label, s in items:
            lines.append(
                f"  {label:<{width}}  "
                f"mean={s.mean()*1000:6.1f}ms  "
                f"p50={s.percentile(0.50)*1000:6.1f}ms  "
                f"p95={s.percentile(0.95)*1000:6.1f}ms  "
                f"n={s.count}"
            )
        return "\n".join(lines)

    def throughput(self, n_games: int) -> str:
        """How many games per hour we're sustaining."""
        wall = time.perf_counter() - self._start_wall
        if wall < 1 or n_games < 1:
            return "throughput: (warming up)"
        per_hour = n_games / wall * 3600.0
        return (
            f"throughput: {n_games} games in {wall:.0f}s "
            f"→ {per_hour:.1f} games/hour"
        )

    def reset(self) -> None:
        """Clear all counters (e.g., at a phase boundary)."""
        self._stages.clear()
        self._start_wall = time.perf_counter()


class _TimerCM:
    """Context manager that accumulates `time.perf_counter` deltas
    into a Profiler stage. Exception-safe."""

    __slots__ = ("prof", "label", "_t0")

    def __init__(self, prof: Profiler, label: str):
        self.prof  = prof
        self.label = label
        self._t0   = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.prof.record(self.label, time.perf_counter() - self._t0)
        return False  # don't suppress exceptions
