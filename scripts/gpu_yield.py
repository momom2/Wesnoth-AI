"""Yield the GPU to the machine's user (docs/local_machine_use_20261029.md,
option 3): watch the GPU use of every process outside a run's own tree and
say when the run should step aside and when it may come back.

Windows reports each process's share of each GPU engine's time as the
performance counter `\\GPU Engine(*)\\Utilization Percentage`, one instance
per process and engine (`pid_<pid>_luid_..._engtype_3D`, `..._Compute 1`,
...). A process's use here is its busiest engine's percentage. Typing,
browsing and the compositor stay in single digits; a game or a 3D program
takes most of an engine.

The policy:
- step aside when some process outside the run has used at least
  `threshold` percent for `busy_samples` samples in a row;
- come back when no process outside the run has reached `threshold` for
  `quiet_minutes`.

The caller stops its work when told to step aside: a match resumes,
losing only its games in flight.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

_COUNTER = r"\GPU Engine(*)\Utilization Percentage"
_INSTANCE = re.compile(r"pid_(\d+)_")


def parse_samples(text: str) -> Dict[int, float]:
    """`instance=value` lines (one counter sample) as the busiest engine's
    percentage per process id."""
    use: Dict[int, float] = {}
    for line in text.splitlines():
        name, sep, value = line.strip().rpartition("=")
        m = _INSTANCE.search(name)
        if not sep or m is None:
            continue
        try:
            pct = float(value)
        except ValueError:
            continue
        pid = int(m.group(1))
        use[pid] = max(use.get(pid, 0.0), pct)
    return use


def gpu_use_by_pid(timeout_s: float = 60.0) -> Dict[int, float]:
    """One counter sample: each process's busiest GPU engine, in percent.
    {} off Windows or when the counter cannot be read (no GPU counters, a
    timeout): the caller then sees no use."""
    if sys.platform != "win32":
        return {}
    cmd = (f"(Get-Counter '{_COUNTER}' -ErrorAction Stop).CounterSamples | "
           f"ForEach-Object {{ $_.InstanceName + '=' + $_.CookedValue }}")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                           text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse_samples(r.stdout) if r.returncode == 0 else {}


def tree_pids(root: int) -> Set[int]:
    """`root` and every live descendant."""
    import psutil
    try:
        p = psutil.Process(root)
        return {root} | {c.pid for c in p.children(recursive=True)}
    except psutil.NoSuchProcess:
        return {root}


def busiest_other(use: Dict[int, float], own: Iterable[int]) -> Tuple[Optional[int], float]:
    """The process outside `own` with the most GPU use, and that use (None,
    0.0 when there is none)."""
    own = set(own)
    others = [(pct, pid) for pid, pct in use.items() if pid not in own]
    if not others:
        return None, 0.0
    pct, pid = max(others)
    return pid, pct


class GpuYield:
    """The policy over successive samples (module docstring)."""

    def __init__(self, threshold: float = 15.0, busy_samples: int = 2, quiet_minutes: float = 10.0,
                 sample: Callable[[], Dict[int, float]] = gpu_use_by_pid):
        self.threshold = float(threshold)
        self.busy_samples = int(busy_samples)
        self.quiet_s = 60.0 * float(quiet_minutes)
        self._sample = sample
        self._busy_run = 0

    def step_aside(self, own: Iterable[int]) -> Optional[str]:
        """Takes one sample. Returns why the run should step aside now (a
        process and its use), or None."""
        pid, pct = busiest_other(self._sample(), own)
        if pid is not None and pct >= self.threshold:
            self._busy_run += 1
            if self._busy_run >= self.busy_samples:
                self._busy_run = 0
                return f"pid {pid} at {pct:.0f}% of a GPU engine"
        else:
            self._busy_run = 0
        return None

    def wait_until_quiet(self, own: Callable[[], Iterable[int]], every_s: float = 30.0,
                         log: Callable[[str], None] = print, clock=time.monotonic,
                         sleep=time.sleep) -> None:
        """Returns once no process outside `own()` has reached the threshold
        for quiet_minutes."""
        quiet_since = clock()
        while clock() - quiet_since < self.quiet_s:
            sleep(every_s)
            pid, pct = busiest_other(self._sample(), own())
            if pid is not None and pct >= self.threshold:
                if clock() - quiet_since > every_s:
                    log(f"the GPU is in use again (pid {pid} at {pct:.0f}%): waiting")
                quiet_since = clock()
