"""Where a game's decision time goes: per-stage wall clock, always on.

A game loop owns a `StageTimer` and makes it current for its thread
(`timing(timer)`); the code a decision runs through records its stages
into whatever timer is current (`stage(name)`, `add(name, seconds)`), so
the encoder, the mask builder and the inference client need no timer
argument. With no current timer the hooks cost one thread-local read.

A timer keeps, per stage, the seconds spent and the number of entries,
and per decision the wall and the thread's CPU time: the gap between the
two is time the thread waited (for the inference server, or for the GIL
when several games share a process). Overhead per stage is two
`perf_counter` calls and a dict update, about a microsecond, against
milliseconds per decision.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_LOCAL = threading.local()


class StageTimer:
    """Seconds and entries per stage, and the decisions' wall and CPU."""

    __slots__ = ("seconds", "entries", "decisions", "decision_wall_s", "decision_cpu_s",
                 "_t_wall", "_t_cpu")

    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}
        self.entries: Dict[str, int] = {}
        self.decisions = 0
        self.decision_wall_s = 0.0
        self.decision_cpu_s = 0.0
        self._t_wall = 0.0
        self._t_cpu = 0.0

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.entries[name] = self.entries.get(name, 0) + 1

    def begin_decision(self) -> None:
        self._t_wall = time.perf_counter()
        self._t_cpu = time.thread_time()

    def end_decision(self) -> None:
        self.decisions += 1
        self.decision_wall_s += time.perf_counter() - self._t_wall
        self.decision_cpu_s += time.thread_time() - self._t_cpu

    def merge(self, other: "StageTimer") -> None:
        for name, s in other.seconds.items():
            self.seconds[name] = self.seconds.get(name, 0.0) + s
            self.entries[name] = self.entries.get(name, 0) + other.entries.get(name, 0)
        self.decisions += other.decisions
        self.decision_wall_s += other.decision_wall_s
        self.decision_cpu_s += other.decision_cpu_s

    def as_dict(self) -> Dict[str, object]:
        """Seconds rounded to the microsecond; read with `from_dict`."""
        return {"decisions": self.decisions,
                "decision_wall_s": round(self.decision_wall_s, 6),
                "decision_cpu_s": round(self.decision_cpu_s, 6),
                "seconds": {k: round(v, 6) for k, v in sorted(self.seconds.items())},
                "entries": dict(sorted(self.entries.items()))}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "StageTimer":
        t = cls()
        t.decisions = int(d.get("decisions", 0))
        t.decision_wall_s = float(d.get("decision_wall_s", 0.0))
        t.decision_cpu_s = float(d.get("decision_cpu_s", 0.0))
        t.seconds = {k: float(v) for k, v in dict(d.get("seconds", {})).items()}
        t.entries = {k: int(v) for k, v in dict(d.get("entries", {})).items()}
        return t


def current() -> Optional[StageTimer]:
    """The timer this thread records into, or None."""
    return getattr(_LOCAL, "timer", None)


@contextmanager
def timing(timer: Optional[StageTimer]) -> Iterator[Optional[StageTimer]]:
    """Make `timer` this thread's current timer for the block."""
    before = getattr(_LOCAL, "timer", None)
    _LOCAL.timer = timer
    try:
        yield timer
    finally:
        _LOCAL.timer = before


class stage:
    """`with stage("encode"): ...` records the block's wall seconds into
    the current timer, when there is one."""

    __slots__ = ("_name", "_timer", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name
        self._timer = getattr(_LOCAL, "timer", None)
        self._t0 = 0.0

    def __enter__(self) -> "stage":
        if self._timer is not None:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        if self._timer is not None:
            self._timer.add(self._name, time.perf_counter() - self._t0)
        return False


__all__ = ["StageTimer", "current", "timing", "stage"]
