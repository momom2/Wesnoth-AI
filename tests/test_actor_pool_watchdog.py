#!/usr/bin/env python3
"""Watchdog tests for ActorPool.run_iteration (2026-06-29 review, A4).

A hard-crashed actor never sends its _R_DONE finally, so the serve
loop would otherwise spin forever with no progress and no error
(violating CLAUDE principle #5). These tests drive the REAL
run_iteration serve loop with fake queues + fake procs -- NO real
multiprocessing, so they're safe in the pytest sweep (actor_pool's
own smoke is deliberately standalone for that reason).
"""
from __future__ import annotations

import queue as _queue
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.actor_pool import (   # noqa: E402
    ActorPool, _R_DONE, _R_EXPS, _R_OUTCOME,
)


class _FakeQ:
    def __init__(self, items=None):
        self._items = list(items or [])

    def put(self, x):
        self._items.append(x)

    def get_nowait(self):
        if not self._items:
            raise _queue.Empty
        return self._items.pop(0)

    def get(self, timeout=None):   # noqa: ARG002 - fake never blocks
        if not self._items:
            raise _queue.Empty
        return self._items.pop(0)


class _FakeProc:
    def __init__(self, alive, exitcode=None, name="actor"):
        self._alive = alive
        self.exitcode = exitcode
        self.name = name

    def is_alive(self):
        return self._alive


def _pool(procs, results, *, iteration_timeout=1800.0, liveness_interval=0.0):
    pool = ActorPool.__new__(ActorPool)
    pool._n = len(procs)
    pool._started = True
    pool._policy = SimpleNamespace(_inference_encoder=SimpleNamespace(
        unit_type_to_id={}, faction_to_id={}))
    pool._iteration_timeout = iteration_timeout
    pool._liveness_interval = liveness_interval
    pool._max_batch = 8
    pool._serve_timeout = 0.0
    pool._serve_threads = 1
    pool._ctrl_qs = [_FakeQ() for _ in procs]
    pool._resp_qs = [_FakeQ() for _ in procs]
    pool._req_q = _FakeQ()              # always empty -> idle path
    pool._result_q = _FakeQ(results)
    pool._procs = list(procs)
    return pool


def test_dead_actor_is_dropped_not_wedged():
    """Actor 1 crashes (is_alive False) and never reports done. The
    loop must drop it and return actor 0's partial results instead of
    spinning forever."""
    procs = [_FakeProc(True, name="actor-0"),
             _FakeProc(False, exitcode=-9, name="actor-1")]
    results = [
        (_R_OUTCOME, 0, "game0"),
        (_R_EXPS, 0, ["e0"]),
        (_R_DONE, 0, None),
        # actor 1 sends nothing -- it "crashed"
    ]
    pool = _pool(procs, results)
    outcomes, experiences = pool.run_iteration(0, games_per_iter=2, base_seed=1)
    assert outcomes == ["game0"]
    assert experiences == ["e0"]


def test_wall_clock_deadline_breaks_out():
    """Both actors hang (alive, never done, no requests). The wall-clock
    deadline must break the loop rather than wedge."""
    procs = [_FakeProc(True), _FakeProc(True)]
    pool = _pool(procs, results=[], iteration_timeout=0.0)
    outcomes, experiences = pool.run_iteration(1, games_per_iter=2, base_seed=1)
    assert outcomes == []
    assert experiences == []


def test_all_actors_done_normal_path():
    """Sanity: when every actor reports done, the loop returns all
    results and the watchdog never fires."""
    procs = [_FakeProc(True), _FakeProc(True)]
    results = [
        (_R_OUTCOME, 0, "g0"), (_R_DONE, 0, None),
        (_R_OUTCOME, 1, "g1"), (_R_EXPS, 1, ["e1"]), (_R_DONE, 1, None),
    ]
    pool = _pool(procs, results)
    outcomes, experiences = pool.run_iteration(2, games_per_iter=2, base_seed=1)
    assert sorted(outcomes) == ["g0", "g1"]
    assert experiences == ["e1"]
