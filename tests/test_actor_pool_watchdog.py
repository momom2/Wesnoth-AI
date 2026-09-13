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


def _pool(procs, results, *, iteration_timeout=1800.0,
          drain_grace=1800.0, liveness_interval=0.0):
    pool = ActorPool.__new__(ActorPool)
    pool._n = len(procs)
    pool._started = True
    # A bare inference model: no weights version, no compile stats
    # (the pool reads both through getattr defaults).
    pool._policy = SimpleNamespace(
        _inference_model=SimpleNamespace(),
        _inference_encoder=SimpleNamespace(unit_type_to_id={}, faction_to_id={}))
    pool._iteration_timeout = iteration_timeout
    pool._drain_grace = drain_grace
    pool._liveness_interval = liveness_interval
    pool._max_batch = 8
    pool._serve_timeout = 0.0
    pool._serve_threads = 1
    pool._coalesce = "fifo"
    pool._coalesce_gap = 0
    pool.value_center = 0.0     # MCTSConfig.value_center broadcast (az4)
    pool.server_priors = False  # server-side priors flag (plan 1.3)
    pool._ctrl_qs = [_FakeQ() for _ in procs]
    pool._game_q = _FakeQ()
    pool._resp_qs = [_FakeQ() for _ in procs]
    pool._req_qs = [_FakeQ()]           # always empty -> idle path
    pool._server = None                 # never asked (no requests)
    pool._result_q = _FakeQ(results)
    pool._procs = list(procs)
    # No serve processes: the learner process is the only server.
    pool._serve_processes = 1
    pool._server_procs = []
    pool._server_ctrl_qs = []
    pool._server_versions = []
    pool._server_q = _FakeQ()
    pool._serving = False
    pool.last_serve_thread_errors = []
    pool.last_stuck_serve_threads = []
    pool._stuck_serve_threads = []
    return pool


def test_dead_actor_with_nonzero_exit_aborts_loudly():
    """Round-35 C0 reversal of the old drop-and-continue: an actor
    killed before its `finally` ran (segfault, OOM-kill, guard trip)
    must ABORT the iteration -- the silent drop hid real deaths
    from the run's exit code. The loop still must not wedge: the
    abort is how it terminates."""
    import pytest
    from tools.actor_pool import ActorFatalError
    procs = [_FakeProc(True, name="actor-0"),
             _FakeProc(False, exitcode=-9, name="actor-1")]
    results = [
        (_R_OUTCOME, 0, "game0"),
        (_R_EXPS, 0, ["e0"]),
        (_R_DONE, 0, None),
        # actor 1 sends nothing -- it "crashed"
    ]
    pool = _pool(procs, results)
    with pytest.raises(ActorFatalError):
        pool.run_iteration(0, games_per_iter=2, base_seed=1)


def test_dead_actor_with_clean_exit_is_dropped():
    """A dead actor whose exitcode is 0 (finished, done message
    lost) keeps the old resilience: dropped, partial results
    returned."""
    procs = [_FakeProc(True, name="actor-0"),
             _FakeProc(False, exitcode=0, name="actor-1")]
    results = [
        (_R_OUTCOME, 0, "game0"),
        (_R_EXPS, 0, ["e0"]),
        (_R_DONE, 0, None),
    ]
    pool = _pool(procs, results)
    outcomes, experiences = pool.run_iteration(0, games_per_iter=2, base_seed=1)
    assert outcomes == ["game0"]
    assert experiences == ["e0"]


def test_wall_clock_deadline_breaks_out():
    """Both actors hang (alive, never done, no requests). With zero
    drain grace, the hard deadline must break the loop rather than
    wedge."""
    procs = [_FakeProc(True), _FakeProc(True)]
    pool = _pool(procs, results=[], iteration_timeout=0.0,
                 drain_grace=0.0)
    outcomes, experiences = pool.run_iteration(1, games_per_iter=2, base_seed=1)
    assert outcomes == []
    assert experiences == []


def test_drain_grace_keeps_late_results():
    """Soft deadline fires immediately, but the actors' results are
    already queued -- the drain-grace window must collect them and
    finish cleanly with nothing abandoned (the leg-3 waste mode was
    discarding exactly these nearly-done games)."""
    procs = [_FakeProc(True), _FakeProc(True)]
    results = [
        (_R_OUTCOME, 0, "g0"), (_R_DONE, 0, None),
        (_R_OUTCOME, 1, "g1"), (_R_EXPS, 1, ["e1"]), (_R_DONE, 1, None),
    ]
    pool = _pool(procs, results, iteration_timeout=0.0,
                 drain_grace=3600.0)
    outcomes, experiences = pool.run_iteration(3, games_per_iter=2, base_seed=1)
    assert sorted(outcomes) == ["g0", "g1"]
    assert experiences == ["e1"]
    assert pool._last_abandoned == 0


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


def test_tickets_are_shared_and_stale_ones_skipped():
    """The manager posts one ticket per game and an end marker per
    actor; an actor takes games in order, skips another iteration's
    leftovers, stops at the end marker, and honours DRAIN and STOP
    while waiting."""
    from tools.actor_worker import _CMD_DRAIN, _CMD_STOP, _TICKET_END, _take_ticket
    pool = _pool([_FakeProc(True), _FakeProc(True)], results=[])
    pool._post_tickets(iter_idx=4, games_per_iter=3, base_seed=100)
    posted = list(pool._game_q._items)
    assert posted[:3] == [(4, 0, 100), (4, 1, 100 + 1_000_003), (4, 2, 100 + 2 * 1_000_003)]
    assert posted[3:] == [(4, _TICKET_END, None)] * 2
    game_q = _FakeQ([(3, 7, 1)] + posted)          # a stale ticket first
    ctrl = _FakeQ()
    assert _take_ticket(game_q, ctrl, 4) == ("game", (0, 100))
    assert _take_ticket(game_q, ctrl, 4) == ("game", (1, 100 + 1_000_003))
    ctrl.put((_CMD_DRAIN,))
    assert _take_ticket(game_q, ctrl, 4) == ("drain", None)
    assert _take_ticket(game_q, ctrl, 4) == ("game", (2, 100 + 2 * 1_000_003))
    assert _take_ticket(game_q, ctrl, 4) == ("end", None)
    ctrl.put((_CMD_STOP,))
    assert _take_ticket(game_q, ctrl, 4) == ("stop", None)
    # Whatever is left after the iteration is flushed, never inherited.
    assert pool._flush_tickets() == 5
    assert pool._flush_tickets() == 0
