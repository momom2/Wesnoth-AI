"""Regression: _ParallelStream must terminate when encode workers die
without their ("worker_exit",) message (BACKLOG item 1, 2026-08-10).

The 2026-08-08 imitation run hung silently at 94% of the epoch: the
consumer's bare blocking out_q.get() waits forever once a worker has
been OOM-killed/segfaulted (its exit message never arrives, so
_workers_alive stays overcounted after every healthy worker retires).
These tests drive the REAL __next__ on a stream whose worker procs
are stubs, so the reconciliation path is pinned without spawning
processes (worker spawn costs ~10s+ each under Windows spawn).
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.supervised_train import _ParallelStream  # noqa: E402


class _StubProc:
    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False

    def join(self, timeout=None) -> None:
        pass


def _stub_stream(procs, alive_count, preload=()):
    """A _ParallelStream with stubbed workers/queues but the real
    consumer logic (__next__/close/_refill_input untouched)."""
    s = _ParallelStream.__new__(_ParallelStream)
    s._closed = False
    s._pair_iter = None
    s._current_gz = None
    s._pending_file_done = None
    s._files = []
    s._next_file = 0
    s._workers_n = len(procs)
    s._workers_alive = alive_count
    s._get_timeout = 0.05
    s._procs = list(procs)
    s._in_q = queue.Queue()
    s._out_q = queue.Queue()
    for item in preload:
        s._out_q.put(item)
    return s


def test_dead_workers_reconciled_instead_of_hanging():
    """Two workers dead, zero exit messages: before the fix this
    blocked forever on out_q.get(); now it must reconcile the corpses
    and raise StopIteration promptly."""
    s = _stub_stream([_StubProc(False), _StubProc(False)], alive_count=2)
    t0 = time.perf_counter()
    with pytest.raises(StopIteration):
        next(s)
    assert time.perf_counter() - t0 < 5.0, "reconciliation took too long"
    assert s._workers_alive <= 0
    assert s._closed


def test_clean_exits_still_terminate():
    """The normal path: every worker sends worker_exit; no
    reconciliation involved."""
    s = _stub_stream([_StubProc(False), _StubProc(False)], alive_count=2,
                     preload=[("worker_exit",), ("worker_exit",)])
    with pytest.raises(StopIteration):
        next(s)
    assert s._workers_alive == 0


def test_mixed_corpse_and_late_message():
    """One worker dead without a message, one alive whose 'file'
    message lands after a timeout cycle: the corpse is reconciled,
    the live worker's data still comes through."""
    live = _StubProc(True)
    s = _stub_stream([_StubProc(False), live], alive_count=2,
                     preload=[("file", [("raw0", "ai0")], "g.json.gz")])
    kind, raw, ai, name = next(s)
    assert (kind, raw, ai, name) == ("pair", "raw0", "ai0", "g.json.gz")
    assert next(s)[0] == "file_done"
    # Queue now empty; next call times out, reconciles the corpse
    # (1 dead > 0 counted exits), then keeps waiting on the live
    # worker -- feed its exit so the stream can finish.
    s._out_q.put(("worker_exit",))
    live._alive = False
    with pytest.raises(StopIteration):
        next(s)
    assert s._workers_alive <= 0
