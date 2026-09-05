"""_ParallelStream's consumer logic on stub workers: it must terminate
when encode workers die without their ("worker_exit",) message
(BACKLOG item 1, 2026-08-10), and it must deliver files in dispatch
order whatever order the workers finish in (2026-09-05: --seed did not
reproduce the pair stream under --workers > 0).

The 2026-08-08 imitation run hung silently at 94% of the epoch: the
consumer's bare blocking out_q.get() waits forever once a worker has
been OOM-killed/segfaulted (its exit message never arrives, so
_workers_alive stays overcounted after every healthy worker retires).
These tests drive the REAL __next__ on a stream whose worker procs
are stubs, so both properties are pinned without spawning processes
(worker spawn costs ~10s+ each under Windows spawn).
"""
from __future__ import annotations

import queue
import random
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


def _stub_stream(procs, alive_count, preload=(), files=(), dispatched=None):
    """A _ParallelStream with stubbed workers/queues but the real
    consumer logic (__next__/refill/reorder untouched). `files` is the
    dispatch order; `dispatched` how many of them are already out
    (default: all)."""
    s = _ParallelStream.__new__(_ParallelStream)
    s._closed = False
    s._files = [Path(f) for f in files]
    s._workers_n = len(procs)
    s._workers_alive = alive_count
    s._get_timeout = 0.05
    s._procs = list(procs)
    s._in_q = queue.Queue()
    s._out_q = queue.Queue()
    s._init_consumer_state()
    s._next_file = len(s._files) if dispatched is None else dispatched
    for item in preload:
        s._out_q.put(item)
    return s


def _file_msg(seq: int, name: str, n_pairs: int = 1):
    return ("file", seq, [(f"raw{seq}_{k}", f"ai{seq}_{k}") for k in range(n_pairs)], name)


def _drain_files(s, stop_after: int):
    """File names in the order the stream emits them, until
    `stop_after` file_done markers (or file_error events) have passed."""
    order = []
    while len(order) < stop_after:
        ev = next(s)
        if ev[0] in ("file_done", "file_error"):
            order.append(ev[1])
    return order


def _queued(q) -> list:
    return list(q.queue)


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
                     preload=[_file_msg(0, "g.json.gz")], files=["g.json.gz"])
    kind, raw, ai, name = next(s)
    assert (kind, raw, ai, name) == ("pair", "raw0_0", "ai0_0", "g.json.gz")
    assert next(s)[0] == "file_done"
    # Queue now empty; next call times out, reconciles the corpse
    # (1 dead > 0 counted exits), then keeps waiting on the live
    # worker -- feed its exit so the stream can finish.
    s._out_q.put(("worker_exit",))
    live._alive = False
    with pytest.raises(StopIteration):
        next(s)
    assert s._workers_alive <= 0


def test_files_are_emitted_in_dispatch_order_and_refilled_per_emit():
    """Workers finish 2, 0, 3, 1: the trainer sees 0, 1, 2, 3 (pairs
    and file_done markers included), and each emitted file dispatches
    exactly one more file, tagged with its dispatch index."""
    files = [f"f{i}.json.gz" for i in range(8)]
    procs = [_StubProc(True), _StubProc(True)]
    s = _stub_stream(procs, alive_count=2, files=files, dispatched=4,
                     preload=[_file_msg(2, files[2], 2), _file_msg(0, files[0], 1),
                              _file_msg(3, files[3], 1), ("file_error", 1, files[1], "boom")])
    events = [next(s) for _ in range(7)]      # 2 + 1 pairs, 3 file_done, 1 file_error
    assert [e[0] for e in events] == ["pair", "file_done", "file_error", "pair", "pair",
                                      "file_done", "pair"]
    assert events[0][3] == files[0] and events[2][1] == files[1]
    assert [e[3] for e in events if e[0] == "pair"] == [files[0], files[2], files[2], files[3]]
    assert next(s) == ("file_done", files[3], 1)
    # Four emitted files, four dispatched; the last dispatch exhausted
    # the list, so the workers' sentinels follow.
    assert _queued(s._in_q) == [(4, Path(files[4])), (5, Path(files[5])),
                                (6, Path(files[6])), (7, Path(files[7])), None, None]
    assert s._next_file == 8 and not s._reorder


def test_same_dispatch_order_gives_same_stream_under_any_completion_order():
    """What --seed promises: the emitted file sequence is the seeded
    dispatch order, whichever permutation the workers complete in."""
    files = [f"g{i}.json.gz" for i in range(12)]
    random.Random(20260905).shuffle(files)
    orders = []
    for arrival_seed in (1, 2):
        arrival = list(range(len(files)))
        random.Random(arrival_seed).shuffle(arrival)
        s = _stub_stream([_StubProc(True)] * 3, alive_count=3, files=files,
                         preload=[_file_msg(k, files[k]) for k in arrival]
                         + [("worker_exit",)] * 3)
        orders.append(_drain_files(s, len(files)))
        with pytest.raises(StopIteration):
            next(s)
    assert orders[0] == orders[1] == files


def test_reorder_buffer_holds_at_most_the_priming_depth():
    """A slow head file: everything else completes and is buffered, but
    no new file is dispatched until the head is emitted, so the
    buffer never exceeds workers * 2 - 1 replays. When the head
    lands, the buffered files drain in order and the refills follow."""
    workers = 3
    files = [f"h{i}.json.gz" for i in range(20)]
    primed = workers * 2
    s = _stub_stream([_StubProc(True)] * workers, alive_count=workers,
                     files=files, dispatched=primed,
                     preload=[_file_msg(k, files[k]) for k in range(primed - 1, 0, -1)])
    s._get_timeout = 0.01
    seen = {}

    def head_lands_on_first_timeout():
        # Every other primed file has been pulled and buffered, nothing
        # was emitted, and no refill went out: the workers wait.
        seen["buffer"] = len(s._reorder)
        seen["in_q"] = _queued(s._in_q)
        s._out_q.put(_file_msg(0, files[0]))
    s._reconcile_corpses = head_lands_on_first_timeout
    t0 = time.perf_counter()
    assert _drain_files(s, primed) == files[:primed]
    assert time.perf_counter() - t0 < 5.0
    assert seen == {"buffer": primed - 1, "in_q": []}
    # Each emitted file paid one refill: the next six files went out,
    # nothing beyond the priming depth.
    assert _queued(s._in_q) == [(k, Path(files[k])) for k in range(primed, 2 * primed)]
    assert not s._reorder


def test_lost_head_does_not_stall_and_buffered_files_still_arrive():
    """A worker dies holding seq 0. The buffered later files must be
    delivered, the pass continues in arrival order, the dead worker's
    refill is dispatched anyway, and the stream still terminates."""
    files = [f"k{i}.json.gz" for i in range(5)]
    live = _StubProc(True)
    s = _stub_stream([_StubProc(False), live], alive_count=2, files=files, dispatched=4,
                     preload=[_file_msg(2, files[2]), _file_msg(1, files[1])])
    assert _drain_files(s, 2) == [files[1], files[2]]
    assert not s._ordered
    # One refill per emitted file plus one for the corpse's lost file.
    assert _queued(s._in_q)[:1] == [(4, Path(files[4]))]
    assert s._next_file == 5
    s._out_q.put(_file_msg(3, files[3]))
    s._out_q.put(_file_msg(4, files[4]))
    s._out_q.put(("worker_exit",))
    live._alive = False
    assert _drain_files(s, 2) == [files[3], files[4]]
    with pytest.raises(StopIteration):
        next(s)
    assert s._workers_alive <= 0 and s._sentinels_sent == 2


def test_fewer_files_than_workers_retires_every_worker():
    """One file, three workers: the sentinels are not tied to refills,
    so all three workers get one and the stream ends."""
    files = ["only.json.gz"]
    s = _stub_stream([_StubProc(True)] * 3, alive_count=3, files=files,
                     preload=[_file_msg(0, files[0])] + [("worker_exit",)] * 3)
    assert _drain_files(s, 1) == files
    assert _queued(s._in_q).count(None) == 3
    with pytest.raises(StopIteration):
        next(s)
