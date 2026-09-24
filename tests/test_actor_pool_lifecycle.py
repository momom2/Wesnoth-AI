#!/usr/bin/env python3
"""Lifecycle guards for the actor pool (2026-09-13 audit).

Four leaks, each one money on a rented box:

  * an orphaned actor (the learner killed with -9 / OOM-killed) never
    exits, because it inherits both ends of its own control queue and
    every wait in it was unbounded;
  * an actor the hard deadline abandoned stayed bound to the OLD
    iteration and silently ate the NEXT one's tickets, end markers
    included;
  * the iteration's serve threads outlived any iteration that ended on
    an unnamed error;
  * shutdown() left the processes unjoined and the queues open.

No real multiprocessing here: the tests drive the production functions
with plain queues, fake processes and a fake parent, so they are safe
(and fast) in the pytest sweep. The pool's own end-to-end smoke is
tests/test_actor_pool_smoke.py.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_actor_pool_watchdog import _FakeProc, _FakeQ, _pool  # noqa: E402
from tools import actor_worker  # noqa: E402
from tools.actor_pool import _R_DONE, _R_EXPS, _R_FATAL, ActorFatalError  # noqa: E402
from tools.actor_worker import (  # noqa: E402
    _CMD_PLAY, _TICKET_END, _IPCInferenceClient, _parent_gone, _take_ticket,
    _TicketSource, _wait_for_command,
)
from tools.serve_worker import _Waiting, _serve_loop  # noqa: E402


def _call_with_deadline(fn, seconds=5.0):
    """Run `fn` on a thread; return its result, or raise if it blocked.
    Every regression guarded here is an actor that never comes back,
    which must FAIL the test instead of hanging the suite."""
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    th.start()
    th.join(timeout=seconds)
    if th.is_alive():
        raise AssertionError("the actor never came back")
    return out["r"]


# ---------------------------------------------------------------- fix 1

class _DeadParent:
    @staticmethod
    def is_alive():
        return False


@pytest.fixture
def dead_parent(monkeypatch):
    """The actor's view of a learner that was killed."""
    monkeypatch.setattr(actor_worker.mp, "parent_process", lambda: _DeadParent())
    monkeypatch.setattr(actor_worker, "_PARENT_POLL", 0.05)


def test_parent_alive_in_the_main_process():
    """The guard must be inert where there is no parent -- every
    in-process test and the standalone smoke run that way."""
    assert _parent_gone() is False


def test_idle_actor_exits_when_the_parent_is_gone(dead_parent):
    """Between iterations an actor blocks on its control queue. Nothing
    ever closes that pipe (the actor holds the write end itself), so
    without the liveness check a killed learner leaves the actors
    running for the life of the box, holding the container's PID
    budget."""
    assert _call_with_deadline(lambda: _wait_for_command(_queue.Queue())) is None


def test_playing_actor_exits_when_the_parent_is_gone(dead_parent):
    """Same guard while waiting for a game ticket."""
    got = _call_with_deadline(
        lambda: _take_ticket(_queue.Queue(), _queue.Queue(), 0))
    assert got == ("stop", None)


def test_orphaned_actor_does_not_walk_the_queued_tickets(dead_parent):
    """With tickets still queued the guard must fire before the next
    one is taken, or the orphan plays on until the queue drains."""
    game_q = _queue.Queue()
    for g in range(5):
        game_q.put((0, g, 1000 + g))
    got = _call_with_deadline(lambda: _take_ticket(game_q, _queue.Queue(), 0))
    assert got == ("stop", None)
    assert game_q.qsize() == 5, "no ticket may be consumed by an orphan"


def test_inference_wait_gives_up_when_the_parent_is_gone(dead_parent):
    """An actor spends most of its cycle blocked on a reply, so that is
    where a killed learner most often catches it -- and the serve
    processes exit on the same guard, so no reply is ever coming."""
    client = _IPCInferenceClient(0, [_queue.Queue()], _queue.Queue())

    def _call():
        try:
            client.infer_batch([object()])
        except RuntimeError as e:
            return str(e)
        return "returned without raising"

    assert "learner process is gone" in _call_with_deadline(_call)


def test_parent_check_does_not_eat_replies(dead_parent):
    """The guard must not lose a reply that is already queued: only an
    EMPTY wait is allowed to conclude anything."""
    resp = _queue.Queue()
    resp.put((0, []))
    client = _IPCInferenceClient(0, [_queue.Queue()], resp)
    assert client.infer_batch([object()]) == []


# ---------------------------------------------------------------- fix 2

def _call_with_deadline(fn, seconds=5.0):
    """Run `fn` on a thread; return its result, or raise if it blocked.
    The regression this guards against is an actor that never returns,
    which must fail the test instead of hanging the suite."""
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    th.start()
    th.join(timeout=seconds)
    if th.is_alive():
        raise AssertionError("the actor never came back")
    return out["r"]


def test_abandoned_actor_resyncs_instead_of_eating_the_next_iteration():
    """The manager abandons an actor at the hard deadline and starts
    iteration N+1 while that actor is still bound to N. Bound to the
    old index it dropped every ticket of the new iteration -- end
    markers included, so the OTHER actors never finished either -- and
    came back only at the next soft deadline (demonstrated 2026-09-13).
    It must hand the PLAY to the main loop and stop instead."""
    ctrl = _queue.Queue()
    game_q = _queue.Queue()
    play = (_CMD_PLAY, 1, 4, 123, {}, {}, 0, False, 0.0, True, 0, False)
    ctrl.put(play)
    for g in range(4):
        game_q.put((1, g, 1000 + g))
    for _ in range(3):                      # one end marker per actor
        game_q.put((1, _TICKET_END, None))

    kind, handed = _call_with_deadline(
        lambda: _take_ticket(game_q, ctrl, iter_idx=0))

    assert (kind, handed) == ("play", play), "the PLAY must reach the main loop"
    assert game_q.qsize() == 7, "the stale actor consumed the new iteration's tickets"


def test_an_actor_bound_to_an_ended_iteration_holds_the_next_sessions_ticket():
    """The same actor when the next session's tickets reach it before
    its PLAY (they go out first, on another queue). It skipped them, and
    those games were lost for good: a stream that lost its first tickets
    this way kept an actor idle for the rest of it (CI loop, 2026-09-24).
    It ends its iteration and plays the ticket under that PLAY."""
    ctrl, game_q = _queue.Queue(), _queue.Queue()
    game_q.put((3, 0, 1000))                # the stream's; its PLAY is not here yet
    tickets = _TicketSource(game_q, ctrl)

    got = _call_with_deadline(lambda: tickets.take(2))

    assert got == ("next", (3, 0, 1000))
    assert tickets.take(3) == ("game", (0, 1000))


def test_done_for_an_abandoned_iteration_does_not_retire_the_actor():
    """That actor then reports the ABANDONED iteration done, while the
    manager is collecting the next one. Counting it there retires an
    actor that has played none of this iteration's games: the iteration
    ends early and every later one is shifted by one."""
    procs = [_FakeProc(True, name="actor-0"), _FakeProc(True, name="actor-1")]
    results = [
        (_R_DONE, 0, (7, {"et_k": 9.0}, 0)),   # stale: iteration 0's report
        (_R_DONE, 1, (1, None, 1)),
    ]
    # The deadline is long enough for both reports to be drained first,
    # then breaks the loop on whoever is left outstanding.
    pool = _pool(procs, results, iteration_timeout=0.3, drain_grace=0.0)
    pool._policy._decision_step = 0          # the anneal counter actors advance

    pool.run_iteration(1, games_per_iter=2, base_seed=1)

    assert pool._last_abandoned == 1, "the stale done retired actor 0"
    # Its decisions still count -- they were made, and the games behind
    # them reached this iteration's queue -- but its distilled means,
    # which describe another iteration, do not.
    assert pool.last_decisions == 8
    assert pool.last_distill_stats is None


def test_done_of_this_iteration_retires_the_actor():
    """The other half of the rule: a correctly dated done still counts
    (the check must not simply ignore every report)."""
    procs = [_FakeProc(True, name="actor-0")]
    results = [(_R_DONE, 0, (7, {"et_k": 9.0}, 3))]
    pool = _pool(procs, results, iteration_timeout=0.3, drain_grace=0.0)
    pool._policy._decision_step = 0

    pool.run_iteration(3, games_per_iter=1, base_seed=1)

    assert pool._last_abandoned == 0
    assert pool.last_decisions == 7
    assert pool.last_distill_stats == {"et_k": 9.0}


def test_legacy_done_payloads_still_retire_the_actor():
    """Older two-field and plain-int payloads carry no iteration; they
    must keep working (a manager restart meeting an old actor)."""
    procs = [_FakeProc(True, name="actor-0"), _FakeProc(True, name="actor-1")]
    results = [(_R_DONE, 0, (2, None)), (_R_DONE, 1, 3)]
    pool = _pool(procs, results)
    pool._policy._decision_step = 0

    pool.run_iteration(0, games_per_iter=2, base_seed=1)

    assert pool.last_decisions == 5


# ---------------------------------------------------------------- fix 3

class _Boom(Exception):
    pass


def _live_serve_threads():
    return {th for th in threading.enumerate()
            if th.name.startswith("serve-") and th.is_alive()}


def test_serve_threads_stop_on_an_unnamed_error_path():
    """`_stop_serving()` used to be called at four NAMED raise sites
    only. Anything else thrown inside the collection loop left the
    iteration's serve threads running on the request queue for the rest
    of the process's life -- one leaked thread per serve_threads per
    failed iteration."""
    before = _live_serve_threads()
    procs = [_FakeProc(True, name="actor-0")]
    pool = _pool(procs, results=[(_R_EXPS, 0, ["e0"])])

    def _explode(_payload):
        raise _Boom("holdout probe failed")

    pool._policy.offer_holdout_game = _explode

    with pytest.raises(_Boom):
        pool.run_iteration(0, games_per_iter=1, base_seed=1)

    leaked = _live_serve_threads() - before
    assert not leaked, f"serve thread(s) outlived the failed iteration: {leaked}"
    assert pool._serving is False


def test_an_aborted_iteration_leaves_no_tickets_behind():
    """The ticket flush sat after the iteration's `finally`, so an
    iteration that raised left its unplayed games and end markers on the
    game queue: the next PLAY of the same index plays those games again,
    and its actors stop at the stale end markers."""
    procs = [_FakeProc(True, name="actor-0"), _FakeProc(True, name="actor-1")]
    pool = _pool(procs, results=[(_R_FATAL, 1, "boom")])

    with pytest.raises(ActorFatalError):
        pool.run_iteration(2, games_per_iter=2, base_seed=1)

    assert pool._game_q._items == []


# ---------------------------------------------------------------- fix 4

class _FakeProcess:
    """A child that dies on join, or (when `stubborn`) only once it has
    been terminated."""

    def __init__(self, name: str, stubborn: bool = False):
        self.name = name
        self.stubborn = stubborn
        self.joins: list = []
        self.terminated = 0
        self.killed = 0
        self._alive = True

    def join(self, timeout=None):
        self.joins.append(timeout)
        if not self.stubborn or self.terminated:
            self._alive = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1


def _queue_pool(n: int = 2, tickets: int = 0):
    """A pool holding REAL mp queues (no children spawned) and fake
    processes, ready for shutdown()."""
    from tools.actor_pool import ActorPool
    ctx = mp.get_context("spawn")
    pool = ActorPool.__new__(ActorPool)
    pool._started = True
    pool._n = n
    pool._ctrl_qs = [ctx.Queue() for _ in range(n)]
    pool._resp_qs = [ctx.Queue() for _ in range(n)]
    pool._req_qs = [ctx.Queue()]
    pool._server_ctrl_qs = [ctx.Queue()]
    pool._result_q = ctx.Queue()
    pool._server_q = ctx.Queue()
    pool._game_q = ctx.Queue()
    for g in range(tickets):
        pool._game_q.put((0, g, g))
    pool._procs = [_FakeProcess(f"actor-{i}", stubborn=(i == 0)) for i in range(n)]
    pool._server_procs = [_FakeProcess("serve-1")]
    pool._open_serving = None
    return pool


def _all_queues(pool):
    return (list(pool._ctrl_qs) + list(pool._resp_qs) + list(pool._req_qs)
            + list(pool._server_ctrl_qs)
            + [pool._result_q, pool._server_q, pool._game_q])


def test_shutdown_joins_the_processes_and_closes_the_queues():
    """terminate() only signals; without the join the child stays a
    live process (and, on the box, a live CUDA context). And each
    mp.Queue holds a pipe pair plus a feeder thread -- ~2n+3 of each
    per pool, which the test suite feels first."""
    pool = _queue_pool(n=2)
    queues = _all_queues(pool)

    pool.shutdown(timeout=0.5)

    for q in queues:
        with pytest.raises(ValueError):
            q.put("after close")
        feeder = getattr(q, "_thread", None)
        assert feeder is None or not feeder.is_alive(), "feeder thread left running"
    assert pool._procs == [] and pool._server_procs == []
    assert pool._ctrl_qs == [] and pool._resp_qs == [] and pool._req_qs == []
    assert pool._started is False


def test_shutdown_terminates_and_then_joins_an_unresponsive_child():
    pool = _queue_pool(n=2)
    procs = list(pool._procs)

    pool.shutdown(timeout=0.5)

    stubborn, willing = procs[0], procs[1]
    assert stubborn.terminated == 1
    assert stubborn.joins == [0.5, 5.0], "terminate() was not followed by a join"
    assert stubborn.killed == 0
    assert willing.terminated == 0 and willing.joins == [0.5]


def test_shutdown_drains_leftover_tickets_without_blocking():
    """close() waits for the feeder to flush; a queue still full of
    tickets no living child will read would hang the join. The drain is
    what makes it safe."""
    pool = _queue_pool(n=2, tickets=500)
    game_q = pool._game_q

    pool.shutdown(timeout=0.5)

    with pytest.raises(ValueError):
        game_q.put("after close")


def test_shutdown_returns_with_big_messages_nobody_read():
    """The 500-small-tickets case above passes even with a NON-blocking
    drain, so it does not test the hazard.

    A `put()` only hands the object to the feeder thread; the bytes
    reach the pipe later. A `get_nowait()` can therefore miss a message
    still in flight, leave it in the pipe, and then the untimed
    `join_thread()` waits on a feeder blocked in `send_bytes` on a full
    pipe whose read end this process still holds -- no EPIPE, no
    return. It needs messages near the pipe buffer (8 KiB on Windows,
    64 KiB on Linux), which is what a real `_CMD_PLAY` is: about 6.6 KB
    once the reference checkpoint's vocab is in it.

    Reachable in production because a C-level-wedged actor is never
    removed from the broadcast, so it collects one unread PLAY per
    iteration and az_loop's `finally` then calls shutdown().

    Bounded by a deadline so a regression FAILS instead of hanging the
    suite.
    """
    payload = {"vocab": {f"unit_type_{i}": i for i in range(400)},
               "blob": "x" * 4096}
    pool = _queue_pool(n=2)
    for q in pool._ctrl_qs:
        for _ in range(24):
            q.put(("play", payload))

    done = threading.Event()
    err = []

    def _go():
        try:
            pool.shutdown(timeout=0.5)
        except Exception as exc:            # noqa: BLE001 -- reported below
            err.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    assert done.wait(20.0), (
        "shutdown() did not return: the drain lost the race with the feeder "
        "and join_thread() is waiting on a pipe nobody reads")
    assert not err, err


def test_shutdown_stops_open_serving_before_closing_its_queue():
    """CI 2026-09-24: a test raised with a stream open, and shutdown()
    closed the request queue under the stream's serve threads, which
    each died on the closed queue with a traceback in the log."""
    pool = _queue_pool(n=1)
    pool._server = None                 # never asked: no requests
    pool._serve_threads = 2
    pool._max_batch = 8
    pool._serve_timeout = 0.01
    pool._coalesce, pool._coalesce_gap = "fifo", 0
    pool._stuck_serve_threads = []
    sv = pool._start_serving(3)

    pool.shutdown(timeout=0.5)

    assert not any(th.is_alive() for th in sv.threads), "shutdown left serve threads running"
    for th in sv.threads:
        th.join(timeout=5.0)
    assert [s.get("error") for s in sv.serve_stats] == [None, None]


def test_shutdown_is_idempotent():
    pool = _queue_pool(n=1)
    pool.shutdown(timeout=0.5)
    pool.shutdown(timeout=0.5)          # must not touch the closed queues


# ---------------------------------------------------------------- fix 5

class _StubPicker:
    """Serves one batch, then whatever `take` is told to do."""

    def __init__(self, batches, parked=(), raise_on_take=None):
        self._batches = list(batches)
        self._parked = list(parked)
        self._raise = raise_on_take

    def take(self, queue, max_batch, timeout):    # noqa: ARG002 - stub
        if self._batches:
            return self._batches.pop(0)
        if self._raise is not None:
            raise self._raise
        return []

    def flush(self):
        parked, self._parked = self._parked, []
        return parked


def _waiting(aid: int, rid: int, payload) -> _Waiting:
    return _Waiting(item=(aid, rid, payload), seq=rid, n_leaves=1, lens=[1], tokens=1)


def test_dying_serve_thread_answers_its_requests_and_reports():
    """Only `server.infer_batch` sat inside a try: a throw anywhere else
    in the loop (the flattening here) killed the thread with its parked
    AND its in-flight requests unanswered -- their actors then blocked
    until the iteration's deadline -- and took its stats with it, so
    the manager could not even see that it was serving with one thread
    fewer."""
    resp = {0: _queue.Queue(), 1: _queue.Queue()}
    # item[2] is an int: the flattening step raises on it, outside the
    # inference try.
    picker = _StubPicker(batches=[[_waiting(0, 7, 5)]],
                         parked=[_waiting(1, 9, 5)])
    stats: list = []

    _serve_loop(server=None, picker=picker, req_q=_FakeQ(), resp_qs=resp,
                max_batch=8, serve_timeout=0.0, stop_ev=threading.Event(),
                stats_out=stats)

    assert resp[0].get_nowait() == (7, None), "the in-flight request went unanswered"
    assert resp[1].get_nowait() == (9, None), "the parked request went unanswered"
    assert len(stats) == 1, "the dead thread's stats never reached the manager"
    assert "TypeError" in stats[0]["error"]


def test_healthy_serve_thread_reports_no_error():
    """The tag must mean something: a thread that stops on its event
    reports clean stats."""
    stop_ev = threading.Event()
    stop_ev.set()
    stats: list = []

    _serve_loop(server=None, picker=_StubPicker(batches=[]), req_q=_FakeQ(),
                resp_qs={}, max_batch=8, serve_timeout=0.0, stop_ev=stop_ev,
                stats_out=stats)

    assert len(stats) == 1 and "error" not in stats[0]


def test_manager_surfaces_a_dead_serve_thread(monkeypatch):
    """And the manager reads the tag back off the stats instead of
    silently serving the rest of the campaign at half its threads."""
    import tools.actor_pool as ap

    def _dying_loop(server, picker, req_q, resp_qs, max_batch, serve_timeout,
                    stop_ev, stats_out):
        stats_out.append({"leaves": 3, "batches": 1, "error": "Traceback: boom"})

    monkeypatch.setattr(ap, "_serve_loop", _dying_loop)
    pool = _pool([_FakeProc(True, name="actor-0")],
                 results=[(_R_DONE, 0, (0, None, 0))])

    pool.run_iteration(0, games_per_iter=1, base_seed=1)

    assert pool.last_serve_thread_errors == ["Traceback: boom"]
