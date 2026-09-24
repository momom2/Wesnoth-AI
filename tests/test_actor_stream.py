"""Continuous generation (tools/actor_stream.py) on the REAL stream
code with fake queues and fake processes -- no multiprocessing, so the
fast tier runs it -- plus the ServeGate the publications go through.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from test_actor_pool_watchdog import _FakeProc, _pool  # noqa: E402
from tools.actor_pool import (  # noqa: E402
    _CMD_DRAIN, _CMD_PLAY, _CMD_UPDATE, _R_DONE, _R_EXPS, _R_GAME, _R_OUTCOME,
    ActorFatalError,
)
from tools.actor_worker import _R_START  # noqa: E402
from tools.actor_stream import _window_delta  # noqa: E402
from tools.inference_seam import ServeGate  # noqa: E402


def _game(aid, g, decisions=4, t0=None, t1=None, outcome="o", exps=("e1", "e2"), dstats=None,
          tag=0):
    """The three messages an actor sends per completed game, played
    under the PLAY tagged `tag` (a stream's tag is 0 unless given)."""
    now = time.time()
    msgs = []
    if outcome is not None:
        msgs.append((_R_OUTCOME, aid, outcome))
    if exps:
        msgs.append((_R_EXPS, aid, list(exps)))
    msgs.append((_R_GAME, aid, (g, decisions, now - 10 if t0 is None else t0,
                                now - 1 if t1 is None else t1, dstats, tag)))
    return msgs


def _stream_pool(results, n=2, **kw):
    pool = _pool([_FakeProc(True, name=f"actor-{i}") for i in range(n)], results, **kw)
    pool._streaming = False
    pool._drain_grace = 5.0
    pool._policy._inference_model._decision_step = 0
    pool._policy._decision_step = 0
    return pool


def test_stream_collects_whole_games_and_keeps_the_queue_topped_up():
    results = _game(0, 0, decisions=5, tag=7) + _game(1, 1, decisions=3, outcome=None, exps=(),
                                                     dstats={"distill_x": 1.0}, tag=7)
    pool = _stream_pool(results)
    stream = pool.stream(base_seed=100, tag=7)
    stream.start()
    try:
        assert pool._streaming and pool._serving
        # One PLAY per actor, carrying the stream flag; two live actors
        # plus two ahead = four tickets, then one more per completion.
        play = pool._ctrl_qs[0]._items[0]
        assert play[0] == _CMD_PLAY and play[1] == 7 and play[12] is True
        assert [t[1] for t in pool._game_q._items] == [0, 1, 2, 3]
        window = stream.collect(2, timeout=5.0)
        assert [t[1] for t in pool._game_q._items[4:]] == [4, 5], "one new ticket per completion"
    finally:
        stream.stop(grace=0.5)
    assert [g.index for g in window.games] == [0, 1]
    assert window.outcomes == ["o"]                  # the failed game sent none
    assert window.experiences == ["e1", "e2"]
    assert window.decisions == 8
    assert pool.last_decisions == 8, "the anneal counter advanced by the window's decisions"
    assert pool.last_distill_stats == {"distill_x": 1.0}
    assert window.straddle_mean == 0.0 and window.straddled_share == 0.0
    assert pool.last_iteration_seconds == pytest.approx(window.seconds)
    assert pool.last_straddle_mean == 0.0
    # After stop: the queue is flushed and the pool is free again.
    assert pool._game_q._items == [] and not pool._streaming and not pool._serving


def test_a_game_counts_the_publications_it_straddled():
    pool = _stream_pool([])
    stream = pool.stream(base_seed=1)
    stream.start()
    try:
        t_start = time.time() - 5
        stream.publish(value_center=0.25, decision_step=42)
        stream.publish()
        t_end = time.time() + 5
        pool._result_q._items.extend(_game(0, 0, t0=t_start, t1=t_end))
        pool._result_q._items.extend(_game(1, 1, t0=time.time() + 1, t1=time.time() + 2))
        window = stream.collect(2, timeout=5.0)
    finally:
        stream.stop(grace=0.5)
    by_index = {g.index: g.straddled for g in window.games}
    assert by_index == {0: 2, 1: 0}
    assert window.straddle_max == 2 and window.straddled_share == 0.5
    # Every live actor got the update between games.
    for q in pool._ctrl_qs:
        updates = [c for c in q._items if c[0] == _CMD_UPDATE]
        assert updates[0] == (_CMD_UPDATE, 0.25, 42)
        assert len(updates) == 2


def test_wait_in_flight_reads_the_starts_and_holds_completed_games_for_collect():
    pool = _stream_pool([])
    stream = pool.stream(base_seed=1)
    stream.start()
    try:
        assert stream.in_flight() == {}
        with pytest.raises(RuntimeError, match=r"actors \[0, 1\] not inside a game"):
            stream.wait_in_flight(timeout=0.3)
        # Actor 0 starts game 0, finishes it and starts game 2; actor 1
        # first reports an iteration abandoned before the stream done
        # (it stays live), then starts game 1. The finished game is
        # read during the wait and must come out of the next collect.
        pool._result_q._items.append((_R_START, 0, (0, time.time() - 10, 0)))
        pool._result_q._items.extend(_game(0, 0))
        pool._result_q._items.append((_R_START, 0, (2, time.time(), 0)))
        pool._result_q._items.append((_R_DONE, 1, (3, None, 5)))
        with pytest.raises(RuntimeError, match=r"actors \[1\] not inside a game"):
            stream.wait_in_flight(timeout=0.3)
        assert stream._live == {0, 1}
        pool._result_q._items.append((_R_START, 1, (1, time.time(), 0)))
        flight = stream.wait_in_flight(timeout=5.0)
        assert {aid: g for aid, (g, _) in flight.items()} == {0: 2, 1: 1}
        window = stream.collect(1, timeout=5.0)
        assert [g.index for g in window.games] == [0]
        assert stream.in_flight() == flight, "a collect does not touch the flight"
        pool._result_q._items.extend(_game(1, 1))
        assert [g.index for g in stream.collect(1, timeout=5.0).games] == [1]
        assert set(stream.in_flight()) == {0}, "actor 1 is between games"
    finally:
        stream.stop(grace=0.5)


def test_a_window_returns_partial_on_timeout_or_raises_below_its_minimum():
    pool = _stream_pool(_game(0, 0))
    stream = pool.stream(base_seed=1)
    stream.start()
    try:
        window = stream.collect(3, timeout=0.3, min_games=1)
        assert len(window.games) == 1 and window.timed_out
        with pytest.raises(RuntimeError, match="fewer than 1"):
            stream.collect(1, timeout=0.3, min_games=1)
    finally:
        stream.stop(grace=0.5)


def test_dead_actors_are_dropped_or_abort_loudly():
    pool = _stream_pool(_game(0, 0), liveness_interval=0.0)
    pool._procs[1] = _FakeProc(False, exitcode=0, name="actor-1")
    stream = pool.stream(base_seed=1)
    stream.start()
    try:
        window = stream.collect(1, timeout=2.0)
        assert window.dropped_actors == [1]
        assert stream._live == {0}
    finally:
        stream.stop(grace=0.5)

    pool = _stream_pool(_game(0, 0), liveness_interval=0.0)
    pool._procs[1] = _FakeProc(False, exitcode=-9, name="actor-1")
    stream = pool.stream(base_seed=1)
    stream.start()
    with pytest.raises(ActorFatalError):
        stream.collect(1, timeout=2.0)
    stream.stop(grace=0.5)
    assert not pool._streaming


def test_stop_drains_the_actors_and_returns_what_completed_meanwhile():
    results = _game(0, 5) + [(_R_DONE, 0, (5, None, 0)), (_R_DONE, 1, (0, None, 0))]
    pool = _stream_pool(results)
    stream = pool.stream(base_seed=1)
    stream.start()
    tail = stream.stop(grace=5.0)
    assert [c for c in pool._ctrl_qs[0]._items if c[0] == _CMD_DRAIN] == [(_CMD_DRAIN,)]
    assert [g.index for g in tail.games] == [5]
    assert stream._live == set()
    assert pool._game_q._items == []
    # Idempotent, and the pool takes iterations again.
    assert stream.stop().games == []
    pool._result_q._items.append((_R_DONE, 0, (0, None, 3)))
    pool._result_q._items.append((_R_DONE, 1, (0, None, 3)))
    outcomes, exps = pool.run_iteration(3, games_per_iter=1, base_seed=1)
    assert outcomes == [] and exps == []


def test_a_stream_drops_the_games_of_an_iteration_that_ended_without_them():
    """CI 2026-09-24 (tests/test_serve_process.py): an iteration aborted
    with its games in flight, and what its actors sent afterwards --
    actor 0's finished game, actor 1's game crashed on the dead-server
    marker, their done reports -- was still on the result queue when the
    next stream opened. The two games filled the stream's first window
    before it had served a leaf. They carry the aborted iteration's tag
    and are dropped with their outcome and experiences."""
    now = time.time()
    stale = (_game(0, 1, outcome="stale", exps=("stale",), tag=2) + [(_R_DONE, 0, (6, None, 2))]
             + _game(1, 0, outcome=None, exps=(), tag=2) + [(_R_DONE, 1, (1, None, 2))])
    own = ([(_R_START, 0, (0, now, 3))] + _game(0, 0, tag=3)
           + [(_R_START, 1, (1, now, 3))] + _game(1, 1, tag=3))
    pool = _stream_pool(stale + own)
    harvested = []
    pool._policy.harvest_boundary_pairs = harvested.append
    stream = pool.stream(base_seed=5, tag=3)
    stream.start()
    try:
        window = stream.collect(2, timeout=5.0)
    finally:
        stream.stop(grace=0.5)
    assert [(g.actor, g.index) for g in window.games] == [(0, 0), (1, 1)]
    assert window.outcomes == ["o", "o"]
    assert window.experiences == ["e1", "e2", "e1", "e2"]
    assert harvested == [["e1", "e2"], ["e1", "e2"]], "a dropped game reached the harvest"


def test_iteration_and_stream_exclude_each_other():
    pool = _stream_pool([])
    stream = pool.stream(base_seed=1)
    stream.start()
    try:
        with pytest.raises(RuntimeError, match="stream is open"):
            pool.run_iteration(0, games_per_iter=1, base_seed=1)
        with pytest.raises(RuntimeError, match="already open"):
            stream.start()
    finally:
        stream.stop(grace=0.5)
    pool._serving = True
    with pytest.raises(RuntimeError, match="during an iteration"):
        pool.stream(base_seed=1).start()


def test_window_delta_diffs_counters_and_cuts_the_timeline():
    prev = {"leaves": 10, "batches": 2, "wait": 1.5, "timeline": [(1.0, 5), (2.0, 10)]}
    cur = {"leaves": 25, "batches": 5, "wait": 4.0, "timeline": [(1.0, 5), (2.0, 10), (3.5, 25)],
           "error": "boom"}
    d = _window_delta(cur, prev, t0=3.0)
    assert d["leaves"] == 15 and d["batches"] == 3 and d["wait"] == pytest.approx(2.5)
    # The window's timeline starts at zero leaves at t0 and counts from
    # the previous snapshot's total, so the merged rate over the window
    # never sees the thread's cumulative count as a jump.
    assert d["timeline"] == [(3.0, 0), (3.5, 15)] and d["error"] == "boom"
    first = _window_delta(cur, None, t0=0.0)
    assert first["leaves"] == 25 and first["timeline"] == [(0.0, 0), (1.0, 5), (2.0, 10), (3.5, 25)]


def test_serve_gate_readers_overlap_and_a_writer_excludes_them():
    gate = ServeGate()
    both_in = threading.Barrier(3, timeout=5.0)     # two readers and the test
    release = threading.Event()
    order = []

    def reader(name):
        with gate.shared():
            order.append(f"{name} in")
            both_in.wait()          # both readers hold the gate at once
            release.wait(5.0)
            order.append(f"{name} out")

    readers = [threading.Thread(target=reader, args=(n,)) for n in ("r1", "r2")]
    for th in readers:
        th.start()
    both_in.wait()

    wrote = threading.Event()

    def writer():
        with gate.exclusive():
            order.append("writer")
            wrote.set()

    w = threading.Thread(target=writer)
    w.start()
    time.sleep(0.1)
    assert not wrote.is_set(), "the writer must wait for the readers in flight"
    # A pending writer blocks NEW readers, so it cannot be starved.
    late_in = threading.Event()

    def late_reader():
        with gate.shared():
            late_in.set()

    late = threading.Thread(target=late_reader)
    late.start()
    time.sleep(0.1)
    assert not late_in.is_set()
    release.set()
    for th in readers:
        th.join(5.0)
    w.join(5.0)
    late.join(5.0)
    assert wrote.is_set() and late_in.is_set()
    assert order.index("writer") > max(order.index("r1 out"), order.index("r2 out"))
