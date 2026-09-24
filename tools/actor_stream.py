"""Continuous generation on the actor pool.

The barrier iteration (`ActorPool.run_iteration`) hands every actor one
game and ends with the longest one; games finish around a median of
200 s while the longest takes 300-430 s, so for the second half of
every iteration fewer than half the actors are alive and the server
starves. Measured 2026-09-18 on three hosts, the iteration rate sits
1.37x, 1.6x, 1.84x and 1.93x below the saturated rate (docs/
box_specs.md "The graphed server on a quiet host"). That gap is the
tail, and this module removes it:

  * the pool's game queue is kept topped up, so an actor that finishes
    a game starts the next one at once;
  * the learner collects completed games in WINDOWS of G games
    (`collect`), steps on them, and publishes the new weights into the
    running servers (`publish`): the in-process server through the
    policy's own snapshot, every serve process through `sync_servers`,
    each under its ServeGate so no batch forwards through half a
    state_dict;
  * a game that started under one publication and ended under the next
    STRADDLES it. Its experiences mix the two policies' decisions; its
    outcome label is the game's, as always. Every completed game
    reports how many publications it straddled (`CompletedGame.
    straddled`), and every window its mean, its maximum and the share
    of games that straddled at all, so the learner's data regime is
    on the record. With as many actors as games per window a game
    straddles about one publication on average.

The actor-side contract (tools/actor_worker.py): one PLAY for the
whole stream with the `stream` flag; per game the actor reports
_R_START (index, start time, tag) as the game begins, then _R_OUTCOME,
_R_EXPS and _R_GAME (index, decisions, start and end times, its
drained distill stats, tag) as it ends, so the stream knows which games
are in flight (`in_flight`, `wait_in_flight`) and which games are its
own (the tag is the PLAY's); UPDATE between games carries the new
value center and the global anneal counter; DRAIN ends the stream, the
actor finishing its game and reporting done.

What the learner sees per step is a batch of G whole games, as under
the barrier, generated at the saturated rate. Whether games may
straddle weight updates is a property of the learner: the loop
records it, and `tools/az_loop.py --stream` is opt-in until a learner
that improves on the prior has been run both ways.
"""
from __future__ import annotations

import logging
import queue as _queue
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tools.actor_worker import (
    _CMD_DRAIN, _CMD_UPDATE, _R_DONE, _R_ERROR, _R_EXPS, _R_FATAL, _R_GAME, _R_OUTCOME,
    _R_START, _done_report,
)

log = logging.getLogger("actor_stream")


def _fatal_error_class():
    """tools.actor_pool.ActorFatalError, imported late: the pool module
    imports this one."""
    from tools.actor_pool import ActorFatalError as cls
    return cls


@dataclass
class CompletedGame:
    """One game the stream collected: its outcome (None when the actor's
    game failed and sent none), its experiences, and its report."""
    index: int
    actor: int
    outcome: object
    experiences: List
    decisions: int
    t_start: float
    t_end: float
    straddled: int              # publications between its start and its end
    distill_stats: Optional[Dict] = None

    @property
    def seconds(self) -> float:
        return self.t_end - self.t_start


@dataclass
class StreamWindow:
    """What one `collect` returned: the games, in completion order, and
    the window's own accounting. The pool's `last_*` readbacks hold
    the serving stats of the same span."""
    games: List[CompletedGame] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    dropped_actors: List[int] = field(default_factory=list)

    @property
    def outcomes(self) -> List:
        return [g.outcome for g in self.games if g.outcome is not None]

    @property
    def experiences(self) -> List:
        return [e for g in self.games for e in g.experiences]

    @property
    def decisions(self) -> int:
        return sum(g.decisions for g in self.games)

    @property
    def straddle_mean(self) -> Optional[float]:
        return statistics.fmean(g.straddled for g in self.games) if self.games else None

    @property
    def straddle_max(self) -> Optional[int]:
        return max((g.straddled for g in self.games), default=None)

    @property
    def straddled_share(self) -> Optional[float]:
        if not self.games:
            return None
        return sum(1 for g in self.games if g.straddled) / len(self.games)

    @property
    def game_seconds_p50(self) -> Optional[float]:
        secs = sorted(g.seconds for g in self.games)
        return secs[len(secs) // 2] if secs else None


class ActorStream:
    """One continuous-generation session on a started ActorPool. Use
    `start()`, then `collect()` / `publish()` in turns, then `stop()`;
    the pool's `shutdown()` still ends the processes."""

    def __init__(self, pool, base_seed: int, *, tag: int = 0,
                 tickets_ahead: Optional[int] = None):
        self._pool = pool
        self._seed = int(base_seed)
        self._tag = int(tag)
        # Unstarted games kept on the queue beyond the actors' own, so
        # no actor ever waits for a ticket. One per actor by default;
        # more only delays a DRAIN by games nobody needed.
        self._ahead = int(tickets_ahead) if tickets_ahead is not None else pool._n
        self._live = set(range(pool._n))    # actors still playing
        self._next_game = 0                 # global game index
        self._completed = 0                 # games reported (any outcome)
        self._publishes: List[float] = []   # time.time() of each publication
        self._pending: Dict[int, Dict] = {} # per actor: the game being reported
        self._in_flight: Dict[int, Tuple[int, float]] = {}  # per actor: (index, start)
        self._done: List[CompletedGame] = []  # completed, not yet cut into a window
        self._serving = None
        self._open = False
        self._draining = False
        self._windows = 0
        self._prev_snapshot: Optional[List[Dict]] = None
        self._prev_pick: Optional[Dict[str, int]] = None
        self._prev_leaves: Optional[List[int]] = None

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        pool = self._pool
        if self._open:
            raise RuntimeError("stream already open")
        if pool._serving:
            raise RuntimeError("stream start during an iteration")
        if pool._server_procs:
            pool._check_servers_synced(self._tag)
        pool._serving = True
        pool._streaming = True
        self._open = True
        self._serving = pool._start_serving(self._tag)
        self._top_up()
        for aid in sorted(self._live):
            pool._ctrl_qs[aid].put(pool._play_command(
                self._tag, 0, self._seed, aid, stream=True))
        self._prev_snapshot, self._prev_pick, self._prev_leaves = pool._serve_snapshot(
            self._serving)
        log.info(f"stream open: {len(self._live)} actors, {self._ahead} games queued "
                 f"ahead, tag {self._tag}")

    def stop(self, *, grace: Optional[float] = None) -> StreamWindow:
        """DRAIN every actor (finish the game in hand, take no new one),
        collect what completes within `grace` seconds (the pool's
        drain grace by default), stop serving, flush the queue. The
        games that completed meanwhile are returned; an actor still
        playing at the end of the grace is abandoned and logged."""
        pool = self._pool
        if not self._open:
            return StreamWindow()
        grace = pool._drain_grace if grace is None else float(grace)
        self._draining = True
        for aid in sorted(self._live):
            pool._ctrl_qs[aid].put((_CMD_DRAIN,))
        window = StreamWindow()
        outstanding = set(self._live)
        t0 = time.monotonic()
        try:
            while outstanding and time.monotonic() - t0 < grace:
                self._pump(outstanding, timeout=0.2)
            self._cut(window, None)
            if outstanding:
                log.error(f"stream stop: actors {sorted(outstanding)} still playing "
                          f"after the {grace:.0f}s drain grace; their games are "
                          f"abandoned")
        finally:
            self._close()
        window.seconds = time.monotonic() - t0
        log.info(f"stream closed: {self._completed} games in "
                 f"{self._windows} windows, {len(self._publishes)} publications, "
                 f"{len(window.games)} games collected during the drain")
        return window

    def _close(self) -> None:
        pool = self._pool
        if not self._open:
            return
        self._open = False
        try:
            pool._stop_serving(self._serving)
        finally:
            pool._streaming = False
            pool._serving = False
            n = pool._flush_tickets()
            if n:
                log.info(f"stream: {n} game tickets left unplayed")

    # -- the learner's turns ------------------------------------------

    def collect(self, n_games: int, *, timeout: Optional[float] = None,
                min_games: int = 1) -> StreamWindow:
        """Block until `n_games` more games have completed (serving
        goes on underneath) and return them. Past `timeout` seconds
        (the pool's iteration timeout by default) the window returns
        with what it has if that is at least `min_games`, else raises:
        a stream on which no game completes in that long is broken.
        The pool's `last_*` readbacks describe this window's serving."""
        pool = self._pool
        if not self._open:
            raise RuntimeError("collect() on a stream that is not open")
        timeout = pool._iteration_timeout if timeout is None else timeout
        window = StreamWindow()
        ds0 = pool._global_decision_step()
        t0 = time.monotonic()
        last_liveness = t0
        self._cut(window, n_games)          # games read during a wait_in_flight
        while len(window.games) < n_games:
            self._pump(set(), timeout=0.2)
            self._cut(window, n_games)
            now = time.monotonic()
            if now - last_liveness > pool._liveness_interval:
                last_liveness = now
                self._drop_dead(window)
                if not self._live:
                    raise RuntimeError("stream: every actor is gone")
            if timeout is not None and now - t0 > timeout:
                if len(window.games) >= min_games:
                    window.timed_out = True
                    log.warning(f"stream window {self._windows}: {len(window.games)} of "
                                f"{n_games} games after {timeout:.0f}s; returning them")
                    break
                raise RuntimeError(
                    f"stream: fewer than {min_games} game(s) completed in {timeout:.0f}s "
                    f"({len(self._live)} actors live)")
        window.seconds = time.monotonic() - t0
        self._windows += 1
        self._record_window(window, ds0, t0)
        return window

    def publish(self, *, value_center: Optional[float] = None,
                decision_step: Optional[int] = None) -> int:
        """After the learner's step: the serve processes load the
        learner's current inference weights (the in-process server
        already serves them, published under its gate), the actors
        get the new value center and the global anneal counter, and
        the publication is dated for the straddle count as it begins:
        the in-process server serves the snapshot already and the
        serve processes load theirs during the sync, so a game that
        ends while the sync runs may have played under both. Returns
        the weights version the servers hold."""
        pool = self._pool
        if not self._open:
            raise RuntimeError("publish() on a stream that is not open")
        self._publishes.append(time.time())
        version = pool.sync_servers()
        vc = float(pool.value_center if value_center is None else value_center)
        ds = int(pool._global_decision_step() if decision_step is None else decision_step)
        for aid in sorted(self._live):
            pool._ctrl_qs[aid].put((_CMD_UPDATE, vc, ds))
        return version

    def publications(self) -> List[float]:
        """time.time() of each publication so far, in order."""
        return list(self._publishes)

    def in_flight(self) -> Dict[int, Tuple[int, float]]:
        """Per live actor, the game it is inside as of the last message
        read: (game index, time.time() at its start). An actor between
        games (its report sent, its next start not yet stamped) is
        absent."""
        return dict(self._in_flight)

    def wait_in_flight(self, timeout: float) -> Dict[int, Tuple[int, float]]:
        """Read messages until every live actor is inside a game, then
        return `in_flight()`; a publication made right after straddles
        each of those games. Games that complete meanwhile wait for the
        next `collect`. Raises past `timeout` seconds."""
        if not self._open:
            raise RuntimeError("wait_in_flight() on a stream that is not open")
        t0 = time.monotonic()
        while set(self._in_flight) != set(self._live):
            if time.monotonic() - t0 > timeout:
                idle = sorted(set(self._live) - set(self._in_flight))
                raise RuntimeError(
                    f"stream: actors {idle} not inside a game after {timeout:.0f}s")
            self._pump(set(), timeout=0.2)
        return self.in_flight()

    def leaves_served(self) -> int:
        """Leaves the in-process server has served so far (its threads'
        live counters), for rates over a span the caller times, such
        as the learner's step."""
        return sum(int(s.get("leaves", 0)) for s in self._serving.serve_stats)

    # -- internals ----------------------------------------------------

    def _top_up(self) -> None:
        """Keep `len(live) + ahead` games posted beyond the completed
        ones: one in hand per actor and `ahead` waiting."""
        pool = self._pool
        if self._draining:
            return
        target = self._completed + len(self._live) + self._ahead
        while self._next_game < target:
            pool._game_q.put(pool._ticket(self._tag, self._next_game, self._seed))
            self._next_game += 1

    def _cut(self, window: StreamWindow, n_games: Optional[int]) -> None:
        """Move completed games into `window` up to `n_games` in it
        (all of them when None); the rest wait for the next window."""
        take = len(self._done) if n_games is None else max(0, n_games - len(window.games))
        window.games.extend(self._done[:take])
        del self._done[:take]

    def _pump(self, outstanding: set, *, timeout: float) -> bool:
        """One read of the result queue. A start report puts its actor
        in flight; a game's outcome and experiences wait for its closing
        report (`_close_game`); a done report retires its actor from
        `outstanding` (the drain) and from the live set. Returns True
        when a message was read."""
        pool = self._pool
        try:
            kind, aid, payload = pool._result_q.get(timeout=timeout)
        except _queue.Empty:
            if pool._server_procs:
                failed = pool._drain_server_replies()
                if failed:
                    pool._abort_on_dead_servers(self._tag, sorted(failed), failures=failed)
            return False
        if kind == _R_START:
            g, t_start, tag = payload
            if tag == self._tag:
                self._in_flight[aid] = (int(g), float(t_start))
        elif kind == _R_OUTCOME:
            self._pending.setdefault(aid, {})["outcome"] = payload
        elif kind == _R_EXPS:
            self._pending.setdefault(aid, {})["exps"] = list(payload)
        elif kind == _R_GAME:
            self._close_game(aid, payload)
        elif kind == _R_DONE:
            _n_dec, _dstats, done_iter = _done_report(payload)
            if done_iter is not None and done_iter != self._tag:
                # An iteration abandoned before this stream opened:
                # the actor reports it done as it takes the stream's
                # PLAY, and stays live here (the barrier path makes
                # the same distinction).
                log.warning(f"stream {self._tag}: actor {aid} reported iteration "
                            f"{done_iter} done (abandoned earlier); it stays live")
            else:
                outstanding.discard(aid)
                self._live.discard(aid)
                self._in_flight.pop(aid, None)
        elif kind == _R_ERROR:
            log.error(f"actor {aid} error:\n{payload}")
        elif kind == _R_FATAL:
            raise _fatal_error_class()(
                f"actor {aid} died on a non-swallowable error (round-35 C0):\n{payload}")
        if pool._server_procs:
            failed = pool._drain_server_replies()
            if failed:
                pool._abort_on_dead_servers(self._tag, sorted(failed), failures=failed)
        return True

    def _close_game(self, aid: int, report) -> None:
        """A game's closing report: with the outcome and experiences its
        actor sent before it, the game goes to `_done` for the next cut.
        A game played under another PLAY is dropped with them: an
        iteration that ended without collecting its games (an abort, a
        hard deadline) or a stream whose drain grace ran out leaves its
        actors' reports on the result queue, and its game indices are
        not this stream's (2026-09-24: an aborted iteration's two games
        filled a stream's first window before the stream served a leaf).
        The holdout probe and the boundary harvest see only accepted
        games."""
        pool = self._pool
        g, decisions, t_start, t_end, dstats, tag = report
        rec = self._pending.pop(aid, {})
        if tag != self._tag:
            log.warning(f"stream {self._tag}: actor {aid} reported game {g} of iteration "
                        f"{tag}, which ended without collecting it; dropped")
            return
        exps = rec.get("exps", [])
        if exps:
            offer = getattr(pool._policy, "offer_holdout_game", None)
            if offer is not None and offer(exps):
                exps = []
            else:
                harvest = getattr(pool._policy, "harvest_boundary_pairs", None)
                if harvest is not None:
                    harvest(exps)
        self._done.append(CompletedGame(
            index=int(g), actor=aid, outcome=rec.get("outcome"),
            experiences=exps, decisions=int(decisions or 0),
            t_start=float(t_start), t_end=float(t_end),
            straddled=sum(1 for t in self._publishes if t_start < t <= t_end),
            distill_stats=dstats))
        self._completed += 1
        self._in_flight.pop(aid, None)
        self._top_up()

    def _drop_dead(self, window: StreamWindow) -> None:
        dead = self._pool._scan_liveness(self._tag, set(self._live))
        for aid in sorted(dead):
            self._live.discard(aid)
            self._pending.pop(aid, None)
            self._in_flight.pop(aid, None)
            window.dropped_actors.append(aid)

    def _record_window(self, window: StreamWindow, ds0: int, t0: float) -> None:
        """The serving stats of this window: the difference between the
        snapshot at its end and the one at the last window's end, per
        serve thread, then the pool's shared recording."""
        pool = self._pool
        cur, pick, leaves = pool._serve_snapshot(self._serving)
        prev = self._prev_snapshot or []
        threads = [_window_delta(c, prev[i] if i < len(prev) else None, t0)
                   for i, c in enumerate(cur)]
        pick_delta = {k: v - (self._prev_pick or {}).get(k, 0) for k, v in pick.items()}
        leaves_delta = [n - (self._prev_leaves[i] if self._prev_leaves and i < len(self._prev_leaves) else 0)
                        for i, n in enumerate(leaves)]
        self._prev_snapshot, self._prev_pick, self._prev_leaves = cur, pick, leaves
        finish = [g.seconds for g in window.games]
        distill = [g.distill_stats for g in window.games if g.distill_stats]
        pool._record_serve_window(
            self._tag, threads, pick_delta, leaves_delta,
            self._serving.server_stats if self._serving.stopped else {},
            t_start=t0, elapsed=window.seconds, n_games=len(window.games),
            n_exps=len(window.experiences), ds0=ds0,
            total_decisions=window.decisions, finish_times=finish,
            distill_dicts=distill)
        pool.last_straddle_mean = window.straddle_mean
        pool.last_straddle_max = window.straddle_max
        pool.last_straddled_share = window.straddled_share
        if window.games:
            log.info(f"stream window {self._windows}: {len(window.games)} games in "
                     f"{window.seconds:.0f}s, game seconds p50 "
                     f"{window.game_seconds_p50:.0f}, straddled mean "
                     f"{window.straddle_mean:.2f} max {window.straddle_max} "
                     f"share {window.straddled_share:.2f}")


def _window_delta(cur: Dict, prev: Optional[Dict], t0: float) -> Dict:
    """One serve thread's stats over a window: counters as differences
    from the previous snapshot, the timeline restricted to the window
    and rebased to zero leaves at its start (the thread's count at the
    previous snapshot), the error tag carried. Without the rebase the
    merged timeline jumps by a thread's whole cumulative count at its
    first mark inside the window, and the best-window rate reads that
    jump as tens of thousands of leaves per second."""
    out: Dict = {}
    base = int((prev or {}).get("leaves", 0) or 0)
    for k, v in cur.items():
        if k == "timeline":
            out[k] = [(t0, 0)] + [(t, n - base) for t, n in v if t >= t0]
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v - ((prev or {}).get(k, 0) or 0)
        else:
            out[k] = v
    return out
