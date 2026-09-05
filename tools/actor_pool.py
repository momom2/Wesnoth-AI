"""Multiprocess actor pool + central inference server (plan §3.1b, B2).

SEED-RL / MonoBeast pattern, built on the B1 inference seam
(`tools/inference_seam.py`):

  * N WEIGHTLESS actor processes each run the pure-Python rollout
    (sim + encode_raw + MCTS bookkeeping) via a seam-backed MCTSPolicy.
    Every leaf forward is shipped as a RawEncoded to the central server
    and the actor blocks for the ModelOutput.
  * The central server is the MAIN (learner) process itself, during the
    rollout phase: it owns the single model on the GPU and dynamically
    BATCHES the forwards arriving from all actors (K actors blocked on
    inference => one batched GPU call of size up to K). Because actors
    hold no weights, there is NO weight sync -- the server always
    forwards with current weights; train_step mutates them in place
    between iterations.
  * Actors ship completed `MCTSExperience`s back; the main drains them
    into its policy queue and runs the existing train_step.

This is the GPU-feeding mechanism: a bare GPU on a CPU-rich host stays
busy because many CPU actors keep the inference batch full. The
intra-search Gumbel batching (plan §3.1a) is a separate, later lever
that enlarges a SINGLE actor's request; this pool enlarges the batch
ACROSS actors. They compose.

Determinism caveat: dynamic cross-actor batching makes the exact
forward grouping (and thus float-reduction order) nondeterministic, so
a pooled training run is NOT bit-reproducible. Eval and tests stay
serial + deterministic. Self-play data generation wants throughput and
diversity, not bit-reproducibility, so this is the right trade.

Vocab consistency: the actor's `encode_raw` and the server's
`encode_from_raw` must agree on the type/faction -> id mapping (the
embedding rows must line up). The current vocab snapshot is re-sent to
actors in every PLAY command, so they always match the server's
encoder; unseen names fall to the overflow bucket on BOTH sides
(consistent). Pre-seed the vocab broadly to minimize overflow
collisions.

Serve processes (2026-09-05): `serve_processes=N` adds N-1 serving
PROCESSES next to the learner's in-process serve threads, each holding
its own copy of the inference model on the same device with the same
switches, its own request queue and its own serve threads. Two serve
threads in one process share one GIL and spend ~33-36 ms of host work
per 16-leaf batch against ~29 ms of GPU work, so the host side was the
ceiling (docs/box_specs.md, "Serve thread host cost"). Actors are
assigned to a server round-robin at PLAY time (actor `aid` asks server
`aid % N`; server 0 is the learner process). The copies serve the
learner's CURRENT inference weights: after every publication the loop
calls `sync_servers()`, which ships the state as one torch.save byte
string on each server's control queue and waits for the acks;
`run_iteration` refuses to start while a server's weights version lags
the learner's (`WesnothModel._weights_version`, bumped by every
load_state_dict). Each server's serve stats merge into the iteration's
(the saturated rate covers all servers). A server death poisons its
actors' reply queues and aborts the iteration; shutdown stops them.

Windows note: uses the 'spawn' start method (the only one on Windows),
so the actor entry + all Process args must be picklable -- they are
(queues, plain dataclasses/dicts). The model is never sent to actors.
"""

from __future__ import annotations

import logging
import dataclasses
import multiprocessing as mp
import os
import queue as _queue
import random
import threading
from dataclasses import dataclass
import time
import traceback
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger("actor_pool")

# Control-queue commands (main -> actor).
_CMD_PLAY = "play"        # (iter_idx, n_games, base_seed, t2i, f2i, decision_step)
_CMD_STOP = "stop"
_CMD_DRAIN = "drain"      # finish the current game, take no new ones

# Result-queue message kinds (actor -> main).
_R_OUTCOME = "outcome"    # a GameOutcome
_R_EXPS    = "experiences"  # List[MCTSExperience]
_R_DONE    = "iter_done"   # actor finished its quota this iteration
_R_ERROR   = "error"       # traceback string (non-fatal; logged)
_R_FATAL   = "fatal"       # non-swallowable death (fork guard, ...)

# Serve-process control commands (main -> serve process).
_SRV_SYNC = "sync"        # (version, state bytes): load these weights
_SRV_SERVE = "serve"      # (iter_idx,): start the serve threads
_SRV_PAUSE = "pause"      # (): stop the serve threads, reply their stats
_SRV_PROBE = "probe"      # (payload,): one infer_batch outside serving
_SRV_STOP = "stop"
# Serve-process replies (serve process -> main), on the shared server queue.
_S_READY = "ready"        # model built
_S_SYNCED = "synced"      # payload: the version loaded
_S_STATS = "stats"        # payload: {"threads": [stats dicts], "picker": {...}}
_S_PROBE = "probe"        # payload: wire outputs
_S_ERROR = "error"        # payload: traceback string
# Reply marker the manager puts on an actor's reply queue when the
# serve process that actor was assigned to died: the client raises on
# it whatever request it is waiting for.
_RID_SERVER_DEAD = -1
# The intra-op pools torch and its BLAS size at import from these
# (start() caps them before every spawn; see the PID-limit note there).
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class ActorFatalError(BaseException):
    """An actor died on a non-swallowable error (round-35 C0: the
    actor's `finally` reported a clean _R_DONE even when a
    ForkGuardViolation escaped, so the pool topology exited 0 on a
    real fork violation). BaseException for the round-34 reason:
    no log-and-continue handler may eat it."""


class ServeProcessDied(ActorFatalError):
    """A serve process died or failed a command: the actors it served
    can get no more replies, so the iteration aborts loudly."""


# =====================================================================
# Actor-side transport
# =====================================================================

class _IPCInferenceClient:
    """RemoteModel transport inside an actor. BATCH-GRANULAR
    (2026-07-22 rework): one message carries a whole forward_batch's
    RawEncodeds, and one reply carries all its wire-serialized
    outputs. The old per-leaf protocol (B messages each way per
    forward_batch, each reply pickling ~9 torch tensors through the
    shm tensor-sharing machinery) capped the central server at
    ~200 req/s with the GPU idle -- the reason spool workers
    replaced the pool. Payloads are plain numpy (inference_seam
    output_to_wire/output_from_wire), which pickle inline."""

    def __init__(self, actor_id: int, req_qs, resp_q):
        """`req_qs`: one request queue per server (index 0 is the
        learner process); `use_server` picks the one this actor asks."""
        self._aid = actor_id
        self._req_qs = list(req_qs)
        self._req = self._req_qs[0]
        self._resp = resp_q
        self._next_id = 0

    def use_server(self, index: int) -> None:
        self._req = self._req_qs[index]

    def infer(self, raw):
        return self.infer_batch([raw])[0]

    def infer_batch(self, raws):
        if not raws:
            return []
        from tools.inference_seam import output_from_wire
        rid = self._next_id
        self._next_id += 1
        payload = list(raws)
        if isinstance(payload[0], tuple):
            # Priors protocol: one buffer per request instead of ~40
            # pickled numpy objects per leaf (wesnoth_ai/leaf_wire.py).
            from wesnoth_ai.leaf_wire import pack_request
            payload = pack_request(payload)
        self._req.put((self._aid, rid, payload))
        while True:
            r_rid, wires = self._resp.get()
            if r_rid == _RID_SERVER_DEAD:
                raise RuntimeError("the serve process this actor was assigned to died "
                                   "(see the pool's log)")
            if r_rid == rid:
                if wires is None:
                    raise RuntimeError("inference server failed on this batch "
                                       "(see the server's log)")
                return [output_from_wire(w) for w in wires]
            # Stale reply from an abandoned request id: drop.


def _zero_reward(_delta) -> float:
    """MCTS discards per-step shaping (policy.observe is a no-op; z
    comes from the winner in finalize_game). play_one_game still calls
    reward_fn at the terminal, so we return a constant 0.0."""
    return 0.0


def _set_fd_safe_sharing() -> None:
    """Switch torch's multiprocessing tensor transport off the
    'file_descriptor' strategy (Linux default). Every tensor shipped
    through the queues (ModelOutput responses, experience payloads)
    holds fds under that strategy, and at 48 actors x thousands of
    forwards/min they LEAK faster than they are reclaimed — observed
    2026-07-03 on the Vast node: fd count grew past ulimit 65536
    overnight, actors died one by one (game counts 48 -> 26), and
    resource_sharer spammed 134MB of Errno-24 tracebacks. The
    'file_system' strategy stages tensors in /dev/shm files instead
    (the standard DataLoader too-many-open-files fix); no-op where
    unsupported (Windows already uses it)."""
    try:
        import torch.multiprocessing as _tmp
        if "file_system" in _tmp.get_all_sharing_strategies():
            _tmp.set_sharing_strategy("file_system")
    except Exception:
        pass


def _actor_loop(
    actor_id: int, ctrl_q, req_qs, resp_q, result_q,
    mcts_cfg, scenario_opts: Dict, max_turns: int,
    max_turns_min,
    pvp_kwargs: Optional[Dict], log_level: int, torch_threads: int,
    turn_cfg=None, gbc_labels: bool = False, pt_cfg=None,
    train_kwargs: dict = None, ground_cfg=None,
) -> None:
    """Persistent actor process body. Builds a seam-backed MCTSPolicy
    once, then loops on the control queue: PLAY -> roll `n_games` and
    ship experiences/outcomes; STOP -> exit."""
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.set_num_threads(max(1, torch_threads))
    _set_fd_safe_sharing()
    # Deprioritize actors below the central server (POSIX only;
    # no-op elsewhere). 2026-07-22 diagnosis: 44 actors saturating
    # all cores scheduler-starved the server's host-side work to
    # ~11.9ms/leaf when the same path runs at 1.1ms/leaf on an idle
    # box -- every actor blocks on the server, so the server MUST
    # win CPU contention or the whole fleet idles at its starved
    # rate.
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass

    # Heavy imports happen here (post-spawn), not at module import time.
    from tools.inference_seam import RemoteEncoder, RemoteModel
    from tools.mcts_policy import MCTSPolicy
    from tools.sim_self_play import _play_one_game_safe, _recruit_cost_lookup
    from tools.scenario_pool import random_setup, roll_mix
    from tools.wesnoth_sim import PvPDefaults

    client = _IPCInferenceClient(actor_id, req_qs, resp_q)
    rmodel = RemoteModel(client)
    cost_lookup = _recruit_cost_lookup()
    pvp = PvPDefaults(**pvp_kwargs) if pvp_kwargs else PvPDefaults()
    cpu = torch.device("cpu")

    while True:
        cmd = ctrl_q.get()
        # A stale DRAIN can sit in the queue when the actor finished
        # its quota before the manager's soft deadline fired: skip it
        # (it referred to the PREVIOUS iteration).
        while cmd[0] == _CMD_DRAIN:
            cmd = ctrl_q.get()
        if cmd[0] == _CMD_STOP:
            return
        (_, iter_idx, n_games, base_seed, t2i, f2i,
         decision_step0) = cmd[:7]
        # The learner's action-space basis rides the PLAY command
        # (project round-2 C3: a hardcoded False here put pool
        # actors on the full-board basis whenever the learner
        # inherited --relevant-set-hexes -- with no tripwire).
        # Legacy 7-tuple PLAY (an old manager) = full-board.
        _rset = bool(cmd[7]) if len(cmd) > 7 else False
        # Search value centering for this iteration (legacy PLAY
        # tuples without it = 0, i.e. off).
        _vc = float(cmd[8]) if len(cmd) > 8 else 0.0
        mcts_cfg = dataclasses.replace(mcts_cfg, value_center=_vc)
        # Server-side priors (wesnoth_ai/server_priors.py); legacy PLAY
        # tuples without the flag = off.
        _sp = bool(cmd[9]) if len(cmd) > 9 else False
        # Which server answers this actor this iteration (module
        # docstring, "Serve processes"); legacy PLAY tuples = the
        # learner process.
        client.use_server(int(cmd[10]) if len(cmd) > 10 else 0)
        # Rebuild the encoder each iteration with the freshly-snapshotted
        # vocab so actor indices line up with the server's encoder.
        renc = RemoteEncoder(t2i, f2i, device=cpu,
                             relevant_set=_rset, server_priors=_sp)
        # MCTSPolicy.select_action reads `_base._lock` / `_base._decision_step`
        # (the combat-oracle anneal, added 2026-06-29). The in-process base is
        # a TransformerPolicy that supplies both; the actor's lightweight base
        # MUST supply them too, or select_action AttributeErrors on the first
        # decision of every game — an error _play_one_game_safe swallows, so
        # the pool would silently produce ZERO experiences. Seed the counter
        # from the main process's GLOBAL decision_step (broadcast in the PLAY
        # command) so the anneal alpha reflects true training progress rather
        # than restarting at 0 in each actor each iteration.
        base = SimpleNamespace(_inference_model=rmodel,
                               _inference_encoder=renc,
                               _lock=threading.Lock(),
                               _decision_step=int(decision_step0))
        # Training-label kwargs (draw_value_weight,
        # train_draw_tiebreak) seal ACTOR-side in finalize_game, so
        # they must ride into the actor's policy (project round-2
        # C1: the pool dropped both, silently nullifying the knobs
        # on the production topology).
        _tk = dict(train_kwargs or {})
        if pt_cfg is not None:
            from tools.plan_tournament import PlanTournamentPolicy
            policy = PlanTournamentPolicy(base, mcts_cfg,
                                          gbc_labels=gbc_labels,
                                          tournament_config=pt_cfg,
                                          **_tk)
        elif turn_cfg is not None:
            from tools.turn_policy import TurnCommitPolicy
            policy = TurnCommitPolicy(base, mcts_cfg,
                                      gbc_labels=gbc_labels,
                                      turn_config=turn_cfg,
                                      grounding_config=ground_cfg,
                                      **_tk)
        else:
            policy = MCTSPolicy(base, mcts_cfg,
                                gbc_labels=gbc_labels, **_tk)
        rng = random.Random(base_seed)
        # Split the mix ratios (absolute, sum to 1; no midgame --
        # the parent CLI rejects --midgame-ratio with --actor-pool)
        # from the pass-through setup options.
        mix = {k: scenario_opts.get(f"{k}_ratio", 0.0)
               for k in ("midgame", "mini", "fogless")}
        # Absent ladder_ratio -> the complement (the old flat 1.0
        # default summed to 2 with any explicit other ratio and blew
        # the sum-to-1 guard in every actor, 2026-07-22 smoke).
        mix["ladder"] = scenario_opts.get(
            "ladder_ratio", max(0.0, 1.0 - sum(mix.values())))
        midgame_dataset = scenario_opts.get("midgame_dataset")
        setup_opts = {k: v for k, v in scenario_opts.items()
                      if not k.endswith("_ratio")
                      and k != "midgame_dataset"}
        try:
            for g in range(n_games):
                # Drain contract (user ruling 2026-08-17, A6): when
                # the manager's soft deadline fires it sends DRAIN --
                # finish the game in progress, start no new one. The
                # check sits BETWEEN games so a completed game is
                # never thrown away (the leg-3 waste mode).
                drained = False
                try:
                    while True:
                        nxt = ctrl_q.get_nowait()
                        if nxt[0] == _CMD_DRAIN:
                            drained = True
                        elif nxt[0] == _CMD_STOP:
                            return
                        # PLAY cannot arrive mid-iteration (the
                        # manager is synchronous); anything else is
                        # dropped with the drain.
                except _queue.Empty:
                    pass
                if drained:
                    break
                from tools.sim_self_play import _roll_max_turns
                mt = _roll_max_turns(rng, max_turns, max_turns_min)
                cat = roll_mix(rng, **mix)
                setup = None
                if cat == "midgame":
                    # Same midgame path as selfplay_worker (2026-07-22
                    # port; the old "actors cannot splice midgame
                    # starts" CLI rejection predates
                    # _play_one_game_safe handling the tuple form).
                    from tools.midgame_starts import sample_midgame_start
                    from pathlib import Path as _P
                    mg = sample_midgame_start(
                        rng, midgame_dataset or _P("replays_dataset"))
                    if mg is not None:
                        setup = ("__midgame__",) + mg
                    else:
                        cat = "ladder"   # degraded sample
                if setup is None:
                    setup = random_setup(rng, category=cat,
                                         **setup_opts)
                gl = f"iter{iter_idx}_a{actor_id}_g{g}"
                outcome = _play_one_game_safe(
                    setup=setup, max_turns=mt, pvp_defaults=pvp,
                    policy=policy, reward_fn=_zero_reward,
                    cost_lookup=cost_lookup, game_label=gl)
                if outcome is not None:
                    result_q.put((_R_OUTCOME, actor_id, outcome))
                # Ship this game's experiences immediately (smaller
                # messages; overlaps with the next game's rollout).
                with policy._lock:
                    exps = policy._queue
                    policy._queue = []
                if exps:
                    result_q.put((_R_EXPS, actor_id, exps))
        except Exception:
            result_q.put((_R_ERROR, actor_id, traceback.format_exc()))
        finally:
            import sys as _sys
            _exc = _sys.exc_info()[1]
            # An in-flight exception here escaped the `except
            # Exception` above, i.e. a BaseException-class death
            # (fork guard, SystemExit): the done report below must
            # NOT fire, or the parent counts this actor as cleanly
            # finished (round-35 C0).
            _is_fatal = (_exc is not None
                         and not isinstance(_exc, Exception))
            # Report how many decisions this actor made this iteration
            # (base._decision_step advanced past decision_step0), so the
            # main process can advance the global anneal counter by the
            # total across all actors.
            local_decisions = (int(getattr(base, "_decision_step",
                                           decision_step0))
                               - int(decision_step0))
            # Distillation-target telemetry rode ONLY the in-process
            # path until 2026-08-12 -- the F3-ruling pool launch ran
            # an entire leg with every distill_* column empty (the
            # blind spot called out by the same-day diagnosis). Ship
            # the actor's drained per-decision means with its done
            # report; the manager averages across actors.
            dstats = None
            drain = getattr(policy, "drain_distill_stats", None)
            if drain is not None:
                try:
                    dstats = drain()
                except Exception:                   # noqa: BLE001
                    dstats = None
            if _is_fatal:
                result_q.put((_R_FATAL, actor_id,
                              traceback.format_exc()))
            else:
                result_q.put((_R_DONE, actor_id,
                              (local_decisions, dstats)))


# =====================================================================
# Main-side pool manager
# =====================================================================

def _request_lengths(payload) -> List[int]:
    """Hex + unit tokens of every leaf of a request (a PackedRequest
    carries them in its headers; a legacy list carries RawEncodeds or
    (RawEncoded, PackedMasks) pairs): the sequence length a batch pads
    to. The packed trunk pads nothing, but the heads and the priors'
    mask kernels still run over the batch's longest leaf."""
    headers = getattr(payload, "headers", None)
    if headers is not None:
        return [h.n_hexes + h.n_units for h in headers]
    return [len(r[0].hex_xs) + len(r[0].unit_xs) if isinstance(r, tuple)
            else len(r.hex_xs) + len(r.unit_xs) for r in payload]


@dataclass(slots=True)
class _Waiting:
    item: tuple             # (actor id, request id, payload), as queued
    seq: int                # arrival order
    n_leaves: int
    lens: List[int]         # hex + unit tokens per leaf
    tokens: int             # the request's longest leaf
    skipped: int = 0        # batches formed while this request waited


class _BatchPicker:
    """Which queued requests share a batch (docs/gpu_forward_design_
    20260904.md section 7), shared by the serve threads. "fifo":
    arrival order until the batch holds max_batch leaves (the rule
    since 2026-07-22). "length": the same whenever everything waiting
    fits one batch; otherwise the oldest request anchors the batch and
    the requests nearest to it in token count fill it (one request is
    one tree on one map, so its token count is its longest leaf), and
    `gap` > 0 refuses any request further than that many tokens from
    the anchor. A request left behind once goes into the next batch
    ahead of the anchor rule, so no request is delayed by more than one
    batch."""

    def __init__(self, policy: str = "fifo", gap: int = 0):
        if policy not in ("fifo", "length"):
            raise ValueError(f"coalesce policy must be 'fifo' or 'length', got {policy!r}")
        self.policy = policy
        self.gap = int(gap)
        self._lock = threading.Lock()
        self._waiting: List[_Waiting] = []
        self._seq = 0
        # Telemetry: requests deferred by the length rule, and the
        # waiting requests seen at each pick (queue depth).
        self.skipped = 0
        self.picks = 0
        self.depth = 0

    def take(self, queue, max_batch: int, timeout: float) -> List[_Waiting]:
        """Moves everything queued into the waiting list (blocking up to
        `timeout` for a first request only when nothing waits) and
        returns the next batch; empty when nothing arrived."""
        fresh = []
        with self._lock:
            idle = not self._waiting
        if idle:
            try:
                fresh.append(queue.get(timeout=timeout))
            except _queue.Empty:
                return []
        while True:
            try:
                fresh.append(queue.get_nowait())
            except _queue.Empty:
                break
        with self._lock:
            for item in fresh:
                self._waiting.append(self._wrap(item))
            return self._pick(max_batch)

    def _wrap(self, item) -> _Waiting:
        lens = _request_lengths(item[2])
        self._seq += 1
        return _Waiting(item=item, seq=self._seq, n_leaves=len(lens), lens=lens,
                        tokens=max(lens) if lens else 0)

    def _pick(self, max_batch: int) -> List[_Waiting]:
        waiting = self._waiting
        if not waiting:
            return []
        self.picks += 1
        self.depth += len(waiting)
        if self.policy == "fifo":
            n = k = 0
            while k < len(waiting) and n < max_batch:
                n += waiting[k].n_leaves
                k += 1
            batch, self._waiting = waiting[:k], waiting[k:]
            return batch
        batch: List[_Waiting] = []
        n = 0
        for w in waiting:                         # left behind last time: first, by age
            if w.skipped and n < max_batch:
                batch.append(w)
                n += w.n_leaves
        rest = [w for w in waiting if not w.skipped]
        if rest and n < max_batch:
            anchor = max(w.tokens for w in batch) if batch else rest[0].tokens
            rest.sort(key=lambda w: (abs(w.tokens - anchor), w.seq))
            for w in rest:
                if n >= max_batch or (self.gap and abs(w.tokens - anchor) > self.gap):
                    break
                batch.append(w)
                n += w.n_leaves
        chosen = {w.seq for w in batch}
        left = [w for w in waiting if w.seq not in chosen]
        for w in left:
            w.skipped += 1
        self.skipped += len(left)
        self._waiting = left
        return batch


def _merge_timelines(per_thread: List[List[Tuple[float, int]]],
                     t_start: float) -> List[Tuple[float, int]]:
    """Total leaves served by time t (seconds since t_start), from the
    threads' (monotonic time, cumulative leaves) marks."""
    marks = sorted((t, i, n) for i, tl in enumerate(per_thread) for (t, n) in tl)
    latest = [0] * len(per_thread)
    out: List[Tuple[float, int]] = []
    for t, i, n in marks:
        latest[i] = n
        out.append((t - t_start, sum(latest)))
    return out


def _best_window_rate(timeline: List[Tuple[float, int]], window: float) -> Optional[float]:
    """Highest leaves/s over any span of at least `window` seconds
    between two marks; None when the timeline is shorter than that."""
    best = None
    j = 0
    for i, (t_i, n_i) in enumerate(timeline):
        while j < len(timeline) and timeline[j][0] < t_i + window:
            j += 1
        if j >= len(timeline):
            break
        t_j, n_j = timeline[j]
        rate = (n_j - n_i) / (t_j - t_i)
        best = rate if best is None or rate > best else best
    return best


def _picker_stats(picker: Optional[_BatchPicker]) -> Dict[str, int]:
    if picker is None:
        return {"skipped": 0, "picks": 0, "depth": 0}
    return {"skipped": picker.skipped, "picks": picker.picks, "depth": picker.depth}


def _serve_loop(server, picker: _BatchPicker, req_q, resp_qs, max_batch: int,
                serve_timeout: float, stop_ev, stats_out: List[Dict]) -> None:
    """One serving thread, in the learner process or a serve process:
    take a batch (the picker coalesces the queued requests) -> unpack
    -> encode+forward -> wire-serialize -> reply. Stage times are
    accumulated locally (no locks on the hot path) and appended to
    `stats_out` on exit; on CUDA the seam adds the host seconds of its
    own stages to the same dict."""
    from tools.inference_seam import output_to_wire
    from wesnoth_ai.leaf_wire import PackedRequest, unpack_request
    # (monotonic time, cumulative leaves) every ~10 s: the iteration
    # average hides the tail where most actors have finished
    # (2026-09-05 whole-pool profile: median game finish at 40% of
    # the wall); run_iteration derives the saturated rate from it.
    # time.monotonic is one system-wide clock, so the marks of every
    # process merge on the manager's time base.
    timeline: List[Tuple[float, int]] = []
    next_mark = time.monotonic()
    st = {"wait": 0.0, "unpack": 0.0, "infer": 0.0, "wire": 0.0, "put": 0.0, "gpu_ms": 0.0,
          "timeline": timeline,
          "leaves": 0, "batches": 0, "requests": 0, "tokens": 0, "padded": 0}
    while not stop_ev.is_set():
        t0 = time.monotonic()
        batch = picker.take(req_q, max_batch, serve_timeout)
        t1 = time.monotonic()
        st["wait"] += t1 - t0
        if not batch:
            continue
        flat = []
        for w in batch:
            payload = w.item[2]
            flat.extend(unpack_request(payload) if isinstance(payload, PackedRequest)
                        else payload)
        t2 = time.monotonic()
        try:
            outs = server.infer_batch(flat, stats=st)
            t3 = time.monotonic()
            wires = [output_to_wire(o) for o in outs]
        except Exception:                       # noqa: BLE001
            # A serve-thread death used to hang every actor waiting
            # on this batch (2026-09-04: the stats line below choked
            # on (raw, masks) items). Reply with a failure marker so
            # the actors raise instead.
            log.error("inference server failed on a batch of %d leaves:\n%s",
                      len(flat), traceback.format_exc())
            for w in batch:
                aid, rid, _payload = w.item
                resp_qs[aid].put((rid, None))
            continue
        t4 = time.monotonic()
        i = 0
        for w in batch:
            aid, rid, _payload = w.item
            resp_qs[aid].put((rid, wires[i:i + w.n_leaves]))
            i += w.n_leaves
        t5 = time.monotonic()
        st["unpack"] += t2 - t1
        st["infer"] += t3 - t2
        st["wire"] += t4 - t3
        st["put"] += t5 - t4
        st["leaves"] += len(flat)
        st["batches"] += 1
        st["requests"] += len(batch)
        if t5 >= next_mark:
            timeline.append((t5, st["leaves"]))
            next_mark = t5 + 10.0
        # Sequence lengths: hex tokens + unit tokens per leaf, and
        # what the batch pads to (its longest leaf).
        lens = [n for w in batch for n in w.lens]
        st["tokens"] += sum(lens)
        st["padded"] += len(lens) * max(lens)
    stats_out.append(st)


def _server_loop(
    server_id: int, ctrl_q, server_q, req_q, resp_qs, blueprint, switches: Dict,
    device_str: str, serve_threads: int, max_batch: int, serve_timeout: float,
    coalesce: str, coalesce_gap: int, log_level: int, torch_threads: int,
) -> None:
    """Serve-process body: build the inference pair at the learner's
    architecture and switches, report READY, then answer the control
    queue (SYNC weights, SERVE / PAUSE the serve threads on this
    process's request queue, PROBE, STOP). Every failure is a reply on
    `server_q`, never a silent death; the process also exits on its
    own when the learner process is gone, so no CUDA context outlives
    the campaign."""
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.set_num_threads(max(1, torch_threads))
    _set_fd_safe_sharing()
    from tools.inference_seam import (
        InferenceServer, build_inference_pair, load_inference_state, output_to_wire,
    )
    try:
        device = torch.device(device_str)
        model, encoder = build_inference_pair(blueprint, device)
        model.infer_autocast_bf16 = bool(switches["model_bf16"])
        model.infer_packed_trunk = bool(switches["packed_trunk"])
        if switches["compile_packed"]:
            model.configure_packed_compile(backend=switches["compile_backend"],
                                           mode=switches["compile_mode"])
            log.info("serve-%d packed compile warmup: %s", server_id,
                     model.warmup_packed_compile())
        server = InferenceServer(model, encoder, device=device,
                                 output_device=torch.device("cpu"),
                                 autocast_bf16=switches["autocast_bf16"],
                                 packed_embed=switches["packed_embed"])
    except Exception:                           # noqa: BLE001
        server_q.put((_S_ERROR, server_id, traceback.format_exc()))
        return
    server_q.put((_S_READY, server_id, None))
    parent = mp.parent_process()
    threads: List[threading.Thread] = []
    stop_ev = threading.Event()
    stats: List[Dict] = []
    picker: Optional[_BatchPicker] = None
    while True:
        try:
            cmd = ctrl_q.get(timeout=2.0)
        except _queue.Empty:
            if parent is not None and not parent.is_alive():
                log.error("serve-%d: the learner process is gone; exiting", server_id)
                break
            continue
        kind = cmd[0]
        try:
            if kind == _SRV_STOP:
                break
            if kind == _SRV_SYNC:
                if threads:
                    raise RuntimeError("SYNC while serving: weights change only "
                                       "between iterations")
                _, version, blob = cmd
                load_inference_state(blob, model, encoder, device)
                server_q.put((_S_SYNCED, server_id, int(version)))
            elif kind == _SRV_SERVE:
                if threads:
                    raise RuntimeError("SERVE while already serving")
                stop_ev = threading.Event()
                stats = []
                picker = _BatchPicker(coalesce, coalesce_gap)
                threads = [threading.Thread(
                    target=_serve_loop,
                    args=(server, picker, req_q, resp_qs, max_batch, serve_timeout,
                          stop_ev, stats),
                    daemon=True, name=f"serve-{server_id}-{i}")
                    for i in range(serve_threads)]
                for th in threads:
                    th.start()
            elif kind == _SRV_PAUSE:
                stop_ev.set()
                for th in threads:
                    th.join(timeout=10.0)
                threads = []
                server_q.put((_S_STATS, server_id,
                              {"threads": list(stats), "picker": _picker_stats(picker)}))
            elif kind == _SRV_PROBE:
                outs = server.infer_batch(cmd[1])
                server_q.put((_S_PROBE, server_id, [output_to_wire(o) for o in outs]))
            else:
                raise RuntimeError(f"unknown serve-process command {kind!r}")
        except Exception:                       # noqa: BLE001
            server_q.put((_S_ERROR, server_id, traceback.format_exc()))
    stop_ev.set()
    for th in threads:
        th.join(timeout=10.0)


class ActorPool:
    """Owns the actor processes and runs the central inference-serve
    loop during each rollout iteration. The model stays in the main
    process (`policy._inference_model`); see module docstring."""

    def __init__(
        self, policy, n_actors: int, mcts_cfg, *,
        turn_cfg=None, pt_cfg=None, gbc_labels: bool = False,
        train_kwargs: dict = None, ground_cfg=None,
        scenario_opts: Optional[Dict] = None, max_turns: int = 60,
        max_turns_min: Optional[int] = None,
        pvp_defaults=None, device: Optional[torch.device] = None,
        max_batch: Optional[int] = None, serve_timeout: float = 0.005,
        log_level: int = logging.WARNING, actor_torch_threads: int = 1,
        iteration_timeout: Optional[float] = 3600.0,
        drain_grace: float = 1800.0,
        liveness_interval: float = 2.0,
        serve_threads: int = 2,
        server_priors: bool = True,
        infer_bf16: Optional[bool] = None,
        packed_embed: bool = False,
        coalesce: str = "fifo",
        coalesce_gap: int = 0,
        serve_processes: int = 1,
        server_torch_threads: int = 4,
        server_start_timeout: float = 600.0,
        server_reply_timeout: float = 120.0,
    ):
        """`server_priors`: actors ship packed legality masks and the
        server returns compact legal actions with priors
        (wesnoth_ai/server_priors.py; measured 2026-09-04, docs/
        box_specs.md). `infer_bf16`: the server's autocast switch
        (None follows the model's `infer_autocast_bf16`).
        `packed_embed`: the server embeds each batch from one pinned
        buffer straight into the trunk's layout
        (tools/inference_seam.InferenceServer). `coalesce` and
        `coalesce_gap`: how the serve threads pick a batch from the
        queued requests (_BatchPicker). `serve_processes`: servers in
        total, the learner process plus N-1 serve processes (module
        docstring); `server_torch_threads` caps each serve process's
        intra-op pools; the two timeouts bound the wait for a serve
        process to build its model and to answer a command."""
        if n_actors < 1:
            raise ValueError("n_actors must be >= 1")
        if serve_processes < 1:
            raise ValueError("serve_processes must be >= 1")
        self._policy = policy
        self._n = n_actors
        self._serve_processes = int(serve_processes)
        self._server_torch_threads = int(server_torch_threads)
        self._server_start_timeout = float(server_start_timeout)
        self._server_reply_timeout = float(server_reply_timeout)
        self._server_procs: List = []
        self._server_ctrl_qs: List = []
        self._server_versions: List[int] = []
        self._serving = False
        self._mcts_cfg = mcts_cfg
        # Per-iteration search value centering (MCTSConfig.value_center),
        # set by the learner after each step; rides the PLAY command.
        self.value_center: float = 0.0
        # Server-side priors (plan 1.3): actors ship packed legality
        # masks, the server returns compact legal actions. Off by
        # default until the box measurement certifies it.
        self.server_priors: bool = bool(server_priors)
        self._infer_bf16 = infer_bf16
        self._packed_embed = bool(packed_embed)
        self._coalesce = coalesce
        self._coalesce_gap = int(coalesce_gap)
        _BatchPicker(coalesce, coalesce_gap)          # validates the policy name
        # TCS (2026-08-14): when set, actors build TurnCommitPolicy
        # instead of MCTSPolicy -- the third generation path of the
        # worker-side-targets symmetry contract.
        self._turn_cfg = turn_cfg
        self._ground_cfg = ground_cfg
        self._pt_cfg = pt_cfg
        # GBC labels (2026-08-14): actors attach hindsight event
        # labels in finalize_game; same symmetry contract.
        self._gbc_labels = bool(gbc_labels)
        self._train_kwargs = dict(train_kwargs or {})
        self._scenario_opts = scenario_opts or {}
        self._max_turns = max_turns
        self._max_turns_min = max_turns_min
        self._pvp_kwargs = (dict(pvp_defaults.__dict__)
                            if pvp_defaults is not None else None)
        self._device = device
        # LEAVES per server-side model batch (2026-07-22: requests
        # are batch-granular, so this caps the coalesced total, not a
        # message count). 2 leaves/actor lets two full B=16 actor
        # requests ride one GPU call at 48 actors without starving
        # latecomers.
        self._max_batch = max_batch or max(64, 2 * n_actors)
        self._serve_timeout = serve_timeout
        self._log_level = log_level
        self._actor_threads = actor_torch_threads
        # Watchdog (CLAUDE principle #5: every wait is finite, failures
        # are visible). A hard-crashed actor (segfault / OOM-kill on the
        # GPU box / C-level hang) never sends its _R_DONE finally, so the
        # serve loop would otherwise spin forever with no progress and no
        # error. `iteration_timeout` is an overall wall-clock deadline
        # (None disables); `liveness_interval` throttles the per-actor
        # is_alive() scan that detects an actor that died without
        # reporting done.
        self._iteration_timeout = iteration_timeout
        # Drain-not-abandon (user ruling 2026-08-17, A6 postmortem):
        # at `iteration_timeout` the manager sends DRAIN (finish the
        # current game, take no new ones) instead of discarding
        # in-flight work; `drain_grace` seconds later the old
        # abandon path is the hard backstop. Leg 3 threw away 30-60%
        # of some iterations' games at the old single hard deadline.
        self._drain_grace = float(drain_grace)
        self._liveness_interval = liveness_interval
        # Serving threads (2026-07-22): >1 overlaps one thread's
        # wire/put with another's encode+forward. The default 1800s
        # iteration timeout also rose to 3600 -- at production game
        # lengths (cap 2000 actions, 100 turns) 1800s truncated the
        # first box bench 28/48 games in.
        self._serve_threads = max(1, int(serve_threads))
        self._started = False

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        from tools.inference_seam import InferenceServer
        _set_fd_safe_sharing()
        # Actors are single-threaded CPU workers (torch_threads=1 in
        # _actor_loop), but torch creates its intra-op pool at import,
        # before that call, from the HOST's core count: ~100 threads per
        # actor on a 128-thread Vast host. A 38-actor pool hit the
        # container's PID limit (pids.max 4352, 2026-09-04): nothing
        # served, sshd unable to fork. The spawned children inherit
        # this environment, so cap the pools before torch is imported.
        for var in _THREAD_ENV_VARS:
            os.environ.setdefault(var, "1")
        ctx = mp.get_context("spawn")
        # One request queue per server; index 0 is the learner process.
        self._req_qs = [ctx.Queue() for _ in range(self._serve_processes)]
        self._result_q = ctx.Queue()
        self._server_q = ctx.Queue()
        self._ctrl_qs = [ctx.Queue() for _ in range(self._n)]
        self._resp_qs = [ctx.Queue() for _ in range(self._n)]
        self._procs = []
        for aid in range(self._n):
            p = ctx.Process(
                target=_actor_loop,
                args=(aid, self._ctrl_qs[aid], self._req_qs,
                      self._resp_qs[aid], self._result_q, self._mcts_cfg,
                      self._scenario_opts, self._max_turns,
                      self._max_turns_min,
                      self._pvp_kwargs, self._log_level,
                      self._actor_threads, self._turn_cfg,
                      self._gbc_labels, self._pt_cfg,
                      self._train_kwargs, self._ground_cfg),
                daemon=True, name=f"actor-{aid}")
            p.start()
            self._procs.append(p)
        if self._serve_processes > 1:
            self._spawn_servers(ctx)
        self._server = InferenceServer(
            self._policy._inference_model, self._policy._inference_encoder,
            device=self._device, output_device=torch.device("cpu"),
            autocast_bf16=self._infer_bf16, packed_embed=self._packed_embed)
        self._started = True
        if self._server_procs:
            self._await_servers_ready()
            self.sync_servers()
        log.info(f"actor pool started: {self._n} actors, "
                 f"max_batch={self._max_batch}, {self._serve_processes} servers")

    def _vocab_snapshot(self) -> Tuple[Dict, Dict]:
        enc = self._policy._inference_encoder
        return dict(enc.unit_type_to_id), dict(enc.faction_to_id)

    # -- serve processes ----------------------------------------------

    def _inference_base(self):
        """The learner's uncompiled inference module: the snapshot
        target, whose `_weights_version` counts publications."""
        owner = getattr(self._policy, "_base", self._policy)
        base = getattr(owner, "_inference_base", None)
        return base if base is not None else self._policy._inference_model

    def _learner_version(self) -> int:
        return int(getattr(self._inference_base(), "_weights_version", 0))

    def _relevant_set(self) -> bool:
        return bool(getattr(
            getattr(self._anneal_base(), "_inference_encoder", None),
            "relevant_set_hexes", False))

    def _server_of(self, actor_id: int) -> int:
        """Round-robin assignment; server 0 is the learner process."""
        return actor_id % self._serve_processes

    def _server_ids(self) -> range:
        return range(1, self._serve_processes)

    def _spawn_servers(self, ctx) -> None:
        from tools.inference_seam import inference_blueprint
        base = self._inference_base()
        encoder = self._policy._inference_encoder
        blueprint = inference_blueprint(base, encoder)
        device = self._device or next(base.parameters()).device
        compiled = getattr(base, "_packed_compile", None)
        switches = dict(
            model_bf16=bool(getattr(base, "infer_autocast_bf16", False)),
            autocast_bf16=self._infer_bf16,
            packed_trunk=bool(getattr(base, "infer_packed_trunk", False)),
            compile_packed=bool(getattr(base, "infer_compile_packed", False)),
            compile_backend=getattr(compiled, "backend", "inductor"),
            compile_mode=getattr(compiled, "mode", None),
            packed_embed=self._packed_embed)
        # The serve process imports torch while unpickling its target,
        # before its body runs, so its pool caps must already be in the
        # environment it inherits at spawn; restored right after.
        saved = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
        for var in _THREAD_ENV_VARS:
            os.environ[var] = str(self._server_torch_threads)
        try:
            for sid in self._server_ids():
                cq = ctx.Queue()
                p = ctx.Process(
                    target=_server_loop,
                    args=(sid, cq, self._server_q, self._req_qs[sid], self._resp_qs,
                          blueprint, switches, str(device), self._serve_threads,
                          self._max_batch, self._serve_timeout, self._coalesce,
                          self._coalesce_gap, self._log_level,
                          self._server_torch_threads),
                    daemon=True, name=f"serve-{sid}")
                p.start()
                self._server_ctrl_qs.append(cq)
                self._server_procs.append(p)
        finally:
            for var, old in saved.items():
                if old is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = old
        self._server_versions = [-1] * len(self._server_procs)

    def _collect_server_replies(self, kind: str, timeout: float) -> Dict[int, object]:
        """One reply of `kind` from every serve process, by server id.
        Raises ServeProcessDied on an error reply, a dead process or
        the timeout: no command may hang the learner."""
        pending = set(self._server_ids())
        got: Dict[int, object] = {}
        deadline = time.monotonic() + timeout
        while pending:
            try:
                r_kind, sid, payload = self._server_q.get(timeout=0.5)
            except _queue.Empty:
                dead = [sid for sid in pending
                        if not self._server_procs[sid - 1].is_alive()]
                if dead:
                    raise ServeProcessDied(
                        f"serve process(es) {dead} died while the pool waited for "
                        f"{kind!r} (exitcodes "
                        f"{[self._server_procs[s - 1].exitcode for s in dead]})")
                if time.monotonic() > deadline:
                    raise ServeProcessDied(
                        f"serve process(es) {sorted(pending)} did not reply "
                        f"{kind!r} within {timeout:.0f}s")
                continue
            if r_kind == _S_ERROR:
                raise ServeProcessDied(f"serve process {sid} failed:\n{payload}")
            if r_kind != kind:
                log.warning(f"serve process {sid}: unexpected reply {r_kind!r} "
                            f"while waiting for {kind!r}; dropped")
                continue
            got[sid] = payload
            pending.discard(sid)
        return got

    def _await_servers_ready(self) -> None:
        self._collect_server_replies(_S_READY, self._server_start_timeout)
        log.info(f"{len(self._server_procs)} serve process(es) ready")

    def sync_servers(self) -> int:
        """Ships the learner's current inference weights (model and
        encoder state_dicts as one torch.save byte string) to every
        serve process and waits for the acks; returns the version the
        servers now hold. Call after every publication
        (TransformerPolicy._snapshot_inference_weights), between
        iterations: run_iteration refuses a lagging server.

        Why a byte string and not torch's CUDA IPC sharing (torch
        2.5.1, docs/multiprocessing "Sharing CUDA tensors"): a shared
        CUDA tensor obliges the sending process to keep the original
        alive as long as any receiver holds it, refcounted through
        handles that a receiver killed by a signal never releases, and
        the strategy setting of this pool ('file_system') does not
        apply to CUDA tensors at all. One copy per iteration (the
        weights, tens of MB against iterations of hundreds of seconds)
        has no such coupling and works the same on CPU and on CUDA."""
        if not self._server_procs:
            return self._learner_version()
        if self._serving:
            raise RuntimeError("sync_servers() during an iteration: weights change only "
                               "between iterations")
        from tools.inference_seam import pack_inference_state
        base = self._inference_base()
        version = self._learner_version()
        t0 = time.monotonic()
        blob = pack_inference_state(base, self._policy._inference_encoder)
        for cq in self._server_ctrl_qs:
            cq.put((_SRV_SYNC, version, blob))
        acks = self._collect_server_replies(_S_SYNCED, self._server_reply_timeout)
        wrong = {sid: v for sid, v in acks.items() if int(v) != version}
        if wrong:
            raise ServeProcessDied(f"serve process(es) acknowledged the wrong weights "
                                   f"version: {wrong} (expected {version})")
        self._server_versions = [version] * len(self._server_procs)
        log.info(f"serve processes synced to weights version {version} "
                 f"({len(blob) / 1e6:.1f} MB in {time.monotonic() - t0:.2f}s)")
        return version

    def _check_servers_synced(self, iter_idx: int) -> None:
        lv = self._learner_version()
        lag = {sid: v for sid, v in zip(self._server_ids(), self._server_versions)
               if v != lv}
        if lag:
            raise RuntimeError(
                f"iter {iter_idx}: serve process(es) hold weights version {lag} but "
                f"the learner is at {lv}; call sync_servers() after publishing weights")
        dead = [sid for sid in self._server_ids()
                if not self._server_procs[sid - 1].is_alive()]
        if dead:
            raise ServeProcessDied(f"iter {iter_idx}: serve process(es) {dead} are dead "
                                   f"(exitcodes "
                                   f"{[self._server_procs[s - 1].exitcode for s in dead]})")

    def _pause_servers(self) -> Dict[int, Dict]:
        """Stops every live serve process's serve threads and returns
        their stats by server id (a dead server contributes nothing
        and is logged; the dead-server abort happens in the serve loop
        or at the next iteration's start)."""
        if not self._server_procs:
            return {}
        live = [sid for sid in self._server_ids() if self._server_procs[sid - 1].is_alive()]
        for sid in live:
            self._server_ctrl_qs[sid - 1].put((_SRV_PAUSE,))
        if len(live) < len(self._server_procs):
            log.error(f"serve process(es) "
                      f"{sorted(set(self._server_ids()) - set(live))} are dead; "
                      f"their serve stats are lost")
        got: Dict[int, Dict] = {}
        pending = set(live)
        deadline = time.monotonic() + self._server_reply_timeout
        while pending and time.monotonic() < deadline:
            try:
                r_kind, sid, payload = self._server_q.get(timeout=0.5)
            except _queue.Empty:
                pending = {s for s in pending if self._server_procs[s - 1].is_alive()}
                continue
            if r_kind == _S_STATS:
                got[sid] = payload
                pending.discard(sid)
            elif r_kind == _S_ERROR:
                log.error(f"serve process {sid} failed:\n{payload}")
                pending.discard(sid)
        if pending:
            log.error(f"serve process(es) {sorted(pending)} did not return their stats")
        return got

    def _abort_on_dead_servers(self, iter_idx: int, dead: List[int]) -> None:
        """A serve process died mid-iteration: its actors would wait on
        their reply queues forever. Poison those queues (the client
        raises on the marker) and abort the iteration."""
        for aid in range(self._n):
            if self._server_of(aid) in dead:
                self._resp_qs[aid].put((_RID_SERVER_DEAD, None))
        raise ServeProcessDied(
            f"iter {iter_idx}: serve process(es) {dead} died (exitcodes "
            f"{[self._server_procs[s - 1].exitcode for s in dead]}); actors "
            f"{[a for a in range(self._n) if self._server_of(a) in dead]} were "
            f"assigned to them -- aborting the iteration instead of hanging.")

    def probe(self, game_states: List) -> List[List]:
        """The same leaves through the learner's server and every serve
        process, outside an iteration: `[server][state]` ModelOutputs.
        The box's parity check (bf16 noise apart, the copies must
        agree) and the sync test's witness."""
        from tools.inference_seam import RemoteEncoder, RemoteModel, output_from_wire
        if self._serving:
            raise RuntimeError("probe() during an iteration")
        t2i, f2i = self._vocab_snapshot()
        renc = RemoteEncoder(t2i, f2i, device=torch.device("cpu"),
                             relevant_set=self._relevant_set(),
                             server_priors=bool(self.server_priors))
        payload = [RemoteModel._payload(renc.encode(gs)) for gs in game_states]
        outs = [self._server.infer_batch(payload)]
        for cq in self._server_ctrl_qs:
            cq.put((_SRV_PROBE, payload))
        replies = self._collect_server_replies(_S_PROBE, self._server_reply_timeout)
        for sid in self._server_ids():
            outs.append([output_from_wire(w) for w in replies[sid]])
        return outs

    def _anneal_base(self):
        """The object holding the combat-oracle anneal counter. In
        actor-pool mode `self._policy` is the MCTSPolicy wrapper, which
        reads `self._base._decision_step` — the counter lives on the
        underlying TransformerPolicy (`_base`). Fall back to the policy
        itself for non-wrapped shapes."""
        return getattr(self._policy, "_base", self._policy)

    def _global_decision_step(self) -> int:
        return int(getattr(self._anneal_base(), "_decision_step", 0))

    def _advance_decision_step(self, n: int) -> None:
        base = self._anneal_base()
        if n and hasattr(base, "_decision_step"):
            # Only run_iteration advances this in the main process (the
            # actors mutate their own private copies), so no lock needed.
            base._decision_step += int(n)

    def run_iteration(
        self, iter_idx: int, games_per_iter: int, base_seed: int,
    ) -> Tuple[List, List]:
        """Broadcast a PLAY command, serve inference until every actor
        reports done, and return (outcomes, experiences)."""
        if not self._started:
            raise RuntimeError("ActorPool.start() not called")
        if self._server_procs:
            self._check_servers_synced(iter_idx)
        self._serving = True
        try:
            return self._run_iteration(iter_idx, games_per_iter, base_seed)
        finally:
            self._serving = False

    def _run_iteration(
        self, iter_idx: int, games_per_iter: int, base_seed: int,
    ) -> Tuple[List, List]:
        # Even split of games across actors (+remainder to the first).
        per = [games_per_iter // self._n] * self._n
        for i in range(games_per_iter % self._n):
            per[i] += 1
        t2i, f2i = self._vocab_snapshot()
        ds0 = self._global_decision_step()
        _rset = self._relevant_set()

        outcomes: List = []
        experiences: List = []
        distill_dicts: List[Dict] = []      # per-actor drained means
        outstanding = set(range(self._n))   # actors not yet _R_DONE
        total_decisions = 0                 # summed across actors this iter
        t_start = time.monotonic()
        last_liveness = t_start
        finish_times: List[float] = []   # per-game wall time since t_start
        drained = False                     # soft deadline fired?
        self._last_abandoned = 0            # discard telemetry (A6)

        # Serving runs in dedicated threads (2026-07-22: the single-
        # threaded serve loop capped the box at ~243 leaves/s with the
        # GPU at 50%). Each thread owns the full get -> coalesce ->
        # infer -> wire -> put chain; numpy/torch stages release the
        # GIL, so N threads overlap one thread's serialization with
        # another's encode+forward. Per-stage timers are accumulated
        # and logged at iteration end so the bottleneck stays visible.
        # The serve processes start their own threads on SERVE.
        stop_ev = threading.Event()
        serve_stats: List[Dict] = []
        self._picker = _BatchPicker(self._coalesce, self._coalesce_gap)
        servers = [threading.Thread(
            target=_serve_loop,
            args=(self._server, self._picker, self._req_qs[0], self._resp_qs,
                  self._max_batch, self._serve_timeout, stop_ev, serve_stats),
            daemon=True, name=f"serve-{i}")
            for i in range(self._serve_threads)]
        for th in servers:
            th.start()
        for cq in self._server_ctrl_qs:
            cq.put((_SRV_SERVE, iter_idx))
        server_stats: Dict[int, Dict] = {}

        def _stop_serving() -> None:
            stop_ev.set()
            for th in servers:
                th.join(timeout=10.0)
            server_stats.update(self._pause_servers())

        for aid in range(self._n):
            self._ctrl_qs[aid].put(
                (_CMD_PLAY, iter_idx, per[aid],
                 base_seed + aid * 1_000_003, t2i, f2i, ds0,
                 _rset, float(self.value_center), bool(self.server_priors),
                 self._server_of(aid)))

        while outstanding:
            # Drain results; blocking with a short timeout (serving no
            # longer happens on this thread, so waiting here is free).
            try:
                kind, aid, payload = self._result_q.get(timeout=0.2)
            except _queue.Empty:
                kind = None
            if kind == _R_OUTCOME:
                outcomes.append(payload)
                finish_times.append(time.monotonic() - t_start)
            elif kind == _R_EXPS:
                # Each _R_EXPS payload is ONE GAME's experiences
                # (actors ship per game) -- exactly the granularity
                # the learner's holdout probe needs. Offer the
                # whole game; only train on it if not diverted.
                offer = getattr(self._policy, "offer_holdout_game",
                                None)
                if offer is None or not offer(payload):
                    # Boundary-pair harvest (T1-F): this drain never
                    # called it, so boundary telemetry read n=0
                    # through the ENTIRE leg-4 campaign (workflow
                    # finding 2026-08-20). Valid here because each
                    # _R_EXPS payload is one game in recorded order.
                    _hv = getattr(self._policy,
                                  "harvest_boundary_pairs", None)
                    if _hv is not None:
                        _hv(payload)
                    experiences.extend(payload)
            elif kind == _R_DONE:
                outstanding.discard(aid)
                if isinstance(payload, tuple):
                    n_dec, dstats = payload
                else:               # legacy shape (plain int)
                    n_dec, dstats = payload, None
                total_decisions += int(n_dec or 0)
                if dstats:
                    distill_dicts.append(dstats)
            elif kind == _R_ERROR:
                log.error(f"actor {aid} error:\n{payload}")
            elif kind == _R_FATAL:
                _stop_serving()
                # The traceback carries the SIM_FORK_GUARD text the
                # launcher greps; propagate, never log-and-drop.
                raise ActorFatalError(
                    f"actor {aid} died on a non-swallowable error "
                    f"(round-35 C0):\n{payload}")
            if not outstanding:
                break
            now = time.monotonic()
            if (self._iteration_timeout is not None and not drained
                    and now - t_start > self._iteration_timeout):
                # Soft deadline: drain, don't abandon. Completed
                # games keep streaming in during the grace window.
                drained = True
                for aid in sorted(outstanding):
                    self._ctrl_qs[aid].put((_CMD_DRAIN,))
                log.warning(
                    f"iter {iter_idx}: soft deadline "
                    f"({self._iteration_timeout:.0f}s) reached; "
                    f"DRAIN sent to actors {sorted(outstanding)} "
                    f"(finish current game, no new ones; hard "
                    f"backstop in {self._drain_grace:.0f}s). "
                    f"{len(outcomes)} games in so far.")
            if (self._iteration_timeout is not None and drained
                    and now - t_start > (self._iteration_timeout
                                         + self._drain_grace)):
                self._last_abandoned = len(outstanding)
                log.error(
                    f"iter {iter_idx}: hard deadline "
                    f"({self._iteration_timeout:.0f}s + "
                    f"{self._drain_grace:.0f}s drain grace) exceeded "
                    f"with actors {sorted(outstanding)} still "
                    f"outstanding; abandoning their in-flight games "
                    f"({len(outcomes)} games, {len(experiences)} "
                    f"exps kept).")
                break
            if now - last_liveness > self._liveness_interval:
                last_liveness = now
                dead_servers = [sid for sid in self._server_ids()
                                if not self._server_procs[sid - 1].is_alive()]
                if dead_servers:
                    _stop_serving()
                    self._abort_on_dead_servers(iter_idx, dead_servers)
                dead = {aid for aid in outstanding
                        if not self._procs[aid].is_alive()}
                if dead:
                    _codes = {aid: self._procs[aid].exitcode
                              for aid in sorted(dead)}
                    if any(c not in (0, None)
                           for c in _codes.values()):
                        # Killed before its `finally` ran (segfault,
                        # OOM-kill, guard trip mid-teardown): a
                        # silent drop hid the death from the run's
                        # exit code (round-35 C0). Loud abort; the
                        # supervisor restarts with backoff.
                        _stop_serving()
                        raise ActorFatalError(
                            f"iter {iter_idx}: actor(s) died "
                            f"without reporting done, exitcodes "
                            f"{_codes} -- aborting the iteration "
                            f"instead of silently degrading.")
                    for aid in sorted(dead):
                        log.error(
                            f"iter {iter_idx}: actor {aid} died without "
                            f"reporting done (exitcode="
                            f"{self._procs[aid].exitcode}); dropping it.")
                    outstanding -= dead

        _stop_serving()
        # Every server's threads report the same stats dict; the
        # per-server leaf counts and the pickers' telemetry merge here.
        leaves_per_server = [sum(int(s.get("leaves", 0)) for s in serve_stats)]
        pick = _picker_stats(self._picker)
        for sid in self._server_ids():
            ss = server_stats.get(sid) or {}
            threads_stats = list(ss.get("threads", []))
            leaves_per_server.append(sum(int(s.get("leaves", 0)) for s in threads_stats))
            serve_stats.extend(threads_stats)
            for k, v in (ss.get("picker") or {}).items():
                pick[k] = pick.get(k, 0) + int(v)
        self.last_leaves_per_server = leaves_per_server
        # Advance the global anneal counter by the decisions generated this
        # iteration (sum across actors), so the combat-oracle bias keeps
        # annealing across the campaign instead of freezing at ds0.
        self._advance_decision_step(total_decisions)
        # Mean of per-actor means (actors carry ~equal decision counts
        # under the even split); None-valued et_* fields are skipped.
        # Consumed by sim_self_play's iteration telemetry in place of
        # the learner-side drain (which never searches under the pool).
        self.last_distill_stats = None
        if distill_dicts:
            keys = set().union(*(d.keys() for d in distill_dicts))
            out = {}
            for k in keys:
                vals = [d[k] for d in distill_dicts
                        if d.get(k) is not None]
                out[k] = (sum(vals) / len(vals)) if vals else None
            self.last_distill_stats = out
        agg = {k: sum(s.get(k, 0) for s in serve_stats)
               for k in ("wait", "unpack", "infer", "wire", "put", "gpu_ms",
                         "leaves", "batches", "requests", "tokens", "padded",
                         "t_encode", "t_forward", "t_priors", "t_finish", "t_reply")
               } if serve_stats else {}
        served = int(agg.get("leaves", 0))
        elapsed = max(1e-9, time.monotonic() - t_start)
        self.last_leaf_timeline = _merge_timelines(
            [s.get("timeline", []) for s in serve_stats], t_start)
        self.last_saturated_leaves_per_s = _best_window_rate(self.last_leaf_timeline, 60.0)
        if agg.get("batches"):
            log.info(
                f"iter {iter_idx}: serve stages ({self._serve_processes} servers x "
                f"{self._serve_threads} threads, leaves per server "
                f"{leaves_per_server}): wait={agg['wait']:.1f}s "
                f"infer={agg['infer']:.1f}s wire={agg['wire']:.1f}s "
                f"put={agg['put']:.1f}s gpu={agg['gpu_ms'] / 1000.0:.1f}s "
                f"({agg['gpu_ms'] / max(served, 1):.2f} ms/leaf) leaves/batch="
                f"{agg['leaves'] / agg['batches']:.1f} "
                f"saturated={self.last_saturated_leaves_per_s or 0:.0f} leaves/s (best 60 s) "
                f"throughput={served / elapsed:.0f} leaves/s")
        # Host milliseconds per batch by stage (the t_* stages come from
        # the seam on CUDA only, so they read 0 on CPU), and what the
        # batch picker saw: requests per batch, waiting requests at each
        # pick, requests deferred by the length rule (summed over the
        # servers' pickers).
        nb = int(agg.get("batches", 0) or 0)
        self.last_host_ms = ({k: 1000.0 * agg[k] / nb for k in (
            "unpack", "t_encode", "t_forward", "t_priors", "t_finish", "t_reply", "wire", "put")}
            if nb else None)
        self.last_skipped_requests = pick["skipped"]
        self.last_queue_depth = pick["depth"] / pick["picks"] if pick["picks"] else None
        if nb:
            hm = self.last_host_ms
            log.info(
                f"iter {iter_idx}: host ms per batch: unpack={hm['unpack']:.2f} "
                f"encode={hm['t_encode']:.2f} forward={hm['t_forward']:.2f} "
                f"priors={hm['t_priors']:.2f} wait={hm['t_finish']:.2f} "
                f"reply={hm['t_reply']:.2f} wire={hm['wire']:.2f} put={hm['put']:.2f} | "
                f"coalesce={self._coalesce} requests/batch={agg['requests'] / nb:.2f} "
                f"queue depth={self.last_queue_depth or 0:.2f} skipped={pick['skipped']}")
        log.info(f"iter {iter_idx}: pool served {served} forwards, "
                 f"{len(outcomes)} games, {len(experiences)} experiences, "
                 f"decision_step {ds0} -> {self._global_decision_step()}")
        # Time-profiling readbacks (minimal loop, 2026-09-03): the
        # numbers above were log-only; the loop's CSV wants them.
        self.last_served_forwards = served
        self.last_iteration_seconds = elapsed
        self.last_decisions = self._global_decision_step() - ds0
        self.last_tokens_per_leaf = (agg["tokens"] / served
                                     if served and agg.get("tokens") else None)
        self.last_pad_ratio = (agg["padded"] / agg["tokens"]
                               if agg.get("tokens") else None)
        ft = sorted(finish_times)
        self.last_game_finish_p50 = ft[len(ft) // 2] if ft else None
        self.last_game_finish_max = ft[-1] if ft else None
        if ft:
            log.info(f"iter {iter_idx}: game finish times p50={ft[len(ft) // 2]:.0f}s "
                     f"p90={ft[int(len(ft) * 0.9)]:.0f}s max={ft[-1]:.0f}s | "
                     f"tokens/leaf={self.last_tokens_per_leaf or 0:.0f} "
                     f"pad_ratio={self.last_pad_ratio or 0:.2f}")
        return outcomes, experiences

    def shutdown(self, timeout: float = 15.0) -> None:
        if not self._started:
            return
        for q in self._ctrl_qs:
            try:
                q.put((_CMD_STOP,))
            except Exception:
                pass
        for q in self._server_ctrl_qs:
            try:
                q.put((_SRV_STOP,))
            except Exception:
                pass
        for p in self._procs + self._server_procs:
            p.join(timeout)
            if p.is_alive():
                log.warning(f"terminating unresponsive process {p.name}")
                p.terminate()
        self._started = False


# =====================================================================
# Standalone correctness smoke (run directly; NOT a pytest -- spawning
# torch subprocesses inside the test sweep risks wedging the machine).
# =====================================================================

def _smoke() -> int:
    logging.basicConfig(level=logging.INFO)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.draw_tiebreak import DrawTiebreakConfig

    pol = TransformerPolicy(device=torch.device("cpu"),
                            d_model=48, num_layers=2, num_heads=4, d_ff=96)
    cfg = MCTSConfig(n_simulations=12, gumbel_root=True, gumbel_m=4,
                     chance_nodes=True, exact_outcome_enumeration=True,
                     draw_tiebreak=DrawTiebreakConfig(cap=0.3), batch_size=1,
                     add_root_noise=False)
    pool = ActorPool(pol, n_actors=2, mcts_cfg=cfg,
                     scenario_opts={"mini_maps": True, "mini_ratio": 1.0},
                     max_turns=10, log_level=logging.INFO)
    pool.start()
    try:
        outcomes, exps = pool.run_iteration(0, games_per_iter=2, base_seed=1)
    finally:
        pool.shutdown()
    print(f"SMOKE: {len(outcomes)} outcomes, {len(exps)} experiences")
    assert len(outcomes) == 2, f"expected 2 games, got {len(outcomes)}"
    assert len(exps) > 0, "expected non-empty experiences"
    # Feed into the learner and train once, exactly as the loop would.
    mp_policy = MCTSPolicy(pol, cfg, ReplayConfig(enabled=False))
    with mp_policy._lock:
        mp_policy._queue.extend(exps)
    stats = mp_policy.train_step()
    print(f"SMOKE: train_step total_loss={stats.total_loss:.4f} "
          f"value={stats.value_loss:.4f} n={stats.n_transitions}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
