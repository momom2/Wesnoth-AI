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
(the saturated rate covers all servers) and carry its compiled packed
trunk's state (`last_packed_compile_per_server`). A server that dies,
or that reports a failed command while serving, poisons its actors'
reply queues and aborts the iteration; shutdown stops them.

Continuous generation (2026-09-18, tools/actor_stream.py): `stream()`
opens a session in which the actors never wait at an iteration
barrier -- the game queue is kept topped up, the learner collects
completed games in windows and publishes weights between them while
serving goes on. The barrier iteration (`run_iteration`) ends with
its longest game, and the server starves for the second half of it
(docs/box_specs.md "The graphed server on a quiet host": iteration
rate 1.4-1.9x below the saturated rate on three hosts); a stream keeps
every actor in a game.

Windows note: uses the 'spawn' start method (the only one on Windows),
so the actor entry + all Process args must be picklable -- they are
(queues, plain dataclasses/dicts). The model is never sent to actors.

Module layout (2026-09-05 split): this module holds the manager
(ActorPool) and the standalone smoke; tools/actor_worker.py the actor
process body and its transport; tools/serve_worker.py the serve threads,
the batch picker and the serve-process body. The workers' names that
tests and scripts import from this module are re-exported below.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from tools.actor_worker import (
    _CMD_DRAIN, _CMD_PLAY, _CMD_UPDATE, _TICKET_END, _CMD_STOP, _R_DONE, _R_ERROR, _R_EXPS,
    _R_FATAL, _R_GAME, _R_OUTCOME, _RID_SERVER_DEAD, _IPCInferenceClient, _actor_loop,
    _done_report, _set_fd_safe_sharing, _zero_reward,
)
from tools.mp_teardown import close_queue, discarding, end_stragglers, join_all, start_child
from tools.serve_worker import (
    _S_ERROR, _S_PROBE, _S_READY, _S_STATS, _S_SYNCED, _SRV_PAUSE, _SRV_PROBE, _SRV_SERVE,
    _SRV_STATS, _SRV_STOP, _SRV_SYNC, _BatchPicker, _best_window_rate, _merge_timelines,
    _picker_stats, _request_lengths, _serve_loop, _server_loop,
)

__all__ = [
    "ActorPool", "ActorFatalError", "ServeProcessDied",
    # Re-exported from tools.actor_worker and tools.serve_worker: tests
    # and scripts import these from here.
    "_CMD_DRAIN", "_CMD_PLAY", "_CMD_STOP", "_CMD_UPDATE", "_R_DONE", "_R_ERROR", "_R_EXPS",
    "_R_FATAL", "_R_GAME", "_R_OUTCOME", "_RID_SERVER_DEAD", "_IPCInferenceClient", "_actor_loop",
    "_set_fd_safe_sharing", "_zero_reward",
    "_S_ERROR", "_S_PROBE", "_S_READY", "_S_STATS", "_S_SYNCED", "_SRV_PAUSE", "_SRV_PROBE",
    "_SRV_SERVE", "_SRV_STATS", "_SRV_STOP", "_SRV_SYNC", "_BatchPicker", "_best_window_rate",
    "_merge_timelines", "_picker_stats", "_request_lengths", "_serve_loop", "_server_loop",
]

log = logging.getLogger("actor_pool")

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


@dataclass
class _Serving:
    """The serving behind one iteration or one stream: the in-process
    serve threads with their live stats dicts and picker, and the
    serve processes' stats once PAUSE brought them back."""
    iter_idx: int
    stop_ev: threading.Event
    picker: _BatchPicker
    t_start: float
    threads: List[threading.Thread] = field(default_factory=list)
    serve_stats: List[Dict] = field(default_factory=list)
    server_stats: Dict[int, Dict] = field(default_factory=dict)
    stopped: bool = False


def _log_failure_report(msg) -> None:
    """Shutdown discards what the children send it; the failures among
    that (an actor's error or fatal report, a serve process's error
    reply) are logged, since nothing else will ever read them."""
    if isinstance(msg, tuple) and len(msg) == 3 and msg[0] in (_R_ERROR, _R_FATAL, _S_ERROR):
        log.error(f"shutdown: {msg[0]} report from child {msg[1]}:\n{msg[2]}")


# =====================================================================
# Main-side pool manager
# =====================================================================

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
        graphed_serve: bool = False,
        game_records_dir: Optional[str] = None,
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
        # A stream (tools/actor_stream.py) is open: serving runs across
        # learner steps and weights publish under the servers' gates.
        self._streaming = False
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
        # The static-shape serve path (wesnoth_ai/graphed_serve.py):
        # every server, in-process and spawned, replays its priors
        # batches from per-bucket CUDA graphs. cuda + bf16 + the
        # packed trunk only; elsewhere the flag is ignored with a log.
        self._graphed_serve = bool(graphed_serve)
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
        # Every game an actor finishes is recorded whole under this
        # directory, one file per actor (tools/game_record.py); None
        # records nothing.
        self._game_records_dir = None if game_records_dir is None else str(game_records_dir)
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
        # Serve-thread health of the last iteration (2026-09-13 audit):
        # the tracebacks of the threads that died inside their loop and
        # the names of those that would not stop at its end. Both mean
        # the iteration served with fewer threads than it was given.
        self.last_serve_thread_errors: List[str] = []
        self.last_stuck_serve_threads: List[str] = []
        # The thread objects behind that second list, kept so the next
        # iteration can re-join them instead of forgetting a thread that
        # is still reading the request queue.
        self._stuck_serve_threads: List[threading.Thread] = []
        # The serving started and not yet stopped (an iteration's, or a
        # stream's): shutdown() stops its threads before closing the
        # queues they read.
        self._open_serving: Optional[_Serving] = None

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
        # The iteration's games, shared by every actor (actor_worker
        # module docstring, "Game tickets").
        self._game_q = ctx.Queue()
        self._procs = []
        for aid in range(self._n):
            self._procs.append(start_child(
                ctx, _actor_loop,
                (aid, self._ctrl_qs[aid], self._req_qs,
                 self._resp_qs[aid], self._result_q, self._game_q,
                 self._mcts_cfg,
                 self._scenario_opts, self._max_turns,
                 self._max_turns_min,
                 self._pvp_kwargs, self._log_level,
                 self._actor_threads, self._turn_cfg,
                 self._gbc_labels, self._pt_cfg,
                 self._train_kwargs, self._ground_cfg,
                 self._game_records_dir),
                name=f"actor-{aid}"))
        if self._serve_processes > 1:
            self._spawn_servers(ctx)
        self._server = InferenceServer(
            self._policy._inference_model, self._policy._inference_encoder,
            device=self._device, output_device=torch.device("cpu"),
            autocast_bf16=self._infer_bf16, packed_embed=self._packed_embed,
            graphed=self._graphed_for(self._policy._inference_model,
                                      self._policy._inference_encoder))
        # Every publication into the inference snapshot (the policy's
        # own after a step, step_control's trial publishes) waits for
        # the batches in flight and excludes the next ones
        # (inference_seam.ServeGate): a stream publishes while serving.
        self._anneal_base()._serve_gate = self._server.gate
        self._started = True
        if self._server_procs:
            self._await_servers_ready()
            self.sync_servers()
        log.info(f"actor pool started: {self._n} actors, "
                 f"max_batch={self._max_batch}, {self._serve_processes} servers")

    def _vocab_snapshot(self) -> Tuple[Dict, Dict]:
        enc = self._policy._inference_encoder
        return dict(enc.unit_type_to_id), dict(enc.faction_to_id)

    def _graphed_serve_applies(self) -> bool:
        base = self._inference_base()
        device = self._device or next(base.parameters()).device
        if not self._graphed_serve:
            return False
        # bf16 as the server resolves it (InferenceServer._use_bf16): the
        # pool's own switch, else the model's (bench_pool sets the model's).
        bf16 = (bool(self._infer_bf16) if self._infer_bf16 is not None
                else bool(getattr(base, "infer_autocast_bf16", False)))
        ok = (device.type == "cuda" and bf16
              and bool(getattr(base, "infer_packed_trunk", False)))
        if not ok:
            log.warning("graphed_serve needs cuda, bf16 inference and the packed trunk; "
                        "serving eager")
        return ok

    def _graphed_for(self, model, encoder):
        """One GraphedServe for the in-process server, shared by its
        serve threads through its lock, or None. One instance per
        thread crashed the pool with illegal memory accesses on
        2026-09-14 while one shared instance ran clean (docs/
        box_specs.md "The graphed server on the pool"). The picker's
        batches run past max_batch leaves (a request's leaves are not
        split), so the segment caps go to twice max_batch."""
        if not self._graphed_serve_applies():
            return None
        from wesnoth_ai.graphed_serve import GraphedServe, pool_caps
        device = self._device or next(model.parameters()).device
        return GraphedServe(model, encoder, device, caps=pool_caps(self._max_batch))

    # -- serve processes ----------------------------------------------

    def _inference_base(self):
        """The learner's uncompiled inference module: the snapshot
        target, whose `_weights_version` counts publications."""
        owner = getattr(self._policy, "_base", self._policy)
        base = getattr(owner, "_inference_base", None)
        return base if base is not None else self._policy._inference_model

    def _learner_version(self) -> int:
        return int(getattr(self._inference_base(), "_weights_version", 0))

    def _fog_hides_enemy_villages(self) -> bool:
        return bool(getattr(
            getattr(self._anneal_base(), "_inference_encoder", None),
            "fog_hides_enemy_villages", False))

    def _terrain_multi_hot(self) -> bool:
        return bool(getattr(
            getattr(self._anneal_base(), "_inference_encoder", None),
            "terrain_multi_hot", False))

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
            packed_embed=self._packed_embed,
            graphed=self._graphed_serve_applies())
        # The serve process imports torch while unpickling its target,
        # before its body runs, so its pool caps must already be in the
        # environment it inherits at spawn; restored right after.
        saved = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
        for var in _THREAD_ENV_VARS:
            os.environ[var] = str(self._server_torch_threads)
        try:
            for sid in self._server_ids():
                cq = ctx.Queue()
                p = start_child(
                    ctx, _server_loop,
                    (sid, cq, self._server_q, self._req_qs[sid], self._resp_qs,
                     blueprint, switches, str(device), self._serve_threads,
                     self._max_batch, self._serve_timeout, self._coalesce,
                     self._coalesce_gap, self._log_level,
                     self._server_torch_threads),
                    name=f"serve-{sid}")
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
        iterations: run_iteration refuses a lagging server. A stream
        calls it between its learner steps while the servers serve;
        each loads under its gate, so no batch forwards through half a
        state_dict.

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
        if self._serving and not getattr(self, "_streaming", False):
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

    def _drain_server_replies(self) -> Dict[int, str]:
        """Replies a serve process posted while serving, read without
        blocking: the error replies by server id (a failed command --
        e.g. a serve thread that could not start under the container's
        PID limit -- after which the process stays alive and answers
        none of its actors). No other reply is pending during an
        iteration; anything else is dropped with a warning."""
        failed: Dict[int, str] = {}
        while True:
            try:
                r_kind, sid, payload = self._server_q.get_nowait()
            except _queue.Empty:
                return failed
            if r_kind == _S_ERROR:
                failed[sid] = str(payload)
            else:
                log.warning(f"serve process {sid}: unexpected reply {r_kind!r} while "
                            f"serving; dropped")

    def _abort_on_dead_servers(self, iter_idx: int, dead: List[int],
                               failures: Optional[Dict[int, str]] = None) -> None:
        """A serve process died or failed mid-iteration: its actors
        would wait on their reply queues forever. Poison those queues
        (the client raises on the marker) and abort the iteration.
        `failures`: the tracebacks of the servers that failed while
        alive."""
        failures = failures or {}
        for aid in range(self._n):
            if self._server_of(aid) in dead:
                self._resp_qs[aid].put((_RID_SERVER_DEAD, iter_idx))
        what = "; ".join(
            f"serve process {sid} failed:\n{failures[sid]}" if sid in failures
            else f"serve process {sid} died (exitcode {self._server_procs[sid - 1].exitcode})"
            for sid in dead)
        raise ServeProcessDied(
            f"iter {iter_idx}: {what}\nactors "
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
                             server_priors=bool(self.server_priors),
                             fog_hides_enemy_villages=self._fog_hides_enemy_villages(),
                             terrain_multi_hot=self._terrain_multi_hot())
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
        if getattr(self, "_streaming", False):
            raise RuntimeError("run_iteration() while a stream is open; stop it first")
        if self._server_procs:
            self._check_servers_synced(iter_idx)
        self._serving = True
        try:
            return self._run_iteration(iter_idx, games_per_iter, base_seed)
        finally:
            self._serving = False

    def stream(self, base_seed: int, *, tag: int = 0, tickets_ahead: Optional[int] = None):
        """Continuous generation (tools/actor_stream.ActorStream): the
        actors take games from a queue the stream keeps topped up, the
        caller collects completed games in windows and publishes
        weights between them while serving goes on. `tag` is the
        iteration index the tickets and the PLAY carry (an actor plays
        only tickets of its PLAY's tag); `tickets_ahead` is how many
        unstarted games the queue holds beyond the actors' own (one
        per actor by default)."""
        from tools.actor_stream import ActorStream
        if not self._started:
            raise RuntimeError("ActorPool.start() not called")
        return ActorStream(self, base_seed, tag=tag, tickets_ahead=tickets_ahead)

    def _post_tickets(self, iter_idx: int, games_per_iter: int, base_seed: int) -> None:
        """One ticket per game (its seed depends on the game index
        only), then one end marker per actor behind them."""
        for g in range(games_per_iter):
            self._game_q.put(self._ticket(iter_idx, g, base_seed))
        for _ in range(self._n):
            self._game_q.put((iter_idx, _TICKET_END, None))

    @staticmethod
    def _ticket(iter_idx: int, g: int, base_seed: int) -> Tuple[int, int, int]:
        return (iter_idx, g, base_seed + g * 1_000_003)

    def _flush_tickets(self) -> int:
        """Drop whatever is left on the game queue (tickets nobody took
        after a drain, end markers of dropped actors) so nothing
        stale reaches the next iteration. Returns the count."""
        n = 0
        while True:
            try:
                self._game_q.get(timeout=0.05)
                n += 1
            except _queue.Empty:
                return n

    def _play_command(self, iter_idx: int, games_per_iter: int, base_seed: int,
                      aid: int, *, stream: bool = False) -> tuple:
        t2i, f2i = self._vocab_snapshot()
        return (_CMD_PLAY, iter_idx, games_per_iter, base_seed, t2i, f2i,
                self._global_decision_step(), self._relevant_set(),
                float(self.value_center), bool(self.server_priors),
                self._server_of(aid), self._fog_hides_enemy_villages(), bool(stream),
                self._terrain_multi_hot())

    # -- serving: the threads behind one iteration or one stream ------

    def _start_serving(self, iter_idx: int) -> "_Serving":
        """Start the in-process serve threads and every serve
        process's, for this iteration or stream.

        Serving runs in dedicated threads (2026-07-22: the single-
        threaded serve loop capped the box at ~243 leaves/s with the
        GPU at 50%). Each thread owns the full get -> coalesce ->
        infer -> wire -> put chain; numpy/torch stages release the
        GIL, so N threads overlap one thread's serialization with
        another's encode+forward. Per-stage timers accumulate in each
        thread's live stats dict (registered at its start) and are
        aggregated at the end of the iteration, or of every window of
        a stream, so the bottleneck stays visible."""
        # Serve threads of an EARLIER iteration that would not stop
        # within their grace: re-join them rather than forget them --
        # while one runs it still reads the request queue.
        for th in self._stuck_serve_threads:
            th.join(timeout=10.0)
        self._stuck_serve_threads = [th for th in self._stuck_serve_threads
                                     if th.is_alive()]
        if self._stuck_serve_threads:
            log.error(f"iter {iter_idx}: {len(self._stuck_serve_threads)} serve "
                      f"thread(s) of an earlier iteration are still running; this "
                      f"one starts alongside them")
        sv = _Serving(iter_idx=iter_idx, stop_ev=threading.Event(),
                      picker=_BatchPicker(self._coalesce, self._coalesce_gap),
                      t_start=time.monotonic())
        self._picker = sv.picker
        self._open_serving = sv
        # `sv.threads` holds started threads only, so a start that fails
        # (the container's PID limit) leaves a serving that stops cleanly.
        for i in range(self._serve_threads):
            th = threading.Thread(
                target=_serve_loop,
                args=(self._server, sv.picker, self._req_qs[0], self._resp_qs,
                      self._max_batch, self._serve_timeout, sv.stop_ev, sv.serve_stats),
                daemon=True, name=f"serve-{i}")
            th.start()
            sv.threads.append(th)
        for cq in self._server_ctrl_qs:
            cq.put((_SRV_SERVE, iter_idx))
        return sv

    def _stop_serving(self, sv: "_Serving") -> None:
        """Stop the serving behind `sv`; idempotent. Called from every
        exit path of an iteration or a stream, named raise sites
        included: started serve threads used to outlive an iteration
        that ended on any UNNAMED error, and their request queues with
        them (2026-09-13 audit). The serve processes' stats come back
        with their PAUSE reply."""
        if sv.stopped:
            return
        self._stop_serve_threads(sv)
        sv.server_stats.update(self._pause_servers())

    def _stop_serve_threads(self, sv: "_Serving") -> None:
        """Stop and join the in-process serve threads behind `sv`. The
        first half of `_stop_serving`, and all of it that shutdown()
        needs: shutdown stops the serve processes with STOP, not PAUSE."""
        sv.stopped = True
        if self._open_serving is sv:
            self._open_serving = None
        sv.stop_ev.set()
        for th in sv.threads:
            th.join(timeout=10.0)
        slow = [th for th in sv.threads if th.is_alive()]
        if slow:
            # Keeping a thread that would not stop OUT of the
            # accounting is the leak: it still reads the request
            # queue and still writes into its stats.
            log.error(f"iter {sv.iter_idx}: serve thread(s) "
                      f"{[th.name for th in slow]} did not stop within 10s; "
                      f"their stats are incomplete and they keep reading the "
                      f"request queue until they do")
        self.last_stuck_serve_threads = [th.name for th in slow]
        self._stuck_serve_threads += slow

    def _serve_snapshot(self, sv: "_Serving") -> Tuple[List[Dict], Dict[str, int], List[int]]:
        """The serving's stats so far: every thread's stats dict (the
        in-process threads' live dicts copied, each serve process's
        as it replies to STATS, or as its PAUSE reply left them once
        serving stopped), the pickers' telemetry summed, and the
        leaves served per server."""
        from tools.serve_worker import snapshot_stats
        threads = [snapshot_stats(s) for s in sv.serve_stats]
        leaves_per_server = [sum(int(s.get("leaves", 0)) for s in threads)]
        pick = _picker_stats(sv.picker)
        if sv.stopped:
            per_server = sv.server_stats
        else:
            per_server = self._server_stats_live(sv.iter_idx)
        for sid in self._server_ids():
            ss = per_server.get(sid) or {}
            ts = list(ss.get("threads", []))
            leaves_per_server.append(sum(int(s.get("leaves", 0)) for s in ts))
            threads.extend(ts)
            for k, v in (ss.get("picker") or {}).items():
                pick[k] = pick.get(k, 0) + int(v)
        return threads, pick, leaves_per_server

    def _server_stats_live(self, iter_idx: int) -> Dict[int, Dict]:
        """Every live serve process's stats while it serves (STATS
        command; a dead one contributes nothing)."""
        if not self._server_procs:
            return {}
        live = [sid for sid in self._server_ids() if self._server_procs[sid - 1].is_alive()]
        for sid in live:
            self._server_ctrl_qs[sid - 1].put((_SRV_STATS,))
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
                self._abort_on_dead_servers(iter_idx, [sid], failures={sid: str(payload)})
        if pending:
            log.error(f"serve process(es) {sorted(pending)} did not return their stats")
        return got

    def _record_serve_window(
        self, iter_idx: int, threads: List[Dict], pick: Dict[str, int],
        leaves_per_server: List[int], server_stats: Dict[int, Dict],
        *, t_start: float, elapsed: float, n_games: int, n_exps: int,
        ds0: int, total_decisions: int, finish_times: List[float],
        distill_dicts: List[Dict],
    ) -> None:
        """Turn one serving window's stats (an iteration's, or the span
        between two learner steps of a stream) into the `last_*`
        readbacks the loops record, and log them. `threads` holds one
        stats dict per serve thread of every server, already reduced
        to the window (a stream diffs two snapshots)."""
        self.last_leaves_per_server = leaves_per_server
        # A serve thread that died inside its loop still ships its
        # stats (tools/serve_worker._serve_loop's finally), so the
        # manager can say so instead of silently serving the rest of
        # the campaign at half the threads it was given.
        self.last_serve_thread_errors = [str(s["error"]) for s in threads if s.get("error")]
        for tb in self.last_serve_thread_errors:
            log.error(f"iter {iter_idx}: a serve thread died mid-iteration; the "
                      f"pool served this iteration with one thread fewer:\n{tb}")
        # Each server's compiled packed trunk state (bench_pool records
        # it): the learner's, then each serve process's own copy; None
        # where a server's stats never arrived.
        pc = getattr(self._inference_base(), "packed_compile_stats", None)
        self.last_packed_compile_per_server: List[Optional[Dict]] = [
            pc() if pc is not None else {"active": False}]
        for sid in self._server_ids():
            ss = server_stats.get(sid)
            self.last_packed_compile_per_server.append(
                None if ss is None else dict(ss.get("packed_compile") or {"active": False}))
        # Advance the global anneal counter by the decisions generated
        # (sum across actors), so the combat-oracle bias keeps annealing
        # across the campaign instead of freezing at ds0.
        self._advance_decision_step(total_decisions)
        # Mean of per-actor (or, on a stream, per-game) means; None-
        # valued et_* fields are skipped. Consumed by sim_self_play's
        # iteration telemetry in place of the learner-side drain (which
        # never searches under the pool).
        self.last_distill_stats = None
        if distill_dicts:
            keys = set().union(*(d.keys() for d in distill_dicts))
            out = {}
            for k in keys:
                vals = [d[k] for d in distill_dicts if d.get(k) is not None]
                out[k] = (sum(vals) / len(vals)) if vals else None
            self.last_distill_stats = out
        agg = {k: sum(s.get(k, 0) for s in threads)
               for k in ("wait", "unpack", "infer", "wire", "put", "gpu_ms",
                         "leaves", "batches", "requests", "tokens", "padded",
                         "t_encode", "t_forward", "t_priors", "t_finish", "t_reply")
               } if threads else {}
        served = int(agg.get("leaves", 0))
        elapsed = max(1e-9, elapsed)
        self.last_leaf_timeline = _merge_timelines(
            [s.get("timeline", []) for s in threads], t_start)
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
                 f"{n_games} games, {n_exps} experiences, "
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

    def _scan_liveness(self, iter_idx: int, outstanding: set) -> set:
        """The dead among `outstanding` actors, and the dead serve
        processes. A serve process that died aborts (its actors would
        wait forever); an actor killed before its `finally` ran
        (segfault, OOM-kill, guard trip mid-teardown) aborts loudly --
        a silent drop hid the death from the run's exit code (round-35
        C0); an actor that exited clean without reporting is returned
        for the caller to drop."""
        dead_servers = [sid for sid in self._server_ids()
                        if not self._server_procs[sid - 1].is_alive()]
        if dead_servers:
            self._abort_on_dead_servers(iter_idx, dead_servers)
        dead = {aid for aid in outstanding if not self._procs[aid].is_alive()}
        if dead:
            codes = {aid: self._procs[aid].exitcode for aid in sorted(dead)}
            if any(c not in (0, None) for c in codes.values()):
                raise ActorFatalError(
                    f"iter {iter_idx}: actor(s) died without reporting done, "
                    f"exitcodes {codes} -- aborting the iteration instead of "
                    f"silently degrading.")
            for aid in sorted(dead):
                log.error(f"iter {iter_idx}: actor {aid} died without reporting done "
                          f"(exitcode={self._procs[aid].exitcode}); dropping it.")
        return dead

    def _run_iteration(
        self, iter_idx: int, games_per_iter: int, base_seed: int,
    ) -> Tuple[List, List]:
        ds0 = self._global_decision_step()
        outcomes: List = []
        experiences: List = []
        distill_dicts: List[Dict] = []      # per-actor drained means
        outstanding = set(range(self._n))   # actors not yet _R_DONE
        total_decisions = 0                 # summed across actors this iter
        finish_times: List[float] = []      # per-game wall time since t_start
        drained = False                     # soft deadline fired?
        self._last_abandoned = 0            # discard telemetry (A6)
        sv = self._start_serving(iter_idx)
        t_start = sv.t_start
        last_liveness = t_start
        self._post_tickets(iter_idx, games_per_iter, base_seed)
        for aid in range(self._n):
            self._ctrl_qs[aid].put(self._play_command(iter_idx, games_per_iter, base_seed, aid))
        try:
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
                    offer = getattr(self._policy, "offer_holdout_game", None)
                    if offer is None or not offer(payload):
                        # Boundary-pair harvest (T1-F): this drain never
                        # called it, so boundary telemetry read n=0
                        # through the ENTIRE leg-4 campaign (workflow
                        # finding 2026-08-20). Valid here because each
                        # _R_EXPS payload is one game in recorded order.
                        _hv = getattr(self._policy, "harvest_boundary_pairs", None)
                        if _hv is not None:
                            _hv(payload)
                        experiences.extend(payload)
                elif kind == _R_GAME:
                    pass                    # the stream's per-game report; the
                    #                         iteration reads the done report
                elif kind == _R_DONE:
                    n_dec, dstats, done_iter = _done_report(payload)
                    # The decisions count wherever they were made: the
                    # anneal counter tracks generated decisions, and the
                    # games behind them are kept too (they arrived on
                    # this iteration's result queue).
                    total_decisions += int(n_dec or 0)
                    if done_iter is not None and done_iter != iter_idx:
                        # But an actor the PREVIOUS iteration abandoned
                        # at its hard deadline is reporting THAT
                        # iteration done while this one collects:
                        # retiring it here would end this iteration
                        # without the actor having played a game of it,
                        # and shift every later iteration by one
                        # (2026-09-13 audit). Its per-decision means
                        # describe another iteration, so they are
                        # dropped rather than averaged in.
                        log.warning(f"iter {iter_idx}: actor {aid} reported iteration "
                                    f"{done_iter} done (abandoned earlier); it stays "
                                    f"outstanding for this one")
                    else:
                        outstanding.discard(aid)
                        if dstats:
                            distill_dicts.append(dstats)
                elif kind == _R_ERROR:
                    log.error(f"actor {aid} error:\n{payload}")
                elif kind == _R_FATAL:
                    # The traceback carries the SIM_FORK_GUARD text the
                    # launcher greps; propagate, never log-and-drop.
                    raise ActorFatalError(
                        f"actor {aid} died on a non-swallowable error "
                        f"(round-35 C0):\n{payload}")
                if self._server_procs:
                    # A serve process that failed a command while serving
                    # stays alive (the liveness scan below sees nothing)
                    # and answers none of its actors: read its error reply
                    # now, not at PAUSE after the timeout (2026-09-05
                    # review).
                    failed = self._drain_server_replies()
                    if failed:
                        self._abort_on_dead_servers(iter_idx, sorted(failed), failures=failed)
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
                    outstanding -= self._scan_liveness(iter_idx, outstanding)
        finally:
            self._stop_serving(sv)
            # On every exit path: tickets left by an iteration that
            # aborted would be played by the next PLAY of the same index
            # and its end markers would end that PLAY's actors early.
            self._last_tickets_flushed = self._flush_tickets()
        if self._last_tickets_flushed:
            log.info(f"iter {iter_idx}: {self._last_tickets_flushed} game tickets "
                     f"and end markers left unplayed (drain or dropped actors)")
        threads, pick, leaves_per_server = self._serve_snapshot(sv)
        self._record_serve_window(
            iter_idx, threads, pick, leaves_per_server, sv.server_stats,
            t_start=t_start, elapsed=time.monotonic() - t_start,
            n_games=len(outcomes), n_exps=len(experiences), ds0=ds0,
            total_decisions=total_decisions, finish_times=finish_times,
            distill_dicts=distill_dicts)
        return outcomes, experiences

    def shutdown(self, timeout: float = 15.0) -> None:
        """Stop every actor and serve process and release the pool's
        queues. Each mp.Queue holds a pipe pair and (once written to) a
        feeder thread, so a pool that is dropped without this leaks
        ~2n+3 of both -- which the test suite feels first, several
        pools living in one pytest process (2026-09-13 audit).

        The children get `timeout` seconds in all, not each: joined one
        after another, every child that would not exit added a full
        timeout, 12 minutes for 48 actors at 15 s. While they exit, the
        manager reads and discards what they send it. An actor whose
        results nobody reads any more (the loop raised mid-iteration or
        mid-stream) cannot exit until its queue's feeder thread has
        written them into the pipe (tools/mp_teardown.py), which holds
        64 KiB on Linux and 8 KiB on Windows, while each experience
        carries a whole game state: 9 KB pickled on a 36-hex mini map,
        47-142 KB on ladder maps (measured 2026-09-24). Unread, every
        such actor was terminated after its timeout (CI 2026-09-24)."""
        if not self._started:
            return
        # Serving still open (a stream the caller never stopped, or a
        # serving whose start failed midway): its threads read the
        # request queue closed below, and each would die on the closed
        # queue with a traceback in the log (2026-09-24 CI).
        if self._open_serving is not None:
            log.warning(f"shutdown with serving {self._open_serving.iter_idx} still open; "
                        f"stopping its serve threads first")
            self._stop_serve_threads(self._open_serving)
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
        children = self._procs + self._server_procs
        with discarding([self._result_q, self._server_q], on_message=_log_failure_report):
            join_all(children, timeout)
        end_stragglers(children)
        self._procs = []
        self._server_procs = []
        # Drained: the queues the manager alone writes to (commands,
        # weight blobs, tickets), which can hold megabytes nobody took.
        # Not drained here: the queues a child writes to, where a killed
        # child may have left half a message behind.
        for q in list(self._ctrl_qs) + list(self._server_ctrl_qs) + [self._game_q]:
            close_queue(q, drain=True)
        for q in (list(self._resp_qs) + list(self._req_qs)
                  + [self._result_q, self._server_q]):
            close_queue(q, drain=False)
        self._ctrl_qs = []
        self._resp_qs = []
        self._req_qs = []
        self._server_ctrl_qs = []
        self._result_q = self._server_q = self._game_q = None
        self._started = False
