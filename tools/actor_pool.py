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

Windows note: uses the 'spawn' start method (the only one on Windows),
so the actor entry + all Process args must be picklable -- they are
(queues, plain dataclasses/dicts). The model is never sent to actors.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue
import random
import time
import traceback
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger("actor_pool")

# Control-queue commands (main -> actor).
_CMD_PLAY = "play"        # (iter_idx, n_games, base_seed, t2i, f2i)
_CMD_STOP = "stop"

# Result-queue message kinds (actor -> main).
_R_OUTCOME = "outcome"    # a GameOutcome
_R_EXPS    = "experiences"  # List[MCTSExperience]
_R_DONE    = "iter_done"   # actor finished its quota this iteration
_R_ERROR   = "error"       # traceback string (non-fatal; logged)


# =====================================================================
# Actor-side transport
# =====================================================================

class _IPCInferenceClient:
    """RemoteModel transport inside an actor: ship a RawEncoded over
    the shared request queue, block on this actor's private response
    queue for the matching reply. Matching by request id keeps
    `forward_batch` (multiple outstanding) correct; the serial Gumbel
    path has a single outstanding request at a time."""

    def __init__(self, actor_id: int, req_q, resp_q):
        self._aid = actor_id
        self._req = req_q
        self._resp = resp_q
        self._next_id = 0

    def _send(self, raw) -> int:
        rid = self._next_id
        self._next_id += 1
        self._req.put((self._aid, rid, raw))
        return rid

    def infer(self, raw):
        rid = self._send(raw)
        while True:
            r_rid, out = self._resp.get()
            if r_rid == rid:
                return out
            # Stale/out-of-order (only possible after a forward_batch);
            # the batch collector below tolerates it, so drop here.

    def infer_batch(self, raws):
        if not raws:
            return []
        ids = [self._send(r) for r in raws]
        need = set(ids)
        got: Dict[int, object] = {}
        while need:
            r_rid, out = self._resp.get()
            if r_rid in need:
                got[r_rid] = out
                need.discard(r_rid)
        return [got[i] for i in ids]


def _zero_reward(_delta) -> float:
    """MCTS discards per-step shaping (policy.observe is a no-op; z
    comes from the winner in finalize_game). play_one_game still calls
    reward_fn at the terminal, so we return a constant 0.0."""
    return 0.0


def _actor_loop(
    actor_id: int, ctrl_q, req_q, resp_q, result_q,
    mcts_cfg, scenario_opts: Dict, max_turns: int,
    pvp_kwargs: Optional[Dict], log_level: int, torch_threads: int,
) -> None:
    """Persistent actor process body. Builds a seam-backed MCTSPolicy
    once, then loops on the control queue: PLAY -> roll `n_games` and
    ship experiences/outcomes; STOP -> exit."""
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.set_num_threads(max(1, torch_threads))

    # Heavy imports happen here (post-spawn), not at module import time.
    from tools.inference_seam import RemoteEncoder, RemoteModel
    from tools.mcts_policy import MCTSPolicy
    from tools.sim_self_play import _play_one_game_safe, _recruit_cost_lookup
    from tools.scenario_pool import random_setup
    from wesnoth_sim import PvPDefaults

    client = _IPCInferenceClient(actor_id, req_q, resp_q)
    rmodel = RemoteModel(client)
    cost_lookup = _recruit_cost_lookup()
    pvp = PvPDefaults(**pvp_kwargs) if pvp_kwargs else PvPDefaults()
    cpu = torch.device("cpu")

    while True:
        cmd = ctrl_q.get()
        if cmd[0] == _CMD_STOP:
            return
        _, iter_idx, n_games, base_seed, t2i, f2i = cmd
        # Rebuild the encoder each iteration with the freshly-snapshotted
        # vocab so actor indices line up with the server's encoder.
        renc = RemoteEncoder(t2i, f2i, device=cpu)
        base = SimpleNamespace(_inference_model=rmodel,
                               _inference_encoder=renc)
        policy = MCTSPolicy(base, mcts_cfg)
        rng = random.Random(base_seed)
        try:
            for g in range(n_games):
                setup = random_setup(rng, **scenario_opts)
                gl = f"iter{iter_idx}_a{actor_id}_g{g}"
                outcome = _play_one_game_safe(
                    setup=setup, max_turns=max_turns, pvp_defaults=pvp,
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
            result_q.put((_R_DONE, actor_id, None))


# =====================================================================
# Main-side pool manager
# =====================================================================

class ActorPool:
    """Owns the actor processes and runs the central inference-serve
    loop during each rollout iteration. The model stays in the main
    process (`policy._inference_model`); see module docstring."""

    def __init__(
        self, policy, n_actors: int, mcts_cfg, *,
        scenario_opts: Optional[Dict] = None, max_turns: int = 60,
        pvp_defaults=None, device: Optional[torch.device] = None,
        max_batch: Optional[int] = None, serve_timeout: float = 0.005,
        log_level: int = logging.WARNING, actor_torch_threads: int = 1,
        iteration_timeout: Optional[float] = 1800.0,
        liveness_interval: float = 2.0,
    ):
        if n_actors < 1:
            raise ValueError("n_actors must be >= 1")
        self._policy = policy
        self._n = n_actors
        self._mcts_cfg = mcts_cfg
        self._scenario_opts = scenario_opts or {}
        self._max_turns = max_turns
        self._pvp_kwargs = (dict(pvp_defaults.__dict__)
                            if pvp_defaults is not None else None)
        self._device = device
        self._max_batch = max_batch or max(8, n_actors)
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
        self._liveness_interval = liveness_interval
        self._started = False

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        from tools.inference_seam import InferenceServer
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._result_q = ctx.Queue()
        self._ctrl_qs = [ctx.Queue() for _ in range(self._n)]
        self._resp_qs = [ctx.Queue() for _ in range(self._n)]
        self._procs = []
        for aid in range(self._n):
            p = ctx.Process(
                target=_actor_loop,
                args=(aid, self._ctrl_qs[aid], self._req_q,
                      self._resp_qs[aid], self._result_q, self._mcts_cfg,
                      self._scenario_opts, self._max_turns,
                      self._pvp_kwargs, self._log_level,
                      self._actor_threads),
                daemon=True, name=f"actor-{aid}")
            p.start()
            self._procs.append(p)
        self._server = InferenceServer(
            self._policy._inference_model, self._policy._inference_encoder,
            device=self._device, output_device=torch.device("cpu"))
        self._started = True
        log.info(f"actor pool started: {self._n} actors, "
                 f"max_batch={self._max_batch}")

    def _vocab_snapshot(self) -> Tuple[Dict, Dict]:
        enc = self._policy._inference_encoder
        return dict(enc.unit_type_to_id), dict(enc.faction_to_id)

    def run_iteration(
        self, iter_idx: int, games_per_iter: int, base_seed: int,
    ) -> Tuple[List, List]:
        """Broadcast a PLAY command, serve inference until every actor
        reports done, and return (outcomes, experiences)."""
        if not self._started:
            raise RuntimeError("ActorPool.start() not called")
        # Even split of games across actors (+remainder to the first).
        per = [games_per_iter // self._n] * self._n
        for i in range(games_per_iter % self._n):
            per[i] += 1
        t2i, f2i = self._vocab_snapshot()
        for aid in range(self._n):
            self._ctrl_qs[aid].put(
                (_CMD_PLAY, iter_idx, per[aid],
                 base_seed + aid * 1_000_003, t2i, f2i))

        outcomes: List = []
        experiences: List = []
        outstanding = set(range(self._n))   # actors not yet _R_DONE
        served = 0
        t_start = time.monotonic()
        last_liveness = t_start
        while outstanding:
            # 1) Drain results (non-blocking).
            while True:
                try:
                    kind, aid, payload = self._result_q.get_nowait()
                except _queue.Empty:
                    break
                if kind == _R_OUTCOME:
                    outcomes.append(payload)
                elif kind == _R_EXPS:
                    experiences.extend(payload)
                elif kind == _R_DONE:
                    outstanding.discard(aid)
                elif kind == _R_ERROR:
                    log.error(f"actor {aid} error:\n{payload}")
            if not outstanding:
                break
            # 2) Serve one batch of inference requests.
            batch = []
            try:
                batch.append(self._req_q.get(timeout=self._serve_timeout))
            except _queue.Empty:
                # No requests in flight. This is the only place the loop
                # can idle, so run the watchdog here (throttled).
                now = time.monotonic()
                if (self._iteration_timeout is not None
                        and now - t_start > self._iteration_timeout):
                    log.error(
                        f"iter {iter_idx}: wall-clock deadline "
                        f"({self._iteration_timeout:.0f}s) exceeded with "
                        f"actors {sorted(outstanding)} still outstanding; "
                        f"abandoning the iteration with partial results "
                        f"({len(outcomes)} games, {len(experiences)} exps).")
                    break
                if now - last_liveness > self._liveness_interval:
                    last_liveness = now
                    dead = {aid for aid in outstanding
                            if not self._procs[aid].is_alive()}
                    if dead:
                        for aid in sorted(dead):
                            log.error(
                                f"iter {iter_idx}: actor {aid} died without "
                                f"reporting done (exitcode="
                                f"{self._procs[aid].exitcode}); dropping it.")
                        outstanding -= dead
                        if not outstanding:
                            break
                continue
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._req_q.get_nowait())
                except _queue.Empty:
                    break
            outs = self._server.infer_batch([it[2] for it in batch])
            for (aid, rid, _raw), out in zip(batch, outs):
                self._resp_qs[aid].put((rid, out))
            served += len(batch)
        log.info(f"iter {iter_idx}: pool served {served} forwards, "
                 f"{len(outcomes)} games, {len(experiences)} experiences")
        return outcomes, experiences

    def shutdown(self, timeout: float = 15.0) -> None:
        if not self._started:
            return
        for q in self._ctrl_qs:
            try:
                q.put((_CMD_STOP,))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout)
            if p.is_alive():
                log.warning(f"terminating unresponsive actor {p.name}")
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
    from transformer_policy import TransformerPolicy
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
