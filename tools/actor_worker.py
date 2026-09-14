"""Actor-process side of the actor pool (tools/actor_pool.py holds the
design overview and the manager).

- The manager -> actor control commands (_CMD_*), the actor -> manager
  result messages (_R_*) and the dead-server reply marker
  (_RID_SERVER_DEAD).
- _IPCInferenceClient: the RemoteModel transport inside an actor.
- _zero_reward, _set_fd_safe_sharing: shared by every generation path
  (tools/selfplay_worker.py imports them through tools.actor_pool).
- _actor_loop: the spawned actor process body (ActorPool.start's
  Process target; spawn pickles it by this module path).

Heavy imports (sim, policies) happen inside _actor_loop, after the
spawn, as before.
"""

from __future__ import annotations

import logging
import dataclasses
import multiprocessing as mp
import os
import queue as _queue
import random
import threading
import traceback
from types import SimpleNamespace
from typing import Dict, Optional

import torch

log = logging.getLogger("actor_pool")

# How often a blocking wait wakes up to check that the learner process
# is still alive (seconds); same guard and same period as the serve
# process (tools/serve_worker._server_loop).
_PARENT_POLL = 2.0

# Control-queue commands (main -> actor).
_CMD_PLAY = "play"        # (iter_idx, games_per_iter, base_seed, t2i, f2i, decision_step, ...)
_CMD_STOP = "stop"
_CMD_DRAIN = "drain"      # finish the current game, take no new ones

# Game tickets (main -> actors, one shared queue): (iter_idx, game
# index, seed); the manager posts every game of the iteration, then
# one end marker per actor. Actors pull until they meet an end
# marker, so an iteration's tail is one game long instead of one
# actor's whole share (2026-09-06; the median game used to finish at
# 40% of the wall with the even split).
_TICKET_END = -1

# Result-queue message kinds (actor -> main).
_R_OUTCOME = "outcome"    # a GameOutcome
_R_EXPS    = "experiences"  # List[MCTSExperience]
_R_DONE    = "iter_done"   # (local_decisions, distill stats, iter_idx)
_R_ERROR   = "error"       # traceback string (non-fatal; logged)
_R_FATAL   = "fatal"       # non-swallowable death (fork guard, ...)
# Reply marker the manager puts on an actor's reply queue when the
# serve process that actor was assigned to died: the client raises on
# it whatever request it is waiting for.
_RID_SERVER_DEAD = -1


# =====================================================================
# Parent liveness
# =====================================================================

def _parent_gone() -> bool:
    """True when the process that spawned this actor has died.

    An actor inherits BOTH ends of every queue it is handed, so its
    control pipe never reaches EOF and a blocking `get()` waits
    forever. `daemon=True` only covers a CLEAN interpreter exit of the
    parent: a kill -9, an OOM-kill or a container-supervisor kill
    leaves the actors running for as long as the box lives, holding
    the container's PID budget -- and a pool that exceeds pids.max
    serves zero leaves (one rental lost that way, 2026-09-04).

    Returns False in the main process (no parent), which is the shape
    the in-process tests drive."""
    parent = mp.parent_process()
    return parent is not None and not parent.is_alive()


def _wait_for_command(ctrl_q):
    """The next control command, or None once the parent is gone."""
    while True:
        try:
            return ctrl_q.get(timeout=_PARENT_POLL)
        except _queue.Empty:
            if _parent_gone():
                return None


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
            try:
                # Bounded so a learner killed WHILE this actor waits for
                # a reply (the likeliest orphan moment: an actor spends
                # ~nine tenths of its cycle blocked right here) cannot
                # park the actor forever. The serve processes exit on
                # the same guard, so nothing would ever answer.
                r_rid, wires = self._resp.get(timeout=_PARENT_POLL)
            except _queue.Empty:
                if _parent_gone():
                    raise RuntimeError(
                        "the learner process is gone; abandoning this request")
                continue
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


def _take_ticket(game_q, ctrl_q, iter_idx: int):
    """The next game of this iteration from the shared queue, honouring
    control commands while waiting. Returns ("game", (index, seed)),
    ("end", None) at the iteration's end marker, ("drain", None) when
    the manager asked for no new games, ("play", cmd) when the next
    iteration's PLAY is already waiting, ("stop", None) on STOP or once
    the parent is gone. Tickets of another iteration are skipped (stale
    after a drain)."""
    while True:
        # Checked before every ticket, not only on an empty queue: an
        # orphaned actor with tickets still queued would otherwise walk
        # them one by one, building a scenario and waiting one
        # inference poll on each, before the empty-queue check let it
        # go (2026-09-14 review).
        if _parent_gone():
            return "stop", None
        try:
            nxt = ctrl_q.get_nowait()
            if nxt[0] == _CMD_STOP:
                return "stop", None
            if nxt[0] == _CMD_DRAIN:
                return "drain", None
            if nxt[0] == _CMD_PLAY:
                # The manager abandoned this actor's iteration at its
                # hard deadline and has already started the next one.
                # Hand the PLAY to the main loop, which owns the
                # per-iteration setup, and end this iteration here:
                # DROPPING it (what this did) left the actor bound to
                # the old iter_idx, where it silently discards every
                # ticket of the new iteration -- end markers included,
                # so the other actors never finish either -- until the
                # next DRAIN (2026-09-13 audit, demonstrated).
                return "play", nxt
            log.warning("actor: unknown control command %r while playing; dropped",
                        nxt[0])
        except _queue.Empty:
            pass
        try:
            t_iter, g, seed = game_q.get(timeout=0.5)
        except _queue.Empty:
            if _parent_gone():
                return "stop", None
            continue
        if t_iter != iter_idx:
            continue
        if g == _TICKET_END:
            return "end", None
        return "game", (g, seed)


def _actor_loop(
    actor_id: int, ctrl_q, req_qs, resp_q, result_q, game_q,
    mcts_cfg, scenario_opts: Dict, max_turns: int,
    max_turns_min,
    pvp_kwargs: Optional[Dict], log_level: int, torch_threads: int,
    turn_cfg=None, gbc_labels: bool = False, pt_cfg=None,
    train_kwargs: dict = None, ground_cfg=None,
) -> None:
    """Persistent actor process body. Builds a seam-backed MCTSPolicy
    once, then loops on the control queue: PLAY -> pull game tickets
    from the shared queue until the iteration's end marker, shipping
    each game's experiences and outcome; STOP -> exit."""
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

    # A PLAY the ticket loop met mid-iteration (the manager abandoned
    # that iteration and started the next one): it is run here rather
    # than read off the control queue.
    pending: Optional[tuple] = None
    while True:
        if pending is not None:
            cmd, pending = pending, None
        else:
            cmd = _wait_for_command(ctrl_q)
            # A stale DRAIN can sit in the queue when the actor finished
            # its quota before the manager's soft deadline fired: skip it
            # (it referred to the PREVIOUS iteration).
            while cmd is not None and cmd[0] == _CMD_DRAIN:
                cmd = _wait_for_command(ctrl_q)
            if cmd is None:
                log.error("actor %d: the learner process is gone; exiting", actor_id)
                return
            if cmd[0] == _CMD_STOP:
                return
        (_, iter_idx, _games_per_iter, _base_seed, t2i, f2i,
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
        # Global feature 5 under fog (visibility.enemy_villages_visible_to);
        # legacy PLAY tuples = the true count, as the seed was trained.
        _fhv = bool(cmd[11]) if len(cmd) > 11 else False
        # Rebuild the encoder each iteration with the freshly-snapshotted
        # vocab so actor indices line up with the server's encoder.
        renc = RemoteEncoder(t2i, f2i, device=cpu,
                             relevant_set=_rset, server_priors=_sp,
                             fog_hides_enemy_villages=_fhv)
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
            while True:
                # Drain contract (user ruling 2026-08-17, A6): when
                # the manager's soft deadline fires it sends DRAIN --
                # finish the game in progress, start no new one. The
                # check sits BETWEEN games so a completed game is
                # never thrown away (the leg-3 waste mode).
                kind, ticket = _take_ticket(game_q, ctrl_q, iter_idx)
                if kind == "stop":
                    return
                if kind == "play":
                    pending = ticket
                    log.warning("actor %d: iteration %d was abandoned by the "
                                "manager, the next one has already started; "
                                "reporting done and running it", actor_id, iter_idx)
                if kind != "game":
                    break
                g, seed = ticket
                # The game's setup depends on (base seed, game index)
                # only, whichever actor plays it.
                rng = random.Random(seed)
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
                gl = f"iter{iter_idx}_g{g}_a{actor_id}"
                outcome = _play_one_game_safe(
                    setup=setup, max_turns=mt, pvp_defaults=pvp,
                    policy=policy, reward_fn=_zero_reward,
                    cost_lookup=cost_lookup, game_label=gl,
                    seed_salt=f"pool:{seed}")     # this game's own combat luck
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
                # The iteration this report belongs to: a done for an
                # ABANDONED iteration can reach the manager while it
                # collects the next one, where it would otherwise be
                # counted as that actor's report and end the iteration
                # early (2026-09-13 audit).
                result_q.put((_R_DONE, actor_id,
                              (local_decisions, dstats, iter_idx)))

