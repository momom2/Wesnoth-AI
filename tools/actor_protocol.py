"""The actor pool's protocol: the messages between the learner (the pool
manager, tools/actor_pool.py, and its continuous stream,
tools/actor_stream.py), the actor processes (tools/actor_worker.py) and
the serve processes (tools/serve_worker.py), and the error a fatal actor
report raises in the learner.

Standard library only: tools/actor_worker.py imports it at module level,
and a spawned child imports that module before any heavy import
(tests/queue_children.py runs one without torch).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Control-queue commands (main -> actor).
_CMD_PLAY = "play"        # (iter_idx, games_per_iter, base_seed, t2i, f2i, decision_step, ...)
_CMD_STOP = "stop"
_CMD_DRAIN = "drain"      # finish the current game, take no new ones
# (value_center, decision_step): the continuous pool's per-step update,
# applied between games -- the search's value center and the global
# anneal counter, which the barrier pool re-sends with every PLAY.
_CMD_UPDATE = "update"

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
# One per completed game, after its _R_OUTCOME and _R_EXPS: (game index,
# decisions made in it, time.time() at its start and end, the distill
# stats drained for it under the continuous pool, else None, the
# iteration or stream tag of the PLAY it was played under). The tag is
# how a stream tells its own games from those of an iteration that
# ended without collecting them, whose reports reach it on the same
# queue (tools/actor_stream.ActorStream._close_game).
_R_GAME    = "game"
# Under the continuous pool only, one per game right after its start
# stamp: (game index, time.time() at its start, the tag). The stream
# keeps the games in flight from it; the barrier pool has no use for it.
_R_START   = "start"
_R_ERROR   = "error"       # traceback string (non-fatal; logged)
_R_FATAL   = "fatal"       # non-swallowable death (fork guard, ...)


def _done_report(payload) -> Tuple[int, Optional[Dict], Optional[int]]:
    """An actor's _R_DONE payload as (decisions, distill stats,
    iteration). The iteration is None for the older two-field and
    plain-int shapes, which the manager then cannot date."""
    if isinstance(payload, tuple):
        if len(payload) >= 3:
            return payload[0], payload[1], int(payload[2])
        return payload[0], payload[1], None
    return payload, None, None


# Reply marker the manager puts on an actor's reply queue when the
# serve process that actor was assigned to died: (_RID_SERVER_DEAD, the
# tag of the iteration or stream it aborted). The client raises on it
# whatever request it is waiting for under that tag. A marker the actor
# reads under a later PLAY is dropped: the aborted iteration left it
# behind, and the pool starts nothing while a serve process is dead.
_RID_SERVER_DEAD = -1


# Serve-process control commands (main -> serve process).
_SRV_SYNC = "sync"        # (version, state bytes): load these weights
_SRV_SERVE = "serve"      # (iter_idx,): start the serve threads
_SRV_PAUSE = "pause"      # (): stop the serve threads, reply their stats
_SRV_STATS = "stats"      # (): reply the serve threads' live stats, serving on
_SRV_PROBE = "probe"      # (payload,): one infer_batch outside serving
_SRV_STOP = "stop"
# Serve-process replies (serve process -> main), on the shared server queue.
_S_READY = "ready"        # model built
_S_SYNCED = "synced"      # payload: the version loaded
_S_STATS = "stats"        # payload: {"threads": [stats dicts], "picker": {...},
                          #           "packed_compile": model.packed_compile_stats()}
_S_PROBE = "probe"        # payload: wire outputs
_S_ERROR = "error"        # payload: traceback string


class ActorFatalError(BaseException):
    """An actor died on a non-swallowable error (round-35 C0: the
    actor's `finally` reported a clean _R_DONE even when a
    ForkGuardViolation escaped, so the pool topology exited 0 on a
    real fork violation). BaseException for the round-34 reason:
    no log-and-continue handler may eat it."""
