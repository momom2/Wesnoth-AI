"""Always-on component timers for production rollouts (2026-08-06).

Arms the same five component timers `tools/profile_rollout.py` uses
(sim.step, sim.fork, encode, forward, enumerate), but built to run
INSIDE a spool worker for the whole campaign: no CUDA synchronize
(production must not serialize async kernels — on CUDA the forward
attribution is therefore skewed toward whatever reads the result;
workers are CPU, where attribution is exact), and accumulation into
a module-level dict that the worker snapshots into its per-game
heartbeat JSON. Fleet-wide readout: `tools/prof_report.py` over
`spool/stats/w*.json` (works over ssh output too).

Overhead: two perf_counter calls + one closure frame per wrapped
call, ~1-3us, against decisions that spend 1-2s — measured target
<0.1%. Armed only when `WESNOTH_PROF=1` (env-inherited from the
learner's `--prof`, same pattern as WESNOTH_RUN_TAG); unarmed runs
carry zero overhead (nothing is patched).
"""

from __future__ import annotations

import time
from typing import Dict, List

ENV_FLAG = "WESNOTH_PROF"

# {label: [n_calls, total_seconds]} — plain list mutation keeps the
# hot path allocation-free.
_ACC: Dict[str, List[float]] = {}
_ORIGINALS: list = []          # (owner, attr, original) for disarm()


def _timed(orig, label: str):
    acc = _ACC.setdefault(label, [0, 0.0])

    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            acc[0] += 1
            acc[1] += time.perf_counter() - t
    timed._prof_wrapped = True      # idempotence marker
    return timed


def armed() -> bool:
    return bool(_ORIGINALS)


def arm(encoder, model) -> None:
    """Patch the five component seams. Idempotent; call once per
    process after the policy is built. `encoder`/`model` are the
    worker's INFERENCE instances (patched per-instance); the sim
    class and the action enumerator are patched at class/module
    level — safe since the dual-import fix (8b68a25) made module
    identity unique."""
    if armed():
        return
    import tools.mcts as mcts_mod
    from tools.wesnoth_sim import WesnothSim

    def _patch(owner, attr, label):
        orig = getattr(owner, attr)
        if getattr(orig, "_prof_wrapped", False):
            return
        _ORIGINALS.append((owner, attr, orig))
        setattr(owner, attr, _timed(orig, label))

    _patch(WesnothSim, "step", "sim.step")
    _patch(WesnothSim, "fork", "sim.fork")
    _patch(encoder, "encode", "encode")
    _patch(model, "forward", "forward")
    _patch(model, "forward_batch", "forward")
    _patch(mcts_mod, "enumerate_legal_actions_with_priors", "enumerate")


def disarm() -> None:
    """Restore originals and clear accumulators (tests)."""
    while _ORIGINALS:
        owner, attr, orig = _ORIGINALS.pop()
        setattr(owner, attr, orig)
    _ACC.clear()


def snapshot() -> Dict[str, Dict[str, float]]:
    """Cumulative {label: {n, s}} since arm(). Cheap; called per
    heartbeat write."""
    return {label: {"n": int(a[0]), "s": round(a[1], 3)}
            for label, a in _ACC.items()}
