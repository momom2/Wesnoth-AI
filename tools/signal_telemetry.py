"""Light in-training signal telemetry (user ruling 2026-09-01:
"record the signal profiling telemetry along the timing telemetry
and all the usual signals at all times").

Per iteration, on a SUBSAMPLE of the incoming batch: the gradient
norm of each signal source (policy distill / game-outcome value /
rollout-grounded value / consistency value), unclipped, optimizer
stubbed — the trend view of the offline signal_profiler's gradient
tree. Plus dv_consult: how far the just-applied update moved the
value head's predictions on THIS iteration's search-consulted
states (the erosion gauge; search accepts plans on ~2-atom ≈ 0.08
differences).

Cost: 4 extra backward passes on <=128 states + <=64*2 value
forwards ≈ a few percent of an iteration. The full update-space
tree stays offline (signal_profiler)."""
from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional

log = logging.getLogger("signal_telemetry")

SIG_SUBSAMPLE = 128
SIG_CONSULT_CAP = 64

_KILL = {"policy_weight": 0.0, "aux_target": None,
         "moves_left_target": None, "gbc_labels": None}


class _NoStep:
    def __init__(self, inner):
        self._inner = inner

    def zero_grad(self, *a, **k):
        return self._inner.zero_grad(*a, **k)

    def step(self, *a, **k):
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _surgeries():
    from tools.value_grounding import is_grounding_experience as ig

    def value_where(pred):
        def fn(e):
            kw = dict(_KILL)
            if not pred(e):
                kw["value_weight"] = 0.0
            return kw
        return fn

    def kind(e):
        k = getattr(e, "label_kind", None)
        if k is not None:
            return k
        if not ig(e):
            return "game"
        return "roll" if e.value_weight >= 0.9 else "consist"

    return {
        "sig_policy_norm": lambda e: {
            "value_weight": 0.0, "aux_target": None,
            "moves_left_target": None, "gbc_labels": None},
        "sig_value_game_norm": value_where(lambda e: kind(e) == "game"),
        "sig_value_ground_norm": value_where(lambda e: kind(e) == "roll"),
        "sig_value_consist_norm": value_where(
            lambda e: kind(e) == "consist"),
    }


def _grad_norm(trainer, exps: List) -> float:
    real_opt = trainer.optimizer
    real_clip = trainer.config.grad_clip
    trainer.optimizer = _NoStep(real_opt)
    trainer.config.grad_clip = 1e9
    try:
        trainer.step_mcts(list(exps))
        tot = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum().item())
        enc = getattr(trainer, "encoder", None)
        if enc is not None:
            for p in enc.parameters():
                if p.grad is not None:
                    tot += float(p.grad.detach().pow(2).sum().item())
        return tot ** 0.5
    finally:
        trainer.optimizer = real_opt
        trainer.config.grad_clip = real_clip


def signal_grad_norms(trainer, batch: List, rng) -> Dict[str, float]:
    """Per-source gradient norms on a fixed-size subsample. Runs
    AFTER the real updates (grads are scratch; the next real step
    zero_grads first). Returns {} on any failure — telemetry must
    never kill training."""
    if not batch:
        return {}
    try:
        sub = (batch if len(batch) <= SIG_SUBSAMPLE
               else rng.sample(batch, SIG_SUBSAMPLE))
        out = {}
        for name, surgery in _surgeries().items():
            exps = [dataclasses.replace(e, **surgery(e)) for e in sub]
            out[name] = _grad_norm(trainer, exps)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"signal telemetry failed: {e!r}")
        return {}


def consult_values(base_policy, batch: List,
                   cap: int = SIG_CONSULT_CAP) -> Optional[List]:
    """(states, values) of up to `cap` grounding-captured states
    under the CURRENT trainer-side weights; None when none exist."""
    import torch
    from tools.value_grounding import is_grounding_experience
    states = [e.game_state for e in batch
              if is_grounding_experience(e)][:cap]
    if not states:
        return None
    model, enc = base_policy._model, base_policy._encoder
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            vals = [float(model(enc.encode(gs)).value.squeeze().item())
                    for gs in states]
    finally:
        if was:
            model.train()
    return [states, vals]


def dv_stats(pre, base_policy) -> Dict[str, float]:
    """Mean |dv| on the pre-captured consult states after updates."""
    if not pre:
        return {}
    try:
        states, v0 = pre
        import torch
        model, enc = base_policy._model, base_policy._encoder
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                v1 = [float(model(enc.encode(gs)).value
                            .squeeze().item()) for gs in states]
        finally:
            if was:
                model.train()
        d = [abs(a - b) for a, b in zip(v0, v1)]
        return {"sig_dv_consult_mean": sum(d) / len(d),
                "sig_dv_consult_n": float(len(d))}
    except Exception as e:  # noqa: BLE001
        log.warning(f"dv_consult telemetry failed: {e!r}")
        return {}
