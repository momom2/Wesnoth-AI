"""Gradient-amplitude tree (signal_profiler v1).

Core mechanics: run the PRODUCTION trainer step per isolated loss
term with the optimizer stubbed to no-op, read the accumulated
(post-clip) gradients, decompose per parameter group, and assemble
the tree. Term isolation is batch surgery through the pipeline's
own magnitude channels — no bespoke loss code.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional

log = logging.getLogger("signal_profiler")

# Parameter-group predicates over `named_parameters()` names, model
# and encoder namespaced as "model." / "encoder.".
PARAM_GROUPS = (
    ("encoder",     lambda n: n.startswith("encoder.")),
    ("value_head",  lambda n: n.startswith("model.value_head")),
    ("actor_head",  lambda n: n.startswith("model.actor_head")),
    ("type_head",   lambda n: n.startswith("model.type_head")),
    ("target_proj", lambda n: n.startswith("model.target_")),
    ("weapon_head", lambda n: n.startswith("model.weapon_head")),
    ("gbc_heads",   lambda n: n.startswith("model.gbc_heads")),
    ("aux_ml",      lambda n: n.startswith(("model.aux_score_head",
                                            "model.moves_left"))),
)


def _group_of(name: str) -> str:
    for g, pred in PARAM_GROUPS:
        if pred(name):
            return g
    return "trunk"


class _NoStepOptimizer:
    """Wraps the real optimizer: zero_grad works (the step needs a
    clean slate), step() is a no-op so accumulated grads survive
    for reading. param_groups delegates for any lr consumers."""

    def __init__(self, inner):
        self._inner = inner

    def zero_grad(self, *a, **k):
        return self._inner.zero_grad(*a, **k)

    def step(self, *a, **k):
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


# Term -> batch surgery. Each entry lists dataclasses.replace kwargs
# applied to EVERY experience (shallow copies; game_state shared,
# read-only in the loss).
_KILL_POLICY = {"policy_weight": 0.0}
_KILL_VALUE = {"value_weight": 0.0}
_KILL_AUX = {"aux_target": None}
_KILL_ML = {"moves_left_target": None}
_KILL_GBC = {"gbc_labels": None}

TERM_SURGERY: Dict[str, Dict] = {
    "policy_distill": {**_KILL_VALUE, **_KILL_AUX, **_KILL_ML,
                       **_KILL_GBC},
    "value_inbatch": {**_KILL_POLICY, **_KILL_AUX, **_KILL_ML,
                      **_KILL_GBC},
    "gbc": {**_KILL_POLICY, **_KILL_VALUE, **_KILL_AUX, **_KILL_ML},
    "aux_margin": {**_KILL_POLICY, **_KILL_VALUE, **_KILL_ML,
                   **_KILL_GBC},
}


def _surgered(batch: List, kwargs: Dict) -> List:
    return [dataclasses.replace(e, **kwargs) for e in batch]


def _named_grads(policy) -> Dict[str, "object"]:
    """name -> grad tensor (post-clip, pre-step), namespaced."""
    out = {}
    base = policy._base if hasattr(policy, "_base") else policy
    for n, p in base._model.named_parameters():
        if p.grad is not None:
            out["model." + n] = p.grad.detach().to("cpu")
    for n, p in base._encoder.named_parameters():
        if p.grad is not None:
            out["encoder." + n] = p.grad.detach().to("cpu")
    return out


def _grad_step(policy, batch) -> Dict[str, "object"]:
    """One production step_mcts with the optimizer stubbed;
    returns named grads. Restores the optimizer either way."""
    base = policy._base if hasattr(policy, "_base") else policy
    trainer = base._trainer
    real_opt = trainer.optimizer
    trainer.optimizer = _NoStepOptimizer(real_opt)
    try:
        policy_queue = getattr(policy, "_queue", None)
        if policy_queue is not None:
            with policy._lock:
                policy._queue = list(batch)
            policy.train_step()
        else:
            trainer.step_mcts(list(batch))
    finally:
        trainer.optimizer = real_opt
    return _named_grads(policy)


def _grads_value_memory(policy, batch) -> Optional[Dict]:
    """Grads of the value-memory step alone (its own train path)."""
    if getattr(policy, "_value_memory_games", 0) <= 0:
        return None
    base = policy._base
    trainer = base._trainer
    real_opt = trainer.optimizer
    trainer.optimizer = _NoStepOptimizer(real_opt)
    try:
        policy._value_memory.clear()
        policy._value_memory_ingest(batch)
        stats = policy.value_memory_step()
        if not stats:
            return None
    finally:
        trainer.optimizer = real_opt
    return _named_grads(policy)


def _norm(t) -> float:
    return float(t.float().norm().item())


def build_tree(policy_factory, batch: List,
               include_value_memory: bool = True) -> Dict:
    """The deliverable. `policy_factory()` returns a FRESH policy
    (same checkpoint) per variant so no state leaks across terms.
    Sequential to bound memory."""
    import torch

    variants = [("total", None)] + [
        (t, s) for t, s in TERM_SURGERY.items()]
    grads: Dict[str, Dict] = {}
    for name, surgery in variants:
        pol = policy_factory()
        b = batch if surgery is None else _surgered(batch, surgery)
        grads[name] = _grad_step(pol, b)
        log.info("term %-15s captured %d grad tensors",
                 name, len(grads[name]))
        del pol
    if include_value_memory:
        pol = policy_factory()
        vm = _grads_value_memory(pol, batch)
        if vm:
            grads["value_memory"] = vm
        del pol

    total = grads["total"]
    names = sorted(total)

    def _flat(g, keys):
        return torch.cat([g[k].reshape(-1) for k in keys
                          if k in g]) if keys else torch.zeros(0)

    tree = {"groups": {}, "terms": {}}
    group_names = {}
    for n in names:
        group_names.setdefault(_group_of(n), []).append(n)

    tot_flat_all = _flat(total, names)
    tot_sq = float(tot_flat_all.pow(2).sum().item()) or 1e-12
    for term, g in grads.items():
        node = {"norm": 0.0, "groups": {}}
        # Flat vector over ALL names, zero-filled where the term
        # produced no grad for a tensor.
        parts = []
        for n in names:
            parts.append(g[n].reshape(-1) if n in g
                         else torch.zeros_like(total[n].reshape(-1)))
        flat = torch.cat(parts)
        node["norm"] = float(flat.norm().item())
        node["cos_total"] = float(
            (flat @ tot_flat_all).item()
            / ((flat.norm().item() or 1e-12)
               * (tot_flat_all.norm().item() or 1e-12)))
        node["proj_frac"] = float((flat @ tot_flat_all).item() / tot_sq)
        for gname, keys in sorted(group_names.items()):
            gp = []
            tp = []
            for n in keys:
                tp.append(total[n].reshape(-1))
                gp.append(g[n].reshape(-1) if n in g
                          else torch.zeros_like(total[n].reshape(-1)))
            gflat = torch.cat(gp)
            tflat = torch.cat(tp)
            gnorm = float(gflat.norm().item())
            tnorm = float(tflat.norm().item())
            node["groups"][gname] = {
                "norm": gnorm,
                "frac_of_total_group": gnorm / (tnorm or 1e-12),
                "cos_total": float(
                    (gflat @ tflat).item()
                    / ((gnorm or 1e-12) * (tnorm or 1e-12))),
            }
        tree["terms"][term] = node
    tree["groups"] = {
        g: {"total_norm": float(_flat(total, keys).norm().item())}
        for g, keys in sorted(group_names.items())}
    tree["n_experiences"] = len(batch)
    return tree
