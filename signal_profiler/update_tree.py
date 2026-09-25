"""Update-space attribution (signal_profiler v2).

The gradient tree measures PRE-optimizer amplitudes. Adam
preconditions per-parameter (elementwise 1/sqrt(v_hat)), so the
gradient-norm split does not automatically survive into weight
space. This stage measures the APPLIED update: per isolated term,
run the production step with the REAL optimizer (checkpoint Adam
moments, production clip) on a fresh policy copy, and diff the
weights. It also measures what each term's step DOES to the value
function on two probe sets — imagined (search-consulted) states vs
real recorded states — the transfer question in one number: a term
whose loss lives on real states but moves imagined valuations just
as much is training the search's input blind.

Sum-linearity does NOT hold here (Adam is nonlinear); there is no
residual check. Norms and cosines against the total step are the
comparable quantities.

Momentum handling (v2 smoke finding): one Adam step applies
~0.9*exp_avg + 0.1*grad elementwise-normalized, and the CHECKPOINT
exp_avg — momentum belonging to past batches — dominates any single
step, making every term's applied update look identical. So term
variants (and "total") step with exp_avg ZEROED and the checkpoint
exp_avg_sq preconditioner kept: "this term's own gradient, through
the production preconditioner." One extra variant,
"total_momentum", steps as-loaded — the realistic single applied
step, for scale reference only (no attribution).
"""
from __future__ import annotations

import logging
from typing import Dict, List

from signal_profiler.gradient_tree import (
    TERM_SURGERY, group_of, _surgered,
)

log = logging.getLogger("signal_profiler")


def _base(policy):
    return policy._base if hasattr(policy, "_base") else policy


def _param_snapshot(policy) -> Dict[str, "object"]:
    """name -> cpu fp32 clone, namespaced like the gradient tree."""
    b = _base(policy)
    out = {}
    for n, p in b._model.named_parameters():
        out["model." + n] = p.detach().float().cpu().clone()
    for n, p in b._encoder.named_parameters():
        out["encoder." + n] = p.detach().float().cpu().clone()
    return out


def _values_on(policy, states: List) -> List[float]:
    """Side-to-move value of each state under the TRAINER-side
    model (the weights the step just moved), eval mode."""
    import torch
    b = _base(policy)
    was_training = b._model.training
    b._model.eval()
    vals = []
    try:
        with torch.no_grad():
            for gs in states:
                enc = b._encoder.encode(gs)
                out = b._model(enc)
                vals.append(float(out.value.squeeze().item()))
    finally:
        if was_training:
            b._model.train()
    return vals


def _zero_momentum(policy) -> None:
    """Zero exp_avg in the loaded Adam state; keep exp_avg_sq."""
    for st in _base(policy)._trainer.optimizer.state.values():
        m = st.get("exp_avg")
        if m is not None:
            m.zero_()


def _real_step(policy, batch) -> None:
    """One production step: real optimizer, production clip."""
    b = _base(policy)
    b._trainer.config.grad_clip = 1.0
    queue = getattr(policy, "_queue", None)
    if queue is not None:
        with policy._lock:
            policy._queue = list(batch)
        policy.train_step()
    else:
        b._trainer.step_mcts(list(batch))


def _dv_stats(pre: List[float], post: List[float]) -> Dict:
    d = sorted(abs(a - b) for a, b in zip(pre, post))
    if not d:
        return {"n": 0}
    return {"n": len(d),
            "mean_abs": sum(d) / len(d),
            "p90_abs": d[int(0.9 * len(d))],
            "max_abs": d[-1]}


def build_update_tree(policy_factory, batch: List,
                      consult_states: List, real_states: List,
                      include_value_memory: bool = True,
                      surgeries=None) -> Dict:
    """Per-term applied-update decomposition + value movement on
    the probe sets. Fresh policy per variant (Adam moments reload
    with the checkpoint, so every variant steps from the identical
    optimizer state)."""
    import torch

    ref = policy_factory()
    pre_snap = _param_snapshot(ref)
    names = sorted(pre_snap)
    pre_consult = _values_on(ref, consult_states)
    pre_real = _values_on(ref, real_states)
    has_opt_state = len(_base(ref)._trainer.optimizer.state) > 0
    del ref

    surgeries = surgeries or TERM_SURGERY
    variants = [("total_momentum", None), ("total", None)] + [
        (t, s) for t, s in surgeries.items()]
    deltas: Dict[str, Dict] = {}
    dv: Dict[str, Dict] = {}
    for name, surgery in variants + (
            [("value_memory", "MEMORY")] if include_value_memory
            else []):
        pol = policy_factory()
        if name != "total_momentum":
            _zero_momentum(pol)
        if surgery == "MEMORY":
            if getattr(pol, "_value_memory_games", 0) <= 0:
                log.info("update term value_memory skipped (disabled)")
                del pol
                continue
            _base(pol)._trainer.config.grad_clip = 1.0
            pol._value_memory.clear()
            pol._value_memory_ingest(batch)
            stats = pol.value_memory_step()
            if not stats:
                log.info("update term value_memory skipped "
                         "(memory step returned no stats)")
                del pol
                continue
        else:
            b = batch if surgery is None else _surgered(batch, surgery)
            _real_step(pol, b)
        post = _param_snapshot(pol)
        deltas[name] = {n: (post[n] - pre_snap[n]) for n in names}
        dv[name] = {
            "consult": _dv_stats(pre_consult,
                                 _values_on(pol, consult_states)),
            "real": _dv_stats(pre_real, _values_on(pol, real_states)),
        }
        log.info("update term %-15s done", name)
        del pol, post

    group_names: Dict[str, List[str]] = {}
    for n in names:
        group_names.setdefault(group_of(n), []).append(n)

    def _flat(d, keys):
        return torch.cat([d[k].reshape(-1) for k in keys])

    tot = deltas["total"]
    tot_flat = _flat(tot, names)
    tot_sq = float(tot_flat.pow(2).sum().item()) or 1e-12
    tree = {"has_optimizer_state": has_opt_state,
            "n_consult_states": len(consult_states),
            "n_real_states": len(real_states),
            "terms": {}}
    for term, d in deltas.items():
        flat = _flat(d, names)
        node = {
            "update_norm": float(flat.norm().item()),
            "cos_total": float(
                (flat @ tot_flat).item()
                / ((flat.norm().item() or 1e-12)
                   * (tot_flat.norm().item() or 1e-12))),
            "proj_frac": float((flat @ tot_flat).item() / tot_sq),
            "value_movement": dv[term],
            "groups": {},
        }
        for gname, keys in sorted(group_names.items()):
            gflat = _flat(d, keys)
            tflat = _flat(tot, keys)
            gnorm = float(gflat.norm().item())
            node["groups"][gname] = {
                "norm": gnorm,
                "frac_of_total_group": gnorm
                / (float(tflat.norm().item()) or 1e-12),
            }
        tree["terms"][term] = node
    return tree
