"""Aleatoric-label probe (signal_profiler round 6).

Discriminates two explanations of the winner/loser gradient
anti-parallelism (round 5: cos -0.83..-0.95):

  H1 "near-random head": the head cannot separate the classes
     anywhere -> per-decade outcome AUC ~0.5 even late, and the
     cancellation is undiminished in the endgame.
  H2 "aleatoric labels": +-1 outcomes on undecided positions are
     label noise; the head is fine where labels are informative ->
     AUC rises toward 1 with game phase, and the informative
     late-game decades carry the smaller share of the opposing
     masses.

Per turn decade: value CE / state-blind floor / outcome AUC / n
(straight from Trainer.eval_value_metrics' by_decade, the same
code that now feeds production telemetry), plus the isolated
value-gradient of the decade's winner states vs loser states and
their cosine.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from signal_profiler.gradient_tree import (
    TERM_SURGERY, _grad_step, _surgered,
)

log = logging.getLogger("signal_profiler")

MIN_CLASS_N = 25    # per-class floor for a decade's gradient cosine

DECADE_KEYS = ("d1_10", "d11_20", "d21_30", "d31_40",
               "d41_50", "d51_60", "d61p")


def _decade_key(turn: int) -> str:
    d = min((int(turn) - 1) // 10, 6)
    return DECADE_KEYS[d]


def _cos(g1: Dict, g2: Dict) -> float:
    names = sorted(set(g1) | set(g2))
    dot = n1 = n2 = 0.0
    for n in names:
        a = g1.get(n)
        b = g2.get(n)
        if a is not None:
            n1 += float(a.pow(2).sum().item())
        if b is not None:
            n2 += float(b.pow(2).sum().item())
        if a is not None and b is not None:
            dot += float((a.reshape(-1) @ b.reshape(-1)).item())
    return dot / (((n1 ** 0.5) or 1e-12) * ((n2 ** 0.5) or 1e-12))


def aleatoric_probe(policy_factory, batch: List) -> Dict:
    pol = policy_factory()
    base = pol._base if hasattr(pol, "_base") else pol
    metrics = base._trainer.eval_value_metrics(batch)
    del pol
    out = {
        "pooled": {"ce": metrics["ce"],
                   "floor": metrics["marginal_ce_floor"],
                   "auc": metrics["value_auc"],
                   "n_decisive": metrics["n_decisive"]},
        "by_decade": metrics["by_decade"],
        "grad_by_decade": {},
    }

    def value_grad(sub):
        pol = policy_factory()
        g = _grad_step(pol, _surgered(sub,
                                      TERM_SURGERY["value_inbatch"]))
        del pol
        return g

    buckets: Dict[str, List] = {}
    for e in batch:
        buckets.setdefault(
            _decade_key(e.game_state.global_info.turn_number),
            []).append(e)
    for key in DECADE_KEYS:
        sub = buckets.get(key, [])
        win = [e for e in sub if e.z > 0]
        lose = [e for e in sub if e.z < 0]
        if len(win) < MIN_CLASS_N or len(lose) < MIN_CLASS_N:
            out["grad_by_decade"][key] = {
                "n_win": len(win), "n_lose": len(lose)}
            continue
        gw = value_grad(win)
        gl = value_grad(lose)

        def _norm(g):
            return sum(float(t.pow(2).sum().item())
                       for t in g.values()) ** 0.5

        out["grad_by_decade"][key] = {
            "n_win": len(win), "n_lose": len(lose),
            "norm_win": _norm(gw), "norm_lose": _norm(gl),
            "cos_win_lose": _cos(gw, gl),
        }
        log.info("decade %s: cos(win,lose)=%.3f (n=%d/%d)",
                 key, out["grad_by_decade"][key]["cos_win_lose"],
                 len(win), len(lose))
    return out
