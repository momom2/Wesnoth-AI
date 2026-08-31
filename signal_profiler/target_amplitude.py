"""Target-amplitude measurement (signal_profiler stage 2).

How much learning signal do the distillation targets CARRY, before
any gradient mechanics? For each harvested experience: rebuild the
prior over legal actions (production predict_priors path at the
experience's own decision_step / combat alphas), normalize the
stored visit_counts into the target distribution, and measure their
divergence. Near-zero KL = homeopathic targets: the teacher spoke,
the link whispered.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List

log = logging.getLogger("signal_profiler")


def target_amplitude(policy, batch: List) -> Dict:
    """Aggregate KL(target || prior) and mass-shift stats over the
    batch. Uses the trainer-side model (same weights as inference
    here; no training has run) and the production prior path."""
    import torch
    from wesnoth_ai.action_sampler import (
        enumerate_legal_actions_with_priors,
    )

    base = policy._base if hasattr(policy, "_base") else policy
    kls, shifts, touched = [], [], []
    skipped = 0
    for e in batch:
        vcs = e.visit_counts or []
        total = float(sum(v[3] for v in vcs))
        if total <= 0:
            skipped += 1
            continue
        with torch.no_grad():
            enc = base._inference_encoder.encode(e.game_state)
            out = base._inference_model(enc)
            legal = enumerate_legal_actions_with_priors(
                enc, out, e.game_state,
                decision_step=int(getattr(e, "decision_step", 0)))
        prior = {}
        for la in legal:
            key = (la.actor_index, la.target_index, la.weapon_index,
                   getattr(la, "type_index", None))
            prior[key] = float(la.prior)
        z = sum(prior.values()) or 1e-12
        prior = {k: v / z for k, v in prior.items()}
        kl = 0.0
        shift = 0.0
        ok = True
        for v in vcs:
            key = (v[0], v[1], v[2],
                   v[4] if len(v) > 4 else None)
            t = float(v[3]) / total
            p = prior.get(key)
            if p is None or p <= 0:
                ok = False
                break
            if t > 0:
                kl += t * math.log(t / p)
                shift += abs(t - p)
        if not ok:
            skipped += 1
            continue
        # Mass the target moved off actions it does NOT list.
        listed_prior = sum(
            prior.get((v[0], v[1], v[2],
                       v[4] if len(v) > 4 else None), 0.0)
            for v in vcs)
        shift += (1.0 - listed_prior)
        kls.append(kl)
        shifts.append(shift / 2.0)     # total-variation distance
        touched.append(len(vcs))
    if not kls:
        return {"n": 0, "skipped": skipped}
    kls.sort()
    return {
        "n": len(kls), "skipped": skipped,
        "kl_mean": sum(kls) / len(kls),
        "kl_median": kls[len(kls) // 2],
        "kl_p90": kls[int(0.9 * len(kls))],
        "tv_mean": sum(shifts) / len(shifts),
        "touched_mean": sum(touched) / len(touched),
    }
