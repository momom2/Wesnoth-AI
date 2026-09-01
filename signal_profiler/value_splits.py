"""Value-gradient provenance splits (signal_profiler v2).

Where does the dominant value gradient COME from? Split the batch
by label side (winner z>0 / loser z<0) and by game phase (turn
terciles), take the isolated value gradient of each sub-batch, and
compare directions. cos(winner, loser) is the proxy question in one
number: if states labeled +1 and states labeled -1 push the trunk
the SAME way, the head is learning a feature that separates games,
not positions (the unit-count-proxy signature from the leg-5
rotation diagnosis).
"""
from __future__ import annotations

import logging
from typing import Dict, List

from signal_profiler.gradient_tree import (
    TERM_SURGERY, _grad_step, _surgered,
)

log = logging.getLogger("signal_profiler")


def _flat_of(grads, names):
    import torch
    return torch.cat([grads[n].reshape(-1) for n in names
                      if n in grads])


def value_grad_splits(policy_factory, batch: List) -> Dict:
    import torch

    def value_grad(sub):
        pol = policy_factory()
        g = _grad_step(pol, _surgered(sub, TERM_SURGERY["value_inbatch"]))
        del pol
        return g

    full = value_grad(batch)
    names = sorted(full)
    full_flat = _flat_of(full, names)

    turns = sorted(e.game_state.global_info.turn_number for e in batch)
    t1 = turns[len(turns) // 3]
    t2 = turns[2 * len(turns) // 3]
    subsets = {
        "winner_states": [e for e in batch if e.z > 0],
        "loser_states": [e for e in batch if e.z < 0],
        "turns_early": [e for e in batch
                        if e.game_state.global_info.turn_number <= t1],
        "turns_mid": [e for e in batch
                      if t1 < e.game_state.global_info.turn_number <= t2],
        "turns_late": [e for e in batch
                       if e.game_state.global_info.turn_number > t2],
    }
    out = {"full_norm": float(full_flat.norm().item()),
           "turn_terciles": [t1, t2], "subsets": {}}
    flats = {}
    for name, sub in subsets.items():
        if not sub:
            out["subsets"][name] = {"n": 0}
            continue
        g = value_grad(sub)
        f = torch.zeros_like(full_flat)
        # zero-filled alignment against the full-grad name order
        off = 0
        for n in names:
            sz = full[n].numel()
            if n in g:
                f[off:off + sz] = g[n].reshape(-1)
            off += sz
        flats[name] = f
        out["subsets"][name] = {
            "n": len(sub),
            "norm": float(f.norm().item()),
            "cos_full": float(
                (f @ full_flat).item()
                / ((f.norm().item() or 1e-12)
                   * (full_flat.norm().item() or 1e-12))),
        }
        log.info("value split %-14s n=%d done", name, len(sub))
    for a, b in [("winner_states", "loser_states"),
                 ("turns_early", "turns_late")]:
        if a in flats and b in flats:
            out[f"cos_{a}|{b}"] = float(
                (flats[a] @ flats[b]).item()
                / ((flats[a].norm().item() or 1e-12)
                   * (flats[b].norm().item() or 1e-12)))
    return out
