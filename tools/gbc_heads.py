"""GBC heads (docs/gbc_spec.md par.3): the reachability predictor.

Rung-1 architecture decision: the heads are a SEPARATE small module
reading the FROZEN trunk's contextualized tokens via a forward hook
on `WesnothModel.encoder` -- zero changes to model.py, zero
checkpoint-identity risk (the review's "trunk features / model.py
unchanged / checkpoint bit-identical: pick two" resolved as the
latter two; the trunk is frozen at rung 1 anyway, so a detached tap
is mathematically identical to an attached head). If GBC graduates
to rung 4 (trunk unfrozen, gradient into the trunk), the heads move
into the model behind a flag with the aux_score peek-and-OR pattern.

Vocabulary after rung 0a (2026-08-14, 500 games): {dies, flips} --
`levels` measured 0.5-1.9% positives, below the pre-registered 5%
bar at every k, PRUNED. Horizons k in {1, 2} game turns (rung-1
scope per review).

Head A (state achievement): C(s,g,k) = sigmoid(MLP([z_g ; z_glob]))
where z_g = pred_embed[predicate] + entity's contextualized token
(unit token by unit.id / hex token by village position -- id-keyed,
never slot indices).

Head B (action-conditioned, staged next): zero-init bilinear logit
correction in the target_logits pointer-attention form; requires the
per-decision label stream (labels exist only for taken actions).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

# The heads class moved to wesnoth_ai/gbc.py (production
# integration 2026-08-14); re-exported here so the offline tap/eval
# tooling and its tests keep one import site.
from wesnoth_ai.gbc import GBC_HORIZONS as HORIZONS  # noqa: E402,F401
from wesnoth_ai.gbc import PRED_IDX  # noqa: E402,F401
from wesnoth_ai.gbc import GBCHeads  # noqa: E402,F401


class TrunkTap:
    """Captures the contextualized token sequence from a frozen
    WesnothModel via a forward hook on its `encoder` (the
    nn.TransformerEncoder trunk). Token layout is the model's fixed
    concatenation order: [hex(H), unit(U), recruit(R), global,
    end_turn] (`_forward_impl`)."""

    def __init__(self, model):
        self._x: Optional[torch.Tensor] = None
        self._h = model.encoder.register_forward_hook(
            lambda _m, _inp, out: setattr(self, "_x", out))

    def remove(self) -> None:
        self._h.remove()

    def slices(self, encoded) -> Dict[str, torch.Tensor]:
        """Split the captured sequence for the LAST forward. Call
        immediately after the model forward that used `encoded`."""
        assert self._x is not None, "no forward captured"
        x = self._x
        H = encoded.hex_tokens.size(1)
        U = encoded.unit_tokens.size(1)
        R = encoded.recruit_tokens.size(1)
        return {
            "hex": x[:, :H],
            "unit": x[:, H:H + U],
            "global": x[:, H + U + R:H + U + R + 1],
        }


def goal_token(encoded, slices: Dict[str, torch.Tensor],
               key: Tuple) -> Optional[torch.Tensor]:
    """The entity's contextualized token for a goal key
    (("u", unit_id) or ("v", x, y)); None when the entity isn't in
    this encoding (e.g. a unit not visible to the encoder's side --
    fog-honest by construction)."""
    if key[0] == "u":
        try:
            i = encoded.unit_ids.index(key[1])
        except ValueError:
            return None
        return slices["unit"][0, i]
    j = encoded.pos_to_hex.get((key[1], key[2]))
    if j is None:
        return None
    return slices["hex"][0, j]


# ---------------------------------------------------------------------
# Evaluation helpers (pure; unit-tested)
# ---------------------------------------------------------------------

def auc(scores: List[float], labels: List[int]) -> float:
    """Rank AUC (Mann-Whitney), tie-averaged, no sklearn."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    vals, inv, cnt = np.unique(s, return_inverse=True,
                               return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(probs: List[float], labels: List[int], bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    import numpy as np
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(out)
