"""GBC — event-prediction auxiliary supervision (value-head repair).

Approved by user 2026-08-14 after the rung-0 attribution test
(docs/gbc_spec.md 0d): observed events predict game outcomes at AUC
0.79 while the value head's event-orthogonal turn movement is noise
(AUC 0.53). GBC's production role is therefore DENSE SUPERVISION:
small heads on the shared trunk predict fog-censored event
probabilities (dies / flips within k game turns), and their BCE
gradient teaches the trunk the who-is-in-danger structure the 1-bit
terminal z cannot — the KataGo-ownership-head pattern, grounded in
this project's own measurements.

This module is the layering-clean core (no tools/ imports): the
heads (built into WesnothModel behind the `gbc` flag, aux_score
precedent) and the hindsight label machinery (shared with the
offline scanner `tools/gbc_labels.py`).

Contracts (docs/gbc_spec.md par.2, review amendment A1):
  * fog-censored CONFIRMED achievement: an event labels 1 for an
    observer only if the event hex was visible to that side when it
    happened; the observer is the side-to-move at the anchor state;
  * entities keyed by unit.id / village (x, y) — never slot indices;
  * vocabulary {dies, flips} (rung 0a pruned `levels`), horizons
    k ∈ {1, 2} GAME turns, window = turns [t0, t0+k-1], events
    strictly after the anchor in sequence order.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

PRED_IDX = {"dies": 0, "flips": 1}
GBC_HORIZONS = (1, 2)

# Label-row schema stored on MCTSExperience.gbc_labels (picklable
# plain tuples; entity resolved to a token index at TRAIN time via
# EncodedState.unit_ids / pos_to_hex):
#   ("u", unit_id, pred_idx, y_k1, y_k2)
#   ("v", x, y,    pred_idx, y_k1, y_k2)
LabelRow = Tuple


class GBCHeads(nn.Module):
    """Per-goal achievement predictor: C-logits over horizons from
    [entity token + predicate embedding ; global token]. ~0.4M
    params at d=384 — all representation lives in the trunk, which
    is the point: the gradient flowing THROUGH the entity tokens is
    the product."""

    def __init__(self, d_model: int,
                 horizons: Tuple[int, ...] = GBC_HORIZONS):
        super().__init__()
        self.horizons = tuple(horizons)
        self.pred_embed = nn.Embedding(len(PRED_IDX), d_model)
        self.head_a = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(self.horizons)),
        )

    def batch_a(self, z_entities: torch.Tensor, z_global: torch.Tensor,
                pred_idx: torch.Tensor) -> torch.Tensor:
        """[N, d] entity tokens + [d] global + [N] predicate indices
        -> [N, |horizons|] logits."""
        z_g = z_entities + self.pred_embed(pred_idx)
        glob = z_global.unsqueeze(0).expand(z_g.size(0), -1)
        return self.head_a(torch.cat([z_g, glob], dim=-1))


# ---------------------------------------------------------------------
# Hindsight events (shared with tools/gbc_labels.py's offline scanner)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """One hindsight-observable event. `observed_by` = sides whose
    fog admitted the event hex when it happened; `seq` orders events
    against anchors so the past is never a prediction target."""
    seq:         int
    turn:        int
    predicate:   str
    key:         Tuple
    entity_side: int
    hex:         Tuple[int, int]
    observed_by: FrozenSet[int]
    prev_side:   int = 0
    cost:        int = 0
    is_leader:   bool = False


def unit_fingerprint(gs) -> Dict:
    """id -> (side, name, pos, cost, is_leader). Advancement keeps
    unit.id and changes name, so name-change-same-id is the levels
    event and a level-up is never mistaken for a death."""
    return {u.id: (u.side, u.name, (u.position.x, u.position.y),
                   int(u.cost), bool(u.is_leader))
            for u in gs.map.units}


def village_fingerprint(gs) -> Dict:
    vo = getattr(gs.global_info, "_village_owner", None) or {}
    return dict(vo)


def diff_events(seq: int, prev_u: Dict, prev_v: Dict,
                gs) -> List[Event]:
    """Events realized between two observed states, observability
    read from the later state (the moment after it happened)."""
    from wesnoth_ai.visibility import visible_hexes_for

    cur_u = unit_fingerprint(gs)
    cur_v = village_fingerprint(gs)
    out: List[Event] = []
    vis: Dict[int, FrozenSet] = {}

    def observed(hex_) -> FrozenSet[int]:
        for s in (1, 2):
            if s not in vis:
                vis[s] = frozenset(visible_hexes_for(gs, s))
        return frozenset(s for s in (1, 2) if hex_ in vis[s])

    turn = int(gs.global_info.turn_number)
    for uid, (side, name, pos, cost, is_leader) in prev_u.items():
        cur = cur_u.get(uid)
        if cur is None:
            out.append(Event(seq, turn, "dies", ("u", uid), side, pos,
                             observed(pos), cost=cost,
                             is_leader=is_leader))
        elif cur[1] != name:
            out.append(Event(seq, turn, "levels", ("u", uid), cur[0],
                             cur[2], observed(cur[2]), cost=cur[3]))
    for pos, owner in cur_v.items():
        if prev_v.get(pos, 0) != owner:
            out.append(Event(seq, turn, "flips", ("v",) + tuple(pos),
                             int(owner), tuple(pos), observed(pos),
                             prev_side=int(prev_v.get(pos, 0))))
    return out


# ---------------------------------------------------------------------
# Training-time labels (called from finalize_game)
# ---------------------------------------------------------------------

def labels_for_game_states(
    states: Sequence, sides: Sequence[int],
    final_gs=None,
) -> List[Optional[List[LabelRow]]]:
    """Per stored decision state: the GBC label rows for its
    side-to-move (amendment A1: the observer IS the mover).

    `states` are the game's recorded pre-action states in decision
    order (both sides interleaved) — the same list finalize_game
    already walks. Consecutive-state diffs capture every event in
    between regardless of playout-cap gaps (a death between stored
    states shows up in the net diff). Event turn/observability carry
    the later state's stamp — at playout-cap gaps this is ±1-turn
    approximate, matching the offline scanner's convention.

    Goal roster per state: units visible to the mover (dies) +
    villages visible to the mover (flips) — fog-honest by
    construction, capped for cost (nearest-N is unnecessary: rosters
    are ~10-40 entities).
    """
    from wesnoth_ai.visibility import (
        units_visible_to, visible_hexes_for,
    )

    seq_states = list(states) + ([final_gs] if final_gs is not None
                                 else [])
    if not seq_states:
        return []
    # Pass 1: the event stream.
    events: List[Event] = []
    prev_u = unit_fingerprint(seq_states[0])
    prev_v = village_fingerprint(seq_states[0])
    for i, gs in enumerate(seq_states[1:], start=1):
        events.extend(diff_events(i, prev_u, prev_v, gs))
        prev_u = unit_fingerprint(gs)
        prev_v = village_fingerprint(gs)
    by_key = defaultdict(list)
    for e in events:
        by_key[(e.predicate, e.key)].append(e)

    # Pass 2: per-state label rows.
    out: List[Optional[List[LabelRow]]] = []
    kmax = max(GBC_HORIZONS)
    for i, (gs, side) in enumerate(zip(states, sides)):
        turn = int(gs.global_info.turn_number)
        rows: List[LabelRow] = []

        def _ys(pred: str, key: Tuple) -> Tuple[int, ...]:
            evs = by_key[(pred, key)]
            ys = []
            for k in GBC_HORIZONS:
                ys.append(int(any(
                    e.seq > i and turn <= e.turn <= turn + k - 1
                    and side in e.observed_by
                    for e in evs)))
            return tuple(ys)

        for u in units_visible_to(gs, side):
            rows.append(("u", u.id, PRED_IDX["dies"])
                        + _ys("dies", ("u", u.id)))
        vis_hex = frozenset(visible_hexes_for(gs, side))
        for pos in village_fingerprint(gs):
            if tuple(pos) in vis_hex:
                key = ("v",) + tuple(pos)
                rows.append(("v", pos[0], pos[1], PRED_IDX["flips"])
                            + _ys("flips", key))
        # Horizon-window guard: states within kmax of the game's end
        # keep their labels (a truncated window under-counts at most;
        # the same censoring every RL horizon has at episode end).
        _ = kmax
        out.append(rows if rows else None)
    return out


def gbc_loss_for_output(model, encoded, output,
                        rows: List[LabelRow]) -> Optional[torch.Tensor]:
    """BCE of head-A predictions vs stored label rows for ONE
    experience's forward. Entities resolved id-keyed against THIS
    encoding; rows whose entity is absent (fog/order drift) are
    skipped. Returns None when nothing resolves."""
    if not rows or getattr(model, "gbc_heads", None) is None:
        return None
    if output.unit_ctx is None or output.global_ctx is None:
        return None
    ents, preds, ys = [], [], []
    unit_index = {uid: i for i, uid in enumerate(encoded.unit_ids)}
    for r in rows:
        if r[0] == "u":
            i = unit_index.get(r[1])
            if i is None:
                continue
            ents.append(output.unit_ctx[0, i])
            preds.append(r[2])
            ys.append(r[3:3 + len(GBC_HORIZONS)])
        else:
            j = encoded.pos_to_hex.get((r[1], r[2]))
            if j is None or output.hex_ctx is None:
                continue
            ents.append(output.hex_ctx[0, j])
            preds.append(r[3])
            ys.append(r[4:4 + len(GBC_HORIZONS)])
    if not ents:
        return None
    z = torch.stack(ents)
    pred_idx = torch.tensor(preds, dtype=torch.long, device=z.device)
    target = torch.tensor(ys, dtype=torch.float32, device=z.device)
    logits = model.gbc_heads.batch_a(
        z, output.global_ctx[0, 0], pred_idx)
    return nn.functional.binary_cross_entropy_with_logits(
        logits, target)


__all__ = ["GBCHeads", "Event", "PRED_IDX", "GBC_HORIZONS",
           "unit_fingerprint", "village_fingerprint", "diff_events",
           "labels_for_game_states", "gbc_loss_for_output"]
