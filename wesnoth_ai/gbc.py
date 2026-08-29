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


def village_hexes(gs) -> List[Tuple[int, int]]:
    """Every village hex by TERRAIN truth (project round-2 C5: a
    roster built from `_village_owner` -- villages already captured
    -- carried no label row for never-captured villages, so every
    FIRST capture was unlabelable)."""
    from wesnoth_ai.classes import Terrain
    return [(h.position.x, h.position.y) for h in gs.map.hexes
            if Terrain.VILLAGE in h.terrain_types]


def _observable_hexes(gs, side):
    """Hexes `side` observes: the WHOLE BOARD when the game runs
    fogless (project round-4: visible_hexes_for is a pure sight-
    disc union with no _fog branch -- unlike units_visible_to and
    the encoder's gates -- so the --fogless-ratio slice trained the
    event head against labels censored by a disc the game does not
    have; docs/gbc_spec.md defines the label as what the observer
    SEES)."""
    from wesnoth_ai.visibility import visible_hexes_for
    if not getattr(gs.global_info, "_fog", True):
        return {(h.position.x, h.position.y) for h in gs.map.hexes}
    return visible_hexes_for(gs, side)


def observe_state(gs) -> Tuple:
    """Fingerprint bundle of one observed state, cheap enough to
    take at EVERY decision (project round-2 C4/C6: diffing only the
    RECORDED states put every fast-turn event at the wrong turn --
    outside its label window -- and censored deaths at the victim's
    stale previous-state hex). (turn, unit_fp, village_fp, vis1,
    vis2)."""
    return (int(gs.global_info.turn_number),
            unit_fingerprint(gs), village_fingerprint(gs),
            frozenset(_observable_hexes(gs, 1)),
            frozenset(_observable_hexes(gs, 2)))


def diff_events_obs(seq: int, prev: Tuple, cur: Tuple) -> List[Event]:
    """Events realized between two OBSERVATIONS (observe_state
    bundles), observability read from the later one."""
    _turn, cur_u, cur_v, vis1, vis2 = cur
    prev_u, prev_v = prev[1], prev[2]
    out: List[Event] = []

    def observed(hex_) -> FrozenSet[int]:
        return frozenset(
            s for s, v in ((1, vis1), (2, vis2)) if hex_ in v)

    for uid, (side, name, pos, cost, is_leader) in prev_u.items():
        cur_e = cur_u.get(uid)
        if cur_e is None:
            # The OWNER always observes its own unit's death -- the
            # roster/sidebar shrinks even when the hex is fogged
            # (round-4 adjacent finding: a lone unit dying deep in
            # enemy territory took its own sight disc with it and
            # its side labeled 0 for its own loss).
            out.append(Event(seq, _turn, "dies", ("u", uid), side,
                             pos, observed(pos) | {side}, cost=cost,
                             is_leader=is_leader))
        elif cur_e[1] != name:
            # Same roster rule: a side always sees its own unit
            # level.
            out.append(Event(seq, _turn, "levels", ("u", uid),
                             cur_e[0], cur_e[2],
                             observed(cur_e[2]) | {cur_e[0]},
                             cost=cur_e[3]))
    for pos, owner in cur_v.items():
        if prev_v.get(pos, 0) != owner:
            out.append(Event(seq, _turn, "flips", ("v",) + tuple(pos),
                             int(owner), tuple(pos), observed(pos),
                             prev_side=int(prev_v.get(pos, 0))))
    return out


def diff_events(seq: int, prev_u: Dict, prev_v: Dict,
                gs) -> List[Event]:
    """Legacy shape over the obs core (the offline scanner's entry
    point): events between two observed states, observability read
    from the later state."""
    return diff_events_obs(
        seq, (0, prev_u, prev_v, frozenset(), frozenset()),
        observe_state(gs))


# ---------------------------------------------------------------------
# Training-time labels (called from finalize_game)
# ---------------------------------------------------------------------

def labels_for_game_states(
    states: Sequence, sides: Sequence[int],
    final_gs=None,
    trace: Optional[Tuple] = None,
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
        units_visible_to,
    )

    if not states and final_gs is None:
        return []
    # Pass 1: the event stream. `trace` = (events, anchor_map) from
    # MCTSPolicy.note_observation's incremental per-decision diff,
    # so events land at action resolution with their true turn
    # stamp and fog view (project round-2 C4/C6: under TCS only
    # turn_full_prob of side-turns record states, so the recorded-
    # only diff stamped fast-turn events with the NEXT recorded
    # state's turn -- outside their label windows -- and censored
    # deaths at stale hexes). trace=None keeps the legacy
    # recorded-states diff (the offline scanner's convention).
    if trace is None:
        seq_states = list(states) + ([final_gs]
                                     if final_gs is not None else [])
        obs_seq = [observe_state(gs) for gs in seq_states]
        anchor_idx = {id(gs): i for i, gs in enumerate(seq_states)}
        events: List[Event] = []
        for i in range(1, len(obs_seq)):
            events.extend(diff_events_obs(i, obs_seq[i - 1],
                                          obs_seq[i]))
    else:
        events, anchor_idx = trace
    by_key = defaultdict(list)
    for e in events:
        by_key[(e.predicate, e.key)].append(e)

    # Pass 2: per-state label rows.
    out: List[Optional[List[LabelRow]]] = []
    kmax = max(GBC_HORIZONS)
    for gs, side in zip(states, sides):
        turn = int(gs.global_info.turn_number)
        rows: List[LabelRow] = []
        # A recorded state absent from the stream (should not
        # happen; defensive) gets no labels rather than wrong ones.
        _ai = anchor_idx.get(id(gs))
        if _ai is None:
            out.append(None)
            continue

        def _ys(pred: str, key: Tuple, _ai=_ai) -> Tuple[int, ...]:
            evs = by_key[(pred, key)]
            ys = []
            for k in GBC_HORIZONS:
                ys.append(int(any(
                    e.seq > _ai and turn <= e.turn <= turn + k - 1
                    and side in e.observed_by
                    for e in evs)))
            return tuple(ys)

        for u in units_visible_to(gs, side):
            rows.append(("u", u.id, PRED_IDX["dies"])
                        + _ys("dies", ("u", u.id)))
        vis_hex = frozenset(_observable_hexes(gs, side))
        for pos in village_hexes(gs):
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
