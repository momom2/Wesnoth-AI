"""The sequence trainer's holdout probe (docs/parity_memory_design_20260929.md
"Training"): the holdout game-sides run whole, in order, at several memory
sizes, and report

  - ce_all: the policy cross-entropy per decision, both sides, as
    `supervised_train._evaluate` computes it for `obs8` (the action-type
    weighted actor term plus the type, target and weapon terms, label
    smoothing included), so the two read the same decisions the same way;
    ce_winners: the same over the winners' decisions;
  - value_auc: per game, the probability that a winner-to-move position's
    expected value exceeds a loser-to-move one's, averaged over games;
  - belief: the belief loss per position;
  - belief_paired: per game-side, the belief loss at the largest size minus
    at 0 slots, with its standard error across game-sides (the memory's
    crash barrier reads it).

The last-seen baseline scores the belief targets with two rates fitted on
training game-sides: the chance that a hidden enemy unit stands on a hex
where the side last saw an enemy it does not see now, and the chance on any
other hex with no visible unit. Positions whose turn ran out of time count
for the value and the belief, not for the policy. Proxies, never verdicts.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from wesnoth_ai.imitation_loss import TIMEOUT_LABEL, build_imitation_targets, imitation_loss_parts
from wesnoth_ai.sequence_loss import belief_loss
from wesnoth_ai.sequence_streams import GameSide

# Holdout game-sides stepped side by side in one batch.
PROBE_BATCH = 48


def _mean_se(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, (v / len(xs)) ** 0.5


def _auc(win: Sequence[float], lose: Sequence[float]) -> Optional[float]:
    if not win or not lose:
        return None
    w = np.asarray(win)[:, None]
    lo = np.asarray(lose)[None, :]
    return float(((w > lo).sum() + 0.5 * (w == lo).sum()) / (w.size * lo.size))


def last_seen_hexes(positions: Sequence) -> List[Set[Tuple[int, int]]]:
    """Per position of one game-side: the hexes where the side last saw an
    enemy unit it does not see now, and where its sighting tokens stand."""
    last: Dict[str, Tuple[int, int]] = {}
    out: List[Set[Tuple[int, int]]] = []
    for pos in positions:
        raw = pos.raw
        visible = set()
        for uid, side_id, p in zip(raw.unit_ids, raw.unit_side_ids, raw.unit_positions):
            if int(side_id) == 1:
                last[uid] = (p.x, p.y)
                visible.add(uid)
        hexes = {h for uid, h in last.items() if uid not in visible}
        hexes.update(zip((int(x) for x in raw.sight_xs), (int(y) for y in raw.sight_ys)))
        out.append(hexes)
    return out


def _baseline_counts(positions: Sequence) -> np.ndarray:
    """[[hidden on a last-seen hex, last-seen hexes], [hidden elsewhere,
    other hexes]] over the tokens with no visible unit of one game-side."""
    counts = np.zeros((2, 2), dtype=np.int64)
    for pos, seen in zip(positions, last_seen_hexes(positions)):
        mask = pos.no_visible_unit
        hidden = np.zeros(mask.shape[0], dtype=bool)
        hidden[pos.hidden_tokens] = True
        on_seen = np.array([(p.x, p.y) in seen for p in pos.raw.hex_positions], dtype=bool)
        for row, sel in ((0, on_seen & mask), (1, ~on_seen & mask)):
            counts[row, 0] += int((hidden & sel).sum())
            counts[row, 1] += int(sel.sum())
    return counts


def fit_last_seen_rates(game_sides: Iterable[Sequence]) -> Tuple[float, float]:
    """(rate on last-seen hexes, rate elsewhere) over the given game-sides'
    positions, each clipped away from 0 and 1."""
    counts = np.zeros((2, 2), dtype=np.int64)
    for positions in game_sides:
        counts += _baseline_counts(positions)
    eps = 1e-4
    rates = [min(1 - eps, max(eps, counts[r, 0] / max(1, counts[r, 1]))) for r in (0, 1)]
    return float(rates[0]), float(rates[1])


def baseline_belief(positions: Sequence, rates: Tuple[float, float]) -> List[float]:
    """The last-seen baseline's belief loss per position of one game-side."""
    out = []
    for pos, seen in zip(positions, last_seen_hexes(positions)):
        mask = pos.no_visible_unit
        if not mask.any():
            out.append(0.0)
            continue
        hidden = np.zeros(mask.shape[0], dtype=bool)
        hidden[pos.hidden_tokens] = True
        p = np.array([rates[0] if (h.x, h.y) in seen else rates[1] for h in pos.raw.hex_positions])
        bce = -(hidden * np.log(p) + (~hidden) * np.log(1 - p))
        out.append(float(bce[mask].mean()))
    return out


def run_side_batch(model, encoder, sides: Sequence[Tuple[GameSide, Sequence]], k: int,
                   device: torch.device, type_loss_weights: Dict[str, float],
                   autocast_dtype=None) -> Dict[GameSide, Dict[str, list]]:
    """Game-sides run side by side, whole and in order, at `k` slots:
    per game-side, per position, the policy cross-entropy (None where the
    turn ran out of time), the expected value and the belief loss."""
    out = {g: {"ce": [], "value": [], "belief": []} for g, _ in sides}
    memories = {g: model.initial_memory(k).to(device) for g, _ in sides}
    t = 0
    while True:
        live = [(g, ps) for g, ps in sides if t < len(ps)]
        if not live:
            return out
        positions = [ps[t] for _, ps in live]
        staged = encoder.stage_raws([p.raw for p in positions], device=device)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            streams = encoder.embed_staged(staged)
            padded = model.forward_embedded(streams, memory=[memories[g] for g, _ in live])
        padded = padded.float32()
        ais = [p.label for p in positions]
        zw = [(None, 0.0, 1.0)] * len(ais)
        targets = build_imitation_targets(
            ais, zw, staged.sizes, n_types=padded.type_logits.shape[2],
            n_weapons=padded.weapon_logits.shape[2], n_atoms=padded.value_logits.shape[1],
            type_loss_weights=type_loss_weights, device=device)
        parts = imitation_loss_parts(padded, targets)
        ce = (parts.actor_weighted + parts.type + parts.target + parts.weapon).float().cpu().tolist()
        H_max = padded.belief_logits.shape[1]
        target = torch.zeros(len(positions), H_max)
        mask = torch.zeros(len(positions), H_max)
        for b, p in enumerate(positions):
            H = p.no_visible_unit.shape[0]
            mask[b, :H] = torch.from_numpy(p.no_visible_unit.astype("float32"))
            if p.hidden_tokens.shape[0]:
                target[b, torch.from_numpy(p.hidden_tokens)] = 1.0
        bl = belief_loss(padded.belief_logits, target.to(device), mask.to(device)).cpu().tolist()
        values = padded.value.float().reshape(-1).cpu().tolist()
        for b, (g, _) in enumerate(live):
            timeout = ais[b].action_type == TIMEOUT_LABEL
            out[g]["ce"].append(None if timeout else ce[b])
            out[g]["value"].append(values[b])
            out[g]["belief"].append(bl[b])
            memories[g] = padded.memory[b].detach() if padded.memory is not None else memories[g]
        t += 1


def probe(model, encoder, holdout: Sequence[GameSide], load: Callable[[GameSide], Sequence],
          winners: Dict[str, int], ks: Sequence[int], device: torch.device,
          type_loss_weights: Dict[str, float], rates: Optional[Tuple[float, float]] = None,
          autocast_dtype=None) -> Dict:
    """The probe's readings at each of `ks` (see the module docstring)."""
    was_training = model.training
    model.eval()
    encoder.eval()
    results: Dict = {"n_game_sides": len(holdout)}
    per_k: Dict[int, Dict[GameSide, Dict[str, list]]] = {}
    try:
        with torch.no_grad():
            for k in ks:
                per_k[k] = {}
                for start in range(0, len(holdout), PROBE_BATCH):
                    chunk = [(g, load(g)) for g in holdout[start:start + PROBE_BATCH]]
                    per_k[k].update(run_side_batch(model, encoder, chunk, k, device,
                                                   type_loss_weights, autocast_dtype))
    finally:
        if was_training:
            model.train()
            encoder.train()
    for k, sides in per_k.items():
        ce_all, ce_win, belief, games = [], [], [], defaultdict(lambda: {"w": [], "l": []})
        for g, r in sides.items():
            won = winners[g.file] == g.side
            ces = [c for c in r["ce"] if c is not None]
            ce_all.extend(ces)
            if won:
                ce_win.extend(ces)
            belief.extend(r["belief"])
            games[g.file]["w" if won else "l"].extend(r["value"])
        m_all, _ = _mean_se(ce_all)
        m_win, _ = _mean_se(ce_win)
        game_ce = [_mean_se([c for c in sides[g]["ce"] if c is not None])[0] for g in sides]
        _, se_all = _mean_se(game_ce)
        auc_m, auc_se = _mean_se([_auc(v["w"], v["l"]) for v in games.values()])
        b_m, _ = _mean_se(belief)
        _, b_se = _mean_se([_mean_se(r["belief"])[0] for r in sides.values()])
        results[f"k{k}"] = {"ce_all": m_all, "ce_all_se": se_all, "ce_winners": m_win,
                            "value_auc": auc_m, "value_auc_se": auc_se, "belief": b_m,
                            "belief_se": b_se, "n_positions": len(belief), "n_decisions": len(ce_all)}
    if len(ks) >= 2 and 0 in per_k:
        top = max(ks)
        diffs = [_mean_se(per_k[top][g]["belief"])[0] - _mean_se(per_k[0][g]["belief"])[0]
                 for g in holdout if per_k[0][g]["belief"]]
        d_m, d_se = _mean_se(diffs)
        results["belief_paired"] = {"k": top, "diff": d_m, "se": d_se, "n": len(diffs)}
    if rates is not None:
        base = []
        for g in holdout:
            base.extend(baseline_belief(load(g), rates))
        results["belief_baseline"] = _mean_se(base)[0]
        results["baseline_rates"] = list(rates)
    return results


def memory_barrier_passes(results: Dict) -> bool:
    """The pre-registered crash barrier: at the largest size the belief loss
    is below the belief loss at 0 slots by more than two standard errors,
    paired over game-sides."""
    paired = results.get("belief_paired") or {}
    if paired.get("diff") is None or paired.get("se") is None:
        return False
    return paired["diff"] < -2.0 * paired["se"]
