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
    value_auc_by_turn: the same-turn AUC by turn bucket, as
    tools/analysis/value_head_by_phase.py reads it for obs8: at each turn
    both sides played, whether the winner's first decision of the turn is
    valued above the loser's, averaged per game in the bucket, then over
    games. The first decision follows the moves the engine makes for
    standing orders at the turn start, where that tool reads the turn's
    starting state, and a game without a leader is read to its end, where
    that tool stops: the two agree closely, not exactly;
  - belief: the belief loss per position;
  - belief_paired: per game, the belief loss at the largest size minus at
    0 slots, both sides pooled, with its standard error across games (the
    memory's crash barrier reads it);
  - belief_carried: at the largest size, the belief loss with the memory
    carried minus with the memory reset to its initial state at every
    decision, per game: what the carried state adds beyond the memory's
    tokens.

The last-seen baseline scores the belief targets with two rates fitted on
training game-sides: the chance that a hidden enemy unit stands on a hex
where the side last saw an enemy it does not see now, and the chance on any
other hex with no visible unit. A last sighting is forgotten once the side
sees its hex empty. Positions whose turn ran out of time count for the
value and the belief, not for the policy. Every reading counts the
non-finite values it met (`n_nonfinite`); the memory barrier fails on any.
Proxies, never verdicts.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from wesnoth_ai.encoder import PARITY_HEX_SEEN_AT
from wesnoth_ai.imitation_loss import TIMEOUT_LABEL, build_imitation_targets, imitation_loss_parts
from wesnoth_ai.sequence_loss import belief_loss
from wesnoth_ai.sequence_streams import GameSide

log = logging.getLogger("sequence_probe")

# Holdout game-sides stepped side by side; a game-side that ends hands its
# row to the next one.
PROBE_BATCH = 48
# The same-turn AUC's turn buckets (tools/analysis/value_head_by_phase.py).
TURN_BUCKETS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10 ** 6))


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
    enemy unit it does not see now, and where its sighting tokens stand. A
    last sighting whose hex the side now sees empty is forgotten: the unit
    moved on or died in view."""
    last: Dict[str, Tuple[int, int]] = {}
    out: List[Set[Tuple[int, int]]] = []
    for pos in positions:
        raw = pos.raw
        visible = set()
        for uid, side_id, p in zip(raw.unit_ids, raw.unit_side_ids, raw.unit_positions):
            if int(side_id) == 1:
                last[uid] = (p.x, p.y)
                visible.add(uid)
        seen_now = raw.hex_dynamic_flags[:, PARITY_HEX_SEEN_AT] > 0.5
        empty_in_view = {(h.x, h.y) for h, seen, empty in zip(raw.hex_positions, seen_now, pos.no_visible_unit)
                         if seen and empty}
        for uid in [u for u, h in last.items() if u not in visible and h in empty_in_view]:
            del last[uid]
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


class _Running:
    """A game-side being probed: its positions, the next one's index, and
    its memory."""
    __slots__ = ("side", "positions", "t", "memory")

    def __init__(self, side: GameSide, positions: Sequence, memory):
        self.side, self.positions, self.t, self.memory = side, positions, 0, memory


def run_sides(model, encoder, sides: Iterable[GameSide], load: Callable[[GameSide], Sequence], k: int,
              device: torch.device, type_loss_weights: Dict[str, float], autocast_dtype=None,
              reset: bool = False, batch: int = PROBE_BATCH) -> Dict[GameSide, Dict[str, list]]:
    """Game-sides run whole and in order at `k` slots, `batch` of them side
    by side, a finished one's row taken by the next: per game-side, per
    position, the policy cross-entropy (None where the turn ran out of
    time), the expected value, the belief loss and the turn. With `reset`,
    every decision reads the initial memory."""
    out: Dict[GameSide, Dict[str, list]] = {}
    queue = iter(sides)
    running: List[_Running] = []

    def refill() -> None:
        while len(running) < batch:
            g = next(queue, None)
            if g is None:
                return
            out[g] = {"ce": [], "value": [], "belief": [], "turn": []}
            positions = load(g)
            if len(positions):
                running.append(_Running(g, positions, model.initial_memory(k).to(device)))

    refill()
    while running:
        live = list(running)
        positions = [r.positions[r.t] for r in live]
        staged = encoder.stage_raws([p.raw for p in positions], device=device)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            streams = encoder.embed_staged(staged)
            padded = model.forward_embedded(streams, memory=[r.memory for r in live])
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
        for b, r in enumerate(live):
            timeout = ais[b].action_type == TIMEOUT_LABEL
            row = out[r.side]
            row["ce"].append(None if timeout else ce[b])
            row["value"].append(values[b])
            row["belief"].append(bl[b])
            row["turn"].append(int(getattr(positions[b], "turn", 0)))
            if padded.memory is not None and not reset:
                r.memory = padded.memory[b].detach()
            r.t += 1
        running[:] = [r for r in running if r.t < len(r.positions)]
        refill()
    return out


def _nonfinite(xs: Iterable) -> int:
    return sum(1 for x in xs if x is not None and not math.isfinite(x))


def probe(model, encoder, holdout: Sequence[GameSide], load: Callable[[GameSide], Sequence],
          winners: Dict[str, int], ks: Sequence[int], device: torch.device,
          type_loss_weights: Dict[str, float], rates: Optional[Tuple[float, float]] = None,
          autocast_dtype=None) -> Dict:
    """The probe's readings at each of `ks` (see the module docstring)."""
    was_training = model.training
    model.eval()
    encoder.eval()
    results: Dict = {"n_game_sides": len(holdout)}
    per_k: Dict[object, Dict[GameSide, Dict[str, list]]] = {}
    top = max(ks) if ks else 0
    runs = [(k, k, False) for k in ks] + ([(f"{top}_reset", top, True)] if top > 0 else [])
    try:
        with torch.no_grad():
            for key, k, reset in runs:
                t0 = time.time()
                per_k[key] = run_sides(model, encoder, holdout, load, k, device, type_loss_weights,
                                       autocast_dtype, reset=reset)
                log.info("probe at %s slots: %d game-sides in %.0f s", key, len(holdout), time.time() - t0)
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
        nonfinite = sum(_nonfinite(r["ce"]) + _nonfinite(r["value"]) + _nonfinite(r["belief"])
                        for r in sides.values())
        results[f"k{k}"] = {"ce_all": m_all, "ce_all_se": se_all, "ce_winners": m_win,
                            "value_auc": auc_m, "value_auc_se": auc_se, "belief": b_m,
                            "belief_se": b_se, "n_positions": len(belief), "n_decisions": len(ce_all),
                            "n_nonfinite": nonfinite, "value_auc_by_turn": same_turn_auc(sides, winners)}
    if top > 0 and 0 in per_k:
        results["belief_paired"] = dict(k=top, **_paired_by_game(per_k[top], per_k[0]))
        results["belief_carried"] = dict(k=top, **_paired_by_game(per_k[top], per_k[f"{top}_reset"]))
    if rates is not None:
        base = []
        for g in holdout:
            base.extend(baseline_belief(load(g), rates))
        results["belief_baseline"] = _mean_se(base)[0]
        results["baseline_rates"] = list(rates)
    return results


def _paired_by_game(a: Dict[GameSide, Dict[str, list]], b: Dict[GameSide, Dict[str, list]]) -> Dict:
    """Per game, the mean belief loss of `a` minus `b` over both sides'
    positions; the mean and its standard error across games (the two sides
    of a game are not independent draws)."""
    by_game: Dict[str, List[Tuple[List[float], List[float]]]] = defaultdict(list)
    for g, r in a.items():
        if r["belief"] and b[g]["belief"]:
            by_game[g.file].append((r["belief"], b[g]["belief"]))
    diffs = []
    for pairs in by_game.values():
        xa = [x for pa, _ in pairs for x in pa]
        xb = [x for _, pb in pairs for x in pb]
        diffs.append(sum(xa) / len(xa) - sum(xb) / len(xb))
    m, se = _mean_se(diffs)
    return {"diff": m, "se": se, "n_games": len(diffs), "n_nonfinite": _nonfinite(diffs)}


def same_turn_auc(sides: Dict[GameSide, Dict[str, list]], winners: Dict[str, int]) -> Dict[str, Dict]:
    """The same-turn AUC per turn bucket: per game and turn where both
    sides decided, 1 when the winner's first decision of the turn is valued
    above the loser's (0.5 on a tie), averaged per game within the bucket,
    then over games, with the standard error between games."""
    first: Dict[Tuple[str, int], Dict[int, float]] = {}
    for g, r in sides.items():
        seen = set()
        for turn, value in zip(r["turn"], r["value"]):
            if turn not in seen:
                seen.add(turn)
                first.setdefault((g.file, g.side), {})[turn] = value
    out: Dict[str, Dict] = {}
    for lo, hi in TURN_BUCKETS:
        per_game = []
        for file, winner in winners.items():
            w, lo_side = first.get((file, winner)), first.get((file, 3 - winner))
            if not w or not lo_side:
                continue
            scores = [1.0 if w[t] > lo_side[t] else 0.5 if w[t] == lo_side[t] else 0.0
                      for t in w if lo <= t <= hi and t in lo_side]
            if scores:
                per_game.append(sum(scores) / len(scores))
        m, se = _mean_se(per_game)
        out[f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"] = {"auc": m, "se": se, "n_games": len(per_game)}
    return out


def memory_barrier_passes(results: Dict) -> bool:
    """The pre-registered crash barrier: at the largest size the belief loss
    is below the belief loss at 0 slots by more than two standard errors,
    paired over holdout games."""
    paired = results.get("belief_paired") or {}
    if paired.get("diff") is None or paired.get("se") is None or paired.get("n_nonfinite", 0):
        return False
    return paired["diff"] < -2.0 * paired["se"]
