"""Stage C of the policy-gradient program
(docs/training_design_20261019/current_design.md §2, §3.4, §5 and §6): from
one gradient pass, read on held-out games; nothing is played or kept.

From pass.pt (build_contracts.md §3) and an Evaluator factory
(wesnoth_ai/pg_protocol.py) over the held-out games:
  1. the gradient's slopes along the head-temperature directions and their
     Fisher (the probe_span arm and the floor): the pass's sampled-score
     Fisher when it carries one (tools/pg_pass.py), else read through the KL;
  2. each arm's direction scaled to one mean per-decision KL eps, the KL
     K_side per side-game, and read by ΔĴ and Ĝ on half A;
  3. the optimizer: another arm replaces reduced_adamw only when its ΔĴ on
     half A exceeds reduced_adamw's by more than 2 SE paired by game (§2),
     both read by importance weights that pass the check (§5);
  4. the ladder on the winner at K_side/2, K_side and 2 K_side, selected on
     half A and confirmed on half B;
  5. the critical batch B with its SE, in reduced AdamW's metric and in the
     winner's when its direction is linear in the gradient (every arm but
     muon);
  6. the gain curve: reduced AdamW's Ĝ on half A from the first 2 and 4 of
     the 8 splits (5,000 and 10,000 games of a 20,000-game pass) against the
     whole pass, each ratio with its SE beside the model's ratio from B;
  7. the floor sqrt(2K s^T F_s^-1 s), as read and less its noise, with its SE;
  8. the gate on round 1's candidate (§6): resolved, rounds at a size the
     gain curve resolves within the round budget, or neither.
The readout JSON is rewritten after each part, so a cut run keeps what it
read. A part, or an arm, that fails is recorded under `errors` (an arm under
its own entry) and the parts that do not need it run.

Usage:
    python tools/pg_stage_c.py --pass pass.pt --factory MODULE:CALLABLE \
        [--factory-arg KEY=VALUE ...] --out stage_c.json \
        --price-per-game DOLLARS --round-fixed-cost DOLLARS [--round-budget 18]

CALLABLE(**factory_args) returns (evaluator_factory, params, sha256): the
factory binds an Evaluator to a direction, params are the weights at theta_k
named as in the pass, and sha256 (or None) is checked against the pass's
`checkpoint_sha256`. On the network it is `pg_eval:network_factory`
(tools/pg_eval.py), with `--factory-arg checkpoint=... games=... labels=...`. The prices are stage 1's: a generated and trained game,
and a round's fixed part (its held-out set, their forwards, its claim share).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wesnoth_ai.pg_protocol import EvaluatorFactory  # noqa: E402
from wesnoth_ai.pg_step import (ARMS, DEFAULT_ARM, SIDE_KL_RANGE, Decision, Direction, HalfReading,  # noqa: E402
                                Reading, decisions_per_side, direction, fisher_quadratic, head_directions,
                                kl_search, ladder, norm, paired_difference, paired_ratio, pooled_gradient,
                                read_candidate, sides_by_half, solve, span_fisher, span_slopes)

REPLACE_SE = 2.0               # an arm replaces the default beyond this many paired SE (§2)
RESOLVED_SE = 2.0              # a reading resolves when it lies this many SE above 0
ROUND_BUDGET = 18.0            # dollars: a quarter of the $73.42 left for rounds (§6)
K_SIDE = 0.25                  # the arms' KL per side-game (§6)
CURVE_SPLITS = (2, 4)          # the gain curve's points besides the whole pass
SPAN_STEP = 0.05               # the head-temperature Fisher's step: a 5% logit scale
LINEAR_ARMS = ("sgd", "reduced_adamw", "heads_only", "probe_span")
# tools/pg_pass.py's head names (wesnoth_ai/pg_loss.py HEADS) for pg_step's HEAD_MAPS.
PASS_HEADS = {"actor": "actor_head", "type": "type_head", "target": "target_q_proj", "weapon": "weapon_head.2"}
NAN = float("nan")


def load_pass(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------- the critical batch

@dataclass(frozen=True)
class CriticalBatch:
    """B = tr(PFP Sigma) / (g^T PFP g) with its two parts and their SEs.
    `status` is "resolved" when both parts read above RESOLVED_SE SE, else
    "noise_unresolved" or "signal_unresolved" and B is NaN: a noise that
    cannot be told from 0 leaves B unknown, not 0."""
    B: float
    B_se: float
    signal: float
    signal_se: float
    noise: float
    noise_se: float
    status: str

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def summary(self) -> Dict:
        return dataclasses.asdict(self)


def critical_batch(split_q: Sequence[float], deviation_q: Sequence[float], total_q: float,
                   split_games: Sequence[int]) -> CriticalBatch:
    """The critical batch of a KL-normalized step (design §2) from Fisher
    quadratic forms q(d) = d^T F d: of the whole pass's direction d = P g, of
    each split's d_i = P g_i, and of each split's deviation d_i - d.

    g_i, the mean gradient of split i's n_i games, has mean g and covariance
    Sigma / n_i; with M = PFP and N = sum n_i:
    - noise: d_i - d = P (g_i - g) has covariance P Sigma P (1/n_i - 1/N),
      since Cov(g_i, g) = Sigma / N, so each split reads tr(M Sigma) as
      q(d_i - d) / (1/n_i - 1/N). The noise is their mean, its SE their
      spread. With equal splits the mean is the design's two-equation
      estimate (mean q_i - q) / (mean 1/n_i - 1/N): the cross terms sum to 0.
    - signal: E[q(d)] = g^T M g + tr(M Sigma) / N. The reading's error is
      mostly its cross term 2 g^T M (g - E g), of variance
      4 g^T M Sigma M g / N; each split's cross form
      b_i = (q_i - q - q(d_i - d)) / 2 = d^T F (d_i - d) has variance about
      g^T M Sigma M g (1/n_i - 1/N), which gives the SE.
    B = noise / signal, its SE by the delta method on the two parts."""
    n = np.asarray(split_games, dtype=np.float64)
    total = n.sum()
    weight = 1.0 / n - 1.0 / total
    deviation = np.asarray(deviation_q, dtype=np.float64)
    per_split = deviation / weight
    noise = float(per_split.mean())
    noise_se = float(per_split.std(ddof=1) / math.sqrt(len(n)))
    cross = (np.asarray(split_q, dtype=np.float64) - total_q - deviation) / 2.0
    signal = float(total_q - noise / total)
    signal_se = 2.0 * math.sqrt(float((cross ** 2 / weight).mean()) / total)
    if not noise > RESOLVED_SE * noise_se:
        return CriticalBatch(NAN, NAN, signal, signal_se, noise, noise_se, "noise_unresolved")
    if not signal > RESOLVED_SE * signal_se:
        return CriticalBatch(NAN, NAN, signal, signal_se, noise, noise_se, "signal_unresolved")
    B = noise / signal
    B_se = B * math.hypot(noise_se / noise, signal_se / signal)
    return CriticalBatch(B, B_se, signal, signal_se, noise, noise_se, "resolved")


def gain_curve(N: float, B: float) -> float:
    """The share of the noise-free first-order gain a step from N games
    reaches at a fixed KL: sqrt(N / (N + B)). The step s = sqrt(2K / q) with
    E[q] = g^T M g (1 + B/N), and its gain s g^T P g, so the gain scales as
    1/sqrt(1 + B/N)."""
    return math.sqrt(N / (N + B))


def gain_ratio(N: float, N_ref: float, B: float) -> float:
    """The model's Ĝ(N)/Ĝ(N_ref), gain_curve(N, B)/gain_curve(N_ref, B)
    (design §5, the gain-curve check)."""
    return math.sqrt(N * (N_ref + B) / (N_ref * (N + B)))


# ---------------------------------------------------------------- the floor

def _pass_head_order(pass_: Mapping, heads: Sequence[str]) -> Optional[List[int]]:
    """The index in the pass's head order (its `head_slopes` keys) of each of
    `heads`, or None when the pass has no head readings or other heads."""
    if "head_slopes" not in pass_ or "head_fisher" not in pass_:
        return None
    order = [PASS_HEADS.get(name) for name in pass_["head_slopes"]]
    if sorted(map(str, order)) != sorted(heads):
        return None
    return [order.index(head) for head in heads]


def fisher_from_pass(pass_: Mapping, heads: Sequence[str]) -> Optional[np.ndarray]:
    """The head-temperature Fisher per decision, ordered as `heads`, from
    tools/pg_pass.py's `head_fisher`: (1/2N) sum over the training decisions
    of the sampled scores' outer products, a per-side-game Fisher, divided
    here by the pass's decisions per side-game. None when the pass lacks it."""
    index = _pass_head_order(pass_, heads)
    if index is None:
        return None
    per_side = pass_["decisions"] / (2.0 * pass_["games"])
    return np.asarray(pass_["head_fisher"], dtype=np.float64)[np.ix_(index, index)] / per_side


def slopes_from_pass(pass_: Mapping, heads: Sequence[str]) -> Optional[List[float]]:
    """The pass's own head slopes, ordered as `heads` (a check on the slopes
    read from `total` and the weights: the two agree up to rounding)."""
    index = _pass_head_order(pass_, heads)
    if index is None:
        return None
    values = list(pass_["head_slopes"].values())
    return [float(values[i]) for i in index]


def floor(s: np.ndarray, F_s: np.ndarray, K: float) -> float:
    """sqrt(2K s^T F_s^-1 s): the first-order gain of the best step at KL K
    within the span of the directions whose slopes are s and Fisher F_s
    (design §3.4). By Cauchy-Schwarz in the Fisher metric the best step over
    every parameter gains at least as much. F_s and K share one
    normalisation (both per decision, or both per side-game)."""
    s = np.asarray(s, dtype=np.float64)
    return math.sqrt(2.0 * K * max(float(s @ solve(np.asarray(F_s, dtype=np.float64), s)), 0.0))


def slope_covariance(total_slopes: np.ndarray, split_slopes: Sequence[np.ndarray],
                     split_games: Sequence[int]) -> np.ndarray:
    """C, the per-game covariance of the slopes, from the splits:
    sum_i n_i (s_i - s)(s_i - s)^T / (k - 1), unbiased since
    E[n_i (s_i - s)(s_i - s)^T] = C (1 - n_i / N)."""
    n = np.asarray(split_games, dtype=np.float64)
    deviations = np.asarray(split_slopes, dtype=np.float64) - np.asarray(total_slopes, dtype=np.float64)
    return np.einsum("i,ij,ik->jk", n, deviations, deviations) / (len(n) - 1)


@dataclass(frozen=True)
class FloorReading:
    floor: float
    floor_se: float
    floor_debiased: float
    quadratic: float            # Q = s^T F_s^-1 s
    quadratic_se: float
    quadratic_noise: float      # E[Q] - Q_true = tr(F_s^-1 C) / N

    def summary(self) -> Dict:
        return dataclasses.asdict(self)


def floor_reading(total_slopes: np.ndarray, split_slopes: Sequence[np.ndarray], split_games: Sequence[int],
                  F_s: np.ndarray, K: float) -> FloorReading:
    """The floor with its SE, and less its noise. The slopes s have
    covariance C / N, and Q = s^T F_s^-1 s has gradient 2 F_s^-1 s, so
    Var(Q) is about 4 s^T F_s^-1 (C / N) F_s^-1 s and the floor
    sqrt(2K Q) has SE sqrt(2K) SE(Q) / (2 sqrt(Q)). Q is biased upward by
    tr(F_s^-1 C) / N, which the debiased floor removes. F_s, read on fixed
    held-out positions, is taken as exact."""
    s = np.asarray(total_slopes, dtype=np.float64)
    F_s = np.asarray(F_s, dtype=np.float64)
    N = float(np.sum(split_games))
    C = slope_covariance(s, split_slopes, split_games)
    x = solve(F_s, s)
    Q = float(s @ x)
    Q_se = 2.0 * math.sqrt(max(float(x @ C @ x), 0.0) / N)
    noise = float(np.trace(solve(F_s, C))) / N
    floor_se = math.sqrt(2.0 * K) * Q_se / (2.0 * math.sqrt(Q)) if Q > 0 else NAN
    return FloorReading(floor(s, F_s, K), floor_se, math.sqrt(2.0 * K * max(Q - noise, 0.0)), Q, Q_se, noise)


# ---------------------------------------------------------------- the optimizer and the gate

def choose_arm(exact: Mapping[str, Reading], readable: Mapping[str, bool], default: str = DEFAULT_ARM,
               threshold: float = REPLACE_SE) -> Tuple[str, Dict[str, Dict], str]:
    """The optimizer by design §2's criterion: `default` unless another arm's
    ΔĴ exceeds it by more than `threshold` paired SE; among several such
    arms, the largest difference. ΔĴ reads an arm only when its importance
    weights pass the check (§5): an unreadable arm cannot replace the
    default, and with the default unreadable no arm is compared. Returns the
    winner, each arm's paired difference and the reason."""
    if not readable[default]:
        return default, {}, f"{default}'s importance weights fail the check: no arm is compared"
    table, winner, margin = {}, default, -math.inf
    for arm, reading in exact.items():
        if arm == default:
            continue
        diff = paired_difference(reading, exact[default])
        beyond = bool(readable[arm] and diff.estimate > threshold * diff.se)
        table[arm] = {**diff.summary(), "readable": bool(readable[arm]), "beyond": beyond}
        if beyond and diff.estimate > margin:
            winner, margin = arm, diff.estimate
    reason = (f"{winner} beats {default} by {margin:.4g} on half A, beyond {threshold:g} paired SE"
              if winner != default else f"no readable arm beats {default} by {threshold:g} paired SE")
    return winner, table, reason


@dataclass(frozen=True)
class RoundPrices:
    per_game: float                 # dollars a generated and trained game
    fixed: float                    # dollars a round's held-out set, its forwards and claim share
    budget: float = ROUND_BUDGET

    def cost(self, games: float) -> float:
        return self.fixed + self.per_game * games


def round_size_for(z: float, games: float, B: float, target: float = RESOLVED_SE) -> float:
    """The smallest round size at which a reading of z from `games` games
    reaches `target` along the gain curve, z(N) = z gain_ratio(N, games, B),
    with the held-out set and so the SE unchanged: N / (N + B) >= rho with
    rho = (target / z)^2 games / (games + B). Infinite when even the curve's
    asymptote, z sqrt((games + B) / games), stays below the target."""
    if not z > 0:
        return math.inf
    rho = (target / z) ** 2 * games / (games + B)
    return rho * B / (1.0 - rho) if rho < 1.0 else math.inf


def stage_c_gate(decision: Decision, confirm: str, batch: Optional[CriticalBatch], games: float,
                 prices: Optional[RoundPrices]) -> Dict:
    """Design §6's verdict on round 1's candidate, the ladder's rung chosen on
    half A and read by ΔĴ on half `confirm`:
    - "resolved": its lower 2-SE bound is above 0; the planted round P,
      then rounds;
    - "rounds_at_size": it is not, but along the gain curve the reading
      reaches z 2 at a round size costing at most the budget; rounds at
      that size;
    - "neither": no rounds; the user chooses the $1,000 case, sized from
      the curve, or a stop.
    The projection needs a resolved B and the prices; without either an
    unresolved reading is "neither"."""
    def verdict(name: str, reason: str, **extra) -> Dict:
        return {"verdict": name, "reason": reason, **extra}

    if decision.chosen is None:
        return verdict("neither", decision.reason)
    half = decision.rungs[decision.chosen].halves[confirm]
    if not half.ess_ok:
        return verdict("neither", f"half {confirm}'s importance weights fail the check")
    reading = half.exact
    z = reading.estimate / reading.se if reading.se > 0 else NAN
    found = {"estimate": reading.estimate, "se": reading.se, "z": z,
             "lower": reading.estimate - RESOLVED_SE * reading.se}
    if found["lower"] > 0:
        return verdict("resolved", f"half {confirm}'s lower {RESOLVED_SE:g}-SE bound is above 0", **found)
    if prices is None:
        return verdict("neither", "unresolved, and no prices to size a round", **found)
    if batch is None or not batch.resolved:
        return verdict("neither", f"unresolved, and B is {batch.status if batch else 'unread'}", **found)
    size = round_size_for(z, games, batch.B)
    cost = prices.cost(size)
    found.update(round_size=math.ceil(size) if math.isfinite(size) else size, round_cost=cost,
                 budget=prices.budget, B=batch.B)
    if cost <= prices.budget:
        return verdict("rounds_at_size", f"the gain curve reaches z {RESOLVED_SE:g} at {math.ceil(size)} games, "
                                         f"${cost:.2f} a round", **found)
    return verdict("neither", f"z {RESOLVED_SE:g} needs a round of ${cost:.2f}, over ${prices.budget:g}", **found)


# ---------------------------------------------------------------- the run

class StageC:
    """One stage C run; each part fills its section of `readout` and writes
    the file."""

    def __init__(self, pass_: Mapping, params: Mapping[str, torch.Tensor], factory: EvaluatorFactory,
                 out: Path, k_side: float = K_SIDE, halves: Tuple[str, str] = ("A", "B"),
                 span_step: float = SPAN_STEP):
        missing = [name for name in pass_["names"] if name not in params]
        if missing:
            raise ValueError(f"the weights lack {len(missing)} parameters of the pass, e.g. {missing[:3]}")
        self.pass_, self.factory, self.out, self.halves = pass_, factory, Path(out), halves
        self.params = {name: params[name].detach().float().cpu() for name in pass_["names"]}
        self.span_step, self.k_side = span_step, k_side
        self.directions: Dict[str, Direction] = {}
        self.points: Dict = {}
        self.arm_readings: Dict[str, HalfReading] = {}
        self.batches: Dict[str, CriticalBatch] = {}
        self.readout: Dict = {"pass": {k: pass_[k] for k in ("games", "decisions", "windows", "split_games",
                                                              "checkpoint_sha256")},
                              "settings": {"k_side": k_side, "halves": list(halves), "span_step": span_step,
                                           "replace_se": REPLACE_SE, "resolved_se": RESOLVED_SE}}
        self.default_evaluator = factory(self.direction(DEFAULT_ARM))
        sides = self.default_evaluator.game_sides()
        self.by_half = sides_by_half(sides, halves)
        per_side = decisions_per_side(sides)
        self.eps = k_side / per_side
        self.readout["heldout"] = {"game_sides": len(sides), "decisions_per_side": per_side, "eps": self.eps,
                                   "games_by_half": {h: len({gs.game for gs in s}) for h, s in self.by_half.items()}}
        self.write()

    def write(self) -> None:
        tmp = self.out.with_name(self.out.name + ".tmp")
        tmp.write_text(json.dumps(self.readout, indent=1))
        os.replace(tmp, self.out)

    def direction(self, arm: str, span_fisher=None, gradient: Optional[Direction] = None) -> Direction:
        return direction(self.pass_, arm, self.params, span_fisher, gradient)

    def span(self) -> None:
        """Part 1: the head-temperature directions' slopes and Fisher, the
        Fisher from the pass's sampled scores when it has them, else read
        through the KL on the held-out positions."""
        self.heads = head_directions(self.params)
        self.slopes = span_slopes(self.direction("sgd"), list(self.heads.values()))
        from_pass = fisher_from_pass(self.pass_, list(self.heads))
        if from_pass is None:
            self.F_s, source = span_fisher(self.factory, list(self.heads.values()), self.span_step), "kl"
        else:
            self.F_s, source = from_pass, "pass"
        self.readout["span"] = {"heads": list(self.heads), "slopes": self.slopes.tolist(),
                                "fisher": self.F_s.tolist(), "fisher_source": source,
                                "pass_slopes": slopes_from_pass(self.pass_, list(self.heads))}
        self.write()

    def _scaled(self, d: Direction, evaluator=None):
        evaluator = evaluator if evaluator is not None else self.factory(d)
        size = norm(d)
        if size == 0:
            raise ValueError("a zero direction")
        return evaluator, kl_search(evaluator.mean_kl, self.eps, s0=1.0 / size)

    def _read_half_a(self, evaluator, point) -> HalfReading:
        select = self.halves[0]
        return read_candidate(evaluator, point.s, point.kl, {select: self.by_half[select]})[select]

    def arms(self, arms: Sequence[str]) -> str:
        """Parts 2 and 3: every arm at KL eps on half A, and the winner. An
        arm that fails is recorded with its error and left out of the
        comparison; the others are read."""
        table = self.readout.setdefault("arms", {})
        for arm in arms:
            try:
                d = self.direction(arm, span_fisher=self.F_s if arm == "probe_span" else None)
                evaluator, point = self._scaled(d, self.default_evaluator if arm == DEFAULT_ARM else None)
                reading = self._read_half_a(evaluator, point)
            except Exception as e:                    # noqa: BLE001 - an arm fails alone
                table[arm] = {"error": f"{type(e).__name__}: {e}"[:500]}
                self.write()
                continue
            self.directions[arm], self.points[arm], self.arm_readings[arm] = d, point, reading
            table[arm] = {"s": point.s, "kl": point.kl, "evals": point.evals, "norm": norm(d),
                          self.halves[0]: reading.summary()}
            self.write()
        if not self.arm_readings:
            raise RuntimeError(f"no arm of {list(arms)} was read")
        exact = {arm: r.exact for arm, r in self.arm_readings.items()}
        in_range = self.k_side <= SIDE_KL_RANGE
        readable = {arm: in_range and r.ess_ok for arm, r in self.arm_readings.items()}
        if DEFAULT_ARM in exact:
            winner, paired, reason = choose_arm(exact, readable)
        else:
            winner = max((a for a in exact if readable[a]), key=lambda a: exact[a].estimate, default=next(iter(exact)))
            paired, reason = {}, f"{DEFAULT_ARM} not run: the readable arm of highest ΔĴ"
        self.readout["optimizer"] = {"winner": winner, "reason": reason, "paired_vs_default": paired}
        self.write()
        return winner

    def ladder(self, winner: str) -> Decision:
        """Part 4: the winner's ladder."""
        self.decision = ladder(self.factory, self.directions[winner], self.eps, self.halves,
                               s0=self.points[winner].s * math.sqrt(0.5))
        self.readout["ladder"] = {"arm": winner, **self.decision.summary()}
        self.write()
        return self.decision

    def _quadratic(self, d: Direction, s_total: float, size_total: float) -> float:
        """d's Fisher form, read at the step that puts it near the whole
        pass's direction's KL at s_total (steps scaled by the ratio of norms)."""
        return fisher_quadratic(self.factory(d), s_total * size_total / norm(d))

    def critical_batch(self, arm: str) -> CriticalBatch:
        """Part 5 for one arm: the Fisher forms of the whole pass's direction,
        of each split's and of each split's deviation from the whole, read
        near KL eps/4."""
        total = pooled_gradient(self.pass_, range(len(self.pass_["splits"])))
        d_total = self.directions[arm]
        s_total, size = self.points[arm].s / 2.0, norm(d_total)
        q_total = fisher_quadratic(self.factory(d_total), s_total)
        q_splits, q_deviations = [], []
        for i in range(len(self.pass_["splits"])):
            split = pooled_gradient(self.pass_, [i])
            q_splits.append(self._quadratic(self.direction(arm, self.F_s, split), s_total, size))
            deviation = {name: split[name] - total[name] for name in split}
            q_deviations.append(self._quadratic(self.direction(arm, self.F_s, deviation), s_total, size))
        result = critical_batch(q_splits, q_deviations, q_total, self.pass_["split_games"])
        self.batches[arm] = result
        self.readout.setdefault("critical_batch", {})[arm] = {**result.summary(), "q_total": q_total,
                                                              "q_splits": q_splits, "q_deviations": q_deviations}
        self.write()
        return result

    def gain_curve(self, batch: CriticalBatch, splits: Sequence[int] = CURVE_SPLITS) -> None:
        """Part 6: reduced AdamW's Ĝ on half A from the first k splits, each
        against the whole pass's paired by game."""
        games = self.pass_["split_games"]
        full_games = sum(games)
        full = self.arm_readings[DEFAULT_ARM].first_order
        points = []
        for k in splits:
            if k >= len(games):
                continue
            d = self.direction(DEFAULT_ARM, gradient=pooled_gradient(self.pass_, range(k)))
            evaluator, point = self._scaled(d)
            reading = self._read_half_a(evaluator, point).first_order
            ratio, ratio_se = paired_ratio(reading, full)
            n = sum(games[:k])
            points.append({"splits": k, "games": n, "s": point.s, "kl": point.kl,
                           "first_order": reading.summary(), "ratio": ratio, "ratio_se": ratio_se,
                           "model_ratio": gain_ratio(n, full_games, batch.B) if batch.resolved else NAN})
            self.readout["gain_curve"] = {"B": batch.B, "B_status": batch.status, "full_games": full_games,
                                          "full_first_order": full.summary(), "points": points}
            self.write()

    def floor(self) -> None:
        """Part 7: the floor at KL eps, F_s per decision."""
        heads = list(self.heads.values())
        splits = [span_slopes(pooled_gradient(self.pass_, [i]), heads) for i in range(len(self.pass_["splits"]))]
        reading = floor_reading(self.slopes, splits, self.pass_["split_games"], self.F_s, self.eps)
        self.readout["floor"] = {"eps": self.eps, **reading.summary()}
        self.write()

    def gate(self, winner: str, prices: Optional[RoundPrices]) -> Dict:
        """Part 8: the verdict, with B in the winner's metric when it was read,
        else in reduced AdamW's."""
        metric = winner if winner in self.batches else DEFAULT_ARM
        result = stage_c_gate(self.decision, self.halves[1], self.batches.get(metric), self.pass_["games"], prices)
        self.readout["gate"] = {**result, "B_metric": metric,
                                "prices": dataclasses.asdict(prices) if prices else None}
        self.write()
        return result


def run_stage_c(pass_: Mapping, params: Mapping[str, torch.Tensor], factory: EvaluatorFactory, out: Path,
                k_side: float = K_SIDE, arms: Sequence[str] = ARMS, halves: Tuple[str, str] = ("A", "B"),
                span_step: float = SPAN_STEP, curve_splits: Sequence[int] = CURVE_SPLITS,
                prices: Optional[RoundPrices] = None) -> Dict:
    """The whole stage C readout (module docstring), written to `out`. Each
    part fails alone: its error goes to the readout's `errors` and the parts
    that do not need it run; a gate without a ladder reads "neither"."""
    run = StageC(pass_, params, factory, out, k_side, halves, span_step)
    errors = run.readout.setdefault("errors", {})

    def part(name: str, fn, *args):
        try:
            return True, fn(*args)
        except Exception as e:                        # noqa: BLE001 - a part fails alone
            errors[name] = f"{type(e).__name__}: {e}"[:500]
            run.write()
            return False, None

    spanned, _ = part("span", run.span)
    _, winner = part("arms", run.arms, arms)
    laddered = winner is not None and part("ladder", run.ladder, winner)[0]
    if DEFAULT_ARM in run.points:
        _, batch = part("critical_batch", run.critical_batch, DEFAULT_ARM)
        if batch is not None:
            part("gain_curve", run.gain_curve, batch, curve_splits)
    if winner is not None and winner != DEFAULT_ARM and winner in LINEAR_ARMS:
        part(f"critical_batch_{winner}", run.critical_batch, winner)
    if spanned:
        part("floor", run.floor)
    if laddered:
        part("gate", run.gate, winner, prices)
    if "gate" not in run.readout:
        run.readout["gate"] = {"verdict": "neither", "reason": f"the gate could not be read: {sorted(errors)} failed"}
        run.write()
    return run.readout


def _factory_args(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--factory-arg takes KEY=VALUE, got {pair!r}")
        out[key] = value
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pass", dest="pass_path", type=Path, required=True)
    parser.add_argument("--factory", required=True, help="MODULE:CALLABLE returning (factory, params, sha256)")
    parser.add_argument("--factory-arg", action="append", default=[], help="KEY=VALUE for the factory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--price-per-game", type=float, required=True,
                        help="dollars a generated and trained game (stage 1's price)")
    parser.add_argument("--round-fixed-cost", type=float, required=True,
                        help="dollars a round's held-out set, its forwards and its claim share")
    parser.add_argument("--round-budget", type=float, default=ROUND_BUDGET)
    parser.add_argument("--k-side", type=float, default=K_SIDE)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--curve-splits", nargs="*", type=int, default=list(CURVE_SPLITS))
    parser.add_argument("--span-step", type=float, default=SPAN_STEP)
    args = parser.parse_args(argv)
    module, _, name = args.factory.partition(":")
    factory, params, sha = getattr(importlib.import_module(module), name)(**_factory_args(args.factory_arg))
    pass_ = load_pass(args.pass_path)
    if sha is not None and sha != pass_["checkpoint_sha256"]:
        raise SystemExit(f"the evaluator's checkpoint {sha} is not the pass's {pass_['checkpoint_sha256']}")
    prices = RoundPrices(args.price_per_game, args.round_fixed_cost, args.round_budget)
    readout = run_stage_c(pass_, params, factory, args.out, args.k_side, args.arms, span_step=args.span_step,
                          curve_splits=args.curve_splits, prices=prices)
    print(json.dumps({"verdict": readout["gate"]["verdict"], "reason": readout["gate"]["reason"],
                      "winner": readout.get("optimizer", {}).get("winner"),
                      "ladder": readout.get("ladder", {}).get("reason"), "errors": readout["errors"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
