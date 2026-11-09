"""The head-temperature floor of stage C (docs/training_design_20261019/
current_design.md §3.4; tools/pg_stage_c.py's part 7): the gradient's slopes
along the four heads' temperature directions and their Fisher, from the
pass's own readings (tools/pg_pass.py `head_slopes`, `head_fisher`), and the
floor sqrt(2K s^T F_s^-1 s) with its SE and less its noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from wesnoth_ai.pg_step import solve

NAN = float("nan")
# tools/pg_pass.py's head names (wesnoth_ai/pg_loss.py HEADS) for pg_step's HEAD_MAPS.
PASS_HEADS = {"actor": "actor_head", "type": "type_head", "target": "target_q_proj", "weapon": "weapon_head.2"}


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
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


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
