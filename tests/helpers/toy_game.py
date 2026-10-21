"""An enumerable two-player game for the policy-gradient step's tests
(design §5, "CI toy game"): the production estimators run on its sampled
games, and its exact values check them.

Play, each player deciding twice with chance between the two rounds:
  1. side 1 picks a1 among 3 actions;
  2. side 2 picks b1, seeing a1;
  3. chance draws c = 1 with probability CHANCE_P, else 0;
  4. side 1 picks a2, seeing (a1, b1, c);
  5. side 2 picks b2, seeing everything.
Side 1 wins with probability q[a1, b1, c, a2, b2], a fixed random table, and
z = +1 or -1 in side 1's frame.

One network plays both seats: logits(h) = w * T[h] over the 3 actions, T the
76 x 3 table "model.table.weight" (a trunk matrix) and w the 1 x 1
"model.actor_head.weight" (a head whose scaling scales the logits).

Labels as the contract's (§2): R_s = sigma_s (z - b - beta S), z, b and the
luck S = c - CHANCE_P in side 1's frame, sigma_1 = +1 and sigma_2 = -1; b
and beta are fixed, so neither biases the estimators.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from wesnoth_ai.pg_protocol import GameSide

ACTIONS = 3
CHANCE_P = 0.4
STATES = 1 + 3 + 18 + 54
TABLE = "model.table.weight"
HEAD = "model.actor_head.weight"
BASELINE, LUCK_COEFFICIENT = 0.1, 0.5
SIDE_OF_DECISION = (1, 2, 1, 2)
Params = Dict[str, torch.Tensor]


def state_ids(a1, b1, c, a2) -> np.ndarray:
    """The information state of each of the four decisions, [..., 4]."""
    a1, b1, c, a2 = (np.asarray(x) for x in (a1, b1, c, a2))
    first = np.zeros_like(a1)
    second = 1 + a1
    third = 4 + (a1 * 3 + b1) * 2 + c
    fourth = 22 + ((a1 * 3 + b1) * 2 + c) * 3 + a2
    return np.stack([first, second, third, fourth], axis=-1)


def initial_params(seed: int = 0) -> Params:
    g = torch.Generator().manual_seed(seed)
    return {TABLE: torch.randn(STATES, ACTIONS, generator=g, dtype=torch.float64) * 0.7,
            HEAD: torch.ones(1, 1, dtype=torch.float64)}


def step_params(params: Mapping[str, torch.Tensor], d: Mapping[str, torch.Tensor], s: float) -> Params:
    return {name: params[name].double() + s * d[name].double() for name in params}


def log_policy(params: Mapping[str, torch.Tensor]) -> np.ndarray:
    """log pi(a | h) for every state, [STATES, ACTIONS]."""
    logits = (params[HEAD].double() * params[TABLE].double()).numpy()
    logits = logits - logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


@dataclass
class ToyLog:
    """Sampled games: each one's states and actions at its four decisions, its
    chance draw and its result z in side 1's frame."""
    states: np.ndarray      # [games, 4]
    actions: np.ndarray     # [games, 4]
    chance: np.ndarray      # [games]
    z: np.ndarray           # [games]

    def labels(self, drop_sign: bool = False) -> np.ndarray:
        """R per game and side, [games, 2]; `drop_sign` leaves out sigma."""
        side1 = self.z - BASELINE - LUCK_COEFFICIENT * (self.chance - CHANCE_P)
        sign = np.array([1.0, 1.0]) if drop_sign else np.array([1.0, -1.0])
        return side1[:, None] * sign


class ToyGame:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.win = rng.uniform(0.1, 0.9, size=(3, 3, 2, 3, 3))

    def sample(self, params: Mapping[str, torch.Tensor], games: int, seed: int) -> ToyLog:
        rng = np.random.default_rng(seed)
        probs = np.exp(log_policy(params))

        def pick(states):
            cdf = probs[states].cumsum(axis=1)
            return np.minimum((rng.random(len(states))[:, None] > cdf).sum(axis=1), ACTIONS - 1)

        zeros = np.zeros(games, dtype=np.int64)
        a1 = pick(zeros)
        b1 = pick(1 + a1)
        c = (rng.random(games) < CHANCE_P).astype(np.int64)
        a2 = pick(state_ids(a1, b1, c, zeros)[:, 2])
        b2 = pick(state_ids(a1, b1, c, a2)[:, 3])
        z = np.where(rng.random(games) < self.win[a1, b1, c, a2, b2], 1.0, -1.0)
        return ToyLog(state_ids(a1, b1, c, a2), np.stack([a1, b1, a2, b2], axis=1), c, z)

    def _paths(self):
        for a1, b1, c, a2, b2 in itertools.product(range(3), range(3), range(2), range(3), range(3)):
            yield (a1, b1, a2, b2), state_ids(a1, b1, c, a2), (CHANCE_P if c else 1 - CHANCE_P), \
                2.0 * self.win[a1, b1, c, a2, b2] - 1.0

    def value(self, seat1: Mapping[str, torch.Tensor], seat2: Mapping[str, torch.Tensor]) -> float:
        """E[z] in side 1's frame, seat 1 played by `seat1`, seat 2 by `seat2`."""
        lp = (log_policy(seat1), log_policy(seat2))
        total = 0.0
        for actions, states, chance, mean_z in self._paths():
            logp = sum(lp[SIDE_OF_DECISION[t] - 1][states[t], actions[t]] for t in range(4))
            total += chance * np.exp(logp) * mean_z
        return total

    def exact_gain(self, base: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor]) -> float:
        """The mean over the two seats of the gain of that seat switched to
        `new` against `base`, each in its own frame."""
        both = self.value(base, base)
        return 0.5 * ((self.value(new, base) - both) + (-self.value(base, new) + both))

    def exact_surrogate(self, base: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor]) -> float:
        """The mean over the seats of sum over the seat's decisions of the gain
        of switching that decision alone to `new`, by Q-values under `base`:
        sum_t E[sum_a (new - base)(a | h_t) Q(h_t, a)]."""
        lp_base, lp_new = log_policy(base), log_policy(new)
        q = np.zeros((4, STATES, ACTIONS))       # sum over paths of P(path without t's factor) * frame z
        for actions, states, chance, mean_z in self._paths():
            factors = [lp_base[states[t], actions[t]] for t in range(4)]
            for t in range(4):
                sign = 1.0 if SIDE_OF_DECISION[t] == 1 else -1.0
                rest = np.exp(sum(factors) - factors[t])
                q[t, states[t], actions[t]] += chance * rest * sign * mean_z
        change = np.exp(lp_new) - np.exp(lp_base)
        return 0.5 * float(sum((change * q[t]).sum() for t in range(4)))

    def gradient_moments(self, params: Mapping[str, torch.Tensor]):
        """The mean g and covariance Sigma of one game's gradient
        x = 1/2 sum_s R_s sum_{t in s} grad log pi(a_t | h_t), the pass's
        per-game term, flattened as `flatten` orders the parameters, by
        enumeration of every path and result."""
        lp = log_policy(params)
        mean, second = np.zeros(DIM), np.zeros((DIM, DIM))
        for a1, b1, c, a2, b2 in itertools.product(range(3), range(3), range(2), range(3), range(3)):
            states, actions = state_ids(a1, b1, c, a2), (a1, b1, a2, b2)
            seats = [np.zeros(DIM), np.zeros(DIM)]
            for t in range(4):
                seats[SIDE_OF_DECISION[t] - 1] += score_vector(params, states[t], actions[t])
            path = np.exp(sum(lp[states[t], actions[t]] for t in range(4))) * (CHANCE_P if c else 1 - CHANCE_P)
            win = self.win[a1, b1, c, a2, b2]
            for z, chance in ((1.0, win), (-1.0, 1.0 - win)):
                x = 0.5 * (z - BASELINE - LUCK_COEFFICIENT * (c - CHANCE_P)) * (seats[0] - seats[1])
                mean += path * chance * x
                second += path * chance * np.outer(x, x)
        return mean, second - np.outer(mean, mean)


DIM = STATES * ACTIONS + 1


def flatten(d: Mapping[str, torch.Tensor]) -> np.ndarray:
    """A direction as one vector: the table row by row, then the head."""
    return np.concatenate([d[TABLE].double().numpy().ravel(), d[HEAD].double().numpy().ravel()])


def _logit_jacobian(params: Mapping[str, torch.Tensor], h: int) -> np.ndarray:
    """d logits(h) / d theta, [ACTIONS, DIM]."""
    J = np.zeros((ACTIONS, DIM))
    J[:, h * ACTIONS:(h + 1) * ACTIONS] = params[HEAD].double().item() * np.eye(ACTIONS)
    J[:, -1] = params[TABLE].double().numpy()[h]
    return J


def score_vector(params: Mapping[str, torch.Tensor], h: int, a: int) -> np.ndarray:
    """grad log pi(a | h), flattened."""
    score = -np.exp(log_policy(params)[h])
    score[a] += 1.0
    return score @ _logit_jacobian(params, h)


def fisher_matrix(params: Mapping[str, torch.Tensor], log: ToyLog) -> np.ndarray:
    """The mean per-decision Fisher over the decisions of `log`, flattened:
    the evaluator's KL is d^T F d s^2 / 2 to second order."""
    probs = np.exp(log_policy(params))
    freq = np.bincount(log.states.ravel(), minlength=STATES) / log.states.size
    F = np.zeros((DIM, DIM))
    for h in np.flatnonzero(freq):
        J = _logit_jacobian(params, h)
        F += freq[h] * J.T @ (np.diag(probs[h]) - np.outer(probs[h], probs[h])) @ J
    return F


def _decision_order(games: int) -> np.ndarray:
    """Decision indices in log order: game by game, side 1's two decisions,
    then side 2's."""
    return (np.arange(games)[:, None] * 4 + np.array([0, 2, 1, 3])[None, :]).ravel()


class ToyEvaluator:
    """The Evaluator protocol on held-out toy games, bound to direction d.
    Halves alternate by game: even games are half A, odd half B. The KL
    subsample is every held-out decision."""

    def __init__(self, params: Mapping[str, torch.Tensor], log: ToyLog, d: Mapping[str, torch.Tensor],
                 drop_sign: bool = False):
        self.params = {name: t.double() for name, t in params.items()}
        self.d = {name: d[name].double() for name in params}
        order = _decision_order(len(log.z))
        self.states = log.states.ravel()[order]
        self.actions = log.actions.ravel()[order]
        R = log.labels(drop_sign)
        self.sides = [GameSide(f"g{g}", side, float(R[g, side - 1]), "A" if g % 2 == 0 else "B",
                               4 * g + 2 * (side - 1), 4 * g + 2 * side)
                      for g in range(len(log.z)) for side in (1, 2)]
        self.base = log_policy(self.params)

    def game_sides(self) -> Sequence[GameSide]:
        return self.sides

    def logp(self, s: float) -> np.ndarray:
        return log_policy(step_params(self.params, self.d, s))[self.states, self.actions]

    def base_logp(self) -> np.ndarray:
        return self.logp(0.0)

    def mean_kl(self, s: float) -> float:
        moved = log_policy(step_params(self.params, self.d, s))[self.states]
        here = self.base[self.states]
        return float((np.exp(here) * (here - moved)).sum(axis=1).mean())


def toy_pass(params: Mapping[str, torch.Tensor], log: ToyLog, splits: int = 8) -> Dict:
    """pass.pt (contract §3) of the games of `log` at `params`: total and
    splits are sums of R grad log pi over 2 x games, a window is one
    game-side, and sq[i] the mean squared window gradient of split i, taken
    before the 1/2N as tools/pg_pass.py takes it."""
    games = len(log.z)
    R = log.labels()
    lp = log_policy(params)
    probs = np.exp(lp)
    w = float(params[HEAD].double().item())
    table = params[TABLE].double().numpy()
    window_table = np.zeros((games, 2, STATES, ACTIONS))
    window_head = np.zeros((games, 2))
    for t in range(4):
        side = SIDE_OF_DECISION[t] - 1
        h, a = log.states[:, t], log.actions[:, t]
        score = -probs[h]
        score[np.arange(games), a] += 1.0                    # d log pi / d logits
        window_table[np.arange(games), side, h] += R[:, side, None] * w * score
        window_head[:, side] += R[:, side] * (score * table[h]).sum(axis=1)
    split_of = np.arange(games) % splits

    def grads(mask) -> Dict[str, torch.Tensor]:
        return {TABLE: torch.from_numpy(window_table[mask].sum(axis=(0, 1)) / (2 * games)).float(),
                HEAD: torch.tensor([[window_head[mask].sum() / (2 * games)]]).float()}

    def squares(mask) -> Dict[str, torch.Tensor]:
        windows = mask.sum() * 2
        return {TABLE: torch.from_numpy((window_table[mask] ** 2).sum(axis=(0, 1)) / windows).float(),
                HEAD: torch.tensor([[(window_head[mask] ** 2).sum() / windows]]).float()}

    masks = [split_of == i for i in range(splits)]
    return {"names": [TABLE, HEAD], "total": grads(np.ones(games, dtype=bool)),
            "splits": [grads(m) for m in masks], "sq": [squares(m) for m in masks],
            "windows": 2 * games, "games": games, "decisions": 4 * games,
            "split_games": [int(m.sum()) for m in masks],
            "probes": {"end_turn": 0.0, "attack": 0.0, "recruit": 0.0}, "checkpoint_sha256": "toy"}


def factory(params: Mapping[str, torch.Tensor], log: ToyLog, drop_sign: bool = False):
    """An evaluator factory over the held-out games of `log`."""
    return lambda d: ToyEvaluator(params, log, d, drop_sign)


def head_direction(params: Mapping[str, torch.Tensor]) -> Params:
    """The head's temperature direction: its own weight, 0 on the table."""
    return {TABLE: torch.zeros_like(params[TABLE]), HEAD: params[HEAD].clone()}
