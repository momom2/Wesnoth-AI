"""The Evaluator protocol of the policy-gradient step
(docs/training_design_20261019/build_contracts.md §4).

An Evaluator reads one direction d on the held-out games: the log
probability of every logged held-out decision at theta_k + s * d, and the
mean per-decision KL from pi_k. The step math (wesnoth_ai/pg_step.py) reads
candidates only through it, so the same code runs on the enumerable toy
game of the tests (tests/helpers/toy_game.py) and on the network
(tools/pg_eval.py, `NetworkEvaluator`).

- The evaluator is bound to one direction d at construction; `logp(s,
  halves)` is log pi at theta_k + s * d for every held-out decision of the
  game-sides in `halves` (every game-side when None), NaN at the others, so
  a candidate judged on one half forwards that half only; `base_logp()` is
  `logp(0.0)` over every game-side as the trainer recomputes it (not the
  logged value).
- The logp arrays are ordered by game-side: each game-side's decisions, in
  play order, are one contiguous range [start, stop). This is not the log's
  play order, where the two sides' decisions interleave.
- R is the game-side's label (contract §2: the result in the side's frame
  less the baseline and the luck term).
- Halves are drawn per game: both seats of a game carry the same `half`
  (`pg_step.sides_by_half` refuses a game split across halves).
- `mean_kl(s)` is the mean per-decision KL(pi_k || pi_{theta_k + s * d})
  over the evaluator's KL subsample.
- s may be negative: the step math reads the Fisher quadratic form of a
  direction from the KL at +s and -s.
- The forwards behind `logp` and `mean_kl` run in float32. bf16 rounds a
  log probability near -5 by about 0.016 nats, more than the KL of about
  0.001 nats per decision the step searches for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class GameSide:
    game: str
    side: int
    R: float
    half: Optional[str]
    start: int      # this game-side's decisions are [start, stop) in the logp arrays
    stop: int


@runtime_checkable
class Evaluator(Protocol):
    def game_sides(self) -> Sequence[GameSide]: ...
    def base_logp(self) -> np.ndarray: ...
    def logp(self, s: float, halves: Optional[Sequence[str]] = None) -> np.ndarray: ...
    def mean_kl(self, s: float) -> float: ...


# An evaluator bound to the direction it is given: d maps each parameter name
# of pass.pt (contract §3) to a tensor of that parameter's shape.
EvaluatorFactory = Callable[[Mapping[str, "torch.Tensor"]], Evaluator]
