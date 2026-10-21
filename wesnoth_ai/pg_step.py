"""The step of a policy-gradient round: its direction, its length and its
reading (docs/training_design_20261019/current_design.md §1 steps 4-5, §2,
§3.4 and §5).

A round's gradient pass (tools/pg_pass.py writes `pass.pt`,
build_contracts.md §3) holds the round's summed gradient g, eight split
accumulators and the mean squared window gradient of each split. From it:

- `direction` turns g into a direction d = P g under one of stage C's arms;
- `kl_search` finds the step s that puts theta_k + s d at a given mean
  per-decision KL from pi_k;
- `first_order_gain` and `exact_gain` read a candidate on held-out games
  through an Evaluator (wesnoth_ai/pg_protocol.py), each with an SE clustered
  by game, and `ess_share` checks its importance weights;
- `ladder` reads three candidates at KL eps/2, eps and 2 eps, selects on half
  A, confirms on half B and returns the round's decision;
- `critical_batch`, `gain_curve` and `floor` read the round size and the
  attainable gain.

Parameters are named as in pass.pt, and a direction maps every name of the
pass to a float32 tensor of that parameter's shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.muon import newton_schulz
from wesnoth_ai.optimizers import IN_PROJECTION, IN_PROJECTION_BLOCKS
from wesnoth_ai.param_groups import group_of
from wesnoth_ai.pg_protocol import Evaluator, EvaluatorFactory, GameSide

Direction = Dict[str, torch.Tensor]

ARMS = ("sgd", "reduced_adamw", "muon", "heads_only", "probe_span")
DEFAULT_ARM = "reduced_adamw"
# The output heads' final linear maps (wesnoth_ai/model.py, the heads): the
# weapon head's last layer is index 2 of its Sequential. Scaling one of them
# scales that head's logits; target_k_proj is left out, since scaling the
# query alone already scales the pointer logits.
HEAD_MAPS = ("actor_head", "type_head", "target_q_proj", "weapon_head.2")
# Every embedding table of the network is a module named `*_embed`.
EMBEDDING_SUFFIX = "_embed.weight"

KL_RTOL = 0.01                     # kl_search: |KL - target| <= KL_RTOL * target
KL_MAX_EVALS = 40
LADDER_MULTIPLIERS = (0.5, 1.0, 2.0)
REVERT_SE = 2.0                    # half B below -REVERT_SE SE keeps pi_k
# ΔĴ reads a step up to this KL per side-game (design §1.2); past it a rung
# is read by a cross-play match, not by importance weights.
SIDE_KL_RANGE = 0.5


# ---------------------------------------------------------------- the pass

def _names(pass_: Mapping) -> List[str]:
    return list(pass_["names"])


def second_moment(pass_: Mapping) -> Direction:
    """The pass's mean squared window gradient: the splits' means weighted by
    their game counts. The contract records windows per pass, not per split;
    the weighting is exact when a split's windows are in proportion to its
    games, as they are on average under the game-index split."""
    weights = np.asarray(pass_["split_games"], dtype=np.float64)
    weights = weights / weights.sum()
    return {name: sum(float(w) * sq[name].float() for w, sq in zip(weights, pass_["sq"]))
            for name in _names(pass_)}


def pooled_gradient(pass_: Mapping, splits: Sequence[int]) -> Direction:
    """The mean gradient of the games of `splits` alone, on `total`'s per-game
    scale: their sum times the pass's games over their own. All eight splits
    give `total` back."""
    own = sum(pass_["split_games"][i] for i in splits)
    if own <= 0:
        raise ValueError(f"splits {list(splits)} hold no games")
    scale = pass_["games"] / own
    return {name: sum(pass_["splits"][i][name].float() for i in splits) * scale for name in _names(pass_)}


# ---------------------------------------------------------------- directions

def head_of(name: str) -> Optional[str]:
    """The head whose final linear map holds parameter `name`, or None."""
    bare = name[len("model."):] if name.startswith("model.") else name
    for head in HEAD_MAPS:
        if bare.startswith(head + "."):
            return head
    return None


def muon_matrices(shapes: Mapping[str, Sequence[int]]) -> List[str]:
    """The parameters the trainer's Muon orthogonalizes (the matrix role of
    wesnoth_ai/optimizers.py): the trunk's 2-D weights other than embedding
    tables. The role finds embeddings by module type; this reads names
    (EMBEDDING_SUFFIX), and tests/test_pg_step.py holds the two to agreement
    on the network."""
    return [name for name, shape in shapes.items()
            if len(shape) == 2 and group_of(name) == "trunk" and not name.endswith(EMBEDDING_SUFFIX)]


def reduced_adamw(g: Direction, v: Direction) -> Direction:
    """AdamW's first step without momentum, weight decay or clipping: g over
    the root of the second moment v, and 0 where v is 0."""
    out = {}
    for name, grad in g.items():
        positive = v[name] > 0
        root = torch.where(positive, v[name], torch.ones_like(v[name])).sqrt()
        out[name] = torch.where(positive, grad / root, torch.zeros_like(grad))
    return out


def _rms(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    return x.pow(2).mean(dim=dims, keepdim=True).sqrt()


def muon_direction(g: Direction, v: Direction) -> Direction:
    """One orthogonalized step on the summed gradient per matrix, the rest
    reduced AdamW. Each matrix (each of an in-projection's query, key and
    value blocks, as the trainer's Muon splits them) is Newton-Schulz
    orthogonalized and scaled to the RMS of the reduced-AdamW direction on the
    same block: Muon's 0.2 sqrt(max(A, B)) factor matches AdamW's RMS per
    training step, which a single KL-scaled step has no use for."""
    adam = reduced_adamw(g, v)
    out = dict(adam)
    for name in muon_matrices({n: t.shape for n, t in g.items()}):
        blocks = IN_PROJECTION_BLOCKS if name.endswith(IN_PROJECTION) else 1
        rows, cols = g[name].shape
        if rows % blocks:
            raise ValueError(f"{name} {tuple(g[name].shape)} does not split into {blocks} row blocks")
        grad = g[name].reshape(blocks, rows // blocks, cols)
        ortho = newton_schulz(grad).float()
        target = _rms(adam[name].reshape(grad.shape), (1, 2))
        have = _rms(ortho, (1, 2))
        scale = torch.where(have > 0, target / torch.where(have > 0, have, torch.ones_like(have)),
                            torch.zeros_like(have))
        out[name] = (ortho * scale).reshape(rows, cols)
    return out


def heads_only(d: Direction) -> Direction:
    """`d` on the output heads' final linear maps, 0 elsewhere."""
    if not any(head_of(name) for name in d):
        raise ValueError("no parameter of the direction belongs to a head (HEAD_MAPS)")
    return {name: t if head_of(name) else torch.zeros_like(t) for name, t in d.items()}


def head_directions(params: Mapping[str, torch.Tensor]) -> Dict[str, Direction]:
    """Each head's temperature direction: its final linear map's own weights
    and bias (moving along it scales the head's logits), 0 elsewhere. Heads
    absent from `params` are left out."""
    out = {}
    for head in HEAD_MAPS:
        members = {name for name in params if head_of(name) == head}
        if members:
            out[head] = {name: (t.detach().float().clone() if name in members else torch.zeros_like(t, dtype=torch.float32))
                         for name, t in params.items()}
    return out


def dot(a: Direction, b: Direction) -> float:
    return float(sum((a[name].double() * b[name].double()).sum() for name in a))


def combine(coefficients: Sequence[float], directions: Sequence[Direction]) -> Direction:
    names = directions[0].keys()
    return {name: sum(float(c) * d[name] for c, d in zip(coefficients, directions)) for name in names}


def span_slopes(g: Direction, directions: Sequence[Direction]) -> np.ndarray:
    """The first-order gain of a unit step along each direction: g . u."""
    return np.array([dot(g, u) for u in directions])


def probe_span(g: Direction, params: Mapping[str, torch.Tensor],
               span_fisher: Optional[np.ndarray] = None) -> Direction:
    """The step within the span of the head-temperature directions u: the
    natural gradient c = F_s^-1 s when the span's Fisher F_s is given (the
    in-span step of largest first-order gain at a fixed KL; its gain at KL K
    is `floor`), else c = s (gradient ascent on the four temperatures)."""
    heads = list(head_directions(params).values())
    if not heads:
        raise ValueError("no head of HEAD_MAPS among the parameters")
    slopes = span_slopes(g, heads)
    coefficients = slopes if span_fisher is None else _solve(span_fisher, slopes)
    return combine(coefficients, heads)


def direction(pass_: Mapping, arm: str, params: Optional[Mapping[str, torch.Tensor]] = None,
              span_fisher: Optional[np.ndarray] = None, gradient: Optional[Direction] = None) -> Direction:
    """The direction of stage C's arm `arm` from the pass (design §1 step 4,
    §2): sgd, reduced_adamw, muon, heads_only (reduced AdamW on the heads,
    so the arm differs from the default by its restriction alone) or
    probe_span (needs `params`, the weights at theta_k). `gradient` replaces
    the pass's total (a split's gradient, say) under the pass's own metric."""
    names = _names(pass_)
    source = pass_["total"] if gradient is None else gradient
    g = {name: source[name].detach().float().cpu() for name in names}
    if arm == "sgd":
        return g
    if arm == "probe_span":
        if params is None:
            raise ValueError("the probe_span arm needs the weights at theta_k")
        return probe_span(g, {name: params[name] for name in names}, span_fisher)
    v = {name: t.float().cpu() for name, t in second_moment(pass_).items()}
    if arm == "reduced_adamw":
        return reduced_adamw(g, v)
    if arm == "muon":
        return muon_direction(g, v)
    if arm == "heads_only":
        return heads_only(reduced_adamw(g, v))
    raise ValueError(f"unknown arm {arm!r}: one of {', '.join(ARMS)}")


def norm(d: Direction) -> float:
    return math.sqrt(dot(d, d))


# ---------------------------------------------------------------- the KL search

@dataclass(frozen=True)
class KLPoint:
    s: float
    kl: float
    evals: int


def kl_search(kl_at: Callable[[float], float], target: float, s0: float = 1.0,
              rtol: float = KL_RTOL, max_evals: int = KL_MAX_EVALS) -> KLPoint:
    """The s > 0 at which kl_at(s) is within rtol * target of `target`.

    Near 0 the KL is quadratic, KL(s) ~ q s^2 / 2, so log KL is nearly linear
    in log s with slope 2. The search first brackets the target, stepping s by
    the quadratic model's factor sqrt(target / KL(s)) clipped to [1/10, 10];
    then it refines by the Illinois variant of regula falsi on (log s,
    log KL - log target), which keeps a bracket and converges superlinearly.
    A KL that is not finite counts as above the target, and a KL of 0 as
    below; a step that meets either bisects in log s. Raises ValueError when
    no bracket is found (the KL never reaches the target, as along a zero
    direction) and RuntimeError when max_evals pass without convergence."""
    if not target > 0 or not s0 > 0:
        raise ValueError(f"kl_search needs a positive target and start, got {target}, {s0}")
    evals = 0

    def residual(log_s: float) -> Tuple[float, float]:
        nonlocal evals
        evals += 1
        kl = float(kl_at(math.exp(log_s)))
        if not math.isfinite(kl):
            return math.inf, math.inf
        if kl <= 0:
            return kl, -math.inf
        return kl, math.log(kl / target)

    below = above = None                     # (log s, residual)
    x = math.log(s0)
    while below is None or above is None:
        if evals >= max_evals:
            raise ValueError(f"kl_search: no bracket of KL {target} within {max_evals} evaluations")
        kl, y = residual(x)
        if abs(kl - target) <= rtol * target:
            return KLPoint(math.exp(x), kl, evals)
        if y < 0:
            below = (x, y)
        else:
            above = (x, y)
        step = math.log(10.0) if y == -math.inf else -math.log(10.0) if y == math.inf else \
            min(max(-y / 2.0, -math.log(10.0)), math.log(10.0))
        x += step
    side = 0
    while evals < max_evals:
        (xb, yb), (xa, ya) = below, above
        if math.isfinite(yb) and math.isfinite(ya):
            x = (xb * ya - xa * yb) / (ya - yb)
        else:
            x = (xb + xa) / 2.0
        kl, y = residual(x)
        if abs(kl - target) <= rtol * target:
            return KLPoint(math.exp(x), kl, evals)
        if y < 0:
            below = (x, y)
            if side == -1 and math.isfinite(ya):
                above = (xa, ya / 2.0)
            side = -1
        else:
            above = (x, y)
            if side == 1 and math.isfinite(yb):
                below = (xb, yb / 2.0)
            side = 1
    raise RuntimeError(f"kl_search: KL {target} not reached within rtol {rtol} in {max_evals} evaluations")


# ---------------------------------------------------------------- the readings

@dataclass(frozen=True)
class Reading:
    """A mean over game-sides with its SE clustered by game. `per_game` holds
    each game's sum over its sides and `counts` its number of sides, keyed by
    `games`, so two readings of the same game-sides pair by game."""
    estimate: float
    se: float
    games: Tuple[str, ...]
    per_game: np.ndarray = field(repr=False)
    counts: np.ndarray = field(repr=False)

    def summary(self) -> Dict[str, float]:
        z = self.estimate / self.se if self.se > 0 else math.nan
        return {"estimate": self.estimate, "se": self.se, "z": z}


def _clustered(games: Tuple[str, ...], per_game: np.ndarray, counts: np.ndarray) -> Reading:
    sides = float(counts.sum())
    mean = float(per_game.sum()) / sides
    n = len(games)
    resid = per_game - counts * mean
    se = math.sqrt(n / (n - 1) * float((resid ** 2).sum())) / sides if n > 1 else math.nan
    return Reading(mean, se, games, per_game, counts)


def clustered_mean(values: np.ndarray, games: Sequence[str]) -> Reading:
    """The mean of `values` (one per game-side) with the cluster-robust SE of
    a mean: sqrt(G/(G-1) sum_g (y_g - n_g m)^2) / M, y_g the sum of game g's
    values, n_g its sides, M the number of game-sides."""
    keys, inverse = np.unique(np.asarray(games, dtype=object).astype(str), return_inverse=True)
    per_game = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64), minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
    return _clustered(tuple(keys), per_game, counts)


def paired_difference(a: Reading, b: Reading) -> Reading:
    """a - b on the same game-sides, its SE clustered by game (the pairing)."""
    if a.games != b.games or not np.array_equal(a.counts, b.counts):
        raise ValueError("a paired difference needs two readings of the same game-sides")
    return _clustered(a.games, a.per_game - b.per_game, a.counts)


def _side_sums(values: np.ndarray, sides: Sequence[GameSide]) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])
    start = np.fromiter((gs.start for gs in sides), dtype=np.int64, count=len(sides))
    stop = np.fromiter((gs.stop for gs in sides), dtype=np.int64, count=len(sides))
    return cumulative[stop] - cumulative[start]


def _labels(sides: Sequence[GameSide]) -> Tuple[np.ndarray, List[str]]:
    return np.array([gs.R for gs in sides], dtype=np.float64), [gs.game for gs in sides]


def first_order_gain(log_ratio: np.ndarray, sides: Sequence[GameSide]) -> Reading:
    """Ĝ: the mean over game-sides of sum_t (r_t - 1) R, r_t = pi'/pi_k at the
    side's decisions. Its expectation is the sum over the side's decisions of
    the gain of switching that decision alone to pi' (the surrogate of
    conservative policy iteration), the first-order gain up to O(s^2), and
    any action-independent baseline in R leaves it unchanged, since
    E[r - 1 | h] = 0."""
    R, games = _labels(sides)
    return clustered_mean(_side_sums(np.expm1(log_ratio), sides) * R, games)


def seat_weights(log_ratio: np.ndarray, sides: Sequence[GameSide]) -> np.ndarray:
    """W = prod_t r_t over each game-side's decisions."""
    return np.exp(_side_sums(log_ratio, sides))


def exact_gain(log_ratio: np.ndarray, sides: Sequence[GameSide]) -> Reading:
    """ΔĴ: the mean of (W - 1) R, W = prod over the seat's decisions of r_t.
    Its expectation is the gain of that seat switched to pi' against pi_k,
    averaged over the seats read."""
    R, games = _labels(sides)
    return clustered_mean(np.expm1(_side_sums(log_ratio, sides)) * R, games)


def ess_share(weights: np.ndarray) -> float:
    """The effective-sample share of importance weights, (sum W)^2 / (n sum W^2)."""
    w = np.asarray(weights, dtype=np.float64)
    square = float((w * w).sum())
    return float(w.sum() ** 2 / (len(w) * square)) if square > 0 else 0.0


def ess_floor(side_kl: float) -> float:
    """Half of e^{-2K}, the share log-normal weights of log-variance 2K have
    (design §5): a share below it marks heavy-tailed ratios."""
    return 0.5 * math.exp(-2.0 * side_kl)


@dataclass(frozen=True)
class HalfReading:
    first_order: Reading
    exact: Reading
    ess_share: float
    side_kl: float          # the candidate's KL per side-game on this half
    ess_ok: bool

    def summary(self) -> Dict:
        return {"first_order": self.first_order.summary(), "exact": self.exact.summary(),
                "ess_share": self.ess_share, "ess_floor": ess_floor(self.side_kl), "ess_ok": self.ess_ok,
                "side_kl": self.side_kl}


def sides_by_half(sides: Sequence[GameSide], halves: Sequence[str]) -> Dict[str, List[GameSide]]:
    out = {half: [gs for gs in sides if gs.half == half] for half in halves}
    for half, members in out.items():
        if len({gs.game for gs in members}) < 2:
            raise ValueError(f"half {half!r} holds fewer than two games")
    return out


def decisions_per_side(sides: Sequence[GameSide]) -> float:
    return float(np.mean([gs.stop - gs.start for gs in sides]))


def read_candidate(evaluator: Evaluator, s: float, kl: float, halves: Mapping[str, Sequence[GameSide]],
                   base: Optional[np.ndarray] = None) -> Dict[str, HalfReading]:
    """Ĝ, ΔĴ and the importance-weight check of theta_k + s d on each half,
    from one forward pass."""
    base = evaluator.base_logp() if base is None else base
    log_ratio = np.asarray(evaluator.logp(s), dtype=np.float64) - base
    out = {}
    for half, sides in halves.items():
        side_kl = kl * decisions_per_side(sides)
        share = ess_share(seat_weights(log_ratio, sides))
        out[half] = HalfReading(first_order_gain(log_ratio, sides), exact_gain(log_ratio, sides),
                                share, side_kl, share >= ess_floor(side_kl))
    return out


# ---------------------------------------------------------------- the ladder

@dataclass(frozen=True)
class Rung:
    multiplier: float
    target_kl: float
    s: float
    kl: float
    in_range: bool          # the rung's KL per side-game is within SIDE_KL_RANGE
    halves: Dict[str, HalfReading]

    def readable(self, half: str) -> bool:
        return self.in_range and self.halves[half].ess_ok

    def summary(self) -> Dict:
        return {"multiplier": self.multiplier, "target_kl": self.target_kl, "s": self.s, "kl": self.kl,
                "in_range": self.in_range, "halves": {h: r.summary() for h, r in self.halves.items()}}


@dataclass(frozen=True)
class Decision:
    adopt: bool
    s: float                # the adopted step; 0 keeps pi_k
    eps: float              # the radius this round read
    next_eps: float
    chosen: Optional[int]   # the rung read on half B
    reason: str
    rungs: List[Rung]

    def summary(self) -> Dict:
        return {"adopt": self.adopt, "s": self.s, "eps": self.eps, "next_eps": self.next_eps,
                "chosen": self.chosen, "reason": self.reason, "rungs": [r.summary() for r in self.rungs]}


def decide(rungs: Sequence[Rung], eps: float, halves: Tuple[str, str] = ("A", "B")) -> Decision:
    """Design §1 step 5: the rung of highest ΔĴ on the selection half among
    those ΔĴ can read (in range, importance weights not heavy-tailed) is read
    on the confirmation half. It is adopted unless that reads below
    -REVERT_SE SE; then pi_k stays and the radius halves. The next round
    starts from the adopted rung's radius. With no readable rung pi_k stays
    and the radius halves."""
    select, confirm = halves
    readable = [i for i, rung in enumerate(rungs) if rung.readable(select)]
    if not readable:
        return Decision(False, 0.0, eps, eps / 2.0, None, "no rung readable by importance weights",
                        list(rungs))
    best = max(readable, key=lambda i: rungs[i].halves[select].exact.estimate)
    check = rungs[best].halves[confirm].exact
    if check.estimate < -REVERT_SE * check.se:
        return Decision(False, 0.0, eps, eps / 2.0, best,
                        f"half {confirm} reads {check.estimate:.4g} below -{REVERT_SE:g} SE ({check.se:.3g})",
                        list(rungs))
    return Decision(True, rungs[best].s, eps, rungs[best].multiplier * eps, best,
                    f"half {confirm} reads {check.estimate:.4g} (SE {check.se:.3g})", list(rungs))


def ladder(evaluator_factory: EvaluatorFactory, d: Direction, eps: float,
           halves: Tuple[str, str] = ("A", "B"), s0: float = 1.0) -> Decision:
    """Three candidates theta_k + s d at mean per-decision KL eps/2, eps and
    2 eps, each read by Ĝ and ΔĴ on both halves; the decision is `decide`'s.
    Every reading is returned. `s0` starts the first rung's KL search."""
    evaluator = evaluator_factory(d)
    by_half = sides_by_half(evaluator.game_sides(), halves)
    per_side = decisions_per_side(evaluator.game_sides())
    base = evaluator.base_logp()
    rungs, s_guess, previous = [], s0, None
    for multiplier in LADDER_MULTIPLIERS:
        target = multiplier * eps
        if previous is not None:
            s_guess = previous.s * math.sqrt(target / previous.kl)
        point = kl_search(evaluator.mean_kl, target, s0=s_guess)
        in_range = target * per_side <= SIDE_KL_RANGE * (1.0 + 1e-9)
        rungs.append(Rung(multiplier, target, point.s, point.kl, in_range,
                          read_candidate(evaluator, point.s, point.kl, by_half, base)))
        previous = point
    return decide(rungs, eps, halves)


# ---------------------------------------------------------------- round size and floor

def fisher_quadratic(evaluator: Evaluator, s: float) -> float:
    """d^T F d per decision, F the mean per-decision Fisher of the KL
    subsample: (KL(s) + KL(-s)) / s^2, where the cubic terms of the two
    readings cancel."""
    return (evaluator.mean_kl(s) + evaluator.mean_kl(-s)) / (s * s)


@dataclass(frozen=True)
class CriticalBatch:
    B: float            # games
    signal: float       # g^T PFP g
    noise: float        # tr(PFP Sigma), Sigma the per-game gradient covariance

    def summary(self) -> Dict[str, float]:
        return {"B": self.B, "signal": self.signal, "noise": self.noise}


def critical_batch(split_q: Sequence[float], total_q: float, split_games: Sequence[int]) -> CriticalBatch:
    """B = tr(PFP Sigma) / (g^T PFP g), the critical batch of a KL-normalized
    step (design §2), from the Fisher quadratic forms q = d^T F d of the
    eight single-split directions d_i = P g_i and of their sum's.

    g_i, the mean gradient of split i's n_i games, has mean g and covariance
    Sigma / n_i, so E[q_i] = g^T M g + tr(M Sigma) / n_i with M = PFP, and the
    total of N = sum n_i games has E[q] = g^T M g + tr(M Sigma) / N. Averaging
    the splits' equations and subtracting the total's:
        tr(M Sigma) = (mean q_i - q) / (mean(1/n_i) - 1/N),
        g^T M g     = q - tr(M Sigma) / N.
    With eight equal splits mean(1/n_i) - 1/N = 7/N. A noise reading below 0
    counts as 0 in B, and B is infinite when the signal reads at or below 0."""
    n = np.asarray(split_games, dtype=np.float64)
    q = np.asarray(split_q, dtype=np.float64)
    total = n.sum()
    noise = (q.mean() - total_q) / ((1.0 / n).mean() - 1.0 / total)
    signal = total_q - noise / total
    B = max(noise, 0.0) / signal if signal > 0 else math.inf
    return CriticalBatch(float(B), float(signal), float(noise))


def gain_curve(N: float, B: float) -> float:
    """The share of the noise-free first-order gain a step from N games
    reaches at a fixed KL: sqrt(N / (N + B)). The step s = sqrt(2K / q) with
    E[q] = g^T M g (1 + B/N), and its gain s g^T P g, so the gain scales as
    1/sqrt(1 + B/N)."""
    return math.sqrt(N / (N + B))


def gain_ratio(N: float, N_ref: float, B: float) -> float:
    """The model's Ĝ(N)/Ĝ(N_ref), gain_curve(N, B)/gain_curve(N_ref, B)
    (design §5, the gain-curve check); sqrt(N/N_ref) when B is infinite."""
    if math.isinf(B):
        return math.sqrt(N / N_ref)
    return math.sqrt(N * (N_ref + B) / (N_ref * (N + B)))


def _solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def floor(s: np.ndarray, F_s: np.ndarray, K: float) -> float:
    """sqrt(2K s^T F_s^-1 s): the first-order gain of the best step at KL K
    within the span of the directions whose slopes are s and Fisher F_s
    (design §3.4). By Cauchy-Schwarz in the Fisher metric the best step over
    every parameter gains at least as much. F_s and K share one
    normalisation (both per decision, or both per side-game)."""
    s = np.asarray(s, dtype=np.float64)
    return math.sqrt(2.0 * K * max(float(s @ _solve(np.asarray(F_s, dtype=np.float64), s)), 0.0))


def debiased_quadratic(total_slopes: np.ndarray, split_slopes: Sequence[np.ndarray],
                       split_games: Sequence[int], F_s: np.ndarray) -> float:
    """s^T F_s^-1 s less its noise, tr(F_s^-1 C) / N. The slopes of split i,
    a mean over n_i games, have covariance C / n_i, and the splits estimate C
    without bias: sum_i n_i (s_i - s)(s_i - s)^T / (k - 1), since
    E[n_i (s_i - s)(s_i - s)^T] = C (1 - n_i / N)."""
    s = np.asarray(total_slopes, dtype=np.float64)
    n = np.asarray(split_games, dtype=np.float64)
    deviations = np.asarray(split_slopes, dtype=np.float64) - s
    C = np.einsum("i,ij,ik->jk", n, deviations, deviations) / (len(n) - 1)
    F_s = np.asarray(F_s, dtype=np.float64)
    return float(s @ _solve(F_s, s) - np.trace(_solve(F_s, C)) / n.sum())


def span_fisher(evaluator_factory: EvaluatorFactory, directions: Sequence[Direction], s: float) -> np.ndarray:
    """The per-decision Fisher of a few directions u_i: F_ii = q(u_i) and
    F_ij = (q(u_i + u_j) - F_ii - F_jj) / 2, q the Fisher quadratic form read
    at step s."""
    k = len(directions)
    F = np.zeros((k, k))
    for i, u in enumerate(directions):
        F[i, i] = fisher_quadratic(evaluator_factory(u), s)
    for i in range(k):
        for j in range(i + 1, k):
            both = combine((1.0, 1.0), (directions[i], directions[j]))
            F[i, j] = F[j, i] = (fisher_quadratic(evaluator_factory(both), s) - F[i, i] - F[j, j]) / 2.0
    return F
