"""Whole-History Rating (Coulom 2008) for the self-play strength curve.

WHR fits a player's TIME-VARYING strength jointly over its entire game
history, instead of re-rating independent snapshots. For us the
"player" is the training run: each periodic checkpoint is a node at its
training-time, games are played BETWEEN checkpoints (and, later, vs
fixed anchors like RCA), and a Brownian-motion prior ties consecutive
checkpoints so the strength curve is smooth and a checkpoint that
played few games BORROWS strength from its neighbors. This is strictly
better than the static Bradley-Terry round-robin in `tools/elo_ladder`
for an ONGOING run: one coherent strength-vs-time curve, no moving
window, no per-window gauge drift (one global anchor pins the scale).

Model (Coulom, "Whole-History Rating", CG 2008):
  * Natural rating r per node; Bradley-Terry P(i beats j) = σ(r_i - r_j),
    σ the logistic. DRAWS ARE DROPPED by default (`draw_weight=0`): a
    Wesnoth "draw" is a turn-budget TIMEOUT — neither side could force a
    win in the allotted budget — NOT evidence the two are equal. (Real
    Wesnoth has no draw outcome; even RCA-vs-itself, maximally equal,
    almost always resolves decisively via RNG/terrain/faction
    asymmetry.) Counting a timeout as half-a-win would wrongly pull
    drawn opponents toward equal Elo — e.g. a passive policy that times
    out against everyone would rate equal to the champion. So a draw
    carries no "who is stronger" signal and is excluded. (`draw_weight=
    0.5` recovers the textbook half-win treatment for games with
    legitimate draws.)
  * Brownian prior: for consecutive timeline nodes,
    r_{k+1} - r_k ~ N(0, w² Δt) — strength does a random walk over
    training-time with variance rate w². Expressed to callers as an
    Elo drift-per-unit-time (intuitive); converted to natural units
    internally (Elo = r · 400/ln 10).
  * MAP estimate of all ratings jointly by Newton's method on the
    log-posterior (Bradley-Terry log-likelihood + Gaussian prior, which
    is strictly concave once the prior is present → a unique optimum).
    Per-node uncertainty comes from the inverse Hessian (a Laplace
    approximation), reported as an Elo standard error.

Gauge: Bradley-Terry is shift-invariant, so exactly one node is PINNED
(`anchor`) to fix the scale. Pin a fixed external anchor (RCA, once
that's wired through the live-Wesnoth eval) for an absolute axis, or
the first checkpoint for a relative one. The pinned node must be
connected to the rest through games (the prior only constrains
DIFFERENCES, not the overall level).

All-draws early regime (expected before the policy can force wins):
with draws dropped, an all-draws history has NO decisive games, so the
likelihood is empty and ratings are determined by the prior alone —
they collapse to the anchor with LARGE (prior-bounded) uncertainty.
That is the honest state: "we can't tell who is stronger yet," NOT
"everyone is equal." It also means a rating only becomes meaningful
once games start resolving — which is exactly why there's no point
standing up a rating pool until training produces decisive results.

This module is pure numpy (torch-free) so it unit-tests in isolation.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Elo per natural-log rating unit: Elo = (400/ln 10)·r ≈ 173.7178·r.
_ELO_PER_LN = 400.0 / math.log(10.0)
_LN_PER_ELO = 1.0 / _ELO_PER_LN


# ---------------------------------------------------------------------
# Core MAP solver (natural-rating space)
# ---------------------------------------------------------------------

# A game between rating nodes i and j: i won `wins_i`, j won `wins_j`,
# `draws` drawn. (Aggregated; one tuple per ordered or unordered pair.)
Game = Tuple[int, int, float, float, float]
# A Brownian link: prior (r_i - r_j) ~ N(0, var).
WalkLink = Tuple[int, int, float]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _grad_hess(
    r: np.ndarray, games: Sequence[Game], walk_links: Sequence[WalkLink],
    draw_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of the log-posterior at `r` (Hessian is
    negative semidefinite; strictly negative-definite with the prior).

    `draw_weight` is the fraction of a draw credited as a win to EACH
    side. Default 0.0 DROPS draws from the likelihood (a Wesnoth "draw"
    is a turn-budget timeout, not evidence of equality — see module
    docstring). 0.5 recovers the textbook draw=half-win treatment for
    games with legitimate draws."""
    n = len(r)
    g = np.zeros(n)
    H = np.zeros((n, n))
    for i, j, wi, d, wj in games:
        eff_i = wi + draw_weight * d
        eff_j = wj + draw_weight * d
        n_ij = eff_i + eff_j          # draws contribute iff draw_weight>0
        if n_ij <= 0:
            continue
        p = _sigmoid(r[i] - r[j])     # P(i beats j)
        gi = eff_i - n_ij * p         # d log-lik / d r_i
        g[i] += gi
        g[j] -= gi
        h = n_ij * p * (1.0 - p)
        H[i, i] -= h
        H[j, j] -= h
        H[i, j] += h
        H[j, i] += h
    for i, j, var in walk_links:
        if var <= 0:
            continue
        inv = 1.0 / var
        diff = r[i] - r[j]
        g[i] -= diff * inv
        g[j] += diff * inv
        H[i, i] -= inv
        H[j, j] -= inv
        H[i, j] += inv
        H[j, i] += inv
    return g, H


def whr_fit(
    n: int, games: Sequence[Game], walk_links: Sequence[WalkLink], *,
    anchor: int = 0, anchor_natural: float = 0.0, draw_weight: float = 0.0,
    iters: int = 500, tol: float = 1e-10, ridge: float = 1e-9,
    max_step: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Newton MAP fit. Returns (r, se) in NATURAL rating units, with
    node `anchor` pinned to `anchor_natural` (se 0 there).

    `max_step` damps each Newton step (caps |Δr|) so an early
    ill-conditioned step on extreme ratings can't diverge; the prior
    makes the problem concave so plain Newton then converges."""
    r = np.zeros(n, dtype=float)
    r[anchor] = anchor_natural
    free = [k for k in range(n) if k != anchor]
    if not free:
        return r, np.zeros(n)
    free_arr = np.array(free)

    for _ in range(iters):
        g, H = _grad_hess(r, games, walk_links, draw_weight)
        gf = g[free_arr]
        Hf = H[np.ix_(free_arr, free_arr)]
        A = -Hf + ridge * np.eye(len(free))     # pos-def (maximization)
        try:
            delta = np.linalg.solve(A, gf)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, gf, rcond=None)[0]
        m = float(np.max(np.abs(delta))) if delta.size else 0.0
        if m > max_step:
            delta *= max_step / m
        r[free_arr] += delta
        if m < tol:
            break

    # Laplace covariance from the inverse Hessian at the optimum.
    _, H = _grad_hess(r, games, walk_links, draw_weight)
    Hf = H[np.ix_(free_arr, free_arr)]
    se = np.zeros(n, dtype=float)
    try:
        cov = np.linalg.inv(-Hf + ridge * np.eye(len(free)))
        var = np.clip(np.diag(cov), 0.0, None)
        se[free_arr] = np.sqrt(var)
    except np.linalg.LinAlgError:
        se[free_arr] = float("inf")
    return r, se


# ---------------------------------------------------------------------
# High-level: names + times + Elo-space drift
# ---------------------------------------------------------------------

def walk_links_from_times(
    node_times: Sequence[Tuple[int, float]], elo_drift_per_time: float,
) -> List[WalkLink]:
    """Build Brownian links between consecutive timeline nodes.
    `node_times` = (node_index, time) for the time-varying player's
    nodes; `elo_drift_per_time` = the Elo std the strength may drift per
    sqrt(unit time). Var (natural²) over a gap Δt is (drift·ln10/400)²·Δt."""
    ordered = sorted(node_times, key=lambda it: it[1])
    drift_nat = elo_drift_per_time * _LN_PER_ELO
    base_var = drift_nat * drift_nat
    links: List[WalkLink] = []
    for (i, ti), (j, tj) in zip(ordered, ordered[1:]):
        dt = max(tj - ti, 1e-9)
        links.append((i, j, base_var * dt))
    return links


@dataclass
class WHRResult:
    names:  List[str]
    elo:    Dict[str, float]
    se_elo: Dict[str, float]
    times:  Dict[str, Optional[float]] = field(default_factory=dict)

    def ranked(self) -> List[Tuple[str, float, float]]:
        """(name, elo, 95% CI half-width) sorted strongest first."""
        out = [(nm, self.elo[nm], 1.96 * self.se_elo[nm]) for nm in self.names]
        return sorted(out, key=lambda t: -t[1])


def fit_whr(
    players: Sequence[str],
    games: Sequence[Tuple[str, str, float, float, float]], *,
    times: Optional[Dict[str, float]] = None,
    anchor: Optional[str] = None, anchor_elo: float = 0.0,
    elo_drift_per_time: float = 20.0, draw_weight: float = 0.0,
) -> WHRResult:
    """Fit WHR over named players.

    `games`: (name_a, name_b, wins_a, draws, wins_b) aggregated pairs.
    `times`: name -> training-time for the time-varying player's nodes
      (the checkpoints). Players absent from `times` are treated as
      FIXED anchors (no Brownian link). Consecutive timed nodes get a
      Brownian prior (see elo_drift_per_time).
    `anchor`: the player pinned to `anchor_elo` (default: the
      earliest-time player, else players[0]).
    """
    idx = {nm: k for k, nm in enumerate(players)}
    n = len(players)
    agg = [(idx[a], idx[b], float(wa), float(d), float(wb))
           for a, b, wa, d, wb in games]

    times = times or {}
    timeline = [(idx[nm], times[nm]) for nm in players if nm in times]
    walk = walk_links_from_times(timeline, elo_drift_per_time)

    if anchor is None:
        anchor = (min(times, key=times.get) if times else players[0])
    a_idx = idx[anchor]

    r, se = whr_fit(n, agg, walk, anchor=a_idx,
                    anchor_natural=anchor_elo * _LN_PER_ELO,
                    draw_weight=draw_weight)
    elo = {nm: float(r[idx[nm]] * _ELO_PER_LN) for nm in players}
    # Recenter so the anchor sits exactly at anchor_elo (guards rounding).
    shift = anchor_elo - elo[anchor]
    elo = {nm: v + shift for nm, v in elo.items()}
    se_elo = {nm: float(se[idx[nm]] * _ELO_PER_LN) for nm in players}
    return WHRResult(names=list(players), elo=elo, se_elo=se_elo,
                     times={nm: times.get(nm) for nm in players})


# ---------------------------------------------------------------------
# CLI: fit from a game-history JSON
# ---------------------------------------------------------------------

def _load_games(payload: dict):
    players = list(payload["players"])
    games = []
    for g in payload["games"]:
        if isinstance(g, dict):
            games.append((g["a"], g["b"],
                          g.get("wins_a", 0), g.get("draws", 0),
                          g.get("wins_b", 0)))
        else:
            games.append(tuple(g))
    return players, games


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True,
                    help="JSON: {players:[...], games:[[a,b,wins_a,"
                         "draws,wins_b],...], times:{name:t}, "
                         "anchor:?, anchor_elo:0, elo_drift_per_time:20}")
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args(argv[1:])

    payload = json.loads(args.games.read_text(encoding="utf-8"))
    players, games = _load_games(payload)
    res = fit_whr(
        players, games,
        times=payload.get("times"),
        anchor=payload.get("anchor"),
        anchor_elo=payload.get("anchor_elo", 0.0),
        elo_drift_per_time=payload.get("elo_drift_per_time", 20.0))

    print()
    print("=" * 64)
    print(f"{'rank':>4}  {'player':<24} {'Elo':>7}  {'+/-95%':>7}  {'t':>8}")
    print("-" * 64)
    for rank, (nm, elo, ci) in enumerate(res.ranked(), 1):
        t = res.times.get(nm)
        print(f"{rank:>4}  {nm:<24} {elo:>7.0f}  {ci:>7.0f}  "
              f"{('' if t is None else f'{t:g}'):>8}")
    print("=" * 64)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps({
            "elo": res.elo, "se_elo": res.se_elo, "times": res.times,
        }, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
