#!/usr/bin/env python3
"""Tests for the two MCTS cliffness consumers added on top of the
distributional value head:

1. **Adaptive sim budget** — root cliffness scales total sims
   between `n_simulations_min` and `n_simulations_max`. Linear
   interpolation, off by default. Default values are uncalibrated
   (see BACKLOG.md "Cliffness adaptive sim budget — calibration
   pending"); tests check the math, not the schedule shape.

2. **Bootstrap weighting** — `_backup` shrinks a non-terminal
   leaf's `v` toward 0 (the prior expectation) using a Bayesian-
   precision blend: treat `v` as a noisy estimate with variance
   `alpha * cliffness²` and mix it with a uniform-on-[-1,+1]
   prior of variance 1/3:

       scale = 1/3 / (1/3 + alpha * cliffness²)

   Terminal leaves bypass the shrink (their value is exact).
   Off by default (`alpha=0`).

A previously-shipped third consumer ("soft TT": refuse cached
high-cliffness nodes in the transposition table) was reverted —
exact-match TT means literally the same state, so cliffness on
the cached node has no role in deciding whether to share.
Cliffness-as-similarity-error gating only makes sense paired
with lossy/similarity hashing, which is a separate design
project (see BACKLOG.md "Similarity-hashing TT").

Dependencies: tools.mcts, classes, action_sampler
Dependents:   pytest only
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from action_sampler import LegalActionPrior  # noqa: E402
from tools.mcts import (  # noqa: E402
    MCTSConfig, MCTSEdge, MCTSNode,
    _adaptive_n_sims, _backup,
    _BOOTSTRAP_PRIOR_VAR,
)


# ---------------------------------------------------------------------
# Fixture helpers (mirrors test_mcts.py)
# ---------------------------------------------------------------------

def _stub_sim(side: int, *, done: bool = False, winner: int = 0):
    gs = SimpleNamespace(global_info=SimpleNamespace(current_side=side))
    return SimpleNamespace(gs=gs, done=done, winner=winner)


def _make_node(side: int, *, done: bool = False) -> MCTSNode:
    return MCTSNode(_stub_sim(side, done=done))


def _make_edge(prior: float = 0.5) -> MCTSEdge:
    lap = LegalActionPrior(
        action={"type": "end_turn"},
        prior=prior,
        actor_idx=0,
        target_idx=None,
        weapon_idx=None,
        type_idx=0,
    )
    return MCTSEdge(lap)


def _attach(parent: MCTSNode, child: MCTSNode,
            prior: float = 0.5) -> MCTSEdge:
    edge = _make_edge(prior=prior)
    edge.child = child
    parent.edges.append(edge)
    return edge


# ---------------------------------------------------------------------
# 1. Adaptive sim budget — _adaptive_n_sims
# ---------------------------------------------------------------------

def test_adaptive_budget_disabled_returns_n_simulations():
    """When `adaptive_sim_budget=False`, the helper just echoes
    `config.n_simulations` regardless of cliffness. The whole
    feature is opt-in; default behavior unchanged."""
    cfg = MCTSConfig(n_simulations=42, adaptive_sim_budget=False)
    assert _adaptive_n_sims(cfg, root_cliffness=0.0) == 42
    assert _adaptive_n_sims(cfg, root_cliffness=0.3) == 42
    assert _adaptive_n_sims(cfg, root_cliffness=10.0) == 42


def test_adaptive_budget_low_cliffness_returns_n_min():
    """At cliffness=0 (network certain about the position), the
    search runs the minimum budget. No reason to waste sims when
    the network has converged on a clear value."""
    cfg = MCTSConfig(
        adaptive_sim_budget=True,
        n_simulations_min=10,
        n_simulations_max=200,
        cliffness_max=0.577,
    )
    assert _adaptive_n_sims(cfg, root_cliffness=0.0) == 10


def test_adaptive_budget_high_cliffness_returns_n_max():
    """At cliffness >= cliffness_max (uniform-ish distribution,
    network admits anything could happen), the search runs the
    maximum budget."""
    cfg = MCTSConfig(
        adaptive_sim_budget=True,
        n_simulations_min=10,
        n_simulations_max=200,
        cliffness_max=0.577,
    )
    assert _adaptive_n_sims(cfg, root_cliffness=0.577) == 200
    # Saturates: even larger cliffness still caps at n_max.
    assert _adaptive_n_sims(cfg, root_cliffness=5.0) == 200


def test_adaptive_budget_interpolates_linearly():
    """At cliffness = 0.5 * cliffness_max, the budget sits at the
    midpoint of [n_min, n_max]. Linear interpolation, no surprises."""
    cfg = MCTSConfig(
        adaptive_sim_budget=True,
        n_simulations_min=20,
        n_simulations_max=100,
        cliffness_max=0.6,
    )
    # Halfway: 20 + 0.5 * (100 - 20) = 60.
    n = _adaptive_n_sims(cfg, root_cliffness=0.3)
    assert n == 60
    # Quarter: 20 + 0.25 * 80 = 40.
    n = _adaptive_n_sims(cfg, root_cliffness=0.15)
    assert n == 40


def test_adaptive_budget_clamps_negative_cliffness():
    """Defensive: a numerically-impossible negative cliffness must
    not produce sub-`n_min` budgets. clamp inside the
    interpolation guards this."""
    cfg = MCTSConfig(
        adaptive_sim_budget=True,
        n_simulations_min=10,
        n_simulations_max=200,
    )
    assert _adaptive_n_sims(cfg, root_cliffness=-0.5) == 10


# ---------------------------------------------------------------------
# 2. Bootstrap weighting — Bayesian-precision shrink in _backup
# ---------------------------------------------------------------------

def _bayes_scale(cliffness: float, alpha: float = 1.0) -> float:
    """The expected scale factor under the Bayesian-precision
    blend, computed independently from the implementation. Used
    as the oracle in the shrinkage tests so that any algebra
    drift between docstring and code surfaces immediately."""
    return _BOOTSTRAP_PRIOR_VAR / (
        _BOOTSTRAP_PRIOR_VAR + alpha * (cliffness ** 2)
    )


def test_backup_alpha_zero_no_shrink():
    """`bootstrap_alpha=0` is the "feature off" switch: full v gets
    backed up regardless of cliffness."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    _backup([(parent, edge)], v=0.8, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=0.5, bootstrap_alpha=0.0,
            leaf_is_terminal=False)
    assert edge.w_value == pytest.approx(0.8)


def test_backup_zero_cliffness_no_shrink():
    """Even with `alpha=1` (Bayes-optimal), a leaf with
    cliffness=0 (network certain) gets the full v. Confident
    bootstraps pass through unchanged."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    _backup([(parent, edge)], v=0.6, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=0.0, bootstrap_alpha=1.0,
            leaf_is_terminal=False)
    assert edge.w_value == pytest.approx(0.6)


def test_backup_uniform_cliffness_half_shrink():
    """At cliffness ≈ 0.577 (≈ std of uniform on [-1, +1] ≈
    sqrt(prior_var)), the cliffness² and prior variance match,
    so Bayes posterior weights v at scale = 1/3 / (1/3 + 1/3) =
    0.5. Halfway-point sanity check on the Bayesian schedule."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    cliffness = (1.0 / 3.0) ** 0.5   # exact sqrt(prior_var)
    _backup([(parent, edge)], v=0.9, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=cliffness, bootstrap_alpha=1.0,
            leaf_is_terminal=False)
    assert edge.w_value == pytest.approx(0.9 * 0.5, abs=1e-6)


def test_backup_high_cliffness_shrinks_toward_zero():
    """At cliffness ≫ sqrt(prior_var), the posterior collapses
    toward the prior (0). Doesn't hit zero exactly under the
    Bayesian schedule — it asymptotes — which is the right
    behavior: the network being maximally uncertain doesn't
    mean its v is meaningless, just that it's mostly noise."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    cliffness = 5.0   # many times larger than prior std
    _backup([(parent, edge)], v=0.9, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=cliffness, bootstrap_alpha=1.0,
            leaf_is_terminal=False)
    expected = 0.9 * _bayes_scale(cliffness, alpha=1.0)
    assert edge.w_value == pytest.approx(expected, abs=1e-6)
    # And the scale is tiny: < 1.5% of v.
    assert abs(edge.w_value) < 0.015


def test_backup_alpha_below_one_softer_shrink():
    """`alpha < 1` makes shrinkage less aggressive than the
    Bayes-optimal default. At cliffness² = prior_var with
    alpha=0.5, the variance term is 0.5 * (1/3) and scale =
    1/3 / (1/3 + 1/6) = 2/3 — gentler than alpha=1's 0.5."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    cliffness = (1.0 / 3.0) ** 0.5   # cliffness² = prior_var
    _backup([(parent, edge)], v=1.0, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=cliffness, bootstrap_alpha=0.5,
            leaf_is_terminal=False)
    # scale = (1/3) / (1/3 + 0.5 * 1/3) = 1/(1 + 0.5) = 2/3.
    assert edge.w_value == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_backup_alpha_above_one_more_aggressive_shrink():
    """`alpha > 1` makes shrinkage more aggressive than Bayes-
    optimal — useful if we believe cliffness underestimates
    actual value-noise variance during early training."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    cliffness = (1.0 / 3.0) ** 0.5
    _backup([(parent, edge)], v=1.0, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=cliffness, bootstrap_alpha=4.0,
            leaf_is_terminal=False)
    # scale = (1/3) / (1/3 + 4 * 1/3) = 1/5.
    assert edge.w_value == pytest.approx(0.2, abs=1e-6)


def test_backup_terminal_leaf_bypasses_shrink():
    """Terminal leaves represent EXACT outcomes (win/loss/draw).
    Cliffness doesn't apply — the network's uncertainty is
    irrelevant when the value is the real game result. Backup
    must not shrink v in that case."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1, done=True)
    edge = _attach(parent, leaf)
    _backup([(parent, edge)], v=1.0, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=99.9, bootstrap_alpha=1.0,
            leaf_is_terminal=True)
    # Despite huge cliffness + alpha=1, terminal flag says
    # "ignore all that"; full +1.0 backed up.
    assert edge.w_value == pytest.approx(1.0)


def test_backup_visit_count_unaffected_by_cliffness():
    """Cliffness scales `w_value` only; `n_visits` and
    `_total_visits` always increment by 1. Visit counts are the
    PUCT exploration signal — independent of how much we trust
    the value backed up."""
    nodes = [_make_node(side=1) for _ in range(4)]
    path = [(p, _attach(p, c)) for p, c in zip(nodes[:-1], nodes[1:])]
    _backup(path, v=0.5, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=10.0, bootstrap_alpha=1.0,
            leaf_is_terminal=False)
    expected_v = 0.5 * _bayes_scale(10.0, alpha=1.0)
    for parent, edge in path:
        assert edge.n_visits == 1, (
            "visit count must increment regardless of shrink"
        )
        assert edge.w_value == pytest.approx(expected_v, abs=1e-6)


def test_backup_scale_never_negative():
    """The Bayesian scale is bounded in (0, 1] for any
    non-negative cliffness and non-negative alpha. No clamping
    needed (unlike the linear-shrink alternative we shipped
    first), but lock it down with a property test against
    pathological inputs."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    # Way-out cliffness with alpha=1: scale > 0, < 1.
    _backup([(parent, edge)], v=1.0, leaf_side=1, virtual_loss=0.0,
            leaf_cliffness=1e6, bootstrap_alpha=1.0,
            leaf_is_terminal=False)
    # Scale is tiny but positive; sign of v preserved.
    assert 0.0 < edge.w_value < 1e-10
