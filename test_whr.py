"""Unit tests for Whole-History Rating (tools/whr).

Sim-free, torch-free: feed synthetic game histories to fit_whr /
whr_fit and pin the invariants — monotone strength recovery, anchor
gauge, Brownian smoothing (a checkpoint with no games borrows strength
from its neighbors), all-draws -> equal, SE shrinks with games, drift
controls smoothing, and a 2-player closed-form match.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.whr import fit_whr, whr_fit, walk_links_from_times   # noqa: E402


def _decisive(a, b, n=20):
    """a beats b n-0."""
    return (a, b, float(n), 0.0, 0.0)


def test_recovers_monotone_strength_curve():
    # 4 checkpoints in training order, each beats every earlier one.
    players = ["s0", "s1", "s2", "s3"]
    times = {"s0": 0.0, "s1": 1.0, "s2": 2.0, "s3": 3.0}
    games = [_decisive(hi, lo, 16)
             for i, hi in enumerate(players)
             for lo in players[:i]]
    res = fit_whr(players, games, times=times, anchor="s0", anchor_elo=0.0,
                  elo_drift_per_time=150.0)   # loose prior
    e = res.elo
    assert e["s0"] < e["s1"] < e["s2"] < e["s3"], e
    assert abs(e["s0"]) < 1e-9                      # anchor pinned


def test_anchor_pinned_at_value():
    players = ["a", "b"]
    res = fit_whr(players, [_decisive("b", "a", 15)],
                  anchor="a", anchor_elo=1000.0)
    assert abs(res.elo["a"] - 1000.0) < 1e-9
    assert res.elo["b"] > 1000.0     # b is stronger, above the anchor


def test_brownian_smoothing_borrows_strength():
    # B played NO games; A (anchor) and C did, with C >> A. B sits
    # between A and C purely via the Brownian prior over the timeline.
    players = ["A", "B", "C"]
    times = {"A": 0.0, "B": 1.0, "C": 2.0}
    games = [_decisive("C", "A", 30)]          # only A vs C
    res = fit_whr(players, games, times=times, anchor="A", anchor_elo=0.0,
                  elo_drift_per_time=100.0)
    e = res.elo
    assert e["C"] > 0.0
    assert 0.0 < e["B"] < e["C"], f"B should interpolate: {e}"
    # B played no games, yet gets a FINITE rating + uncertainty purely
    # from the Brownian prior linking it to A and C (not inf/NaN).
    assert math.isfinite(res.se_elo["B"]) and res.se_elo["B"] > 0.0


def test_draws_carry_no_signal_and_are_uncertain():
    # A Wesnoth draw is a timeout, NOT equality evidence. Dropped by
    # default => an all-draws history has no decisive games => ratings
    # collapse to the anchor (we can't tell who's stronger) and are FAR
    # MORE uncertain than the same pairings played out decisively.
    players = ["s0", "s1", "s2"]
    times = {"s0": 0.0, "s1": 1.0, "s2": 2.0}
    drawn = [("s0", "s1", 0.0, 30.0, 0.0),
             ("s1", "s2", 0.0, 30.0, 0.0),
             ("s0", "s2", 0.0, 30.0, 0.0)]
    rd = fit_whr(players, drawn, times=times, anchor="s0",
                 elo_drift_per_time=50.0)
    for nm in players:
        assert abs(rd.elo[nm]) < 1e-6, rd.elo            # no decisive info
        assert math.isfinite(rd.se_elo[nm])

    decisive = [_decisive("s1", "s0", 30), _decisive("s2", "s1", 30),
                _decisive("s2", "s0", 30)]
    rc = fit_whr(players, decisive, times=times, anchor="s0",
                 elo_drift_per_time=50.0)
    assert rc.elo["s2"] > 0.0                             # decisive => spread
    assert rd.se_elo["s2"] > rc.se_elo["s2"]              # draws => uncertain


def test_draws_are_ignored_by_default():
    # Padding a decisive record with draws (timeouts) must not move the
    # rating: a timeout is a non-result.
    base = fit_whr(["i", "j"], [("i", "j", 15.0, 0.0, 5.0)], anchor="j")
    padded = fit_whr(["i", "j"], [("i", "j", 15.0, 100.0, 5.0)], anchor="j")
    assert abs(base.elo["i"] - padded.elo["i"]) < 1e-9


def test_draw_weight_half_recovers_equality_pull():
    # Opt-in: with draw_weight=0.5 (textbook half-win, for games with
    # LEGITIMATE draws), many draws DO pull the two toward equal Elo.
    gap_drop = fit_whr(["i", "j"], [("i", "j", 15.0, 200.0, 5.0)],
                       anchor="j", draw_weight=0.0).elo["i"]
    gap_half = fit_whr(["i", "j"], [("i", "j", 15.0, 200.0, 5.0)],
                       anchor="j", draw_weight=0.5).elo["i"]
    assert 0.0 < gap_half < gap_drop


def test_se_shrinks_with_more_games():
    few = fit_whr(["a", "b"], [("a", "b", 6.0, 0.0, 4.0)], anchor="a")
    many = fit_whr(["a", "b"], [("a", "b", 60.0, 0.0, 40.0)], anchor="a")
    assert many.se_elo["b"] < few.se_elo["b"]


def test_drift_controls_smoothing():
    # Heavier (smaller-drift) prior compresses the curve toward equal;
    # looser (larger-drift) prior lets decisive results spread it.
    players = ["s0", "s1", "s2"]
    times = {"s0": 0.0, "s1": 1.0, "s2": 2.0}
    games = [_decisive("s2", "s0", 30), _decisive("s2", "s1", 30),
             _decisive("s1", "s0", 30)]
    tight = fit_whr(players, games, times=times, anchor="s0",
                    elo_drift_per_time=5.0)
    loose = fit_whr(players, games, times=times, anchor="s0",
                    elo_drift_per_time=300.0)
    assert loose.elo["s2"] > tight.elo["s2"] > 0.0


def test_two_player_closed_form_no_prior():
    # No `times` => no Brownian link => pure Bradley-Terry. With i over
    # j 15/5, the MLE Elo gap is 400*log10(15/5).
    res = fit_whr(["i", "j"], [("i", "j", 15.0, 0.0, 5.0)], anchor="j")
    expected = 400.0 * math.log10(15.0 / 5.0)
    assert abs(res.elo["i"] - expected) < 0.5, (res.elo["i"], expected)


def test_walk_links_variance_scales_with_gap():
    links = walk_links_from_times([(0, 0.0), (1, 1.0), (2, 5.0)],
                                  elo_drift_per_time=400.0 / math.log(10.0))
    # drift chosen so drift_nat == 1 -> var == Δt exactly.
    assert abs(links[0][2] - 1.0) < 1e-9     # gap 1
    assert abs(links[1][2] - 4.0) < 1e-9     # gap 4


def test_fixed_anchor_not_in_timeline_has_no_walk_link():
    # 'rca' is a fixed anchor (absent from `times`); only the snapshots
    # form the Brownian timeline. The snapshot that beats rca rates
    # above it.
    players = ["rca", "s0", "s1"]
    times = {"s0": 0.0, "s1": 1.0}
    games = [_decisive("s0", "rca", 12), _decisive("s1", "rca", 18),
             _decisive("s1", "s0", 12)]
    res = fit_whr(players, games, times=times, anchor="rca",
                  anchor_elo=0.0, elo_drift_per_time=100.0)
    assert abs(res.elo["rca"]) < 1e-9
    assert res.elo["s1"] > res.elo["s0"] > 0.0
