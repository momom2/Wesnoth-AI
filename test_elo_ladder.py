"""Unit tests for the pure Elo / Bradley-Terry math in tools/elo_ladder.

Sim-free and torch-free: feeds synthetic pairwise records to fit_elo /
fit_bradley_terry / elo_standard_errors and pins the invariants
(ordering recovery, anchor gauge, finiteness under degeneracy, equal
players -> equal Elo, anchor SE == 0). The round-robin + policy plumbing
is exercised separately by a CLI smoke.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.elo_ladder import (   # noqa: E402
    PairRecord, fit_elo, fit_bradley_terry, _win_and_game_matrices,
    elo_standard_errors,
)


def test_recovers_dominance_ordering():
    # 3 players, strict A > B > C with clean separation.
    pairs = {
        (0, 1): PairRecord(wins_i=18, draws=2, wins_j=0),   # A >> B
        (0, 2): PairRecord(wins_i=20, draws=0, wins_j=0),   # A >>> C
        (1, 2): PairRecord(wins_i=16, draws=2, wins_j=2),   # B > C
    }
    elo, se = fit_elo(3, pairs, anchor_idx=2, anchor_elo=0.0)
    assert elo[0] > elo[1] > elo[2], f"ordering not recovered: {elo}"
    # Anchor pinned exactly.
    assert abs(elo[2] - 0.0) < 1e-9
    # Dominant player well clear of the floor.
    assert elo[0] - elo[2] > 200
    # SEs finite and positive for non-anchor; anchor SE == 0.
    assert se[2] == 0.0
    assert np.all(np.isfinite(se)) and se[0] > 0 and se[1] > 0


def test_anchor_choice_only_shifts_not_reorders():
    pairs = {
        (0, 1): PairRecord(wins_i=15, draws=5, wins_j=0),
        (0, 2): PairRecord(wins_i=18, draws=1, wins_j=1),
        (1, 2): PairRecord(wins_i=12, draws=4, wins_j=4),
    }
    elo_a, _ = fit_elo(3, pairs, anchor_idx=2, anchor_elo=0.0)
    elo_b, _ = fit_elo(3, pairs, anchor_idx=0, anchor_elo=1000.0)
    # Differences between players are gauge-invariant.
    d_a = elo_a[0] - elo_a[1]
    d_b = elo_b[0] - elo_b[1]
    assert abs(d_a - d_b) < 1e-6
    # Anchor b pinned at 1000.
    assert abs(elo_b[0] - 1000.0) < 1e-9


def test_equal_players_get_equal_elo():
    pairs = {(0, 1): PairRecord(wins_i=10, draws=0, wins_j=10)}
    elo, _ = fit_elo(2, pairs, anchor_idx=0, anchor_elo=0.0)
    assert abs(elo[0] - elo[1]) < 1e-6


def test_draws_dropped_by_default():
    # A Wesnoth draw is a timeout, not equality evidence: padding a
    # decisive record with draws must NOT move the Elo (default
    # draw_weight=0), and a draws-only pair contributes nothing.
    base = {(0, 1): PairRecord(wins_i=15, draws=0, wins_j=5)}
    padded = {(0, 1): PairRecord(wins_i=15, draws=200, wins_j=5)}
    e_base, _ = fit_elo(2, base, anchor_idx=1, prior_games=0.0)
    e_pad, _ = fit_elo(2, padded, anchor_idx=1, prior_games=0.0)
    assert abs(e_base[0] - e_pad[0]) < 1e-6
    # draw_weight=0.5 (textbook) instead pulls them toward equal.
    e_half, _ = fit_elo(2, padded, anchor_idx=1, prior_games=0.0,
                        draw_weight=0.5)
    assert 0.0 < e_half[0] < e_base[0]


def test_winless_and_undefeated_stay_finite():
    # Without the prior, a 0-win player would have gamma -> 0 (Elo
    # -inf) and an undefeated player Elo +inf. The ghost-games prior
    # must keep both finite.
    pairs = {
        (0, 1): PairRecord(wins_i=20, draws=0, wins_j=0),   # A undefeated
        (0, 2): PairRecord(wins_i=20, draws=0, wins_j=0),
        (1, 2): PairRecord(wins_i=20, draws=0, wins_j=0),   # C winless
    }
    elo, se = fit_elo(3, pairs, anchor_idx=1, anchor_elo=0.0,
                      prior_games=1.0)
    assert np.all(np.isfinite(elo)), elo
    assert np.all(np.isfinite(se)), se
    assert elo[0] > elo[1] > elo[2]


def test_stronger_prior_shrinks_spread():
    pairs = {(0, 1): PairRecord(wins_i=19, draws=0, wins_j=1)}
    wide, _ = fit_elo(2, pairs, anchor_idx=1, anchor_elo=0.0,
                      prior_games=0.5)
    narrow, _ = fit_elo(2, pairs, anchor_idx=1, anchor_elo=0.0,
                        prior_games=20.0)
    # Both put player 0 above player 1, but the heavier prior pulls
    # the gap toward 0 (regularization toward equality).
    assert wide[0] > narrow[0] > 0


def test_more_games_tighten_se():
    few = {(0, 1): PairRecord(wins_i=6, draws=0, wins_j=4)}
    many = {(0, 1): PairRecord(wins_i=60, draws=0, wins_j=40)}
    _, se_few = fit_elo(2, few, anchor_idx=0, anchor_elo=0.0)
    _, se_many = fit_elo(2, many, anchor_idx=0, anchor_elo=0.0)
    # Non-anchor SE shrinks with more games at the same win rate.
    assert se_many[1] < se_few[1]


def test_mm_matches_closed_form_two_player():
    # For two players with w wins out of n (no draws, no prior), the
    # BT MLE is gamma_0/gamma_1 = w/(n-w); Elo gap = 400*log10(that).
    pairs = {(0, 1): PairRecord(wins_i=15, draws=0, wins_j=5)}
    W, N = _win_and_game_matrices(2, pairs, prior_games=0.0)
    gamma = fit_bradley_terry(W, N)
    import math
    expected_gap = 400.0 * math.log10(15.0 / 5.0)
    got_gap = (400.0 / math.log(10.0)) * (math.log(gamma[0]) -
                                          math.log(gamma[1]))
    assert abs(got_gap - expected_gap) < 1e-3, (got_gap, expected_gap)
