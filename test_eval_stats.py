#!/usr/bin/env python3
"""Tests for eval_vs_builtin's statistical helpers.

Wilson interval values cross-checked against scipy.stats.binom_test
(implicitly via standard tables). Hand-checked points:

  - 0/0     -> (0.0, 1.0)            (no data, full range)
  - 5/10    -> ~(23.7, 76.3)         (50% point estimate, ±26%)
  - 10/10   -> ~(72.2, 100)          (top-clamped)
  - 0/10    -> ~(0, 27.7)            (bottom-clamped)
  - 50/100  -> ~(40.4, 59.6)         (50%, tighter)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools"))

import pytest

from eval_vs_builtin import _wilson_interval, _wr


def test_wilson_zero_n_returns_full_range():
    lo, hi = _wilson_interval(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_50_percent_n10():
    """Standard table: 5/10 with z=1.96 -> [0.237, 0.763]."""
    lo, hi = _wilson_interval(5, 10)
    assert lo == pytest.approx(0.237, abs=0.005)
    assert hi == pytest.approx(0.763, abs=0.005)


def test_wilson_full_wins():
    """10/10: lower bound should be ~0.722, upper bound 1.0."""
    lo, hi = _wilson_interval(10, 10)
    assert lo == pytest.approx(0.722, abs=0.01)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_zero_wins():
    """0/10: lower bound 0.0, upper bound ~0.278."""
    lo, hi = _wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.278, abs=0.01)


def test_wilson_tighter_with_more_data():
    """50/100 should give a tighter interval than 5/10 -- both
    50% point estimates."""
    _, hi_10 = _wilson_interval(5, 10)
    _, hi_100 = _wilson_interval(50, 100)
    width_10 = hi_10 - 0.5
    width_100 = hi_100 - 0.5
    assert width_100 < width_10
    # 100-game CI should be roughly half the width of the 10-game CI.
    assert width_100 == pytest.approx(width_10 / (10 ** 0.5), rel=0.2)


def test_wilson_bounds_always_in_unit_interval():
    """Random spot check across boundary cases."""
    for w, n in [(0, 1), (1, 1), (0, 100), (100, 100),
                 (1, 1000), (999, 1000)]:
        lo, hi = _wilson_interval(w, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wr_includes_ci_in_string():
    """The user-facing win-rate string includes the CI bounds."""
    s = _wr({"win": 5, "loss": 5, "draw": 0, "timeout": 0, "errored": 0})
    assert "95% CI" in s
    assert "(5/10" in s


def test_wr_handles_zero_decisive():
    """No decisive games -> 'n/a', no CI fragment."""
    s = _wr({"win": 0, "loss": 0, "draw": 5, "timeout": 0, "errored": 0})
    assert s == "n/a"
