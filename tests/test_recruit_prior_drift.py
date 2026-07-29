"""The cycle-30 recruit-prior drift tripwire.

Cycle 29 found the "tried-and-cut tax" in Gumbel target extraction;
cycle 30 chose NOT to change the target and to arm a tripwire instead.
These pin the tripwire's analysis core -- the part that decides whether
the tax has graduated from "recorded" to "actionable" -- so the rule
cannot silently drift from the prose that justifies it.

The core is deliberately torch-free, which is why it is testable at all:
collection needs a checkpoint and a sim, the DECISION does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.recruit_prior_drift import (   # noqa: E402
    escalates, paired_delta, summarize,
)


def _row(turn, rec, end=0.01):
    return {"turn": turn, "gold": 100, "rec": rec, "end": end}


def test_buckets_split_at_turn_3():
    """turn<=2 is the opening (recruiting near-forced); turn>=3 is the
    diagnostic midgame bucket. The split point is the whole claim."""
    rows = [_row(1, 0.8), _row(2, 0.7), _row(3, 0.1), _row(9, 0.2)]
    s = summarize(rows)
    assert s["n_early"] == 2 and s["n_mid"] == 2
    assert s["rec_mean_early"] == pytest.approx(0.75)
    assert s["rec_mean_mid"] == pytest.approx(0.15)


def test_summarize_tolerates_an_empty_bucket():
    """A thin snapshot set must not raise -- the tripwire is meant to be
    runnable on whatever states exist."""
    s = summarize([_row(1, 0.9)])
    assert s["n_mid"] == 0
    assert s["rec_mean_mid"] == 0.0


def test_paired_delta_is_paired_and_direction_correct():
    a = [_row(1, 0.70), _row(4, 0.26), _row(5, 0.30)]
    b = [_row(1, 0.73), _row(4, 0.12), _row(5, 0.10)]
    d = paired_delta(a, b)
    assert d["n"] == 3
    assert d["up"] == 1                     # only the opening row rose
    assert d["mean"] == pytest.approx((0.03 - 0.14 - 0.20) / 3)
    # the midgame-only view is the diagnostic one
    assert d["n_mid"] == 2 and d["up_mid"] == 0
    assert d["mean_mid"] == pytest.approx((-0.14 - 0.20) / 2)


def test_paired_delta_refuses_mismatched_state_lists():
    """A length mismatch means the checkpoints did not score the same
    states, so the pairing -- the entire point -- would be a lie. It
    must raise, not truncate."""
    with pytest.raises(ValueError, match="same snapshots"):
        paired_delta([_row(3, 0.2)], [_row(3, 0.2), _row(4, 0.1)])


def test_escalation_rule_matches_the_documented_floor():
    """Cycle 30: escalate when midgame recruit prior bleeds below ~0.05.
    Pinned as a function so the rule cannot drift from the prose."""
    below = summarize([_row(5, 0.04), _row(6, 0.03)])
    above = summarize([_row(5, 0.12), _row(6, 0.20)])
    assert escalates(below) is True
    assert escalates(above) is False


def test_escalation_needs_midgame_states_to_fire():
    """An all-opening snapshot set has no midgame evidence, so it must
    NOT escalate -- otherwise an empty bucket reads as a collapse."""
    assert escalates(summarize([_row(1, 0.9), _row(2, 0.8)])) is False


def test_leg_measurements_reproduce_the_recorded_direction():
    """Sanity-check the instrument against the numbers cycle 29 actually
    recorded: midgame recruit prior fell 0.264 -> 0.122 while the
    opening ROSE 0.700 -> 0.731. A tool that cannot reproduce the sign
    of the finding it was built for is not measuring it."""
    seed = [_row(1, 0.700), _row(4, 0.264), _row(6, 0.264)]
    live = [_row(1, 0.731), _row(4, 0.122), _row(6, 0.122)]
    d = paired_delta(seed, live)
    assert d["mean_mid"] < 0            # midgame bled
    assert d["up_mid"] == 0
    assert live[0]["rec"] > seed[0]["rec"]   # opening rose
    assert summarize(live)["rec_mean_mid"] == pytest.approx(0.122)
