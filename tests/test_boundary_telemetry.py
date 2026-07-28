"""Boundary-pair harvesting (T1-F telemetry).

A turn handoff pairs the LAST recorded state of side p with the FIRST
recorded state of side q. Zero-sum calibration predicts V(pre)+V(post) ~ 0;
measured 2026-07-29 it is +0.4..+0.65 fogged and ~0 fogless (WYSIATI: the
head under-discounts unseen enemy assets, so both sides read optimistic).
These tests pin the EXTRACTION contract only -- no loss term exists yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


class _St:
    """Stand-in for _PendingMCTSState: only .side and .gs are read."""
    def __init__(self, side, tag):
        self.side = side
        self.gs = tag


def _pairs(states):
    """Mirror of the harvest in MCTSPolicy.finalize_game."""
    return [(a.gs, b.gs) for a, b in zip(states, states[1:]) if a.side != b.side]


def test_pairs_at_side_switches_only():
    states = [_St(1, "a"), _St(1, "b"), _St(2, "c"), _St(2, "d"), _St(1, "e")]
    assert _pairs(states) == [("b", "c"), ("d", "e")]


def test_single_side_game_yields_no_pairs():
    assert _pairs([_St(1, "a"), _St(1, "b"), _St(1, "c")]) == []


def test_side_three_interleave_is_not_assumed_alternating():
    """Sides are READ from the records, so a neutral/monster side-3 turn
    produces its own boundaries rather than corrupting the 1<->2 pairing."""
    states = [_St(1, "a"), _St(3, "b"), _St(2, "c")]
    assert _pairs(states) == [("a", "b"), ("b", "c")]


def test_playout_cap_gaps_do_not_break_pairing():
    """Unrecorded fast moves just widen the gap; the pair is still
    'last recorded of p' -> 'first recorded of q'."""
    states = [_St(1, "a"), _St(2, "z")]
    assert _pairs(states) == [("a", "z")]


def test_stats_fields_exist_and_default_to_nan():
    from wesnoth_ai.trainer import TrainStats
    s = TrainStats()
    assert s.boundary_sum != s.boundary_sum      # NaN default
    assert s.boundary_pairs_n == 0
