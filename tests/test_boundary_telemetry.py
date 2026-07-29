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


# --------------------------------------------------------------------
# Spool path (T1-H): the box runs 100 WORKER PROCESSES, so the learner's
# finalize_game never sees those games. Pairs are reconstructed at ingest
# from each experience's own recorded side.
# --------------------------------------------------------------------

class _Exp:
    def __init__(self, side, tag):
        class _GI:
            current_side = side
        class _GS:
            global_info = _GI()
        self.game_state = _GS()
        self.game_state._tag = tag


def _harvest(exps):
    from tools.mcts_policy import MCTSPolicy
    sink = []

    class _Shim:
        _boundary_pairs = sink

        class _L:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        _lock = _L()
    MCTSPolicy.harvest_boundary_pairs(_Shim(), exps)
    return sink


def test_spool_harvest_pairs_on_side_switches():
    exps = [_Exp(1, "a"), _Exp(1, "b"), _Exp(2, "c"), _Exp(2, "d"), _Exp(1, "e")]
    assert len(_harvest(exps)) == 2


def test_spool_harvest_single_side_and_short_games():
    assert _harvest([_Exp(1, "a"), _Exp(1, "b")]) == []
    assert _harvest([_Exp(1, "a")]) == []
    assert _harvest([]) == []


def test_combine_stats_carries_advice_fields():
    """REGRESSION: _combine_stats built a fresh TrainStats naming only 11
    fields, so every advice_* stat set inside step_mcts reverted to its NaN
    default under --replay-buffer -- telemetry that looked fine on the
    in-process path was dead in production. This test pins the chokepoint
    so the NEXT stat added in step_mcts cannot die silently."""
    from tools.mcts_policy import MCTSPolicy
    from wesnoth_ai.trainer import TrainStats
    a = TrainStats(advice_fire_rate=0.10, advice_opps_mean=1.5,
                   advice_grad_share=0.04, advice_out_norm=0.0)
    b = TrainStats(advice_fire_rate=0.20, advice_opps_mean=2.5,
                   advice_grad_share=0.06, advice_out_norm=7.5)
    out = MCTSPolicy._combine_stats([a, b], 2)
    assert abs(out.advice_fire_rate - 0.15) < 1e-9
    assert abs(out.advice_opps_mean - 2.0) < 1e-9
    assert abs(out.advice_grad_share - 0.05) < 1e-9
    assert out.advice_out_norm == 7.5          # LAST non-NaN, like grad_norm
    # all-NaN must stay NaN so the log guard still suppresses the fields
    n = MCTSPolicy._combine_stats([TrainStats(), TrainStats()], 2)
    assert n.advice_fire_rate != n.advice_fire_rate
