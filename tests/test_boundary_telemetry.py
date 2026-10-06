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


# --------------------------------------------------------------------
# Per-game harvest (T1-H): the pool's actors play in other processes, so
# the learner's finalize_game never sees those games. Pairs are
# reconstructed at the drain from each experience's own recorded side.
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
        _trace = None                   # a player without a memory

        class _L:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        _lock = _L()
    MCTSPolicy.harvest_boundary_pairs(_Shim(), exps)
    return sink


def test_harvest_pairs_on_side_switches():
    exps = [_Exp(1, "a"), _Exp(1, "b"), _Exp(2, "c"), _Exp(2, "d"), _Exp(1, "e")]
    assert len(_harvest(exps)) == 2


def test_harvest_single_side_and_short_games():
    assert _harvest([_Exp(1, "a"), _Exp(1, "b")]) == []
    assert _harvest([_Exp(1, "a")]) == []
    assert _harvest([]) == []


# --------------------------------------------------------------------
# Sampling contract (2026-07-29). The reading is a MEAN over sampled
# pairs, so its sampling SE scales 1/sqrt(k) -- and it is watched
# against a +-0.25 band that the campaign's first readings swing most
# of on their own. These pin that the sample cap is deliberate and
# that the POOL size is reported, since boundary_pairs_n saturates at
# the cap and so cannot on its own tell you whether more sampling
# would help.
# --------------------------------------------------------------------

def _attach(n_pairs, k=None):
    """Run the production _attach_boundary_sum over `n_pairs` synthetic
    pairs, stubbing ONLY at the model/encoder boundary."""
    import random
    import torch
    from wesnoth_ai.trainer import TrainStats
    from tools.mcts_policy import MCTSPolicy

    class _Out:
        value = torch.tensor(0.25)

    class _Model:
        def __call__(self, x):
            return _Out()

    class _Enc:
        def encode(self, gs):
            return gs

    class _Base:
        _model = _Model()
        _encoder = _Enc()

    class _L:
        def __enter__(self): return None
        def __exit__(self, *a): return False

    class _Shim:
        _boundary_pairs = [(object(), object()) for _ in range(n_pairs)]
        _boundary_rng = random.Random(0)
        _lock = _L()
        _base = _Base()

    stats = TrainStats()
    if k is None:
        MCTSPolicy._attach_boundary_sum(_Shim(), stats)
    else:
        MCTSPolicy._attach_boundary_sum(_Shim(), stats, k=k)
    return stats


def test_pool_size_is_reported_separately_from_sample_size():
    """The whole point: n saturates at the cap, pool does not."""
    stats = _attach(500)
    assert stats.boundary_pairs_n == 64      # capped
    assert stats.boundary_pool_n == 500      # true population
    # 0.25 per state, two states per pair -> mean 0.5
    assert abs(stats.boundary_sum - 0.5) < 1e-6


def test_small_pool_reports_equal_n_and_pool():
    """Below the cap the two agree -- which is exactly why reporting
    only n was ambiguous."""
    stats = _attach(10)
    assert stats.boundary_pairs_n == 10
    assert stats.boundary_pool_n == 10


def test_too_few_pairs_leaves_defaults_untouched():
    stats = _attach(3)
    assert stats.boundary_sum != stats.boundary_sum   # still NaN
    assert stats.boundary_pairs_n == 0
    assert stats.boundary_pool_n == 0
