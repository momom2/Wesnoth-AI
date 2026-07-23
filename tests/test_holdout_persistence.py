#!/usr/bin/env python3
"""Persistent holdout probe (2026-07-18).

Every supervisor relaunch used to resample the holdout set, so the
logged holdout value CE jumped between sessions (0.44<->0.88 on set
changes) and network-capacity trends were unreadable. The probe now
persists beside the campaign checkpoint: partial sets resume
collecting, full sets stay frozen, and the CE series is comparable
across restarts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _tiny_policy(holdout_size):
    import torch
    from tools.draw_tiebreak import DrawTiebreakConfig
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from wesnoth_ai.transformer_policy import TransformerPolicy

    torch.manual_seed(0)
    base = TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))
    return MCTSPolicy(
        base,
        MCTSConfig(n_simulations=2,
                   draw_tiebreak=DrawTiebreakConfig(cap=0.3)),
        holdout_size=holdout_size, holdout_per_game_cap=4)


def _fake_game(policy, n_states, label):
    """Run one REAL tiny game through finalize_game so the holdout
    diversion path produces genuine MCTSExperiences (tests drive the
    production path, not a mirror)."""
    import copy
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(seed=hash(label) % 1000, max_turns=2,
                             mini=True)
    n = 0
    while not sim.done and n < n_states:
        pre = copy.deepcopy(sim.gs)
        action = policy.select_action(pre, game_label=label, sim=sim)
        sim.step(action)
        n += 1
    policy.finalize_game(label, sim.winner if sim.done else 0,
                         final_gs=sim.gs)


@pytest.mark.slow          # ~18s: fills a real holdout via 3 tiny games
def test_holdout_round_trip_and_freeze(tmp_path):
    p1 = _tiny_policy(holdout_size=4)
    for g in range(3):
        _fake_game(p1, 4, f"g{g}")
        if len(p1._holdout) >= 4:
            break
    assert len(p1._holdout) >= 4, "holdout never filled"
    path = tmp_path / "camp.pt.holdout"
    assert p1.save_holdout(path)
    assert path.exists()

    p2 = _tiny_policy(holdout_size=4)
    assert p2.load_holdout(path)
    assert len(p2._holdout) == len(p1._holdout)
    assert p2._holdout_games == p1._holdout_games
    # Frozen: further games are NOT diverted.
    assert p2.offer_holdout_game(p1._holdout[:2]) is False
    # Metrics evaluate on the restored set.
    m = p2.holdout_metrics()
    assert m is not None and m[1] == len(p2._holdout)


def test_partial_holdout_resumes_collecting(tmp_path):
    p1 = _tiny_policy(holdout_size=64)      # won't fill from one game
    _fake_game(p1, 4, "partial")
    n_partial = len(p1._holdout)
    assert 0 < n_partial < 64
    path = tmp_path / "camp.pt.holdout"
    assert p1.save_holdout(path)

    p2 = _tiny_policy(holdout_size=64)
    assert p2.load_holdout(path)
    assert len(p2._holdout) == n_partial
    # Still collecting: a new game IS diverted.
    assert p2.offer_holdout_game(p1._holdout[:2]) is True
    assert len(p2._holdout) == n_partial + 2


def test_load_missing_or_corrupt_is_safe(tmp_path):
    p = _tiny_policy(holdout_size=4)
    assert p.load_holdout(tmp_path / "nope.holdout") is False
    bad = tmp_path / "bad.holdout"
    bad.write_bytes(b"not a pickle")
    assert p.load_holdout(bad) is False
    assert p._holdout == []                 # untouched


def test_maybe_persist_saves_on_growth_only(tmp_path):
    p = _tiny_policy(holdout_size=4)
    path = tmp_path / "camp.pt.holdout"
    p._holdout_persist_path = path
    p.maybe_persist_holdout()               # empty -> no file
    assert not path.exists()
    _fake_game(p, 4, "grow")
    p.maybe_persist_holdout()
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    p.maybe_persist_holdout()               # no growth -> no rewrite
    assert path.stat().st_mtime_ns == mtime
