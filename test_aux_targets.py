"""Auxiliary prediction target (KataGo §3.5): the model predicts the
final MATERIAL margin as a denser companion to the win/loss z.

Pins: the pure margin target (antisymmetry, range, monotonicity); the
config-gated aux head (present + bounded when on, absent from the
default arch when off); finalize_game attaching aux_target; and the
trainer's aux MSE term firing only when the head + targets are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from transformer_policy import TransformerPolicy             # noqa: E402
from tools.mcts import MCTSConfig                             # noqa: E402
from tools.mcts_policy import MCTSPolicy                      # noqa: E402
from tools.draw_tiebreak import (                             # noqa: E402
    DrawTiebreakConfig, material_margin, draw_tiebreak_z,
)
from sim_test_helpers import fresh_scenario_sim               # noqa: E402


# ---- pure target ----------------------------------------------------

def test_material_margin_antisymmetric_and_bounded():
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    cfg = DrawTiebreakConfig(cap=0.3)
    m1 = material_margin(sim.gs, 1, cfg)
    m2 = material_margin(sim.gs, 2, cfg)
    assert abs(m1 + m2) < 1e-9, "margin must be zero-sum between sides"
    assert -1.0 < m1 < 1.0 and -1.0 < m2 < 1.0


def test_material_margin_is_uncapped_vs_draw_z():
    # Force a large, deterministic material lead for side 1 (huge gold
    # gap) and confirm the margin saturates toward 1.0 while the draw z
    # stays clamped at the cap -- the whole point of the denser signal.
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    gs = sim.gs
    gs.sides[0].current_gold = 1000
    gs.sides[1].current_gold = 0
    cfg = DrawTiebreakConfig(cap=0.3, score_scale=1.0)
    m = material_margin(gs, 1, cfg)
    z = draw_tiebreak_z(gs, 1, cfg)
    assert m > cfg.cap, f"uncapped margin should exceed the draw cap: {m}"
    assert abs(z) <= cfg.cap + 1e-9, f"draw z must respect the cap: {z}"
    assert m > z


def test_material_margin_disabled_when_scale_zero():
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    cfg = DrawTiebreakConfig(score_scale=0.0)
    assert material_margin(sim.gs, 1, cfg) == 0.0


# ---- model head -----------------------------------------------------

def _pol(aux):
    return TransformerPolicy(device=torch.device("cpu"), d_model=48,
                             num_layers=2, num_heads=4, d_ff=96,
                             aux_score=aux)


def test_aux_head_present_and_bounded_when_enabled():
    pol = _pol(True)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    with torch.no_grad():
        out = pol._inference_model(pol._inference_encoder.encode(sim.gs))
    assert out.aux_score is not None
    assert tuple(out.aux_score.shape) == (1, 1)
    assert -1.0 < float(out.aux_score.item()) < 1.0


def test_aux_head_none_and_arch_unchanged_when_disabled():
    pol = _pol(False)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    with torch.no_grad():
        out = pol._inference_model(pol._inference_encoder.encode(sim.gs))
    assert out.aux_score is None
    # The default arch's state_dict must carry NO aux head keys, so
    # existing checkpoints are byte-compatible.
    keys = pol._model.state_dict().keys()
    assert not any("aux_score_head" in k for k in keys)
    assert any("aux_score_head" in k for k in _pol(True)._model.state_dict())


# ---- end-to-end -----------------------------------------------------

def _cfg():
    return MCTSConfig(n_simulations=10, gumbel_root=True, gumbel_m=4,
                      chance_nodes=True, exact_outcome_enumeration=True,
                      draw_tiebreak=DrawTiebreakConfig(cap=0.3),
                      batch_size=1, add_root_noise=False)


def _play_and_finalize(mp, seed=21):
    # Drive the REAL production rollout (tools.sim_self_play.play_one_game)
    # rather than re-implementing the select_action/step/finalize loop:
    # mirroring the loop by hand is exactly what previously got the
    # snapshot-deepcopy contract wrong. play_one_game deepcopies the
    # per-decision state and calls finalize_game itself, so the recorded
    # MCTSExperiences land in mp._queue. Seed the search RNG for
    # reproducibility. (MCTS ignores per-step rewards -> a zero reward_fn.)
    import numpy as np
    from tools.sim_self_play import play_one_game, _recruit_cost_lookup
    mp._rng = np.random.default_rng(seed)
    sim = fresh_scenario_sim(seed=seed, max_turns=8, mini=True)
    play_one_game(sim, mp, lambda delta: 0.0,
                  game_label="g0", cost_lookup=_recruit_cost_lookup())


def test_select_action_rejects_live_sim_gs():
    """Contract guard: passing the LIVE sim.gs (not a deepcopy snapshot)
    must fail loudly -- otherwise sim.step would mutate the recorded
    training target (the 'actor-slot drift' that bit the tests)."""
    import pytest
    mp = MCTSPolicy(_pol(False), _cfg())
    sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)
    with pytest.raises(ValueError, match="deepcopy snapshot"):
        mp.select_action(sim.gs, game_label="x", sim=sim)


def test_finalize_sets_aux_target_when_aux_on():
    mp = MCTSPolicy(_pol(True), _cfg())
    _play_and_finalize(mp)
    assert mp._queue, "expected recorded experiences"
    assert all(e.aux_target is not None for e in mp._queue)
    assert all(-1.0 <= e.aux_target <= 1.0 for e in mp._queue)


@pytest.mark.slow          # ~21s: see pytest.ini two-tier note
def test_train_step_aux_loss_fires_only_when_on():
    mp_on = MCTSPolicy(_pol(True), _cfg())
    _play_and_finalize(mp_on, seed=21)
    stats_on = mp_on.train_step()
    assert stats_on.aux_loss > 0.0, "aux loss should be active with head+targets"
    assert stats_on.total_loss == stats_on.total_loss  # finite (not NaN)

    mp_off = MCTSPolicy(_pol(False), _cfg())
    _play_and_finalize(mp_off, seed=22)
    stats_off = mp_off.train_step()
    assert stats_off.aux_loss == 0.0, "no aux head -> no aux loss"
