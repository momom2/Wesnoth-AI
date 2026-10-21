"""Auxiliary prediction head (KataGo §3.5).

Pins: the config-gated aux head (present + bounded when on, absent from
the default arch when off) and the trainer's aux MSE term firing only
when the head + targets are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.transformer_policy import TransformerPolicy             # noqa: E402
from tools.mcts import MCTSConfig                             # noqa: E402
from tools.mcts_policy import MCTSPolicy                      # noqa: E402
from sim_test_helpers import fresh_scenario_sim               # noqa: E402


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
                      batch_size=1, add_root_noise=False)


def _play_and_finalize(mp, seed=21):
    # Drive the REAL production rollout (tools.selfplay_game.play_one_game)
    # rather than re-implementing the select_action/step/finalize loop:
    # mirroring the loop by hand is exactly what previously got the
    # snapshot-deepcopy contract wrong. play_one_game deepcopies the
    # per-decision state and calls finalize_game itself, so the recorded
    # MCTSExperiences land in mp._queue. Seed the search RNG for
    # reproducibility.
    import numpy as np
    from tools.selfplay_game import play_one_game
    mp._rng = np.random.default_rng(seed)
    sim = fresh_scenario_sim(seed=seed, max_turns=8, mini=True)
    play_one_game(sim, mp, game_label="g0")


def test_select_action_rejects_live_sim_gs():
    """Contract guard: passing the LIVE sim.gs (not a deepcopy snapshot)
    must fail loudly -- otherwise sim.step would mutate the recorded
    training target (the 'actor-slot drift' that bit the tests)."""
    import pytest
    mp = MCTSPolicy(_pol(False), _cfg())
    sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)
    with pytest.raises(ValueError, match="deepcopy snapshot"):
        mp.select_action(sim.gs, game_label="x", sim=sim)


@pytest.mark.slow          # ~21s: see pytest.ini two-tier note
def test_train_step_aux_loss_fires_only_when_on():
    mp_on = MCTSPolicy(_pol(True), _cfg())
    _play_and_finalize(mp_on, seed=21)
    assert mp_on._queue, "expected recorded experiences"
    for e in mp_on._queue:
        e.aux_target = 0.5
    stats_on = mp_on.train_step()
    assert stats_on.aux_loss > 0.0, "aux loss should be active with head+targets"
    assert stats_on.total_loss == stats_on.total_loss  # finite (not NaN)

    mp_off = MCTSPolicy(_pol(False), _cfg())
    _play_and_finalize(mp_off, seed=22)
    stats_off = mp_off.train_step()
    assert stats_off.aux_loss == 0.0, "no aux head -> no aux loss"
