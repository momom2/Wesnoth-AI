"""Trainer robustness: a stale MCTS actor-slot index must NOT crash
train_step.

Regression for the pre-existing 'MCTS actor-slot drift' bug surfaced
2026-06-17: an experience's recorded action indices assume the actor-
slot layout at SEARCH time, but re-encoding at train time can yield a
different actor count (chiefly the recruit-phantom count), so a recorded
index can fall outside the current legal set. The trainer must drop that
experience's POLICY term (value + aux still train) rather than
IndexError out of the whole gradient step.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from transformer_policy import TransformerPolicy             # noqa: E402
from trainer import MCTSExperience                           # noqa: E402
from sim_test_helpers import fresh_scenario_sim              # noqa: E402


def _gs():
    return fresh_scenario_sim(seed=21, max_turns=12, mini=True).gs


def test_out_of_range_actor_idx_does_not_crash_train_step(caplog):
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=48,
                            num_layers=2, num_heads=4, d_ff=96)
    gs = _gs()
    # A single visit tuple with an actor_idx far past any legal slot:
    # (actor_idx, target_idx, weapon_idx, count, type_idx).
    stale = MCTSExperience(
        game_state=gs,
        visit_counts=[(9999, None, None, 5, None)],
        z=1.0,
    )
    with caplog.at_level(logging.WARNING, logger="trainer"):
        stats = pol._trainer.step_mcts([stale])   # must not raise
    # The policy term was dropped (logged), but the step still ran and
    # produced a value-loss gradient against z.
    assert any("actor-slot" in r.message for r in caplog.records), \
        "expected the slot-drift skip warning"
    assert stats.n_transitions == 1
    assert stats.value_loss > 0.0          # value head still trained on z


def test_mixed_valid_and_stale_experiences_still_train(caplog):
    """A stale experience is skipped for policy; valid ones train
    normally in the same batch."""
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=48,
                            num_layers=2, num_heads=4, d_ff=96)
    from action_sampler import enumerate_legal_actions_with_priors
    gs = _gs()
    enc, mdl = pol._inference_encoder, pol._inference_model
    with torch.no_grad():
        encoded = enc.encode(gs)
        out = mdl(encoded)
    priors = enumerate_legal_actions_with_priors(encoded, out, gs)
    assert priors, "scenario should have legal actions"
    p = priors[0]
    valid = MCTSExperience(
        game_state=gs,
        visit_counts=[(p.actor_idx, p.target_idx, p.weapon_idx, 3,
                       p.type_idx)],
        z=1.0,
    )
    stale = MCTSExperience(
        game_state=gs,
        visit_counts=[(9999, None, None, 3, None)],
        z=-1.0,
    )
    with caplog.at_level(logging.WARNING, logger="trainer"):
        stats = pol._trainer.step_mcts([valid, stale])
    assert stats.n_transitions == 2
    assert any("1/2" in r.message for r in caplog.records), \
        "exactly one experience's policy term should be dropped"
