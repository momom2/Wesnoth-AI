"""The batched factored policy loss step_mcts runs
(trainer._batched_factored_policy_loss) against the per-state
reference (trainer._mcts_factored_policy_loss_reference, the original
per-tuple loop) on real mid-game positions: the policy loss, the
"entropy" field and the gradient must agree within float32
reassociation noise through step_mcts at batch 1 and at batch 4
(chunks with different hex counts), with every kind of term present
(actor, type, attack / move / recruit targets, weapons, end_turn), a
legacy 4-tuple experience, an empty experience, unequal game and
policy weights, two combat-oracle anneal steps, and out-of-range
stored indices. Also the stage-timing hook.

Positions: the first bench states of configs/bench_states.json,
rebuilt from the replay dataset (skipped when it is not on this
machine; WESNOTH_BENCH_DATASET points at a packed copy).
"""
from __future__ import annotations

import logging
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from helpers.policy_loss_parity import (  # noqa: E402
    _assert_parity, _bench_states, batched_policy_step, reference_policy_step,
)
from tools.az_recipe import configure_az_trainer  # noqa: E402
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from wesnoth_ai.trainer import STEP_MCTS_STAGES, MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

N_STATES = 6


@pytest.fixture(scope="module")
def case():
    """A small policy configured as az_loop configures its trainer,
    and experiences on real positions whose visit tables cover
    every term kind."""
    states = _bench_states(N_STATES)
    torch.manual_seed(0)
    policy = TransformerPolicy(device=torch.device("cpu"),
                               d_model=64, num_layers=2, num_heads=4, d_ff=128)
    configure_az_trainer(policy._trainer)
    enc, mdl = policy._inference_encoder, policy._inference_model
    rng = random.Random(0)
    exps = []
    kinds = Counter()
    for i, gs in enumerate(states):
        enc.register_names(gs)
        with torch.no_grad():
            e0 = enc.encode(gs)
            legal = enumerate_legal_actions_with_priors(e0, mdl(e0), gs)
        assert legal
        if i % 2 == 0:
            # Every legal action, varied counts: the Gumbel-root regime.
            chosen = [(la, float((j % 7) + 1)) for j, la in enumerate(legal)]
        else:
            # 32 draws: the loop's search budget.
            counts = Counter(rng.choices(range(len(legal)), k=32))
            chosen = [(legal[j], float(c)) for j, c in sorted(counts.items())]
        for la, _c in chosen:
            kinds[la.action.get("type")] += 1
        visits = [(la.actor_idx, la.target_idx, la.weapon_idx, c, la.type_idx)
                  for la, c in chosen]
        exps.append(MCTSExperience(
            game_state=gs, visit_counts=visits, z=rng.choice((-1.0, 1.0)),
            game_weight=1.0 / (1 + i % 3), policy_weight=0.5 if i == 1 else 1.0,
            game_id=f"g{i // 2}", decision_step=0 if i < 3 else 5000))
    # Legacy 4-tuples (no type: union target rows, no type term).
    exps[2].visit_counts = [v[:4] for v in exps[2].visit_counts]
    # No visits at all (step_control's held-out probes build these).
    exps.append(MCTSExperience(game_state=states[0], visit_counts=[], z=1.0,
                               policy_weight=0.0, game_id="empty"))
    for kind in ("attack", "move", "recruit", "end_turn"):
        assert kinds[kind] > 0, f"the bench positions carry no {kind} target: {kinds}"
    return policy, exps


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_loss_matches_reference_through_step_mcts(case, batch_size):
    policy, exps = case
    ref = reference_policy_step(policy, exps)
    cand = batched_policy_step(policy, exps, batch_size)
    _assert_parity(ref, cand, f"B={batch_size}")


def test_out_of_range_indices_skip_loudly_and_match_reference(case, caplog):
    """Stored-index bounds guard: tuples whose indices exceed the
    re-encoded basis are skipped with a log.error by both paths, and
    the surviving terms still agree."""
    policy, exps = case
    bad = list(exps)
    good = bad[0].visit_counts
    a0, t0 = good[0][0], good[0][4]
    bad[0] = MCTSExperience(
        game_state=bad[0].game_state, z=bad[0].z, game_weight=bad[0].game_weight,
        game_id=bad[0].game_id,
        visit_counts=good + [(10_000, None, None, 3.0, None),
                             (a0, 10_000, None, 3.0, t0)])
    with caplog.at_level(logging.ERROR, logger="trainer"):
        ref = reference_policy_step(policy, bad)
        caplog.clear()
        cand = batched_policy_step(policy, bad, 4)
    assert sum("index out of range" in r.message for r in caplog.records) >= 2
    _assert_parity(ref, cand, "oob")


def test_stage_timings_hook(case):
    policy, exps = case
    tr = policy._trainer
    tr.optimizer.step = lambda *a, **k: None    # weights stay put for the other tests
    try:
        # The attribute channel (MCTSPolicy.train_step callers).
        tr.stage_timings = {}
        tr.step_mcts(exps)
        first = dict(tr.stage_timings)
        assert set(first) == set(STEP_MCTS_STAGES)
        assert all(v > 0.0 for v in first.values()), first
        # A second call adds to the same dict.
        tr.step_mcts(exps)
        assert all(tr.stage_timings[s] > first[s] for s in STEP_MCTS_STAGES)
        # The keyword wins over the attribute, which stays untouched.
        second = dict(tr.stage_timings)
        own = {}
        tr.step_mcts(exps, timings=own)
        assert set(own) == set(STEP_MCTS_STAGES)
        assert tr.stage_timings == second
        tr.stage_timings = None
        tr.step_mcts(exps)      # no sink: no timing, no error
    finally:
        del tr.optimizer.step
        tr.optimizer.zero_grad(set_to_none=True)
