"""The batched factored policy loss step_mcts runs
(trainer._batched_factored_policy_loss) against the per-state
reference (trainer._mcts_factored_policy_loss_reference) on SYNTHETIC
positions, so the pin holds on any machine: a mini scenario from
wesnoth_ai/rules/scenario_pool (as tests/test_encode_raw_cache.py builds them),
advanced by a scripted charger until attacks are legal. Covers every
term kind (actor, type, attack / move / recruit targets, weapons,
end_turn), a legacy 4-tuple experience, an empty (value-only)
experience, unequal game and policy weights, two combat-oracle anneal
steps and out-of-range stored indices, through step_mcts at batch 1
and batch 4. The value-only experience must not pay the host mask
build. tests/test_batched_policy_loss.py runs the same pins on real
mid-game positions when the replay dataset is present.
"""
from __future__ import annotations

import copy
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
sys.path.insert(0, str(Path(__file__).parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from helpers.policy_loss_parity import (  # noqa: E402
    _assert_parity, batched_policy_step, reference_policy_step,
)
from tools.az_recipe import configure_az_trainer  # noqa: E402
from wesnoth_ai import trainer as trainer_module  # noqa: E402
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from wesnoth_ai.rewards import hex_distance  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

N_STATES = 6
MAX_STEPS = 400


def _legal(policy, gs):
    enc, mdl = policy._inference_encoder, policy._inference_model
    enc.register_names(gs)
    with torch.no_grad():
        e0 = enc.encode(gs)
        return enumerate_legal_actions_with_priors(e0, mdl(e0), gs)


def _charge(legal, gs, rng: random.Random):
    """The scripted opponent of itself: attack when possible, else the
    move that brings a unit closest to an enemy, else recruit, else
    end the turn. Reaches contact on a mini map within a few turns."""
    by_kind = {}
    for la in legal:
        by_kind.setdefault(la.action.get("type"), []).append(la)
    if "attack" in by_kind:
        return rng.choice(by_kind["attack"])
    side = gs.global_info.current_side
    enemies = [u.position for u in gs.map.units if u.side != side]
    if "move" in by_kind and enemies:
        def closest(la):
            t = la.action["target_hex"]
            return min(hex_distance(t.x, t.y, e.x, e.y) for e in enemies)
        return min(by_kind["move"], key=closest)
    if "recruit" in by_kind:
        return rng.choice(by_kind["recruit"])
    return by_kind["end_turn"][0]


def _synthetic_states(policy, n: int, seed: int):
    """`n` decision states from mini games: each game's opening position
    (recruits), every position where an attack is legal and every
    eighth position; another seed when a game ends early."""
    states = []
    for game_seed in range(seed, seed + 8):
        sim = fresh_scenario_sim(seed=game_seed, max_turns=20, mini=True)
        rng = random.Random(game_seed)
        states.append(copy.deepcopy(sim.gs))
        steps = 0
        while not sim.done and len(states) < n and steps < MAX_STEPS:
            legal = _legal(policy, sim.gs)
            if any(la.action.get("type") == "attack" for la in legal) or steps % 8 == 7:
                states.append(copy.deepcopy(sim.gs))
            sim.step(_charge(legal, sim.gs, rng).action)
            steps += 1
        if len(states) >= n:
            break
    return states[:n]


@pytest.fixture(scope="module")
def case():
    """A small policy configured as az_loop configures its trainer, and
    experiences on synthetic positions whose visit tables cover every
    term kind."""
    torch.manual_seed(0)
    policy = TransformerPolicy(device=torch.device("cpu"),
                               d_model=64, num_layers=2, num_heads=4, d_ff=128)
    configure_az_trainer(policy._trainer)
    states = _synthetic_states(policy, N_STATES, seed=11)
    assert len(states) == N_STATES, f"only {len(states)} synthetic states reached"
    rng = random.Random(0)
    exps = []
    kinds = Counter()
    for i, gs in enumerate(states):
        legal = _legal(policy, gs)
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
    # A value-only experience (no visits); its own state object so the
    # mask-build pin can tell it apart.
    exps.append(MCTSExperience(game_state=copy.deepcopy(states[0]), visit_counts=[],
                               z=1.0, policy_weight=0.0, game_id="empty"))
    for kind in ("attack", "move", "recruit", "end_turn"):
        assert kinds[kind] > 0, f"the synthetic positions carry no {kind} target: {kinds}"
    assert any(v[2] is not None for e in exps for v in e.visit_counts), "no weapon term"
    return policy, exps


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_loss_matches_reference_through_step_mcts(case, batch_size):
    policy, exps = case
    ref = reference_policy_step(policy, exps)
    cand = batched_policy_step(policy, exps, batch_size)
    _assert_parity(ref, cand, f"synthetic B={batch_size}")


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
    _assert_parity(ref, cand, "synthetic oob")


def test_value_only_experience_skips_the_host_mask_build(case, monkeypatch):
    """An experience with no visit counts contributes no policy term:
    step_mcts must not build its legality masks on the host (the
    largest host item of the step), while the chunk layout stays
    index-aligned and the loss stays at the reference."""
    policy, exps = case
    empty = [e for e in exps if not e.visit_counts]
    assert len(empty) == 1
    empty_gs = empty[0].game_state
    built = []
    real = trainer_module._host_packed_masks

    def counting(raw, game_state, decision_step):
        built.append(game_state)
        return real(raw, game_state, decision_step)
    monkeypatch.setattr(trainer_module, "_host_packed_masks", counting)
    ref = reference_policy_step(policy, exps)
    cand = batched_policy_step(policy, exps, 4)
    assert len(built) == len(exps) - 1
    assert all(gs is not empty_gs for gs in built)
    _assert_parity(ref, cand, "value-only in batch")
