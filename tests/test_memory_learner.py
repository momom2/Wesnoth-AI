"""The self-play learner over a model with a memory (tools/memory_trace.py,
wesnoth_ai/memory_step.py): every decision of a memory player's game
reaches the learner in its game-side, the learner rebuilds the memory the
player held at each, its step back-propagates through the memory within a
window and not across, and the probes read each state with its memory."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.game_core import snapshot_view  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
SLOTS = 4
SEED = 3  # Hamlets: the leaders start on their keeps


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(relevant_set_hexes=True, observation_parity=True, memory_slots=SLOTS,
                             relevant_set_version=2, device=torch.device("cpu"),
                             d_model=32, num_layers=1, num_heads=2, d_ff=64)


def _play(policy, decisions: int = 14, winner: int = 1, label: str = "g"):
    """A memory player plays `decisions` decisions, about half of them fast
    moves without a target; returns the player, each decision's (side,
    the side's memory after it) and the game's experiences."""
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    cfg = MCTSConfig(n_simulations=4, playout_cap_randomization=True, playout_cap_prob=0.5,
                     playout_cap_fast_sims=1)
    player = MCTSPolicy(policy, cfg, memory_slots=SLOTS, rng_seed=0)
    sim = fresh_scenario_sim(seed=SEED, max_turns=10)
    held = []
    while len(held) < decisions and not sim.done:
        snapshot = snapshot_view(sim.gs)
        side = int(snapshot.global_info.current_side)
        action = player.select_action(snapshot, game_label=label, sim=sim)
        sim.step(action)
        if sim.last_step_rejected:
            player.drop_last_pending(label)
            continue
        held.append((side, player._memories[(label, side)].clone()))
    player.finalize_game(label, winner, final_gs=sim.gs)
    return player, held, list(player._queue)


@needs_core
def test_every_decision_reaches_the_learner_in_its_game_side():
    policy = _policy()
    _, held, exps = _play(policy)
    assert len(exps) == len(held)
    for side in (1, 2):
        mine = [e for e in exps if e.side == side]
        assert [e.side_step for e in mine] == list(range(sum(1 for s, _ in held if s == side)))
    carry = [e for e in exps if e.label_kind == "carry"]
    targets = [e for e in exps if e.label_kind == "game"]
    assert carry and targets, "the premise: fast moves and full searches both"
    assert all(e.game_state is None and e.game_weight == 0 and not e.visit_counts for e in carry)
    assert all(e.visit_counts and e.game_state is not None for e in targets)
    for e in exps:
        assert e.memory_k == SLOTS and e.raw is not None and e.raw.observation is None
        assert e.no_visible_unit.shape[0] == e.raw.hex_xs.shape[0]


@needs_core
def test_the_learner_rebuilds_the_memory_the_player_held():
    from wesnoth_ai.memory_step import memory_inputs
    policy = _policy()
    _, held, exps = _play(policy)
    memories = memory_inputs(policy._inference_model, policy._inference_encoder, exps)
    after = {}
    for side, state in held:
        after.setdefault(side, []).append(state)
    for e in exps:
        state = memories[id(e)].state
        if e.side_step == 0:
            assert state is None
        else:
            torch.testing.assert_close(state, after[e.side][e.side_step - 1], rtol=1e-4, atol=1e-5)


@needs_core
def test_the_step_reads_each_position_with_the_memory_of_its_game_side():
    """The step's losses are the ones the per-position loss reads with each
    position's rebuilt memory, through windows shorter than the game-sides
    and fewer streams side by side than game-sides."""
    from wesnoth_ai.memory_step import memory_inputs
    policy = _policy()
    _, _, exps = _play(policy)
    trainer = policy._trainer
    trainer.config.train_batch_size, trainer.config.memory_window = 1, 3
    with torch.no_grad():
        stats = trainer.step_mcts(exps, no_grad=True)
    memories = memory_inputs(trainer.model, trainer.encoder, exps)
    sums = {}

    def add(terms):
        for name, t in terms.items():
            sums[name] = sums.get(name, 0.0) + float(t)

    with torch.no_grad():
        trainer.mcts_loss_terms(exps, add, memories=memories)
    policy_loss = sum(sums[h] for h in ("actor", "type", "target", "weapon"))
    assert stats.policy_loss == pytest.approx(policy_loss, rel=1e-4, abs=1e-6)
    assert trainer.config.value_coef * stats.value_loss == pytest.approx(sums["value"], rel=1e-4, abs=1e-6)
    assert stats.belief_loss > 0
    assert trainer.config.belief_coef * stats.belief_loss == pytest.approx(sums["belief"], rel=1e-4, abs=1e-6)


def _writer_gradient(policy, exps, window: int) -> float:
    trainer = policy._trainer
    trainer.config.memory_window = window
    trainer.optimizer.zero_grad(set_to_none=True)
    real_step = trainer.optimizer.step
    trainer.optimizer.step = lambda *a, **k: None
    try:
        trainer.step_mcts(exps)
    finally:
        trainer.optimizer.step = real_step
    writer = [*trainer.model.slot_memory.gate.parameters(), *trainer.model.slot_memory.candidate.parameters()]
    return float(sum(p.grad.abs().sum() for p in writer if p.grad is not None))


@needs_core
def test_the_memory_carries_gradient_within_a_window_and_not_across():
    """Only side 1's last decision carries a loss: the memory writer gets its
    gradient through the memory that decision reads, which crosses from the
    decision before inside a window and is cut at a window's edge."""
    import dataclasses
    policy = _policy()
    _, _, exps = _play(policy)
    last = max((e for e in exps if e.side == 1 and e.label_kind == "game"), key=lambda e: e.side_step)
    quiet = dict(policy_weight=0.0, value_weight=0.0, game_weight=0.0, no_visible_unit=None)
    exps = [e if e is last else dataclasses.replace(e, **quiet) for e in exps]
    ones = [e for e in exps if e.side == 1]
    assert last.side_step >= 1 and last.side_step == len(ones) - 1, "the premise: a decision before it"
    assert _writer_gradient(policy, exps, window=16) > 0
    assert _writer_gradient(policy, exps, window=1) == 0


@needs_core
def test_a_game_side_with_a_missing_decision_is_refused():
    policy = _policy()
    _, _, exps = _play(policy)
    gap = [e for e in exps if not (e.side == 1 and e.side_step == 1)]
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        policy._trainer.step_mcts(gap)


@needs_core
def test_the_probes_read_each_state_with_its_memory():
    from tools.step_control import action_priors, backtracking_step
    from wesnoth_ai.memory_step import memory_inputs
    policy = _policy()
    _, _, exps = _play(policy)
    _, _, held = _play(policy, label="h", winner=2)
    states = [e for e in held if e.label_kind == "game"]
    with pytest.raises(ValueError, match="memory"):
        policy._trainer.eval_value_metrics(states)
    memories = memory_inputs(policy._trainer.model, policy._trainer.encoder, held)
    metrics = policy._trainer.eval_value_metrics(states, memories=memories)
    assert metrics["ce"] == metrics["ce"]

    def step():
        policy._trainer.step_mcts(exps)
        return None

    result = backtracking_step(
        policy, step, exps, held, states,
        memories_of=lambda: memory_inputs(policy._inference_model, policy._inference_encoder, held))
    assert result.shift["n"] == len(states)
    first = min(states, key=lambda e: (e.side, e.side_step))
    priors, _, value = action_priors(policy, first, memory_inputs(
        policy._inference_model, policy._inference_encoder, held))
    assert abs(sum(priors.values()) - 1.0) < 1e-6 and -1.0 <= value <= 1.0


@needs_core
def test_an_in_process_memory_player_trains_through_its_own_queue():
    """sim_self_play's path: the player's train_step over the games it
    finalized, several turns long (the boundary telemetry's pairs stay
    empty: it reads states without their games)."""
    policy = _policy()
    player, held, _ = _play(policy, decisions=40)
    assert len({side for side, _ in held}) == 2
    stats = player.train_step()
    assert stats.n_transitions == len(held) and stats.belief_loss > 0
    assert stats.boundary_pairs_n == 0
