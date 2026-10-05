"""Turn search over a model with a memory (tools/turn_search.py,
tools/turn_policy.py): every line the search walks reads, at each
position, what the side to move wrote at its last decision on that line,
else its memory at the turn's start; and the turn-search player's memory
after each command it serves is what that decision wrote at the live
state."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.game_core import snapshot_view  # noqa: E402
from wesnoth_ai.memory import MemoryState, SideMemories  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
SLOTS = 8
SEED = 3  # Hamlets: the leaders start on their keeps, so a turn can recruit four times


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(relevant_set_hexes=True, observation_parity=True, memory_slots=SLOTS,
                             relevant_set_version=2, device=torch.device("cpu"),
                             d_model=32, num_layers=1, num_heads=2, d_ff=64)


def _start_memories(model) -> SideMemories:
    """Distinct memories for the two sides, so a read of the wrong one shows."""
    init = model.initial_memory(SLOTS).detach()
    return SideMemories(SLOTS, {1: init + 0.1, 2: init - 0.1})


def _decide(model, encoder, gs, held):
    with torch.no_grad():
        return model(encoder.encode(gs), memory=MemoryState(SLOTS, held))


def _busy_turn(policy, sim, side, start: SideMemories, n: int = 4):
    """The mover's turn as `n` decisions other than end_turn (recruits
    first, the most likely at each), then end_turn."""
    from tools.turn_search import forward_state
    r, track, actions = sim.fork(), start.copy(), []
    for _ in range(n):
        _, _, legal = forward_state(policy, r.gs, 0, track)
        act = max((a for a in legal if a.action["type"] != "end_turn"),
                  key=lambda a: (a.action["type"] == "recruit", a.prior)).action
        r.step(act)
        assert r.gs.global_info.current_side == side
        actions.append(act)
    return actions + [{"type": "end_turn"}]


def _salted_fork(sim, salt):
    r = sim.fork()
    r._is_search_fork = True
    r._seed_salt = salt
    return r


@needs_core
def test_a_materialized_turn_is_graded_with_the_memory_its_line_holds():
    """Mover frame: the pre-flip boundary reads what the mover's decisions
    before its last one wrote. Opponent frame: the post-flip boundary reads
    the opponent's memory, which the mover's turn leaves as it was."""
    from tools.turn_search import _value_for, batch_boundary_values, materialize
    policy = _policy()
    model, encoder = policy._inference_model, policy._inference_encoder
    sim = fresh_scenario_sim(seed=SEED, max_turns=10)
    side = sim.gs.global_info.current_side
    start = _start_memories(model)
    commands = _busy_turn(policy, sim, side, start)

    mover = materialize(policy, sim, side, commands, "salt", 0, mover_frame=True, memory=start)
    assert mover.executed == commands
    r = _salted_fork(sim, "salt")
    held = start.read(side).state
    for cmd in mover.executed[:-1]:
        held = _decide(model, encoder, r.gs, held).memory
        r.step(cmd)
    want = _value_for(_decide(model, encoder, r.gs, held), r.gs, side)
    assert mover.value == pytest.approx(want, abs=1e-6)

    opponent = materialize(policy, sim, side, commands, "salt", 0, skip_value=True, memory=start)
    batch_boundary_values(policy, [opponent], side, 0)
    r.step(mover.executed[-1])
    other = r.gs.global_info.current_side
    assert other != side
    want = _value_for(_decide(model, encoder, r.gs, start.read(other).state), r.gs, side)
    assert opponent.value == pytest.approx(want, abs=1e-6)


@needs_core
def test_the_spine_reads_the_memory_of_its_own_walk():
    from tools.turn_search import _value_for, record_spine
    policy = _policy()
    model, encoder = policy._inference_model, policy._inference_encoder
    sim = fresh_scenario_sim(seed=SEED, max_turns=10)
    side = sim.gs.global_info.current_side
    start = _start_memories(model)
    commands = _busy_turn(policy, sim, side, start)
    steps, _ = record_spine(policy, sim, side, 0, np.random.default_rng(0), max_spine=8,
                            actions=commands, memory=start)
    assert [s.action for s in steps] == commands
    held = start.read(side).state
    for step in steps:
        out = _decide(model, encoder, step.pre_fork.gs, held)
        assert step.pre_value == pytest.approx(_value_for(out, step.pre_fork.gs, side), abs=1e-6)
        held = out.memory


class _Spy:
    """The model, recording each call's memory in and memory out."""

    def __init__(self, model):
        self.model, self.calls = model, []

    def __call__(self, encoded, memory=None):
        out = self.model(encoded, memory=memory)
        held = memory.state if memory.state is not None else self.model.initial_memory(memory.k)
        self.calls.append((held, out.memory))
        return out


@needs_core
def test_a_projection_carries_each_sides_memory_through_its_turns():
    """From a boundary where the opponent moves: each of its decisions reads
    what the one before wrote, the first reads its memory at the boundary,
    and the final value read, the mover to move again, reads the mover's."""
    from tools.turn_search import project_value
    policy = _policy()
    model, encoder = policy._inference_model, policy._inference_encoder
    sim = fresh_scenario_sim(seed=SEED, max_turns=10)
    side = sim.gs.global_info.current_side
    boundary = sim.fork()
    boundary.step({"type": "end_turn"})
    other = boundary.gs.global_info.current_side
    start = _start_memories(model)
    spy = _Spy(model)
    project_value(SimpleNamespace(_inference_model=spy, _inference_encoder=encoder), boundary, side, 0,
                  half_turns=1, max_actions=4, rng=np.random.default_rng(0), memory=start)
    decisions, final = spy.calls[:-1], spy.calls[-1]
    assert decisions
    torch.testing.assert_close(decisions[0][0], start.read(other).state)
    for (_, wrote), (read, _) in zip(decisions, decisions[1:]):
        torch.testing.assert_close(read, wrote)
    torch.testing.assert_close(final[0], start.read(side).state)


@needs_core
def test_the_turn_search_players_memory_follows_its_served_decisions():
    from tools.mcts import MCTSConfig
    from tools.turn_policy import TurnCommitPolicy
    from tools.turn_search import TurnSearchConfig
    policy = _policy()
    model, encoder = policy._inference_model, policy._inference_encoder
    cfg = TurnSearchConfig(n_alt=2, rounds=1, fast_rounds=0, reval_salts=2, max_spine=6, turn_full_prob=1.0)
    player = TurnCommitPolicy(policy, MCTSConfig(), turn_config=cfg, rng_seed=0, memory_slots=SLOTS)
    sim = fresh_scenario_sim(seed=SEED, max_turns=10)
    held = {1: None, 2: None}
    for _ in range(4):
        side = sim.gs.global_info.current_side
        snapshot = snapshot_view(sim.gs)
        action = player.select_action(snapshot, game_label="g", sim=sim)
        held[side] = _decide(model, encoder, snapshot, held[side]).memory
        torch.testing.assert_close(player._memories[("g", side)], held[side])
        sim.step(action)
    player.finalize_game("g", 0)
    assert not player._memories
    shipped = list(player._queue)
    assert len(shipped) == 4 and [e.side_step for e in shipped] == list(range(4))
    assert all(e.raw is not None and e.memory_k == SLOTS for e in shipped)
