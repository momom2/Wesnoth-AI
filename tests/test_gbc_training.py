"""GBC production-integration tests: labels in finalize_game, the
trainer's event-supervision loss, and checkpoint stickiness --
driving the real policy/trainer/model paths end to end.
"""
from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.gbc import labels_for_game_states  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from wesnoth_ai.visibility import units_visible_to  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.turn_policy import TurnCommitPolicy  # noqa: E402
from tools.turn_search import TurnSearchConfig  # noqa: E402


def _tiny_policy(gbc: bool) -> TransformerPolicy:
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64,
                             gbc=gbc)


def test_labels_for_game_states_synthetic_kill():
    """A visible enemy death between two stored states labels 1 for
    the mover at every horizon; the same rows carry 0 when the event
    never happens."""
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    s0 = copy.deepcopy(sim.gs)
    victim = next((u for u in units_visible_to(s0, 1) if u.side == 2),
                  None)
    if victim is None:
        # Plant a visible enemy next to side 1's leader.
        leader1 = next(u for u in s0.map.units if u.side == 1)
        donor = next(u for u in s0.map.units if u.side == 2)
        victim = dataclasses.replace(
            donor, id="gbc_train_victim",
            is_leader=False,
            position=dataclasses.replace(
                leader1.position, x=leader1.position.x + 1))
        s0.map.units.add(victim)
    s1 = copy.deepcopy(s0)
    dead = next(u for u in s1.map.units if u.id == victim.id)
    s1.map.units.discard(dead)

    rows_per_state = labels_for_game_states([s0], [1], final_gs=s1)
    assert len(rows_per_state) == 1
    rows = rows_per_state[0]
    assert rows is not None
    dies = [r for r in rows if r[0] == "u" and r[1] == victim.id]
    assert dies, "the visible enemy must be in the goal roster"
    assert dies[0][3] == 1 and dies[0][4] == 1, \
        "observed death labels 1 at k=1 and k=2"
    # A different visible unit that did NOT die labels 0.
    others = [r for r in rows if r[0] == "u" and r[1] != victim.id]
    assert others and all(r[3] == 0 for r in others)


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, 4])
def test_full_game_trains_with_gbc_loss(batch_size):
    """End to end through the production pipeline: TCS plays a mini
    game with gbc labeling on, finalize attaches labels, train_step
    reports a positive gbc_loss and MOVES the gbc head params.

    Parametrized over trainer batch size because the two sizes take
    DIFFERENT forward paths (single vs forward_batch) -- the 2026-08-15
    leg-2 incident: the batched path's missing ctx tap made GBC a
    silent no-op on CUDA while every CPU smoke passed at B=1."""
    from wesnoth_ai.trainer import TrainerConfig
    sim = fresh_scenario_sim(seed=5, max_turns=4, mini=True)
    torch.manual_seed(0)
    base = TransformerPolicy(
        device=torch.device("cpu"), d_model=32, num_layers=1,
        num_heads=4, d_ff=64, gbc=True,
        trainer_config=TrainerConfig(train_batch_size=batch_size))
    policy = TurnCommitPolicy(
        base, MCTSConfig(), gbc_labels=True,
        turn_config=TurnSearchConfig(n_alt=2, rounds=1, fast_rounds=0,
                                     reval_salts=2, max_spine=6,
                                     turn_full_prob=1.0))
    guard = 0
    while not sim.done and guard < 400:
        pre = copy.deepcopy(sim.gs)
        action = policy.select_action(pre, game_label="g", sim=sim)
        sim.step(action)
        guard += 1
    assert sim.done
    policy.finalize_game("g", sim.winner, final_gs=sim.gs)
    labeled = [e for e in policy._queue if e.gbc_labels]
    assert labeled, "finalize_game must attach gbc labels"
    w0 = base._model.gbc_heads.pred_embed.weight.detach().clone()
    stats = policy.train_step()
    assert stats.gbc_loss > 0.0, "gbc loss must be reported"
    w1 = base._model.gbc_heads.pred_embed.weight.detach()
    assert not torch.equal(w0, w1), "gbc head params must move"


def test_checkpoint_stickiness(tmp_path):
    """gbc heads round-trip through save/load; a gbc-on model
    resuming a pre-gbc checkpoint grafts fresh heads (whitelisted
    partial load); a gbc-off model loading a gbc checkpoint merely
    warns (unexpected keys dropped)."""
    p_gbc = _tiny_policy(gbc=True)
    ck = tmp_path / "gbc.pt"
    p_gbc.save_checkpoint(ck)

    fresh = _tiny_policy(gbc=True)
    fresh.load_checkpoint(ck)
    assert torch.equal(
        fresh._model.gbc_heads.pred_embed.weight,
        p_gbc._model.gbc_heads.pred_embed.weight)

    p_off = _tiny_policy(gbc=False)
    ck_off = tmp_path / "plain.pt"
    p_off.save_checkpoint(ck_off)
    grafted = _tiny_policy(gbc=True)
    grafted.load_checkpoint(ck_off)   # must not raise
    assert grafted._model.gbc_heads is not None

    dropper = _tiny_policy(gbc=False)
    dropper.load_checkpoint(ck)       # must not raise
    assert dropper._model.gbc_heads is None
