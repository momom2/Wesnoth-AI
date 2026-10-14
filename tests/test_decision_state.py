"""A player that only reads the state while it decides (the raw player)
decides on the simulator's own view instead of a snapshot of it
(`eval_players._decision_state`): the same game, command for command."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.game_core import game_core_class  # noqa: E402

needs_core = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core wheel absent or stale")


def _player(server, encoder, seed: int):
    from tools.inference_seam import RemoteEncoder, RemoteModel
    from tools.raw_player import RawPolicyPlayer
    renc = RemoteEncoder(encoder.unit_type_to_id, encoder.faction_to_id, relevant_set=True,
                         server_priors=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                         observation_parity=True, relevant_set_version=2)
    base = SimpleNamespace(_inference_model=RemoteModel(server), _inference_encoder=renc,
                           _lock=threading.Lock(), _decision_step=0)
    return RawPolicyPlayer(base, 0.0, seed=seed, end_turn_offset=-6.0, memory_slots=8)


def _game(server, encoder):
    import random
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    setup = random_setup(random.Random(4))
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=4)
    sim._seed_salt = "elo:4"
    _play_one_eval_game(sim, _PolicyPair(_player(server, encoder, 8), "a", 1),
                        _PolicyPair(_player(server, encoder, 9), "b", 2), game_label="g")
    return [list(rc.cmd) for rc in sim.command_history]


@needs_core
def test_the_raw_player_plays_the_same_game_on_the_simulators_view(monkeypatch):
    from tools import eval_players
    from tools.inference_seam import InferenceServer
    from tools.raw_player import RawPolicyPlayer
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(6)
    policy = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=torch.device("cpu"),
                               relevant_set_hexes=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                               observation_parity=True, memory_slots=8, relevant_set_version=2)
    server = InferenceServer(policy._inference_model, policy._inference_encoder)
    snapshots = []
    real = eval_players.snapshot_view
    monkeypatch.setattr(eval_players, "snapshot_view", lambda gs: snapshots.append(1) or real(gs))
    with torch.no_grad():
        live = _game(server, policy._inference_encoder)
        assert not snapshots, "the raw player decided on snapshots"
        monkeypatch.setattr(RawPolicyPlayer, "reads_state_only", False)
        snapped = _game(server, policy._inference_encoder)
    assert snapshots, "the second game decided on snapshots"
    assert len(live) > 30 and live == snapped
