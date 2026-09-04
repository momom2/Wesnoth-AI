"""Joint-temperature raw player (tools/raw_player.py): the raw-argmax
control for the search-vs-raw question (review 2026-09-04)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def test_pick_index_temperatures():
    from tools.raw_player import pick_index
    priors = np.array([0.2, 0.5, 0.3])
    rng = np.random.default_rng(0)
    assert pick_index(priors, 0.0, rng) == 1
    # Temperature 1 reproduces the prior; 0.1 concentrates on the mode.
    draws = np.array([pick_index(priors, 1.0, rng) for _ in range(4000)])
    freq = np.bincount(draws, minlength=3) / len(draws)
    assert np.allclose(freq, priors, atol=0.04)
    cold = [pick_index(priors, 0.1, rng) for _ in range(200)]
    assert cold.count(1) >= 195


def test_procedure_tag_carries_temperature():
    from tools.eval_procedure import procedure_of
    assert procedure_of(0, False, False) == "raw"
    assert procedure_of(0, False, False, 0.0) == "raw:t0"
    assert procedure_of(0, False, False, 0.5) == "raw:t0.5"
    assert procedure_of(32, False, True, None) == "mcts:32"


def _tiny_policy():
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def test_argmax_player_picks_max_joint_prior():
    import torch
    from tools.raw_player import RawPolicyPlayer
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    policy = _tiny_policy()
    gs = build_scenario_gamestate(random_setup(random.Random(3)))
    with torch.no_grad():
        enc = policy._inference_encoder.encode(gs)
        out = policy._inference_model(enc)
        legal = enumerate_legal_actions_with_priors(
            enc, out, gs, decision_step=policy._decision_step)
    assert len(legal) > 1
    expected = max(legal, key=lambda la: la.prior).action
    player = RawPolicyPlayer(policy, 0.0)
    assert player.select_action(gs, game_label="t", sim=None) == expected
    sampled = RawPolicyPlayer(policy, 1.0, seed=1).select_action(
        gs, game_label="t", sim=None)
    assert sampled in [la.action for la in legal]


def test_argmax_player_drives_the_eval_loop():
    """The wrapper survives the real eval loop (recruit-bounce retry,
    drop_pending at game end) and the forward counter sees its
    forwards."""
    from tools.elo_eval_game import _CountingModel
    from tools.eval_sim import _PolicyPair, _play_one_eval_game
    from tools.raw_player import RawPolicyPlayer
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    policy = _tiny_policy()
    counter = _CountingModel(policy._inference_model)
    policy._inference_model = counter
    setup = random_setup(random.Random(5))
    sim = WesnothSim(build_scenario_gamestate(setup),
                     scenario_id=setup.scenario_id, max_turns=2)
    r = _play_one_eval_game(
        sim,
        _PolicyPair(policy=RawPolicyPlayer(policy, 0.0), label="argmax",
                    side=1),
        _PolicyPair(policy=policy, label="sampler", side=2),
        game_label="g")
    assert sim.done and r.our_actions > 0
    assert counter.n_forwards >= r.our_actions


def test_eval_game_refuses_temperature_with_search(tmp_path):
    from tools.elo_eval_game import main
    with pytest.raises(SystemExit, match="raw player only"):
        main(["x", "A", "dummy", "B", "dummy", "1", "7", str(tmp_path),
              "--mcts-sims", "32", "--raw-temperature-a", "0",
              "--no-turn-search", "--device", "cpu"])


def test_eval_game_guard_separates_temperature_estimands(tmp_path):
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    base = ["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
            "--mcts-sims", "0", "--device", "cpu"]
    (out / "game_A_B_s1_7.json").write_text(
        json.dumps({"procedure_a": "raw", "procedure_b": "raw",
                    "max_turns": 200}), encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to mix"):
        main(base + ["--raw-temperature-a", "0"])
    (out / "game_A_B_s1_7.json").write_text(
        json.dumps({"procedure_a": "raw:t0", "procedure_b": "raw",
                    "max_turns": 200}), encoding="utf-8")
    assert main(base + ["--raw-temperature-a", "0"]) == 0


def test_batch_driver_forwards_temperature():
    repo = Path(__file__).parent.parent
    src = (repo / "tools/run_elo_batch.py").read_text(encoding="utf-8")
    assert "--raw-temperature-a" in src and "--raw-temperature-b" in src
    assert "args.raw_temperature_a" in src


def test_search_root_procedure_is_per_player():
    """Plan 1.6: the eval harness records and applies the root
    procedure per player ('mcts:<sims>' Gumbel root, 'puct:<sims>'
    plain PUCT, what tools/az_loop.py trains with)."""
    import torch
    from tools import elo_eval_game as g
    from tools.eval_procedure import procedure_of
    assert procedure_of(32, False, True) == "mcts:32"
    assert procedure_of(32, False, True, gumbel_root=False) == "puct:32"
    assert procedure_of(32, False, False, gumbel_root=False) == "tcs:32"
    cpu = torch.device("cpu")
    puct, _ = g._build_player("random", "A", 2, cpu, turn_search=False, gumbel_root=False)
    gumbel, _ = g._build_player("random", "B", 2, cpu, turn_search=False)
    assert puct._mcts_config.gumbel_root is False and gumbel._mcts_config.gumbel_root is True
