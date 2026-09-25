"""end_turn decided at the actor level, and the end_turn logit offset
(tools/raw_player.py; docs/training_signal_panel_20260905.md test 1):
the two decode rules on synthetic priors, on a real state through the
list path, and their procedure tags and plumbing.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.raw_player import (  # noqa: E402
    RawPolicyPlayer, actor_rule_index, pick_index, shift_end_turn_prior,
)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def test_end_turn_offset_is_an_exact_logit_shift():
    """Four actors, the last is end_turn; unit actors split their mass
    over several actions. Shifting the end_turn actor logit and
    recomputing the joint priors must give what the client-side shift
    gives, for every action."""
    rng = np.random.default_rng(3)
    actor_logits = rng.normal(size=4)
    subs = [rng.dirichlet(np.ones(k)) for k in (3, 2, 4)]     # per unit actor

    def joint(logits):
        pa = _softmax(logits)
        priors, actors, is_end = [], [], []
        for a, sub in enumerate(subs):
            for q in sub:
                priors.append(pa[a] * q)
                actors.append(a)
                is_end.append(False)
        priors.append(pa[3])
        actors.append(3)
        is_end.append(True)
        return np.array(priors), np.array(actors), np.array(is_end)

    priors, actors, is_end = joint(actor_logits)
    for offset in (-1.5, -0.75, 0.4):
        shifted = shift_end_turn_prior(priors, is_end, offset)
        truth, _, _ = joint(actor_logits + np.array([0, 0, 0, offset]))
        assert np.allclose(shifted, truth, atol=1e-12)
        assert shifted.sum() == pytest.approx(1.0)
    assert shift_end_turn_prior(priors, is_end, 0.0) is priors
    no_end = shift_end_turn_prior(priors[:-1], is_end[:-1], -1.0)
    assert np.array_equal(no_end, priors[:-1])


def test_actor_rule_ends_the_turn_only_when_its_mass_leads():
    rng = np.random.default_rng(0)
    # Joint argmax would end the turn (0.3 beats every single action),
    # the actor rule does not: actor 0's marginal is 0.5.
    priors = np.array([0.2, 0.2, 0.1, 0.2, 0.3])
    actors = np.array([0, 0, 0, 1, 2])
    is_end = np.array([False, False, False, False, True])
    assert pick_index(priors, 0.0, rng) == 4
    assert actor_rule_index(priors, actors, is_end, 0.0, rng) == 0
    # With end_turn's mass at the top marginal it ends the turn.
    priors2 = np.array([0.2, 0.2, 0.1, 0.5])
    actors2 = np.array([0, 0, 0, 2])
    is_end2 = np.array([False, False, False, True])
    assert actor_rule_index(priors2, actors2, is_end2, 0.0, rng) == 3
    # Ties go to end_turn (>=), as the pre-registration states it.
    priors3 = np.array([0.25, 0.25, 0.5])
    assert actor_rule_index(priors3, np.array([0, 0, 1]), np.array([False, False, True]),
                            0.0, rng) == 2
    # Without an end_turn action the rule is the plain pick.
    assert actor_rule_index(priors[:-1], actors[:-1], is_end[:-1], 0.0, rng) == 0
    # At a temperature the non-end branch is sampled among non-end actions only.
    draws = {actor_rule_index(priors, actors, is_end, 1.0, rng) for _ in range(200)}
    assert 4 not in draws and draws <= {0, 1, 2, 3}


def test_procedure_tag_carries_the_end_turn_decode():
    from tools.eval_procedure import procedure_of
    assert procedure_of(0, False, False, 0.0) == "raw:t0"
    assert procedure_of(0, False, False, 0.0, raw_end_turn="actor") == "raw:t0+endm"
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-0.75) == "raw:t0+eo-0.75"
    assert procedure_of(0, False, False, 0.0, raw_end_turn="actor",
                        raw_end_turn_offset=-1.5) == "raw:t0+endm+eo-1.5"
    # A searched player carries none of it.
    assert procedure_of(32, False, True, None, raw_end_turn="actor") == "mcts:32"


def test_player_refuses_an_unknown_rule():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="end_turn_rule"):
        RawPolicyPlayer(SimpleNamespace(), 0.0, end_turn_rule="sometimes")


def test_actor_rule_player_matches_the_rule_on_a_real_state():
    import torch
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    policy = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                               device=torch.device("cpu"))
    gs = build_scenario_gamestate(random_setup(random.Random(3)))
    with torch.no_grad():
        enc = policy._inference_encoder.encode(gs)
        out = policy._inference_model(enc)
        legal = enumerate_legal_actions_with_priors(
            enc, out, gs, decision_step=policy._decision_step)
    priors = np.array([la.prior for la in legal])
    actors = np.array([la.actor_idx for la in legal])
    is_end = np.array([la.action.get("type") == "end_turn" for la in legal])
    assert is_end.sum() == 1
    for offset in (0.0, -1.5, 3.0):
        expected = legal[actor_rule_index(
            shift_end_turn_prior(priors, is_end, offset), actors, is_end, 0.0,
            np.random.default_rng(0))].action
        player = RawPolicyPlayer(policy, 0.0, end_turn_rule="actor", end_turn_offset=offset)
        assert player.select_action(gs, game_label="t", sim=None) == expected
    # A large positive offset makes the joint player end the turn; a
    # large negative one makes it act.
    ends = RawPolicyPlayer(policy, 0.0, end_turn_offset=20.0).select_action(gs, game_label="t")
    acts = RawPolicyPlayer(policy, 0.0, end_turn_offset=-20.0).select_action(gs, game_label="t")
    assert ends == {"type": "end_turn"} and acts != {"type": "end_turn"}


def test_actor_rule_on_the_compact_arrays_matches_the_list_path():
    """Behind a shared inference server the player picks on the compact
    legal arrays (server-side priors); the rule and the offset must
    choose what the list path chooses on the same state."""
    import threading
    from types import SimpleNamespace
    import torch
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    policy = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                               device=torch.device("cpu"))
    enc, model = policy._inference_encoder, policy._inference_model
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, terrain_multi_hot=enc.terrain_multi_hot,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages,
                         server_priors=True)
    rmodel = RemoteModel(InferenceServer(model, enc))
    states = [build_scenario_gamestate(random_setup(random.Random(s))) for s in (3, 5, 8)]
    for rule, offset in (("actor", 0.0), ("actor", -1.5), ("joint", -0.75)):
        for gs in states:
            base = SimpleNamespace(_inference_model=rmodel, _inference_encoder=renc,
                                   _lock=threading.Lock(), _decision_step=0)
            compact = RawPolicyPlayer(base, 0.0, end_turn_rule=rule, end_turn_offset=offset)
            listed = RawPolicyPlayer(base, 0.0, end_turn_rule=rule, end_turn_offset=offset,
                                     compact_selection=False)
            assert compact.select_action(gs, game_label="t") == listed.select_action(
                gs, game_label="t"), (rule, offset)


def test_batch_driver_forwards_the_end_turn_decode():
    repo = Path(__file__).parent.parent
    src = (repo / "tools/run_elo_batch.py").read_text(encoding="utf-8")
    assert "--raw-end-turn-a" in src and "--raw-end-turn-offset-b" in src
    assert "raw_end_turn=args.raw_end_turn_a" in src
    game = (repo / "tools/elo_eval_game.py").read_text(encoding="utf-8")
    assert '"raw_end_turn_a": args.raw_end_turn_a' in game


def _batch_argv(tmp_path, *extra):
    return ["x", "--label-a", "A", "--spec-a", "random", "--label-b", "B",
            "--spec-b", "dummy", "--outdir", str(tmp_path / "games"), "--games", "2",
            "--device", "cpu", "--jobs", "1", "--time-budget-min", "0",
            "--min-free-mb", "0", *extra]


def test_the_driver_refuses_an_end_turn_decode_that_would_not_apply(tmp_path):
    """The decode flags are knobs of the raw player at a temperature. A
    searched side, the legacy sampler and the scripted dummy dropped
    them while the result recorded them; the driver refuses up front."""
    from tools.run_elo_batch import main
    for extra in (["--mcts-sims", "8", "--raw-end-turn-offset-a", "-1.5"],
                  ["--mcts-sims", "0", "--raw-end-turn-a", "actor"],
                  ["--mcts-sims", "0", "--raw-temperature-b", "0",
                   "--raw-end-turn-offset-b", "-1.5"]):
        with pytest.raises(SystemExit) as refused:
            main(_batch_argv(tmp_path, *extra))
        assert refused.value.code == 2, extra                    # argparse's refusal
    # Control: the raw player at a temperature takes the decode.
    main(_batch_argv(tmp_path, "--mcts-sims", "0", "--raw-temperature-a", "0",
                     "--raw-end-turn-offset-a", "-1.5"))


def test_a_game_refuses_an_end_turn_decode_that_would_not_apply(tmp_path):
    """Per game too: the legacy sampler (no temperature) on a real
    checkpoint would play without the offset and record it."""
    import torch
    from tools.elo_eval_game import main
    from wesnoth_ai.transformer_policy import TransformerPolicy
    spec = str(tmp_path / "net.pt")
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(spec)
    out = tmp_path / "games"
    with pytest.raises(SystemExit, match="raw-end-turn"):
        main(["x", "A", spec, "B", "dummy", "1", "7", str(out), "--mcts-sims", "0",
              "--raw-end-turn-offset-a", "-1.5", "--max-turns", "1", "--device", "cpu"])
    assert not list(out.glob("game_*.json"))
