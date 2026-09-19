"""The root's legality masks ride from the actor's encoder to the
training target: a PackedMasks the encoder attached to the encoding
(RemoteEncoder under server priors) is kept on the expanded MCTSNode,
carried through MCTSPolicy's pending record, and shipped as
MCTSExperience.masks -- equal, field for field, to pack_masks on the
recorded state -- so step_mcts stages it without rebuilding. An
encoder that packs nothing (the in-process GameStateEncoder) yields
experiences with masks None. Drives the real _expand / _populate_leaf,
select_action / finalize_game and step_mcts (no mirroring).
"""
from __future__ import annotations

import copy
import random as _random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import tools.mcts as mcts  # noqa: E402
from tools.mcts import MCTSConfig, MCTSNode, _expand, _populate_leaf  # noqa: E402
from wesnoth_ai.action_sampler import LegalActionPrior  # noqa: E402
from wesnoth_ai.server_priors import PackedMasks  # noqa: E402


def assert_same_pack(got, want) -> None:
    assert isinstance(got, PackedMasks)
    for f in PackedMasks.__dataclass_fields__:
        a, b = getattr(got, f), getattr(want, f)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert a is not None and b is not None, f
            assert a.dtype == b.dtype and np.array_equal(a, b), f
        else:
            assert a == b, f


def _packed_on_the_state(policy, game_state) -> PackedMasks:
    """pack_masks as the actor runs it: on a light EncodedState of the
    state's RawEncoded, the policy's vocabulary."""
    from tools.inference_seam import build_light_encoded
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.server_priors import pack_masks
    enc = policy._inference_encoder
    raw = encode_raw(game_state, type_to_id=enc.unit_type_to_id,
                     faction_to_id=enc.faction_to_id,
                     relevant_set=enc.relevant_set_hexes)
    return pack_masks(build_light_encoded(raw, torch.device("cpu")), game_state)


# ---------------------------------------------------------------------
# Node level: the pack survives expansion, on the encoding's basis
# ---------------------------------------------------------------------

def _node(side: int = 1) -> MCTSNode:
    gs = SimpleNamespace(global_info=SimpleNamespace(current_side=side))
    return MCTSNode(SimpleNamespace(gs=gs, done=False, winner=0))


class _Model:
    def __call__(self, encoded):
        return SimpleNamespace(value=torch.tensor([[0.2]]),
                               cliffness=torch.tensor([[0.1]]),
                               moves_left=None, aux_score=None)


def _encoding(masks, U: int = 2, R: int = 3, H: int = 5):
    enc = SimpleNamespace(unit_tokens=torch.zeros(1, U, 1),
                          recruit_tokens=torch.zeros(1, R, 1),
                          hex_tokens=torch.zeros(1, H, 1))
    if masks is not None:
        enc._masks = masks
    return enc


def _pack(U: int = 2, R: int = 3, H: int = 5) -> PackedMasks:
    A, HB = U + R + 1, (H + 7) // 8
    return PackedMasks(actor_mask=np.ones(A, np.uint8), type_valid=np.ones((A, 2), np.uint8),
                       attack_valid=np.zeros((A, HB), np.uint8),
                       move_valid=np.zeros((A, HB), np.uint8),
                       union_valid=np.zeros((A, HB), np.uint8),
                       n_attacks=np.zeros(A, np.int8), end_turn_bias=0.0,
                       n_units=U, n_recruits=R, n_hexes=H)


def _one_end_turn(*a, **k):
    return [LegalActionPrior(action={"type": "end_turn"}, prior=1.0,
                             actor_idx=0, target_idx=None,
                             weapon_idx=None, type_idx=None)]


@pytest.fixture
def stubbed_enumeration(monkeypatch):
    monkeypatch.setattr(mcts, "enumerate_legal_actions_with_priors", _one_end_turn)
    # The CUDA path's dataclass replace drops `_masks`: the pack must be
    # read before it.
    monkeypatch.setattr(mcts, "_leaf_to_cpu",
                        lambda e, o: (SimpleNamespace(**{k: v for k, v in vars(e).items()
                                                         if k != "_masks"}), o))


def test_expand_and_populate_keep_the_encoders_pack(stubbed_enumeration):
    pack = _pack()
    encoder = SimpleNamespace(encode=lambda gs: _encoding(pack))
    root = _node()
    _expand(root, _Model(), encoder, tiebreak=None)
    assert root.masks is pack and root.expanded

    leaf = _node()
    _populate_leaf(leaf, _encoding(pack), _Model()(None))
    assert leaf.masks is pack and leaf.expanded

    bare = _node()
    _expand(bare, _Model(), SimpleNamespace(encode=lambda gs: _encoding(None)),
            tiebreak=None)
    assert bare.masks is None and bare.expanded


def test_pack_on_another_basis_than_the_encoding_is_refused(stubbed_enumeration):
    encoder = SimpleNamespace(encode=lambda gs: _encoding(_pack(H=6)))
    with pytest.raises(ValueError, match="attached masks"):
        _expand(_node(), _Model(), encoder, tiebreak=None)


# ---------------------------------------------------------------------
# Policy level: real sim, real search, real trainer
# ---------------------------------------------------------------------

def _sim(seed: int):
    from sim_test_helpers import require_scenario_data
    from tools.scenario_pool import (
        build_scenario_gamestate, load_factions, random_setup,
    )
    from tools.wesnoth_sim import WesnothSim
    require_scenario_data()
    load_factions()
    setup = random_setup(_random.Random(seed), forced_faction=None, mini_maps=True)
    return WesnothSim(build_scenario_gamestate(setup),
                      scenario_id=setup.scenario_id, max_turns=6)


def _learner():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(5)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                             device=torch.device("cpu"))


def _seam_base(policy):
    """The actor-worker topology in one process: a RemoteEncoder that
    packs masks at encode time over a RemoteModel on an in-process
    InferenceServer (tools/actor_worker.py builds the same namespace)."""
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    enc, model = policy._inference_encoder, policy._inference_model
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, terrain_multi_hot=enc.terrain_multi_hot,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages,
                         relevant_set=enc.relevant_set_hexes, server_priors=True)
    return SimpleNamespace(_inference_model=RemoteModel(InferenceServer(model, enc)),
                           _inference_encoder=renc, _lock=threading.Lock(),
                           _decision_step=0)


def _play_and_seal(base, sim, n_decisions: int):
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    cfg = MCTSConfig(n_simulations=2, batch_size=1, add_root_noise=False)
    pol = MCTSPolicy(base, cfg, replay_config=ReplayConfig(enabled=False))
    for _ in range(n_decisions):
        if sim.done:
            break
        action = pol.select_action(copy.deepcopy(sim.gs), game_label="g", sim=sim)
        sim.step(action)
    pol.finalize_game("g", winner=1, final_gs=sim.gs)
    assert pol._queue, "expected queued experiences"
    return pol._queue


def test_in_process_encoder_ships_no_masks():
    policy = _learner()
    exps = _play_and_seal(policy, _sim(seed=3), n_decisions=3)
    assert all(e.masks is None for e in exps)


def test_server_priors_pack_reaches_the_experience_and_the_trainer(monkeypatch):
    from tools.bench_train_step import configure_trainer_like_az_loop
    from wesnoth_ai import trainer as trainer_module
    policy = _learner()
    exps = _play_and_seal(_seam_base(policy), _sim(seed=3), n_decisions=3)
    for e in exps:
        assert_same_pack(e.masks, _packed_on_the_state(policy, e.game_state))

    def no_build(*a, **k):
        raise AssertionError("step_mcts rebuilt masks an experience already carried")
    monkeypatch.setattr(trainer_module, "_host_packed_masks", no_build)
    configure_trainer_like_az_loop(policy._trainer)
    policy._trainer.config.train_batch_size = 2
    stats = policy._trainer.step_mcts(exps)
    assert np.isfinite(float(stats.policy_loss))
