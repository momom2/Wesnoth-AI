"""A player of a model with a memory carries each side's state from one of
its decisions to the next (docs/parity_memory_design_20260929.md "Serving
and play"): the raw player keeps one state per game and side, the shared
inference server's wire carries the parity streams and the state both
ways, a served forward writes what the local forward writes, and the
memory size a player uses is an estimand every result records."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.eval_provenance import effective_memory, memory_refusal  # noqa: E402
from tools.raw_player import RawPolicyPlayer  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.memory import MemoryState  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
D = 32


class _SpyModel:
    """Answers every decision with end_turn and a memory that names the call."""

    def __init__(self):
        self.seen = []

    def __call__(self, encoded, memory=None):
        self.seen.append(memory)
        state = torch.full((memory.k, 4), float(len(self.seen)))
        return SimpleNamespace(legal_compact=SimpleNamespace(prior=[], kind=[], actor=[]), memory=state)


def _state(side):
    return SimpleNamespace(global_info=SimpleNamespace(current_side=side))


def test_a_raw_player_feeds_each_side_its_own_memory_and_forgets_a_finished_game():
    spy = _SpyModel()
    base = SimpleNamespace(_inference_model=spy, _inference_encoder=SimpleNamespace(encode=lambda gs: gs),
                           _lock=threading.Lock(), _decision_step=0)
    player = RawPolicyPlayer(base, 0.0, memory_slots=3)
    for label, side in (("g", 1), ("g", 2), ("g", 1), ("g", 1), ("h", 1)):
        assert player.select_action(_state(side), game_label=label) == {"type": "end_turn"}
    states = [None if m.state is None else float(m.state[0, 0]) for m in spy.seen]
    assert states == [None, None, 1.0, 3.0, None], "side 1 of g reads what its previous decision wrote"
    assert {m.k for m in spy.seen} == {3}
    player.drop_pending("g")
    assert set(player._memories) == {("h", 1)}


def test_a_refused_decision_leaves_the_memory_as_it_was():
    """A bounced recruit is decided again from the memory its side had
    before it: the engine records nothing for it, so training saw no
    position there."""
    spy = _SpyModel()
    base = SimpleNamespace(_inference_model=spy, _inference_encoder=SimpleNamespace(encode=lambda gs: gs),
                           _lock=threading.Lock(), _decision_step=0)
    player = RawPolicyPlayer(base, 0.0, memory_slots=3)
    player.select_action(_state(1), game_label="g")
    player.select_action(_state(1), game_label="g")
    player.drop_last_pending("g")
    player.select_action(_state(1), game_label="g")
    player.select_action(_state(2), game_label="g")
    player.drop_last_pending("g")
    player.select_action(_state(2), game_label="g")
    states = [None if m.state is None else float(m.state[0, 0]) for m in spy.seen]
    assert states == [None, 1.0, 1.0, None, None]


def test_the_memory_size_is_an_estimand():
    assert effective_memory(0, None) is None and effective_memory(0, 0) is None
    assert effective_memory(64, None) == 64 and effective_memory(64, 16) == 16
    for slots, flag in ((0, 8), (64, 65), (64, -1)):
        with pytest.raises(ValueError):
            effective_memory(slots, flag)
    assert memory_refusal("g.json", {"memory_a": 64, "memory_b": None}, (64, None)) is None
    assert "memories" in memory_refusal("g.json", {}, (64, None)), "a file before the field had none"


def _parity_pair():
    from helpers.parity_games import vocab_of
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(0)
    encoder = GameStateEncoder(d_model=D, unit_type_to_id=vocab_of(["Lieutenant", "Spearman"]),
                               relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True,
                               relevant_set_version=2).eval()
    encoder.freeze_vocab()
    model = WesnothModel(d_model=D, num_layers=1, num_heads=2, d_ff=64, observation_parity=True,
                         memory_slots=8).eval()
    return encoder, model


def _position():
    from helpers.parity_games import core_of, record
    cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 6, 1, False),
                         ("Lieutenant", 2, 18, 3, True), ("Spearman", 2, 9, 2, False)], fog=True))
    cs.apply_command(["init_side", 1])
    gs = cs.to_state()
    gc.bind_view(gs, cs.fork())
    return gs


@needs_core
def test_a_served_forward_writes_the_memory_the_local_forward_writes():
    from tools.inference_seam import (InferenceServer, RemoteEncoder, RemoteModel, output_from_wire,
                                      output_to_wire)
    from wesnoth_ai.leaf_wire import pack_request, unpack_request
    encoder, model = _parity_pair()
    gs = _position()
    with torch.no_grad():
        local = model(encoder.encode(gs), memory=MemoryState(4, None))
    remote_encoder = RemoteEncoder(encoder.unit_type_to_id, encoder.faction_to_id, relevant_set=True,
                                   server_priors=True, fog_hides_enemy_villages=True,
                                   terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    enc = remote_encoder.encode(gs)
    served = RemoteModel(InferenceServer(model, encoder, device=torch.device("cpu")))(
        enc, memory=MemoryState(4, None))
    assert served.memory.shape == (4, D) and served.legal_compact is not None
    assert torch.allclose(served.memory, local.memory, atol=1e-5)
    assert not torch.allclose(served.memory, model.initial_memory(4)), "the decision wrote to it"
    # The next decision reads the state the player sends, locally and served.
    with torch.no_grad():
        local_next = model(encoder.encode(gs), memory=MemoryState(4, local.memory))
    served_next = RemoteModel(InferenceServer(model, encoder, device=torch.device("cpu")))(
        remote_encoder.encode(gs), memory=MemoryState(4, served.memory))
    assert torch.allclose(served_next.memory, local_next.memory, atol=1e-5)
    assert not torch.allclose(served_next.memory, served.memory, atol=1e-5)
    # The wire both ways: the parity streams and the state out, the state back.
    raw, masks = enc._raw, enc._masks
    [(back, _, sent)] = unpack_request(pack_request([(raw, masks, MemoryState(4, local.memory))]))
    for name in ("their_faction_probs", "sight_type_ids", "sight_xs", "sight_feats", "unit_feats"):
        assert np.array_equal(getattr(back, name), getattr(raw, name)), name
    assert sent.k == 4 and np.array_equal(sent.state, local.memory.numpy())
    [(_, _, first)] = unpack_request(pack_request([(raw, masks, MemoryState(4, None))]))
    assert first.k == 4 and first.state is None
    assert torch.equal(output_from_wire(output_to_wire(served)).memory, served.memory)


@needs_core
@pytest.mark.slow
def test_an_eval_game_between_memory_players_records_their_sizes(tmp_path):
    """A real game, per process: the same checkpoint at 8 slots and at 0."""
    from tools.elo_eval_game import main
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    spec = tmp_path / "memory.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=D, num_layers=1, num_heads=2, d_ff=64,
                      relevant_set_hexes=True, observation_parity=True, memory_slots=8,
                      relevant_set_version=2).save_checkpoint(spec)
    out = tmp_path / "games"
    assert main(["x", "A", str(spec), "B", str(spec), "1", "7", str(out), "--mcts-sims", "0",
                 "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--memory-a", "8",
                 "--memory-b", "0", "--max-turns", "2", "--device", "cpu"]) == 0
    rec = json.loads((out / "game_A_B_s1_7.json").read_text(encoding="utf-8"))
    assert (rec["memory_a"], rec["memory_b"]) == (8, 0)
    from tools.game_record import read_records
    game = next(read_records(out / "game_A_B_s1_7.game.jsonl.gz"))
    assert (game["players"]["a"]["memory"], game["players"]["b"]["memory"]) == (8, 0)
    with pytest.raises(SystemExit, match="memor"):
        main(["x", "A", str(spec), "B", str(spec), "1", "7", str(out), "--mcts-sims", "0",
              "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--memory-a", "4",
              "--memory-b", "0", "--max-turns", "2", "--device", "cpu"])
