"""The turn-gap measurement with a memory model (tools/turn_gap.py,
tools/corpus_memory.py): the players at a position start from the memory
each side held there in its corpus game, which is what the network writes
reading the trainer's own positions, and a playout continues each side's
memory from the candidate turn."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.synthetic_replay import move, turn, two_sides, write_replay  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.memory import MemoryState  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
SLOTS = 8


def _game(tmp_path):
    from tools.replay_extract import extract_replay
    commands = [*turn(1, move(1, [(3, 5), (3, 6)])), *turn(2, move(2, [(11, 4), (11, 5)])),
                *turn(1), *turn(2), *turn(1)]
    return extract_replay(write_replay(tmp_path / "g.bz2", two_sides(extra1=[("Cavalryman", 3, 5, False)]),
                                       commands))


def _parity_base():
    from tools.preencode_sequences import fresh_vocab
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    type_to_id, faction_to_id = fresh_vocab()
    torch.manual_seed(0)
    encoder = GameStateEncoder(d_model=32, unit_type_to_id=type_to_id, faction_to_id=faction_to_id,
                               relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True,
                               relevant_set_version=2).eval()
    encoder.freeze_vocab()
    model = WesnothModel(d_model=32, num_layers=1, num_heads=2, d_ff=64, observation_parity=True,
                         memory_slots=SLOTS).eval()
    return SimpleNamespace(_inference_model=model, _inference_encoder=encoder,
                           _lock=threading.Lock(), _decision_step=0), type_to_id, faction_to_id


@needs_core
def test_the_memory_at_a_point_of_a_game_is_what_the_trainer_wrote(tmp_path):
    """Side 2's turn 2 begins: side 1 has read its turns 1 and 2, side 2
    its turn 1, each through its own memory. Reading the trainer's
    pre-encoded positions of the same game gives the same states."""
    from tools.corpus_memory import memories_before
    from tools.preencode_sequences import encode_game_sequence
    from tools.raw_player import RawPolicyPlayer
    data = _game(tmp_path)
    base, type_to_id, faction_to_id = _parity_base()
    got = memories_before(data, RawPolicyPlayer(base, 0.0, memory_slots=SLOTS), turn=2, side=2)
    seq = encode_game_sequence(data, "g", 1, type_to_id, faction_to_id)
    want, read = {}, {}
    with torch.no_grad():
        for side, positions in seq.sides.items():
            state = None
            read[side] = 0
            for p in positions:
                if (p.turn, side) >= (2, 2):
                    break
                state = base._inference_model(base._inference_encoder.encode_from_raw(p.raw),
                                              memory=MemoryState(SLOTS, state)).memory
                read[side] += 1
            want[side] = state
    assert (read[1], read[2]) == (3, 2), "side 1 read its turns 1 and 2, side 2 its turn 1"
    for side in (1, 2):
        torch.testing.assert_close(got[side], want[side])


class _SpyModel:
    """Ends every turn; the memory it writes names the call, and every
    call is kept with the side to move and the memory it read."""

    def __init__(self):
        self.calls = []

    def __call__(self, encoded, memory=None):
        self.calls.append((int(encoded.global_info.current_side), memory))
        state = torch.full((memory.k, 4), float(len(self.calls)))
        return SimpleNamespace(legal_compact=SimpleNamespace(prior=[], kind=[], actor=[]),
                               memory=state, value=torch.tensor(0.0))


def test_a_candidate_turn_starts_from_the_boundary_memories_and_its_playouts_carry_them_on():
    from sim_test_helpers import fresh_scenario_sim
    from tools import turn_gap as tg
    spy = _SpyModel()
    base = SimpleNamespace(_inference_model=spy, _inference_encoder=SimpleNamespace(encode=lambda gs: gs),
                           _lock=threading.Lock(), _decision_step=0)
    cfg = tg.GapConfig(k_alternatives=0, playouts=1, cap_turns=2, memory=3)
    sim0 = fresh_scenario_sim(seed=5, max_turns=10)
    position = tg.BoundaryPosition(index=0, gs=sim0.gs, scenario_id=sim0.scenario_id)
    boundary = {1: torch.full((3, 4), 0.5), 2: torch.full((3, 4), 0.25)}
    player = tg.player_for(base, cfg, 0.0)
    candidate, sim = tg._candidate_turn(position, player, 10, "salt", None, "L", boundary)
    side, read = spy.calls[0]
    assert side == 1 and read.k == 3 and torch.equal(read.state, boundary[1])
    written = sim.player_memories[1]
    assert torch.equal(written, player.memory_of("L", 1)) and not torch.equal(written, boundary[1])
    assert torch.equal(sim.player_memories[2], boundary[2]), "the mover's turn leaves the other side's memory"
    before = len(spy.calls)
    tg._play_next(candidate, sim, position, 1, 10, cfg, 0, base, "L")
    first = {}
    for side, read in spy.calls[before:]:
        first.setdefault(side, read.state)
    assert torch.equal(first[2], boundary[2]) and torch.equal(first[1], written)
