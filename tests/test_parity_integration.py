"""The parity recipe end to end on a real position: the Rust core encodes
it with the parity observation and the relevant set's version 2, and the
memory model reads it and carries a side's memory to its next decision."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from wesnoth_ai import game_core as gc  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")

D = 32


def test_a_core_encoding_runs_through_the_memory_model():
    from helpers.parity_games import core_of, parity_raw, record, vocab_of
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    names = ["Lieutenant", "Spearman"]
    cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 6, 1, False),
                         ("Lieutenant", 2, 18, 3, True)], fog=True))
    cs.apply_command(["init_side", 1])
    raw = parity_raw(cs, vocab_of(names))
    encoder = GameStateEncoder(d_model=D, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True,
                               relevant_set_version=2).eval()
    model = WesnothModel(d_model=D, num_layers=2, num_heads=2, d_ff=64,
                         observation_parity=True, memory_slots=64).eval()
    state = model.initial_memory(16)
    with torch.no_grad():
        first = model.forward_embedded(encoder.encode_from_raw_embedded([raw]), memory=[state])
        second = model.forward_embedded(encoder.encode_from_raw_embedded([raw]), memory=first.memory)
    n_units, n_recruits, n_hexes = first.sizes[0]
    assert n_hexes == len(raw.hex_positions) and first.belief_logits.shape[1] >= n_hexes
    assert first.actor_logits.shape[1] >= n_units + n_recruits + 1
    assert first.memory[0].shape == (16, D) and first.memory[0].dtype == torch.float32
    assert not torch.equal(first.memory[0], state)          # the decision wrote to the memory
    assert not torch.equal(second.memory[0], first.memory[0])
