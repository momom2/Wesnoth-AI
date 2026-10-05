"""The memory each side holds at a point of a corpus game.

A side's memory there is what its network wrote reading every decision
position of the side before that point, in order, from the learned
initial memory: the positions the sequence trainer reads
(`replay_dataset.iter_record_pairs` with timeouts, as
tools/preencode_sequences.py walks a game), so a player started there
holds what the trained network held (docs/parity_memory_design_20260929.md,
"Training" and "Serving and play").
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict

from wesnoth_ai.classes import PLAYER_SIDES


def load_game(path: Path) -> dict:
    """An extracted corpus game (`*.json.gz`)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def memories_before(data: dict, player, turn: int, side: int, *,
                    game_label: str = "history") -> Dict[int, object]:
    """{side: memory state} for each player side after every decision
    position of the game before `side`'s turn `turn` begins (sides play
    in their order within a turn), each read through `player`'s own
    forward (its encoder, model and slots; `RawPolicyPlayer.advance_memory`).
    A side with no decision before that point holds None, its learned
    initial memory."""
    from tools.replay_dataset import iter_record_pairs
    boundary = (int(turn), int(side))
    for gs, _label in iter_record_pairs(data, relevant_set=False, timeouts=True):
        if (int(gs.global_info.turn_number), int(gs.global_info.current_side)) >= boundary:
            break
        player.advance_memory(gs, game_label=game_label)
    out = {s: player.memory_of(game_label, s) for s in PLAYER_SIDES}
    player.drop_pending(game_label)
    return out
