"""The sequence pre-encoding (tools/preencode_sequences.py) on a replay
written from scratch: every decision of both player sides, in order, each
label re-indexed into its encoding's own hex tokens and checked, the
observation that carries the true state left out, and a turn that ran out
of time kept as a position that names no action."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.synthetic_replay import countdown_update, move, turn, two_sides, write_replay  # noqa: E402
from tools.encode_worker import LabelSlotMismatch, label_in_raw_basis, label_slot_mismatch  # noqa: E402
from tools.replay_dataset import ActionIndices  # noqa: E402
from tools.replay_extract import extract_replay  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.classes import Position  # noqa: E402

TIMER = {"mp_countdown": "yes", "mp_countdown_init_time": 240, "mp_countdown_turn_bonus": 240,
         "mp_countdown_action_bonus": 0, "mp_countdown_reservoir_time": 360}
RIDE = [(3, 5), (3, 6), (3, 7)]               # WML coordinates, as a replay carries them


@pytest.fixture(scope="module")
def vocab():
    from tools.preencode_sequences import fresh_vocab
    return fresh_vocab()


def _record(tmp_path):
    commands = [*turn(1, move(1, RIDE), countdown_update(1, 240_000)),     # side 1 runs out of time
                *turn(2, move(2, [(11, 4), (11, 5)])),
                *turn(1), *turn(2)]
    return extract_replay(write_replay(tmp_path / "g.bz2",
                                       two_sides(extra1=[("Cavalryman", 3, 5, False)]), commands,
                                       multiplayer=TIMER))


needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


@needs_core
def test_every_decision_of_both_sides_in_order(tmp_path, vocab):
    from tools.preencode_sequences import encode_game_sequence, read_record, record_path, write_record
    seq = encode_game_sequence(_record(tmp_path), "g", 1, *vocab)
    assert [p.label.action_type for p in seq.sides[1]] == ["move", "timeout", "end_turn"]
    assert [p.label.action_type for p in seq.sides[2]] == ["move", "end_turn", "end_turn"]
    assert seq.counts["timeout_positions"] == 1 and seq.counts["positions"] == 6
    for side, positions in seq.sides.items():
        for p in positions:
            assert p.raw.observation is None, "the true state never travels with an input"
            assert p.no_visible_unit.shape == (len(p.raw.hex_positions),)
            if p.label.action_type != "timeout":
                assert label_slot_mismatch(p.raw, p.label) is None
    ride = seq.sides[1][0]
    target = ride.raw.hex_positions[ride.label.target_idx]
    assert (target.x, target.y) == (2, 6), "the move's label points at its destination's token"
    path = record_path(tmp_path, "g")
    write_record(path, seq)
    back = read_record(path)
    assert [p.label for p in back.sides[2]] == [p.label for p in seq.sides[2]]


@needs_core
def test_a_label_left_in_the_full_board_basis_is_refused(tmp_path, vocab, monkeypatch):
    """Without the re-indexing, a move's full-board target index points at
    another token of the relevant set, and the game is refused."""
    import tools.encode_worker as ew
    from tools.preencode_sequences import encode_game_sequence
    monkeypatch.setattr(ew, "label_in_raw_basis", lambda ai, raw: ai)
    with pytest.raises(LabelSlotMismatch):
        encode_game_sequence(_record(tmp_path), "g", 1, *vocab)


def test_a_label_follows_the_tokens_of_its_encoding():
    raw = SimpleNamespace(hex_positions=[Position(4, 1), Position(2, 6), Position(5, 5)])
    ai = ActionIndices(action_type="move", actor_idx=0, target_idx=74, source_hex=(2, 4), target_hex=(2, 6))
    assert label_in_raw_basis(ai, raw).target_idx == 1
    gone = ActionIndices(action_type="move", actor_idx=0, target_idx=80, source_hex=(2, 4), target_hex=(7, 7))
    moved = label_in_raw_basis(gone, raw)
    assert moved.target_idx is None and moved.target_off_subset
    end = ActionIndices(action_type="end_turn", actor_idx=3)
    assert label_in_raw_basis(end, raw) is end
