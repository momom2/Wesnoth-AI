"""The label the core builds beside the encoding (rust/wesnoth_core/src/
core_label.rs) and the record walks that read it without a view
(tools/preencode_sequences.py, tools/own_trajectories.py).

On every player command of the fixture games, the core's label is the
Python builder's (`replay_dataset._action_indices` on a view of a fork,
re-indexed into the encoding's tokens), in the production basis and on the
full board; and the records the walks write are the ones the walk over views
wrote, byte for byte. The fixtures:
  - tests/fixtures/strict_sync_hamlets_t9.bz2, an engine replay of two AIs:
    moves with and without the engine's order, attacks, recruits, the
    engine's own goto moves;
  - replays written from scratch (helpers/synthetic_replay.py): a move cut
    short whose clicked hex the unit can reach, one whose clicked hex it
    cannot, one that reached its turn's end, a turn that ran out of time, a
    move from a hex with no unit;
  - tests/fixtures/own1_selfplay_silverhead.game.jsonl.gz, a game `own1`
    played against itself at 64 slots (2026-10-12): the simulator's
    approach moves before attacks, attacks, recruits.
The failure these guard against is a label that points at a token other
than the one its command names (the 2026-07-16 class, 19% of pairs).
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core wheel absent or stale")

FIXTURES = Path(__file__).parent / "fixtures"
HAMLETS = FIXTURES / "strict_sync_hamlets_t9.bz2"
OWN = FIXTURES / "own1_selfplay_silverhead.game.jsonl.gz"
TIMER = {"mp_countdown": "yes", "mp_countdown_init_time": 240, "mp_countdown_turn_bonus": 240,
         "mp_countdown_action_bonus": 0, "mp_countdown_reservoir_time": 360}
FULL_BOARD = dict(relevant_set=False, fog_hides_enemy_villages=True, terrain_multi_hot=True)


@pytest.fixture(scope="module")
def vocab():
    from tools.preencode_sequences import fresh_vocab
    return fresh_vocab()


def _synthetic(tmp_path):
    """The replays written from scratch, extracted: name -> record."""
    from helpers.synthetic_replay import countdown_update, move, turn, two_sides, write_replay
    from tools.replay_extract import extract_replay
    ride = [(3, 5), (4, 5), (5, 5), (6, 5)]
    cavalry = two_sides(extra1=[("Cavalryman", 3, 5, False)])
    games = {
        "clicked": (cavalry, [*turn(1, move(1, ride, final=(4, 5), stopped_early="yes")), *turn(2)], None),
        "turn_end": (cavalry, [*turn(1, move(1, ride, final=(5, 5), stopped_early="no")), *turn(2)], None),
        "unreachable": (two_sides(extra1=[("Spearman", 2, 5, False)]),
                        [*turn(1, move(1, [(x, 5) for x in range(2, 12)], final=(4, 5), stopped_early="yes")),
                         *turn(2)], None),
        "timeout": (cavalry, [*turn(1, move(1, ride[:3]), countdown_update(1, 240_000)),
                              *turn(2, move(2, [(11, 4), (11, 5)])), *turn(1), *turn(2)], TIMER),
        "unpaired": (cavalry, [*turn(1, move(1, ride[:2])), *turn(2)], None),
    }
    records = {}
    for name, (sides, commands, timer) in games.items():
        records[name] = extract_replay(write_replay(tmp_path / f"{name}.bz2", sides, commands, multiplayer=timer))
    for c in records["unpaired"]["commands"]:
        if c[0] == "move":
            c[1], c[2] = [7, 8], [1, 1]              # no unit there
    return records


@pytest.fixture(scope="module")
def corpus_games(tmp_path_factory):
    from tools.replay_extract import extract_replay
    games = {"hamlets": extract_replay(HAMLETS)}
    games.update(_synthetic(tmp_path_factory.mktemp("synthetic")))
    return games


@pytest.fixture(scope="module")
def own_record():
    from tools.own_trajectories import read_game_record
    return read_game_record(OWN)


def _own_positions(rec):
    """(core, command) before each player command of an own game record."""
    from tools.game_record import _walk_core
    from tools.replay_dataset import PAIRED_KINDS
    for _k, cs, cmd in _walk_core(rec, verify=True):
        if int(cs.core.current_side) in (1, 2) and cmd and cmd[0] in PAIRED_KINDS:
            yield cs, cmd


def _python_label(cs, cmd, encoding, vocab):
    """The oracle: the Python builder's label on a view of a fork of `cs`,
    re-indexed into the tokens of the view's encoding; None when the command
    names no action."""
    from tools.encode_worker import label_in_raw_basis
    from tools.replay_dataset import _action_indices
    from wesnoth_ai.encoder import encode_raw
    gs = gc.view_of(cs.fork())
    ai = _action_indices(gs, cmd, relevant_set=False)
    if ai is None:
        return None
    return label_in_raw_basis(ai, encode_raw(gs, type_to_id=vocab[0], faction_to_id=vocab[1], **encoding))


@needs_core
@pytest.mark.parametrize("basis", ["production", "full_board"])
def test_the_core_labels_every_player_command_as_the_python_builder(basis, corpus_games, own_record, vocab):
    from tools.preencode_sequences import ENCODING
    from tools.replay_dataset import core_label, core_label_args, iter_record_decisions
    encoding = ENCODING if basis == "production" else FULL_BOARD
    seen: Counter = Counter()
    walks = [(name, iter_record_decisions(data)) for name, data in corpus_games.items()]
    walks.append(("own", _own_positions(own_record)))
    for name, walk in walks:
        for k, (cs, cmd) in enumerate(walk):
            want = _python_label(cs, cmd, encoding, vocab)
            raw, fields = cs.encode_labelled(core_label_args(cmd), type_to_id=vocab[0], faction_to_id=vocab[1],
                                             **encoding)
            got = None if raw is None else core_label(cmd, fields, seen)
            assert got == want, (name, k, cmd[:4])
            seen[got.action_type if got is not None else "unpaired"] += 1
    # Every kind of label occurs, a move cut short reached and not.
    for kind in ("move", "attack", "recruit", "end_turn", "unpaired", "move_label_path_end",
                 "move_label_turn_end", "move_label_clicked", "move_label_clicked_unreachable"):
        assert seen[kind] > 0, (kind, seen)


def _view_walk_sequence(data, file, winner, vocab):
    """`encode_game_sequence` as the walk over views built it (core phase
    34): each position a view of a fork of the record's core, its label the
    Python builder's."""
    from tools.preencode_sequences import PLAYER_SIDES, encode_position
    from tools.replay_dataset import iter_record_pairs
    from tools.replay_engine_actions import TurnTimer
    from wesnoth_ai.encoder import unknown_type_counts
    from wesnoth_ai.faction_posterior import posterior_counts
    from wesnoth_ai.sequence_records import GameSequence
    stats: Counter = Counter()
    before = posterior_counts()
    overflow_before = sum(unknown_type_counts().values())
    timer = data.get("turn_timer")
    timeouts_hidden = timer is not None and not TurnTimer(*timer).timeouts_detectable()
    seq = GameSequence(file=file, winner=int(winner), n_commands=len(data.get("commands", [])),
                       sides={s: [] for s in PLAYER_SIDES})
    for gs, ai in iter_record_pairs(data, relevant_set=False, stats=stats, timeouts=True):
        side = int(gs.global_info.current_side)
        seq.sides[side].append(encode_position(
            gs, ai, f"{file}, side {side} decision {len(seq.sides[side])}", *vocab, stats, timeouts_hidden))
    after = posterior_counts()
    stats["posteriors"] += after["posteriors"] - before["posteriors"]
    stats["posterior_inconsistent"] += after["inconsistent"] - before["inconsistent"]
    stats["posterior_misses"] += after["excludes_truth"] - before["excludes_truth"]
    stats["posterior_errors"] += stats["posterior_inconsistent"] + stats["posterior_misses"]
    stats["overflow_type_lookups"] += sum(unknown_type_counts().values()) - overflow_before
    stats["positions"] = sum(len(v) for v in seq.sides.values())
    stats["n_commands"] = seq.n_commands
    for s, positions in seq.sides.items():
        stats[f"positions_side{s}"] = len(positions)
    seq.counts = {k: int(v) for k, v in stats.items()}
    return seq


def _view_walk_candidates(rec, sides, vocab):
    """`own_trajectories.candidates_of` as the walk over views built it
    (core phase 34): a view of a snapshot per command, the Python label
    builder, the Python mask builder."""
    import torch
    from tools.game_record import _walk_core
    from tools.inference_seam import build_light_encoded
    from tools.own_trajectories import PLAYER_COMMANDS
    from tools.preencode_sequences import checked_label, encode_position
    from tools.replay_dataset import _action_indices
    from wesnoth_ai.sequence_records import OwnCandidate, OwnCandidateGame, OwnCandidateSide
    from wesnoth_ai.server_priors import pack_masks
    by_side = {s.side: s for s in sides}
    stats: Counter = Counter()
    found = {s: [] for s in by_side}
    commands = rec["commands"]
    for k, cs, cmd in _walk_core(rec, verify=True):
        side = int(cs.core.current_side)
        if side not in by_side or cmd[0] not in PLAYER_COMMANDS:
            continue
        snapshot = cs.fork()
        gs = snapshot.to_state()
        gc.bind_view(gs, snapshot)
        where = f"{sides[0].name}, side {side}, command {k}"
        ai = _action_indices(gs, cmd, relevant_set=False, stats=stats)
        pos = encode_position(gs, ai, where, *vocab, stats)
        pos.masks = pack_masks(build_light_encoded(pos.raw, torch.device("cpu")), gs)
        attack = _view_approach_attack(gs, commands, k, side)
        if attack is not None:
            attack = checked_label(attack, pos.raw, where + " (its approach's attack)", stats)
            stats["approach_pairs"] += 1
        found[side].append(OwnCandidate(position=pos, attack_label=attack))
    game = OwnCandidateGame(file=sides[0].name, winner=int(rec.get("winner", 0)), n_commands=len(commands))
    for side, s in by_side.items():
        game.sides[side] = OwnCandidateSide(side=side, candidates=found[side], forwards=s.forwards,
                                            rejections=0, offset=s.offset)
        stats[f"candidates_side{side}"] = len(found[side])
    stats["candidates"] = sum(len(v) for v in found.values())
    game.counts = {k: int(v) for k, v in stats.items()}
    return game


def _view_approach_attack(gs, commands, k, side):
    """`own_trajectories.approach_attack` on a view."""
    from tools.abilities import hex_neighbors
    from tools.replay_dataset import _action_indices
    from tools.wesnoth_sim import attack_hex
    from wesnoth_ai.classes import Position
    cmd, nxt = commands[k], commands[k + 1] if k + 1 < len(commands) else None
    if cmd[0] != "move" or nxt is None or nxt[0] != "attack":
        return None
    start, end = (int(cmd[1][0]), int(cmd[2][0])), (int(cmd[1][-1]), int(cmd[2][-1]))
    target = (int(nxt[3]), int(nxt[4]))
    if (int(nxt[1]), int(nxt[2])) != end or target in hex_neighbors(*start):
        return None
    units = {(u.position.x, u.position.y): u for u in gs.map.units}
    attacker = units.get(start)
    if attacker is None or attacker.side != side or target not in units:
        return None
    hex_ = attack_hex(gs, attacker, Position(*target))
    if hex_ is None or (hex_.x, hex_.y) != end:
        return None
    return _action_indices(gs, ["attack", start[0], start[1], target[0], target[1], int(nxt[5])])


def _first_difference(new_positions, old_positions) -> str:
    from wesnoth_ai.sequence_columns import position_difference
    if len(new_positions) != len(old_positions):
        return f"{len(new_positions)} positions against {len(old_positions)}"
    for i, (a, b) in enumerate(zip(new_positions, old_positions)):
        why = position_difference(a, b) or (None if a.label == b.label else "label")
        if why is not None:
            return f"position {i}: {why}"
    return "every position agrees; the bytes differ elsewhere (the counts?)"


@needs_core
def test_the_corpus_records_are_the_view_walks_byte_for_byte(corpus_games, vocab):
    from tools.preencode_sequences import encode_game_sequence
    for name, data in corpus_games.items():
        new = encode_game_sequence(data, name, 1, *vocab)
        old = _view_walk_sequence(data, name, 1, vocab)
        if pickle.dumps(new, protocol=pickle.HIGHEST_PROTOCOL) != pickle.dumps(old, protocol=pickle.HIGHEST_PROTOCOL):
            where = [f"side {s}: {_first_difference(new.sides[s], old.sides[s])}" for s in old.sides]
            pytest.fail(f"{name}: {where}, counts {new.counts} against {old.counts}")


@needs_core
def test_the_own_candidates_are_the_view_walks_byte_for_byte(own_record, vocab):
    """The candidates' encodings, labels, approach labels and masks (the
    core's against the Python builder's) are the view walk's."""
    from tools.own_trajectories import OwnSide, candidates_of
    sides = [OwnSide(name="own", record=str(OWN), side=s, forwards=None, offset=-1.5,
                     winner=int(own_record["winner"])) for s in (1, 2)]
    new = candidates_of(own_record, sides, *vocab)
    old = _view_walk_candidates(own_record, sides, vocab)
    assert new.counts["approach_pairs"] > 0 and new.counts["candidates"] > 100, new.counts
    if pickle.dumps(new, protocol=pickle.HIGHEST_PROTOCOL) != pickle.dumps(old, protocol=pickle.HIGHEST_PROTOCOL):
        where = []
        for s in (1, 2):
            a, b = new.sides[s].candidates, old.sides[s].candidates
            where.append(f"side {s}: {_first_difference([c.position for c in a], [c.position for c in b])}")
            where += [f"side {s} candidate {i}: attack label" for i, (x, y) in enumerate(zip(a, b))
                      if x.attack_label != y.attack_label][:1]
        pytest.fail(f"{where}, counts {new.counts} against {old.counts}")


@needs_core
def test_a_label_off_its_command_is_refused_at_its_position(corpus_games, vocab, monkeypatch):
    """A label whose target slot is not its command's hex raises at the
    position (`label_slot_mismatch`), so the game is left out."""
    from tools import replay_dataset
    from tools.encode_worker import LabelSlotMismatch
    from tools.preencode_sequences import encode_game_sequence
    real = replay_dataset.core_label

    def off_by_one(cmd, fields, stats=None):
        ai = real(cmd, fields, stats)
        if ai.action_type == "move" and ai.target_idx is not None:
            ai.target_idx += 1
        return ai
    monkeypatch.setattr(replay_dataset, "core_label", off_by_one)
    with pytest.raises(LabelSlotMismatch, match="move target"):
        encode_game_sequence(corpus_games["hamlets"], "hamlets", 1, *vocab)
