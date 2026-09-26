"""The imitation corpus's version-2 rules, on replays written from scratch
(tests/helpers/synthetic_replay.py), so they run where no corpus is:

  - a move the engine stopped short keeps the hex its player clicked, and
    its label names that hex when the unit could end a move there;
  - a record is cut at the first action after a surrender, and at the
    first action of a side's turn its opponent took;
  - the outcome: leader death, else the surrendering side loses, unless
    it was ahead on material;
  - pre-encoded records carry their corpus's version and a label's slots
    are checked against its command.
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.synthetic_replay import (end_turn, init_side, move, server, side_block,  # noqa: E402
                                      surrender, turn, two_sides, write_replay)
from tools.replay_dataset import iter_record_pairs  # noqa: E402
from tools.replay_extract import extract_replay  # noqa: E402

CAVALRY = [("Cavalryman", 3, 5, False)]


def _pairs(record):
    stats = Counter()
    pairs = [(gs.global_info.current_side, ai) for gs, ai in iter_record_pairs(record, stats=stats)]
    return pairs, stats


def _only_move(pairs):
    moves = [ai for _side, ai in pairs if ai.action_type == "move"]
    assert len(moves) == 1
    return moves[0]


# ---- a stopped move's label ------------------------------------------

def test_a_move_stopped_on_sighting_is_labelled_with_the_clicked_hex(tmp_path):
    path = [(3, 5), (4, 5), (5, 5), (6, 5)]
    replay = write_replay(tmp_path / "g.bz2", two_sides(extra1=CAVALRY),
                          [*turn(1, move(1, path, final=(4, 5), stopped_early="yes")), *turn(2)])
    record = extract_replay(replay)
    stopped = next(c for c in record["commands"] if c[0] == "move")
    assert stopped[1:3] == [[2, 3], [4, 4]]                 # the state follows the stop
    assert stopped[4] == {"clicked": [5, 4], "stopped_early": True}
    pairs, stats = _pairs(record)
    label = _only_move(pairs)
    assert (label.source_hex, label.target_hex) == ((2, 4), (5, 4))
    assert stats["move_label_clicked"] == 1


def test_a_move_that_reached_its_turn_end_keeps_the_stop(tmp_path):
    """stopped_early=no: the order's part for this turn ended at the stop
    (a multi-turn order), which is this turn's target."""
    path = [(3, 5), (4, 5), (5, 5), (6, 5)]
    replay = write_replay(tmp_path / "g.bz2", two_sides(extra1=CAVALRY),
                          [*turn(1, move(1, path, final=(5, 5), stopped_early="no")), *turn(2)])
    label = _only_move(_pairs(extract_replay(replay))[0])
    assert label.target_hex == (4, 4)


def test_a_clicked_hex_out_of_the_units_reach_keeps_the_stop(tmp_path):
    """A Spearman has 5 moves; a clicked hex 9 hexes away is not a hex it
    can end this turn's move on."""
    path = [(x, 5) for x in range(2, 12)]
    sides = two_sides(extra1=[("Spearman", 2, 5, False)])
    replay = write_replay(tmp_path / "g.bz2", sides,
                          [*turn(1, move(1, path, final=(4, 5), stopped_early="yes")), *turn(2)])
    pairs, stats = _pairs(extract_replay(replay))
    assert _only_move(pairs).target_hex == (3, 4)
    assert stats["move_label_clicked_unreachable"] == 1


def test_a_record_from_before_the_order_field_still_loads(tmp_path):
    path = [(3, 5), (4, 5), (5, 5), (6, 5)]
    replay = write_replay(tmp_path / "g.bz2", two_sides(extra1=CAVALRY),
                          [*turn(1, move(1, path, final=(4, 5), stopped_early="yes")), *turn(2)])
    record = extract_replay(replay)
    record["commands"] = [c[:4] if c[0] == "move" else c for c in record["commands"]]
    pairs, stats = _pairs(record)
    assert _only_move(pairs).target_hex == (3, 4)
    assert stats["move_label_path_end"] == 1


# ---- where a game ends -----------------------------------------------

def _two_turns_then(*tail):
    return [*turn(1), *turn(2), init_side(1), *tail]


def test_play_after_a_surrender_is_cut(tmp_path):
    sides = two_sides(extra1=CAVALRY)
    commands = _two_turns_then(
        server("alice takes control of side 2."), server("bob has surrendered."), surrender(2),
        move(1, [(3, 5), (4, 5)]), end_turn(), init_side(2), end_turn())
    replay = write_replay(tmp_path / "g.bz2", sides, commands)
    whole = extract_replay(replay)
    cut = extract_replay(replay, cut_at_game_end=True)
    assert [c[0] for c in whole["commands"]][-4:] == ["move", "end_turn", "init_side", "end_turn"]
    assert cut["commands"][-1] == ["init_side", 1]
    assert cut["game_end"]["reason"] == "surrender"
    assert cut["game_end"]["surrender_side"] == 2 and cut["game_end"]["cut"]


def test_a_side_played_by_its_opponent_is_cut_at_that_sides_turn(tmp_path):
    """bob leaves during alice's turn and alice takes his side: her own
    move after that is hers to make; side 2's next turn is not bob's."""
    sides = two_sides(extra1=CAVALRY)
    commands = _two_turns_then(
        server("bob has left the game."), server("alice takes control of side 2."),
        move(1, [(3, 5), (4, 5)]), end_turn(), init_side(2), end_turn())
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, commands),
                            cut_at_game_end=True)
    assert [c[0] for c in record["commands"]][-3:] == ["init_side", "move", "end_turn"]
    assert record["game_end"]["reason"] == "played_by_opponent"
    assert record["game_end"]["played_by_opponent_side"] == 2


@pytest.mark.parametrize("control", [
    ["bob has disconnected.", "alice takes control of side 2.", "bob takes control of side 2."],
    ["bob has left the game.", "alice takes control of side 2.", "carol takes control of side 2."],
], ids=["back_before_his_turn", "replaced"])
def test_a_side_its_own_player_or_a_replacement_plays_on_is_not_cut(tmp_path, control):
    commands = _two_turns_then(*[server(m) for m in control], end_turn(), *turn(2))
    record = extract_replay(write_replay(tmp_path / "g.bz2", two_sides(), commands),
                            cut_at_game_end=True)
    assert record["game_end"]["reason"] == "end"
    assert record["commands"][-2:] == [["init_side", 2], ["end_turn"]]


def test_holding_the_opponents_side_during_ones_own_turn_is_not_cut(tmp_path):
    """alice disconnects during bob's turn; bob holds her side while he
    finishes his own turn, and she takes it back before hers: every
    command is still its side's own player's."""
    sides = two_sides(extra2=[("Cavalryman", 10, 5, False)])
    commands = [*turn(1), init_side(2), server("alice has disconnected."),
                server("bob takes control of side 1."), move(2, [(10, 5), (9, 5)]),
                server("alice takes control of side 1."), end_turn(), *turn(1)]
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, commands),
                            cut_at_game_end=True)
    assert record["game_end"]["reason"] == "end"
    assert [c[0] for c in record["commands"]][-4:] == ["move", "end_turn", "init_side", "end_turn"]


def test_one_name_on_both_sides_leaves_no_play(tmp_path):
    record = extract_replay(write_replay(tmp_path / "g.bz2", two_sides("alice", "alice"),
                                         [*turn(1), *turn(2)]), cut_at_game_end=True)
    assert record["commands"] == [] and record["game_end"]["cut_before_play"]


# ---- the outcome -----------------------------------------------------

@pytest.mark.parametrize("led, alive, material, end, expected", [
    ({1, 2}, {1}, {1: 50, 2: 900}, {"reason": "surrender", "surrender_side": 1}, ("leader_death", 1)),
    ({1, 2}, {1, 2}, {1: 100, 2: 104}, {"reason": "surrender", "surrender_side": 2}, ("surrender", 1)),
    ({1, 2}, {1, 2}, {1: 100, 2: 106}, {"reason": "surrender", "surrender_side": 2}, ("abandoned", None)),
    ({1, 2}, {1, 2}, {1: 100, 2: 90},
     {"reason": "surrender", "surrender_side": 2, "surrender_sides_agree": False}, ("abandoned", None)),
    ({1, 2}, {1, 2}, {1: 100, 2: 90},
     {"reason": "played_by_opponent", "played_by_opponent_side": 1}, ("left", 2)),
    ({1, 2}, {1, 2}, {1: 100, 2: 90}, {"reason": "end"}, ("inconclusive", None)),
    ({2}, {2}, {1: 100, 2: 90}, {"reason": "end"}, ("inconclusive", None)),
], ids=["leader_death_first", "surrender_behind", "surrender_ahead", "sources_disagree",
        "left", "nothing", "no_leader_from_the_start"])
def test_the_outcome_rules(led, alive, material, end, expected):
    from tools.replay_outcome import decide_outcome
    outcome = decide_outcome(led, alive, {k: float(v) for k, v in material.items()}, end)
    assert (outcome.outcome, outcome.winner_side) == expected


@pytest.mark.parametrize("extra, expected", [((), ("surrender", 2)),
                                             (CAVALRY, ("abandoned", None))],
                         ids=["even", "surrenderer_ahead"])
def test_the_labeller_reads_the_surrender_and_the_material(tmp_path, extra, expected):
    """Side 1 surrenders with gold 100 against 100; with a Cavalryman
    (cost 17) more it is ahead by more than 5%."""
    from tools.replay_outcome import label_outcome
    commands = _two_turns_then(server("bob takes control of side 1."),
                               server("alice has surrendered."), surrender(1))
    record = extract_replay(write_replay(tmp_path / "g.bz2", two_sides(extra1=extra), commands),
                            cut_at_game_end=True)
    outcome = label_outcome(record)
    assert (outcome.outcome, outcome.winner_side) == expected


# ---- match keys ------------------------------------------------------

def _game(commands):
    return {"commands": commands, "map_data": "Gg", "starting_units": []}


def test_a_reloaded_game_shares_the_key_of_its_original():
    from tools.replay_dataset import match_key
    shared = [["recruit", "Spearman", 1, 1, "5eed"]] + [["end_turn"]] * 39
    original = _game(shared + [["move", [1, 2], [1, 1], 1]] * 200)
    reloaded = _game(shared + [["move", [1, 1], [1, 2], 1]] * 50)
    assert match_key(original) == match_key(reloaded)
    assert match_key(original) != match_key(_game([["recruit", "Spearman", 1, 1, "0the"]] + shared[1:]))


def test_a_prefix_without_a_seed_reads_on_to_the_first_one():
    from tools.replay_dataset import match_key
    seedless = [["end_turn"]] * 40
    a = _game(seedless + [["recruit", "Ghoul", 1, 1, "aaaa"]])
    b = _game(seedless + [["recruit", "Ghoul", 1, 1, "bbbb"]])
    assert match_key(a) != match_key(b)


# ---- guards ----------------------------------------------------------

def _record_file(tmp_path, commands, extra1=CAVALRY):
    record = extract_replay(write_replay(tmp_path / "g.bz2", two_sides(extra1=extra1), commands))
    path = tmp_path / "g.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(record, f)
    return path


VOCAB = ({"Lieutenant": 1, "Cavalryman": 2, "Spearman": 3}, {"Loyalists": 1})


@pytest.mark.parametrize("relevant_set", [False, True], ids=["full_board", "relevant_set"])
def test_encode_game_checks_every_label_against_the_tokens(tmp_path, monkeypatch, relevant_set):
    from tools import replay_dataset
    from tools.encode_worker import LabelSlotMismatch, encode_game
    path = _record_file(tmp_path, [*turn(1, move(1, [(3, 5), (4, 5), (5, 5)],
                                                  final=(4, 5), stopped_early="yes")), *turn(2)])
    pairs = encode_game(path, *VOCAB, relevant_set)
    assert [ai.action_type for _raw, ai in pairs] == ["move", "end_turn", "end_turn"]
    real = replay_dataset._action_indices

    def off_by_one(gs, cmd, **kw):
        ai = real(gs, cmd, **kw)
        if ai is not None and ai.action_type == "move":
            ai.target_idx += 1
        return ai
    monkeypatch.setattr(replay_dataset, "_action_indices", off_by_one)
    with pytest.raises(LabelSlotMismatch, match="move target"):
        encode_game(path, *VOCAB, relevant_set)


def test_a_player_command_without_a_pair_is_counted(tmp_path):
    from tools.replay_dataset import iter_replay_pairs
    path = _record_file(tmp_path, [*turn(1, move(1, [(3, 5), (4, 5)])), *turn(2)])
    record = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
    for c in record["commands"]:
        if c[0] == "move":
            c[1], c[2] = [7, 8], [1, 1]              # no unit there
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(record, f)
    stats = Counter()
    assert [ai.action_type for _gs, ai in iter_replay_pairs(path, stats=stats)] == ["end_turn"] * 2
    assert stats["unpaired"] == 1


# ---- the corpus's version --------------------------------------------

def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_records_of_another_corpus_version_are_refused(tmp_path):
    from types import SimpleNamespace

    from tools.preencode_corpus import MANIFEST_NAME, vocab_fingerprint
    from tools.replay_dataset import corpus_version_of
    from tools.supervised_train import check_preencoded
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    types, factions = VOCAB
    assert vocab_fingerprint(types, factions, True) == vocab_fingerprint(types, factions, True,
                                                                         corpus_version=1)
    assert vocab_fingerprint(types, factions, True, corpus_version=2) != \
        vocab_fingerprint(types, factions, True)
    dataset, encoded = tmp_path / "corpus", tmp_path / "encoded"
    dataset.mkdir()
    encoded.mkdir()
    _write_jsonl(dataset / "manifest.jsonl", [{"file": "g.json.gz", "corpus_version": 2}])
    assert corpus_version_of(dataset) == 2
    (encoded / MANIFEST_NAME).write_text(json.dumps({
        "fingerprint": vocab_fingerprint(types, factions, True),
        "observation_epoch": int(OBSERVATION_EPOCH),
        "unit_types": len(types), "factions": len(factions)}), encoding="utf-8")
    encoder = SimpleNamespace(unit_type_to_id=dict(types), faction_to_id=dict(factions))
    with pytest.raises(RuntimeError, match="corpus of version 1"):
        check_preencoded(encoded, [dataset / "g.json.gz"], encoder, True)
    _write_jsonl(dataset / "manifest.jsonl", [{"file": "g.json.gz", "corpus_version": 2},
                                              {"file": "h.json.gz"}])
    with pytest.raises(ValueError, match="mixes"):
        corpus_version_of(dataset)


# ---- the builder -----------------------------------------------------

def test_the_builder_keeps_decided_human_games_and_says_why_it_leaves_out_the_rest(tmp_path):
    from tools.build_imitation_dataset import CORPUS_VERSION, build, is_holdout
    raw = tmp_path / "raw"
    surrendered = _two_turns_then(server("alice takes control of side 2."),
                                  server("bob has surrendered."), surrender(2),
                                  move(1, [(3, 5), (4, 5)]), end_turn())
    games = {
        "surrender": (two_sides(extra1=CAVALRY), surrendered),
        "unfinished": (two_sides(), [*turn(1), *turn(2)]),
        "ai": (two_sides(controller2="ai"), surrendered),
        "hotseat": ([side_block(1, "alice", [("Lieutenant", 2, 4, True)]),
                     side_block(2, "alice", [("Lieutenant", 11, 4, True)])], [*turn(1), *turn(2)]),
    }
    ledger = []
    for name, (sides, commands) in games.items():
        write_replay(raw / "replays_raw" / "2026-09-26" / f"{name}.bz2", sides, commands)
        ledger.append(f"replays_raw\\2026-09-26\\{name}.bz2")
    config = {"outcome_classes": ["leader_death", "surrender"], "holdout_fraction": 0.02,
              "quarantine_ai_player_sides": True}
    out = tmp_path / "corpus"
    counts = build(ledger, raw, out, config, workers=1)
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [(r["file"], r["outcome"], r["winner_side"]) for r in manifest] == [
        ("2026-09-26_surrender.json.gz", "surrender", 1)]
    row = manifest[0]
    assert row["corpus_version"] == CORPUS_VERSION
    assert row["holdout"] == is_holdout(ledger[0], 0.02)
    assert row["winner_actions"] == 0                      # alice's move came after the surrender
    quarantined = {json.loads(line)["source"].split("\\")[-1]: json.loads(line)["quarantined"]
                   for line in (out / "quarantined.jsonl").read_text().splitlines()}
    assert quarantined == {"ai.bz2": "ai_player_side", "hotseat.bz2": "one_player_both_sides"}
    assert counts["inconclusive"] == 1 and counts["games"] == 1
    index = [json.loads(line) for line in (out / "value_corpus_index.jsonl").read_text().splitlines()]
    assert index == [{"file": row["file"], "winner": 1, "n_commands": row["n_commands"]}]


def test_the_builder_refuses_an_outcome_class_the_labeller_never_gives(tmp_path):
    """The version-1 config's "explicit" would keep no game at all."""
    from tools.build_imitation_dataset import build
    with pytest.raises(ValueError, match="explicit"):
        build([], tmp_path, tmp_path / "corpus",
              {"outcome_classes": ["explicit"], "holdout_fraction": 0.02}, workers=1)

def test_the_staged_raw_corpus_is_what_the_builder_reads(tmp_path):
    """tools/stage_raw_corpus.py packs the ledger and its candidates at
    their ledger paths; unpacked elsewhere, the builder finds them."""
    import tarfile

    from tools.build_imitation_dataset import DISPOSITIONS, build, load_candidates
    from tools.stage_raw_corpus import write_tarball
    laptop, box = tmp_path / "laptop", tmp_path / "box"
    ledger = laptop / DISPOSITIONS
    ledger.parent.mkdir(parents=True)
    rows = [{"path": "replays_raw\\2026-09-26\\won.bz2", "era_class": "accept", "mod_class": "mod_free"},
            {"path": "replays_raw\\2026-09-26\\other_era.bz2", "era_class": "set_aside_other_era",
             "mod_class": "mod_free"}]
    with gzip.open(ledger, "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    commands = _two_turns_then(server("alice takes control of side 2."),
                               server("bob has surrendered."), surrender(2))
    write_replay(laptop / "replays_raw" / "2026-09-26" / "won.bz2", two_sides(), commands)
    assert write_tarball(tmp_path / "raw.tar", laptop, ledger) == 1
    with tarfile.open(tmp_path / "raw.tar") as tf:
        tf.extractall(box, filter="data")
    candidates = load_candidates(box / DISPOSITIONS)
    config = {"outcome_classes": ["surrender"], "holdout_fraction": 0.0}
    counts = build(candidates, box, tmp_path / "corpus", config, workers=1)
    assert (candidates, counts["games"], counts["errors"]) == ([rows[0]["path"]], 1, 0)
