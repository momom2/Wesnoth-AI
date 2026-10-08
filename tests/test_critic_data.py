"""The step-1 critics' positions (wesnoth_ai/critic_data.py,
tools/critic_positions.py): which positions a game offers and which are
drawn, the two encodings, the labels, what a build leaves out and counts,
and the benchmark's holdout."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import critic_positions as builder  # noqa: E402
from wesnoth_ai import critic_data as cd  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


# ---------------------------------------------------------------------
# Which positions
# ---------------------------------------------------------------------
COMMANDS = [["init_side", 1], ["recruit", "Spearman", 2, 3, "a"], ["move", [2, 3], [3, 3], 1], ["end_turn"],
            ["init_side", 2], ["move", [9, 3], [8, 3], 2], ["end_turn"],
            ["init_side", 3], ["end_turn"],
            ["init_side", 1], ["attack", 3, 3, 4, 3, 0, 0, "b"], ["end_turn"]]


def test_a_game_offers_its_player_turn_starts_and_decisions_once_each():
    points = cd.eligible_points(COMMANDS)
    assert [(p.index, p.side, p.kind) for p in points] == [
        (1, 1, "turn_start"), (2, 1, "decision"), (3, 1, "decision"),
        (5, 2, "turn_start"), (6, 2, "decision"),
        (10, 1, "turn_start"), (11, 1, "decision")], "the neutral side's turn offers nothing"
    engine = cd.eligible_points(COMMANDS, engine_issued=[2, 6])
    assert [p.index for p in engine] == [1, 3, 5, 10, 11], "the engine's commands are no decision"


def test_at_most_the_cap_is_drawn_by_the_games_own_seed():
    points = [cd.Point(k, 1 + k % 2, "decision") for k in range(100)]
    drawn = cd.sample_points(points, seed=7)
    assert len(drawn) == cd.MAX_POSITIONS_PER_GAME == 24
    assert [p.index for p in drawn] == sorted(p.index for p in drawn), "in game order"
    assert drawn == cd.sample_points(points, seed=7), "the same seed draws the same positions"
    assert drawn != cd.sample_points(points, seed=8)
    assert cd.sample_points(points[:10], seed=7) == points[:10], "a short game gives all its positions"
    assert cd.game_seed("a.json.gz") == cd.game_seed("a.json.gz") != cd.game_seed("b.json.gz")


def test_the_split_and_the_size_draw_follow_the_key():
    draws = [cd.split_of(f"m/g{k}") for k in range(4000)]
    assert draws[0] == cd.split_of("m/g0")
    holdout = sum(s == "holdout" for s, _ in draws) / len(draws)
    assert 0.04 < holdout < 0.06
    assert 0.23 < sum(u < 0.25 for _, u in draws) / len(draws) < 0.27


# ---------------------------------------------------------------------
# Encodings and labels
# ---------------------------------------------------------------------
def _fogged_core():
    from helpers.parity_games import core_of, record
    cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 2, 3, False),
                         ("Lieutenant", 2, 18, 3, True), ("Spearman", 2, 17, 3, False)], fog=True))
    cs.apply_command(["init_side", 1])
    return cs


@needs_core
def test_the_true_state_shows_a_unit_the_observation_hides():
    from helpers.parity_games import FACTION_IDS, vocab_of
    vocab = vocab_of(["Lieutenant", "Spearman"])
    cs = _fogged_core()
    hidden = cs.core.unit_id_at(17, 3)
    obs = cd.encode_view(cs, "obs", vocab, FACTION_IDS)
    true = cd.encode_view(cs, "true", vocab, FACTION_IDS)
    assert hidden not in obs.unit_ids and hidden in true.unit_ids
    assert set(obs.unit_ids) < set(true.unit_ids)
    assert obs.observation is None and true.observation is None and true.their_faction_id == 0
    assert bool(cs.core.globals_export()["fog_on"]), "the core the walk goes on with keeps its fog"
    again = cd.unpack_raw(cd.pack_raw(true))
    assert again.unit_ids == true.unit_ids
    with pytest.raises(ValueError):
        cd.encode_view(cs, "god", vocab, FACTION_IDS)


@needs_core
def test_the_true_state_shows_a_hider_no_one_has_found():
    """A Wose in forest ambushes: with fog or without, no side sees it
    until an enemy stands next to it. The true state shows it, and the
    walk's own core keeps it hidden."""
    from helpers.parity_games import FACTION_IDS, core_of, record, vocab_of
    vocab = vocab_of(["Lieutenant", "Spearman", "Wose"])
    for fog in (True, False):
        cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 4, 3, False),
                             ("Lieutenant", 2, 18, 3, True), ("Wose", 2, 6, 3, False)], fog=fog,
                            special={(6, 3): "Gs^Fds"}))
        cs.apply_command(["init_side", 1])
        wose = cs.core.unit_id_at(6, 3)
        assert wose not in cd.encode_view(cs, "obs", vocab, FACTION_IDS).unit_ids
        assert wose in cd.encode_view(cs, "true", vocab, FACTION_IDS).unit_ids, f"fog={fog}"
        assert wose not in cd.encode_view(cs, "obs", vocab, FACTION_IDS).unit_ids
        assert cs.core.uncovered_export() == []


def test_the_margin_is_the_movers_hit_points_less_every_other_sides():
    from types import SimpleNamespace as NS
    units = [NS(side=1, current_hp=30), NS(side=2, current_hp=12), NS(side=3, current_hp=1)]
    assert cd.hp_margin(NS(map=NS(units=units)), 1) == 17
    assert cd.hp_margin(NS(map=NS(units=units)), 2) == -19


def test_the_aux_margin_counts_the_two_player_sides_only():
    from types import SimpleNamespace as NS
    units = [NS(side=1, current_hp=30), NS(side=2, current_hp=12), NS(side=3, current_hp=400)]
    assert cd.player_hp_margin(NS(map=NS(units=units)), 1) == 18
    assert cd.player_hp_margin(NS(map=NS(units=units)), 2) == -18


class _Core:
    """A stand-in for a walk's core: the side to move and the turn follow
    the init_side commands; each side's margin is fixed per turn."""

    def __init__(self):
        from types import SimpleNamespace as NS
        self.core = NS(current_side=0, turn_number=0)
        self._ns = NS

    def apply(self, cmd):
        if cmd[0] == "init_side":
            self.core.current_side = int(cmd[1])
            if int(cmd[1]) == 1:
                self.core.turn_number += 1

    def to_state(self):
        # Side 1's units total 10 x turn, side 2's 5 x turn; a neutral
        # side's 50 count for neither.
        t = self.core.turn_number
        units = [self._ns(side=1, current_hp=10 * t), self._ns(side=2, current_hp=5 * t),
                 self._ns(side=3, current_hp=50)]
        return self._ns(map=self._ns(units=units))


def _steps(commands, end):
    core = _Core()
    for k, cmd in enumerate(commands):
        yield k, core, cmd
        core.apply(cmd)
    end.append(core)


def test_labels_are_the_outcome_for_the_mover_and_its_margin_at_its_next_turn_start():
    end = []
    chosen = cd.eligible_points(COMMANDS)
    positions, raws = cd.collect_positions(_steps(COMMANDS, end), end, chosen, winner=2,
                                           encode=lambda cs, view: f"{view}{cs.core.turn_number}".encode())
    by_index = {p.index: p for p in positions}
    assert [p.z for p in positions] == [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    # Side 1 at turn 1 is read at its turn-2 start: 20 - 10.
    assert by_index[1].aux == by_index[3].aux == 10.0
    # Side 2 has no turn start after its turn 1 in this stream.
    assert math.isnan(by_index[5].aux) and math.isnan(by_index[6].aux)
    # The game ends before side 1's turn 3.
    assert math.isnan(by_index[10].aux)
    assert raws["true"][0] == b"true1" and len(raws["obs"]) == len(positions)


def test_a_stream_and_a_core_that_disagree_on_the_mover_stop_the_game():
    end = []
    chosen = [cd.Point(2, 2, "decision")]
    with pytest.raises(cd.PositionError):
        cd.collect_positions(_steps(COMMANDS, end), end, chosen, winner=1, encode=lambda cs, view: b"")


# ---------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------
def test_a_benchmark_game_on_the_training_side_stops_the_build():
    rows = [{"file": "bench.json.gz", "holdout": False, "winner_side": 1},
            {"file": "other.json.gz", "holdout": False, "winner_side": 2}]
    with pytest.raises(builder.HoldoutLeak):
        builder.corpus_training_rows(rows, ["bench.json.gz"])
    rows[0]["holdout"] = True
    copy_of_bench = rows + [{"file": "renamed.json.gz", "holdout": False, "winner_side": 1, "match_key": "k1"}]
    with pytest.raises(builder.HoldoutLeak, match="match"):
        builder.corpus_training_rows(copy_of_bench, ["bench.json.gz"], bench_keys={"k1"})
    train, check = builder.corpus_training_rows(rows, ["bench.json.gz", "dropped.json.gz"])
    assert [r["file"] for r in train] == ["other.json.gz"]
    assert check == {"bench_games": 2, "in_holdout": 1, "absent_from_manifest": 1, "leaked": 0}


def _write(path, rec):
    from tools.game_record import GameRecordLog
    GameRecordLog(path).write(rec)


@needs_core
def test_a_build_keeps_the_decided_games_and_counts_the_capped_and_unlabelled(tmp_path):
    from helpers.parity_games import FACTION_IDS
    from tools.preencode_sequences import fresh_vocab
    from helpers.critic_games import played_record
    rec = dict(played_record(), seed=5)
    games = tmp_path / "games_m1"
    games.mkdir()
    _write(games / "game_x_s1_5.game.jsonl.gz", dict(rec, ended_by="max_turns", winner=0))
    decided = dict(copy.deepcopy(rec), game_label="game_x_s1_6", ended_by="leader_killed", winner=2, seed=6)
    _write(games / "game_x_s1_6.game.jsonl.gz", decided)
    _write(games / "game_x_s1_7.game.jsonl.gz", dict(decided, game_label="game_x_s1_7", winner=0))
    _write(games / "game_x_s1_8.game.jsonl.gz", dict(decided, game_label="another_game"))
    type_to_id, _ = fresh_vocab()
    vocab = tmp_path / "ref.pt"
    torch.save({"unit_type_to_id": type_to_id, "faction_to_id": FACTION_IDS}, vocab)
    out = tmp_path / "positions"
    assert builder.main(["--out", str(out), "--vocab-from", str(vocab), "--matches", str(games),
                         "--workers", "0"]) == 0
    rows = {r["record"]: r for r in map(json.loads, (out / "manifest.jsonl").read_text().splitlines())}
    assert rows["game_x_s1_5.game.jsonl.gz"]["status"] == "capped"
    assert rows["game_x_s1_7.game.jsonl.gz"]["status"] == "no_winner"
    ok = rows["game_x_s1_6.game.jsonl.gz"]
    assert ok["status"] == "ok" and ok["key"] == "m1/game_x_s1_6" and ok["seed"] == 6
    assert ok["positions"] == min(24, ok["eligible"]) > 0
    assert rows["game_x_s1_8.game.jsonl.gz"]["status"] == "error", "a record of another game than its name"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["done"] and summary["counts"]["M"]["games_capped"] == 1
    assert summary["counts"]["M"]["games_no_winner"] == 1 and summary["counts"]["M"]["games_ok"] == 1
    assert summary["counts"]["M"]["games_error"] == 1 and summary["errors"][0]["key"] == "m1/game_x_s1_8"
    from wesnoth_ai import unpickle
    shard = unpickle.loads((out / f"{ok['shard']}.true.pkl").read_bytes())
    assert len(shard["raws"]) == len(shard["positions"]) == ok["positions"]
    assert {p["z"] for p in shard["positions"]} <= {1.0, -1.0}
    assert all(p["z"] == (1.0 if p["side"] == 2 else -1.0) for p in shard["positions"])
    # A second run builds again only the game that failed, and counts it once.
    assert builder.main(["--out", str(out), "--vocab-from", str(vocab), "--matches", str(games),
                         "--workers", "0"]) == 0
    assert len((out / "manifest.jsonl").read_text().splitlines()) == 5
    assert json.loads((out / "summary.json").read_text())["counts"]["M"]["games_error"] == 1
