"""tools/player_ratings.py: rating fit, shrinkage, index parsing, and
the subset a trainer-ready dataset is built from. Torch-free; synthetic
tournaments and headers only."""
from __future__ import annotations

import bz2
import gzip
import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools import player_ratings as pr   # noqa: E402

STRENGTHS_ELO = {"ann": 300.0, "bob": 200.0, "cid": 100.0, "dee": 0.0,
                 "eve": -100.0, "fay": -200.0, pr.AI_PLAYER_ID: -300.0}


def _simulate(rng: random.Random, strengths: dict, games_per_pair: int):
    """(winner, loser) pairs drawn from the Bradley-Terry model."""
    out = []
    names = sorted(strengths)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p_a = 1.0 / (1.0 + math.exp(-(strengths[a] - strengths[b])
                                        / pr.ELO_PER_NATURAL))
            for _ in range(games_per_pair):
                out.append((a, b) if rng.random() < p_a else (b, a))
    return out


def _records(pairs):
    return [pr.GameRecord(f"g{i}.json.gz", w, lo) for i, (w, lo) in enumerate(pairs)]


def test_fit_recovers_known_ordering():
    pairs = _simulate(random.Random(1), STRENGTHS_ELO, games_per_pair=60)
    fit = pr.fit_bradley_terry(_records(pairs))
    fitted = [p.id for p in fit.players]
    truth = sorted(STRENGTHS_ELO, key=lambda k: -STRENGTHS_ELO[k])
    assert fitted == truth
    by_id = fit.by_id()
    assert by_id["ann"].rating_elo - by_id["fay"].rating_elo > 250.0
    assert by_id["ann"].games == 6 * 60
    assert by_id[pr.AI_PLAYER_ID].wins < by_id["ann"].wins


def test_shrinkage_pulls_two_game_player_toward_mean():
    pairs = _simulate(random.Random(2), STRENGTHS_ELO, games_per_pair=40)
    pairs += [("newcomer", "ann"), ("newcomer", "ann")]     # 2-0 vs the best
    records = _records(pairs)
    tight = pr.fit_bradley_terry(records, prior_sd_elo=200.0).by_id()
    loose = pr.fit_bradley_terry(records, prior_sd_elo=5000.0).by_id()
    assert 0.0 < tight["newcomer"].rating_elo < loose["newcomer"].rating_elo
    assert tight["newcomer"].rating_elo < tight["ann"].rating_elo
    assert tight["newcomer"].se_elo > max(p.se_elo for k, p in tight.items()
                                          if k != "newcomer")
    assert tight["newcomer"].games == 2 and tight["newcomer"].wins == 2


def _wml_header(side1: str, side2: str, extra_side: str = "") -> bytes:
    return f"""version="1.18.6"
[replay_start]
\tid="multiplayer_Hamlets"
\t[side]
\t\tside="1"
\t\tfaction="Rebels"
{side1}
\t[/side]
\t[side]
\t\tside="2"
\t\tfaction="Undead"
{side2}
\t[/side]
{extra_side}
[/replay_start]
""".encode("utf-8")


HUMAN_ALICE = '\t\tcontroller="human"\n\t\tplayer_id="Alice"\n\t\tcurrent_player="Alice"'
AI_HOSTED = '\t\tcontroller="ai"\n\t\tplayer_id="Host"\n\t\tcurrent_player="Host"'
NULL_SIDE = '\t\tcontroller="null"\n\t\tfaction="Custom"\n\t\tuser_team_name="Referee"'
REFEREE = '\t[side]\n\t\tside="3"\n' + NULL_SIDE + '\n\t[/side]'


def test_index_reads_side_ids_version_and_date(tmp_path):
    raw = tmp_path / "replays_raw" / "2025-01-02"
    raw.mkdir(parents=True)
    with bz2.open(raw / "game_(7).bz2", "wb") as f:
        f.write(_wml_header(HUMAN_ALICE, AI_HOSTED, REFEREE))
    with bz2.open(raw / "game_(8).bz2", "wb") as f:
        f.write(_wml_header(HUMAN_ALICE, NULL_SIDE))
    rows = [{"file": "a.json.gz", "source": "replays_raw\\2025-01-02\\game_(7).bz2"},
            {"file": "b.json.gz", "source": "replays_raw\\2025-01-02\\game_(8).bz2"},
            {"file": "c.json.gz", "source": "replays_raw\\2025-01-02\\missing.bz2"}]
    out = tmp_path / "players.jsonl"
    stats = pr.build_player_index(rows, tmp_path, out)
    assert (stats["ok"], stats["error"], stats["sides_missing"]) == (2, 1, 1)
    got = {r["file"]: r for r in pr.read_jsonl(out)}
    assert got["a.json.gz"]["sides"] == {"1": "Alice", "2": pr.AI_PLAYER_ID}
    assert got["a.json.gz"]["version"] == "1.18.6"
    assert got["a.json.gz"]["date"] == "2025-01-02"
    assert got["b.json.gz"]["sides"] == {"1": "Alice", "2": None}
    assert "error" in got["c.json.gz"]


def _write_dataset(root: Path, pairs, holdout_every: int):
    """A dataset dir in the tools/build_imitation_dataset.py schema plus
    its players.jsonl; the winner sits on side 1 or 2 alternately."""
    rng = random.Random(3)
    rows, index = [], []
    for i, (w, lo) in enumerate(pairs):
        name = f"2025-01-0{1 + i % 9}_game_({i}).json.gz"
        side = 1 + i % 2
        with gzip.open(root / name, "wt", encoding="utf-8") as f:
            json.dump({"game_id": i, "commands": []}, f)
        rows.append({"file": name, "source": f"replays_raw\\d\\g{i}.bz2",
                     "winner_side": side, "outcome": "explicit",
                     "n_turns": rng.randint(6, 30), "n_commands": rng.randint(50, 400),
                     "winner_actions": rng.randint(20, 200),
                     "holdout": i % holdout_every == 0})
        index.append({"file": name, "source": rows[-1]["source"], "date": "d",
                      "version": "1.18.6",
                      "sides": {str(side): w, str(3 - side): lo}})
    pr.write_jsonl(root / "manifest.jsonl", rows)
    pr.write_jsonl(root / "players.jsonl", index)
    return rows


def _trainer_split(dataset_dir: Path):
    """The trainer's imitation-mode split (tools/supervised_train.py
    train(): every *.json.gz in the directory, minus the manifest's
    holdout names) -> (training files, holdout files)."""
    rows = pr.read_manifest(dataset_dir)
    holdout = {r["file"] for r in rows if r["holdout"]}
    files = sorted(p.name for p in dataset_dir.glob("*.json.gz"))
    return [f for f in files if f not in holdout], [f for f in files if f in holdout]


def test_subset_excludes_holdout_and_builds_trainer_dataset(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pairs = _simulate(random.Random(4), STRENGTHS_ELO, games_per_pair=20)
    rows = _write_dataset(dataset, pairs, holdout_every=10)
    out = tmp_path / "ratings"
    built = tmp_path / "toprated"
    rc = pr.main(["player_ratings.py", "fit", "--dataset", str(dataset),
                  "--min-games", "30", "--quantile", "0.75", "--out", str(out),
                  "--report", "--build-dataset", str(built)])
    assert rc == 0

    ratings = json.loads((out / "ratings.json").read_text(encoding="utf-8"))
    top = set(ratings["top_ids"])
    assert top and top <= {"ann", "bob", "cid"} and pr.AI_PLAYER_ID not in top
    assert {p["id"] for p in ratings["players"]} == set(STRENGTHS_ELO)
    assert all(p["games"] == 120 for p in ratings["players"])

    subset = pr.read_jsonl(out / "subset_manifest.jsonl")
    source = {r["file"]: r for r in rows}
    winners = {r["file"]: w for r, (w, _lo) in zip(rows, pairs)}
    expected = {r["file"] for r in rows if not r["holdout"] and winners[r["file"]] in top}
    assert {r["file"] for r in subset} == expected
    for r in subset:
        assert r["holdout"] is False and r["winner_id"] == winners[r["file"]]
        assert r["winner_games"] == 120 and r["winner_rating_elo"] >= ratings["threshold_elo"]
        stripped = {k: v for k, v in r.items() if k not in pr.SUBSET_EXTRA_KEYS}
        assert stripped == source[r["file"]]
    assert pr.estimate_pairs(subset) == sum(source[r["file"]]["winner_actions"] for r in subset)

    train_files, holdout_files = _trainer_split(built)
    assert set(train_files) == expected
    assert set(holdout_files) == {r["file"] for r in rows
                                  if r["holdout"] and winners[r["file"]] in top}
    assert holdout_files and not (set(holdout_files) & set(train_files))
    value_index = pr.read_jsonl(built / "value_corpus_index.jsonl")
    assert {(v["file"], v["winner"]) for v in value_index} == {
        (r["file"], r["winner_side"]) for r in pr.read_manifest(built)}
    with gzip.open(built / train_files[0], "rt", encoding="utf-8") as f:
        assert json.load(f)["commands"] == []

    printed = capsys.readouterr().out
    assert "KILL 0" in printed and f"subset: {len(subset)} of" in printed
    with pytest.raises(FileExistsError):
        pr.build_dataset_dir(dataset, built, subset)
