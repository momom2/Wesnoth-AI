"""The corpus split is one decision, taken by the manifest, and every
consumer respects it (2026-09-08 contamination review: the value
tools' own shuffled splits trained on 98% of the manifest holdout;
copies of one match under two names straddled the split)."""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.build_imitation_dataset import dedup_rows  # noqa: E402
from tools.dedup_corpus import dedup_corpus  # noqa: E402
from tools.replay_dataset import command_hash, manifest_holdout_split, match_key  # noqa: E402


def _game(n_commands: int, seed: int = 7):
    return {"commands": [{"turn": i, "seed": seed, "i": i} for i in range(n_commands)],
            "map_data": "Gg, Gg\nGg, Gg",
            "starting_units": [{"type": "Elvish Fighter", "x": 1, "y": 1}]}


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_manifest_split_drops_unlisted_rows_and_keeps_holdout_flags(tmp_path):
    rows = [{"file": f"g{i}.json.gz", "winner": 1} for i in range(4)]
    assert manifest_holdout_split(rows, tmp_path) is None
    _write_jsonl(tmp_path / "manifest.jsonl", [
        {"file": "g0.json.gz", "holdout": False}, {"file": "g1.json.gz", "holdout": True},
        {"file": "g2.json.gz", "holdout": False}])          # g3 quarantined: absent
    train, hold = manifest_holdout_split(rows, tmp_path)
    assert [r["file"] for r in train] == ["g0.json.gz", "g2.json.gz"]
    assert [r["file"] for r in hold] == ["g1.json.gz"]


def test_match_key_identifies_a_game_saved_at_two_turns_but_not_two_games():
    early = _game(250)                      # no seeds: the key reads the first 200 commands
    late = {**early, "commands": early["commands"] + [{"turn": 20, "i": 999}]}
    other = _game(250, seed=8)
    assert match_key(early) == match_key(late)
    assert match_key(early) != match_key(other)
    assert command_hash(early) != command_hash(late)


def test_dedup_keeps_the_longest_copy_and_promotes_the_cluster_to_holdout():
    rows = [
        {"file": "b_turn10.json.gz", "n_commands": 100, "holdout": False, "match_key": "k1"},
        {"file": "a_turn12.json.gz", "n_commands": 130, "holdout": False, "match_key": "k1"},
        {"file": "c_turn12.json.gz", "n_commands": 130, "holdout": True, "match_key": "k1"},
        {"file": "d.json.gz", "n_commands": 50, "holdout": False, "match_key": "k2"},
    ]
    kept, dropped = dedup_rows(rows)
    assert sorted(r["file"] for r in kept) == ["a_turn12.json.gz", "d.json.gz"]
    survivor = next(r for r in kept if r["match_key"] == "k1")
    assert survivor["holdout"] is True           # one copy was holdout: the match stays out
    assert {r["file"] for r in dropped} == {"b_turn10.json.gz", "c_turn12.json.gz"}
    assert all(r["duplicate_of"] == "a_turn12.json.gz" for r in dropped)


def test_dedup_pass_moves_files_and_rewrites_manifest_and_index(tmp_path):
    ds = tmp_path / "corpus"
    ds.mkdir()
    games = {"x_t10.json.gz": _game(220), "x_t14.json.gz": _game(260),
             "y.json.gz": _game(220, seed=3)}
    for name, data in games.items():
        with gzip.open(ds / name, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    manifest = [{"file": n, "n_commands": len(d["commands"]),
                 "holdout": n == "x_t10.json.gz", "winner_side": 1} for n, d in games.items()]
    _write_jsonl(ds / "manifest.jsonl", manifest)
    _write_jsonl(ds / "value_corpus_index.jsonl", [
        {"file": r["file"], "winner": 1, "n_commands": r["n_commands"]} for r in manifest])
    kept, dropped = dedup_corpus(ds, workers=1, log=lambda *_: None)
    assert [r["file"] for r in kept] == ["x_t14.json.gz", "y.json.gz"]
    assert kept[0]["holdout"] is True
    assert not (ds / "x_t10.json.gz").exists()
    assert (tmp_path / "corpus_duplicates" / "x_t10.json.gz").exists()
    assert [r["file"] for r in _read_jsonl(ds / "value_corpus_index.jsonl")] == ["x_t14.json.gz", "y.json.gz"]
    assert [r["file"] for r in _read_jsonl(ds / "manifest.jsonl")] == ["x_t14.json.gz", "y.json.gz"]
    assert [r["duplicate_of"] for r in _read_jsonl(ds / "duplicates.jsonl")] == ["x_t14.json.gz"]
    kept2, dropped2 = dedup_corpus(ds, workers=1, log=lambda *_: None)   # idempotent
    assert len(kept2) == 2 and dropped2 == []
