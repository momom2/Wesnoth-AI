"""Loaders say what they skip: the replay size filter when it cannot
apply its command cap or cannot read a file, the holdout probe when a
game fails to reconstruct, the value loaders when a game fails to load
(2026-09-25 crawl: each of these skipped without a word)."""
import gzip
import logging

import torch

from tools import supervised_train as st
from tools.value_pretrain import LoadFailures, _load_worker
from wesnoth_ai.encoder import GameStateEncoder
from wesnoth_ai.model import WesnothModel


def _names(files):
    return [p.name for p in files]


def test_the_size_filter_says_when_its_command_cap_cannot_apply(tmp_path, caplog):
    files = [tmp_path / "a.json.gz", tmp_path / "b.json.gz"]
    with caplog.at_level(logging.INFO, logger="supervised_train"):
        kept = st._apply_size_filters(files, tmp_path, max_commands=1500, max_starting=0)
    assert _names(kept) == ["a.json.gz", "b.json.gz"]
    assert "cap is not applied" in caplog.text and "dropped 0" not in caplog.text

    caplog.clear()
    (tmp_path / "index.jsonl").write_text('{"file": "b.json.gz", "n_commands": 2000}\n',
                                          encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="supervised_train"):
        kept = st._apply_size_filters(files, tmp_path, max_commands=1500, max_starting=0)
    assert _names(kept) == ["a.json.gz"] and "dropped 1 (>1500 cmds)" in caplog.text


def test_the_size_filter_counts_the_files_it_cannot_read(tmp_path, caplog):
    good, bad = tmp_path / "good.json.gz", tmp_path / "bad.json.gz"
    with gzip.open(good, "wt", encoding="utf-8") as f:
        f.write('{"starting_units": []}')
    bad.write_bytes(b"not gzip")
    with caplog.at_level(logging.INFO, logger="supervised_train"):
        kept = st._apply_size_filters([good, bad], tmp_path, max_commands=0, max_starting=10)
    assert _names(kept) == ["good.json.gz"]
    assert "1 unreadable (dropped)" in caplog.text and "bad.json.gz unreadable" in caplog.text


def test_the_holdout_probe_counts_the_games_it_cannot_reconstruct(tmp_path, caplog):
    broken = tmp_path / "broken.json.gz"
    broken.write_bytes(b"not a replay")
    torch.manual_seed(0)
    model, enc = WesnothModel(d_model=16, num_layers=1, num_heads=2, d_ff=32), GameStateEncoder(d_model=16)
    with caplog.at_level(logging.WARNING, logger="supervised_train"):
        stats = st._evaluate(model, enc, [broken], torch.device("cpu"), eval_pairs=4)
    assert stats["n"] == 0
    assert stats["probe_skipped"] == {"file_errors": 1, "encode": 0, "forward": 0}
    assert "broken.json.gz not reconstructed" in caplog.text


def test_a_value_loader_reports_the_game_it_cannot_load(tmp_path, caplog):
    exps, error = _load_worker((str(tmp_path), "missing.json.gz", 1, 6, 0, 7))
    assert exps == [] and "missing.json.gz" in error
    failures = LoadFailures("pass 0")
    with caplog.at_level(logging.WARNING, logger="value_pretrain"):
        failures.note(None)
        failures.note(error)
        failures.summary(2)
    assert len(failures.errors) == 1 and "1 of 2 games not loaded" in caplog.text
