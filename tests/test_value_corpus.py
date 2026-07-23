#!/usr/bin/env python3
"""Value-corpus build + load (tools/build_value_corpus.py,
tools/value_corpus.py).

Pure-function tests are hermetic; the loader test consumes the real
built corpus and SKIPS when it hasn't been built yet (same pattern as
the replay-dependent tests that skip without replays_raw/).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pytest

from tools.build_value_corpus import (_raw_outcome_scan,
                                      _side_player_map)

_INDEX = Path(__file__).parent.parent / "replays_dataset" \
    / "value_corpus_index.jsonl"


def test_surrender_scan_names_the_other_side_winner():
    raw = (
        '[command]\n\tundo=no\n\t[surrender]\n\t\tside_number=1\n'
        '\t[/surrender]\n[/command]\n'
    )
    winner, source = _raw_outcome_scan(raw, {})
    assert (winner, source) == (2, "surrender")


def test_leaver_scan_maps_player_name_to_side():
    raw = (
        '[command]\nundo="no"\n[speak]\nid="server"\n'
        'message="alice has left the game."\n[/speak]\n[/command]\n'
    )
    winner, source = _raw_outcome_scan(raw, {"alice": 2, "bob": 1})
    assert (winner, source) == (1, "leaver")


def test_spectator_leaver_is_ignored():
    raw = (
        '[speak]\nid="server"\n'
        'message="lurker99 has left the game."\n[/speak]\n'
        '[speak]\nid="server"\n'
        'message="bob has left the game."\n[/speak]\n'
    )
    # lurker99 is not a player side; the first PLAYER leaver (bob,
    # side 1) loses.
    winner, source = _raw_outcome_scan(raw, {"alice": 2, "bob": 1})
    assert (winner, source) == (2, "leaver")


def test_no_signal_returns_none():
    assert _raw_outcome_scan("just chat\n", {"a": 1}) == (None, None)


def test_side_player_map_reads_player_keys():
    header = {"sides": [
        {"side": "1", "player_id": "alice", "controller": "human"},
        {"side": "2", "current_player": "bob", "controller": "network"},
    ]}
    m = _side_player_map(header)
    assert m == {"alice": 1, "bob": 2}


@pytest.mark.skipif(not _INDEX.exists(),
                    reason="value corpus not built "
                           "(tools/build_value_corpus.py)")
def test_raw_loader_and_raw_trainer_path():
    """The parallel fine-tune's producer/consumer: game_raw_experiences
    yields picklable (RawEncoded, z, ml) with correct perspective, and
    the raw-fed trainer methods run (chunked, no OOM) on a small net."""
    import json
    import pickle
    import random
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.value_corpus import game_raw_experiences

    net = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64,
                            moves_left=True)
    t2i = dict(net._encoder.unit_type_to_id)
    f2i = dict(net._encoder.faction_to_id)
    row = json.loads(_INDEX.read_text(encoding="utf-8").splitlines()[0])
    recs = game_raw_experiences(_INDEX.parent / row["file"], row["winner"],
                                type_to_id=t2i, faction_to_id=f2i,
                                stride=10, rng=random.Random(1))
    assert recs, "a game must yield raw records"
    raw0, z0, ml0 = recs[0]
    assert z0 in (+1.0, -1.0)
    assert 0.0 <= ml0 <= 1.0
    # RawEncoded must survive the worker->trainer pickle boundary.
    assert pickle.loads(pickle.dumps(raw0)) is not None

    net._trainer.config.train_batch_size = 2   # chunked forward
    raws = [r[0] for r in recs][:6]
    zs = [r[1] for r in recs][:6]
    mls = [r[2] for r in recs][:6]
    m = net._trainer.eval_value_metrics_from_raw(raws, zs)
    assert all(k in m for k in ("ce", "pred_entropy", "marginal_ce_floor"))
    st = net._trainer.step_value_from_raw(raws, zs, mls)
    import math
    assert math.isfinite(st["value_loss"]) and st["grad_norm"] >= 0.0


@pytest.mark.skipif(not _INDEX.exists(),
                    reason="value corpus not built "
                           "(tools/build_value_corpus.py)")
def test_loader_yields_perspective_consistent_experiences():
    import json
    from tools.value_corpus import game_experiences

    row = json.loads(_INDEX.read_text(encoding="utf-8")
                     .splitlines()[0])
    exps = game_experiences(_INDEX.parent / row["file"], row["winner"],
                            stride=8)
    assert exps, "an accepted game must produce states"
    for e in exps:
        assert e.z in (+1.0, -1.0)
        side = e.game_state.global_info.current_side
        assert e.z == (+1.0 if side == row["winner"] else -1.0), \
            "z must be from the side-to-move perspective"
        assert e.visit_counts == []
        assert 0.0 <= e.moves_left_target <= 1.0
