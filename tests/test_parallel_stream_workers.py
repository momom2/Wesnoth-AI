"""The trainer's `--workers N` stream through real encode worker
processes (tools/supervised_train._ParallelStream,
tools/encode_worker.py). The only test that spawns them, so the only
check on CI that a worker encodes with the trainer's vocab and encoder
switches. Slow: each worker imports torch."""
import dataclasses
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE = Path(__file__).parent / "fixtures" / "strict_sync_hamlets_t9.bz2"
# Every switch off its default, so a worker that lost them encodes
# otherwise (asserted below).
SWITCHES = dict(relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True)


def _same_raw(a, b) -> bool:
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True


@pytest.mark.slow
def test_workers_encode_what_the_trainer_would(tmp_path):
    from tools.encode_worker import encode_game
    from tools.replay_extract import extract_replay
    from tools.supervised_train import _pair_stream_parallel
    from wesnoth_ai.encoder import GameStateEncoder

    game = tmp_path / "hamlets.json.gz"
    with gzip.open(game, "wt", encoding="utf-8") as f:
        json.dump(extract_replay(FIXTURE), f)
    missing = tmp_path / "missing.json.gz"
    enc = GameStateEncoder(d_model=8)
    t2i, f2i = dict(enc.unit_type_to_id), dict(enc.faction_to_id)
    want = encode_game(game, t2i, f2i, **SWITCHES)
    defaults = encode_game(game, t2i, f2i, relevant_set=False)
    assert want and not any(_same_raw(a, b) for (a, _), (b, _) in zip(want, defaults))

    stream = _pair_stream_parallel([game, missing], workers=2, type_to_id=t2i,
                                   faction_to_id=f2i, prefetch_factor=2, **SWITCHES)
    events = list(stream)

    pairs = events[:len(want)]
    assert [e[0] for e in pairs] == ["pair"] * len(want)
    for (_, raw, ai, name), (raw_w, ai_w) in zip(pairs, want):
        assert name == game.name and ai == ai_w and _same_raw(raw, raw_w)
    assert events[len(want)] == ("file_done", game.name, len(want))
    assert [e[:2] for e in events[len(want) + 1:]] == [("file_error", missing.name)]
    # Retired by their sentinels, not terminated by close().
    assert [p.exitcode for p in stream._procs] == [0, 0]
