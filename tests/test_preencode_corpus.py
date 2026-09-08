"""Pre-encoded corpus (tools/preencode_corpus.py): the records are
byte-for-byte what the encode workers produce, the trainer's stream
over them is the worker stream, and a corpus of another vocab is
refused. Uses two games of the local imitation corpus (skipped when
it is absent) and the local seed checkpoint's vocab."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "replays_dataset_imitation"
SEED = ROOT / "training/checkpoints/seed_imit_tierb_start.pt"

pytestmark = pytest.mark.skipif(
    not (DATASET / "manifest.jsonl").exists() or not SEED.exists(),
    reason="local imitation corpus or seed checkpoint absent")


def _two_games():
    rows = [json.loads(line) for line in
            (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    small = sorted(rows, key=lambda r: r["n_commands"])[:2]
    return [DATASET / r["file"] for r in small]


def _same_raw(a, b) -> bool:
    import dataclasses
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    from tools.preencode_corpus import main as preencode
    out = tmp_path_factory.mktemp("enc")
    files = _two_games()
    rc = preencode(["--dataset", str(DATASET), "--out", str(out), "--vocab-from", str(SEED),
                    "--workers", "2", "--limit", "0"])
    assert rc == 0
    # --limit 0 encodes nothing; encode the two chosen games explicitly.
    from tools.preencode_corpus import (encode_game, record_path, vocab_from_checkpoint,
                                        write_record)
    t2i, f2i = vocab_from_checkpoint(SEED)
    for gz in files:
        write_record(record_path(out, gz.name), encode_game(gz, t2i, f2i, False))
    return out, files, t2i, f2i


def test_records_equal_the_live_encoding(encoded):
    out, files, t2i, f2i = encoded
    from tools.preencode_corpus import encode_game, read_record, record_path
    for gz in files:
        stored = read_record(record_path(out, gz.name))
        live = encode_game(gz, t2i, f2i, False)
        assert len(stored) == len(live) > 0
        for (raw_s, ai_s), (raw_l, ai_l) in zip(stored, live):
            assert _same_raw(raw_s, raw_l)
            assert ai_s == ai_l


def test_preencoded_stream_is_the_worker_stream(encoded):
    out, files, t2i, f2i = encoded
    from tools.supervised_train import _pair_stream_parallel, _pair_stream_preencoded
    pre = list(_pair_stream_preencoded(files, out))
    live = list(_pair_stream_parallel(files, workers=2, type_to_id=t2i, faction_to_id=f2i,
                                      prefetch_factor=2))
    assert [e[0] for e in pre] == [e[0] for e in live]
    for a, b in zip(pre, live):
        if a[0] == "pair":
            assert _same_raw(a[1], b[1]) and a[2] == b[2] and a[3] == b[3]
        else:
            assert a == b


def test_other_vocab_or_missing_record_is_refused(encoded, tmp_path):
    out, files, t2i, f2i = encoded
    from tools.supervised_train import check_preencoded
    from types import SimpleNamespace
    enc = SimpleNamespace(unit_type_to_id=dict(t2i), faction_to_id=dict(f2i))
    check_preencoded(out, files, enc, False)
    other = SimpleNamespace(unit_type_to_id=dict(t2i, Zzz=999), faction_to_id=dict(f2i))
    with pytest.raises(RuntimeError, match="vocab"):
        check_preencoded(out, files, other, False)
    with pytest.raises(RuntimeError, match="no record"):
        check_preencoded(out, files + [tmp_path / "missing.json.gz"], enc, False)
    with pytest.raises(RuntimeError, match="manifest"):
        check_preencoded(tmp_path, files, enc, False)
