"""A resumed imitation run continues the pass it was cut from
(tools/supervised_train.py, `PassPosition`): cut at K pairs and
resumed, it trains the same pairs in the same order with the same
value-selection draws as the uncut run, and with the generators the
checkpoint saved, the same weights. A checkpoint written before pass
positions were saved continues the same pass in a first run's first
epoch. Runs the recipe's path (imitation config, pre-encoded
relevant-set records, batched flow, holdout evaluations inside the
pass) on four small games of the local imitation corpus; skipped when
the corpus is absent."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import tools.supervised_train as st  # noqa: E402

CORPUS = ROOT / "replays_dataset_imitation"
REAL_FLUSH = st._flush_batch     # every run wraps this one, not the previous run's wrapper

pytestmark = pytest.mark.skipif(
    not (CORPUS / "manifest.jsonl").exists() or not (CORPUS / "value_corpus_index.jsonl").exists(),
    reason="local imitation corpus absent")

CUT = 24          # pairs before the cut: three batches of 8


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Five small games (one of them held out), their value index, the
    unit stats the trainer seeds its vocab from, and their pre-encoded
    records in the recipe's basis."""
    from tools.preencode_corpus import main as preencode
    from wesnoth_ai.encoder import GameStateEncoder
    root = tmp_path_factory.mktemp("resume")
    ds = root / "dataset"
    ds.mkdir()
    rows = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8")]
    small = sorted((r for r in rows if not r["holdout"] and r["winner_actions"] > 0),
                   key=lambda r: r["n_commands"])[:5]
    small[-1] = dict(small[-1], holdout=True)
    index = {json.loads(line)["file"]: line
             for line in (CORPUS / "value_corpus_index.jsonl").open(encoding="utf-8")}
    for r in small:
        shutil.copyfile(CORPUS / r["file"], ds / r["file"])
    (ds / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in small), encoding="utf-8")
    (ds / "value_corpus_index.jsonl").write_text(
        "".join(index[r["file"]] if index[r["file"]].endswith("\n") else index[r["file"]] + "\n"
                for r in small if r["file"] in index), encoding="utf-8")
    shutil.copyfile(ROOT / "unit_stats.json", root / "unit_stats.json")
    enc = GameStateEncoder(d_model=32)
    st._seed_vocab_from_unit_stats(enc, root / "unit_stats.json")
    vocab = root / "vocab.pt"
    torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
                "faction_to_id": dict(enc.faction_to_id)}, vocab)
    out = root / "encoded"
    assert preencode(["--dataset", str(ds), "--out", str(out), "--vocab-from", str(vocab),
                      "--relevant-set-hexes", "--fog-hides-enemy-villages", "--terrain-multi-hot",
                      "--workers", "1"]) == 0
    return ds, out


def _train(corpus, ckpt: Path, monkeypatch, *, max_pairs=0, resume=None):
    """One run of the recipe's path on the small corpus; the pairs it
    trained, in order, as (action, value draw, policy weight, digest of
    the encoded position)."""
    ds, encoded = corpus
    trained = []

    def flush(model, encoder, batch_raws, batch_ais, batch_zw, *args, **kwargs):
        for raw, ai, zw in zip(batch_raws, batch_ais, batch_zw):
            digest = hashlib.sha1(raw.global_feats.tobytes() + raw.unit_feats.tobytes()).hexdigest()[:12]
            trained.append((ai.action_type, ai.actor_idx, ai.target_idx, ai.weapon_idx, tuple(zw), digest))
        return REAL_FLUSH(model, encoder, batch_raws, batch_ais, batch_zw, *args, **kwargs)

    monkeypatch.setattr(st, "_flush_batch", flush)
    st.train(ds, ckpt, epochs=1, batch_size=8, max_pairs=max_pairs, resume=resume, seed=7,
             d_model=16, num_layers=1, num_heads=2, d_ff=32, competitive_only=False,
             max_replay_commands=0, batched_forward=True, preencoded=encoded,
             relevant_set_hexes=True, terrain_multi_hot=True, fog_hides_enemy_villages=True,
             imitation_config=ROOT / "configs" / "imitation.json", value_states_per_game=40,
             eval_every=40, eval_pairs=16, log_every=1000, ckpt_every=1000)
    return trained


def _weights(ckpt: Path):
    return torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]


def test_a_cut_and_resumed_run_trains_the_uncut_pass(corpus, tmp_path, monkeypatch, caplog):
    full = _train(corpus, tmp_path / "full.pt", monkeypatch)
    assert len(full) > 2 * CUT
    assert any(zw[0] is not None for *_, zw, _d in full), "no value state drawn: the replay is untested"
    cut = _train(corpus, tmp_path / "cut.pt", monkeypatch, max_pairs=CUT)
    assert cut == full[:CUT]
    shutil.copyfile(tmp_path / "cut.pt", tmp_path / "resumed.pt")
    with caplog.at_level("INFO", logger="supervised_train"):
        rest = _train(corpus, tmp_path / "resumed.pt", monkeypatch, resume=tmp_path / "resumed.pt")
    assert cut + rest == full
    assert "the replayed draws land on the checkpoint's state" in caplog.text
    a, b = _weights(tmp_path / "full.pt"), _weights(tmp_path / "resumed.pt")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_checkpoint_without_a_saved_position_continues_a_first_runs_first_epoch(
        corpus, tmp_path, monkeypatch):
    """The box's case (2026-09-24): a checkpoint written before positions
    were saved, cut inside the first epoch of a first run."""
    full = _train(corpus, tmp_path / "full.pt", monkeypatch)
    cut = _train(corpus, tmp_path / "cut.pt", monkeypatch, max_pairs=CUT)
    old = torch.load(tmp_path / "cut.pt", map_location="cpu", weights_only=False)
    old.pop("supervised_resume", None)
    torch.save(old, tmp_path / "old.pt")
    rest = _train(corpus, tmp_path / "old.pt", monkeypatch, resume=tmp_path / "old.pt")
    assert cut + rest == full
