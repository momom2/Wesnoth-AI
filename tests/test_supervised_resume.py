"""A resumed imitation run continues the pass it was cut from
(tools/supervised_train.py, `PassPosition`): cut at K pairs and resumed,
in any epoch, it trains the same pairs in the same order, with the same
value-selection draws and at the same learning rate as the uncut run,
and with the generators the checkpoint saved, the same weights. A
checkpoint written before pass positions were saved continues the same
pass in a first run's first epoch.

Two harnesses. The scripted one drives `train()` over a scripted pair
stream, with recording stubs where the loss would be computed; it needs
no corpus, so it runs on CI. The corpus one runs the recipe's path
(imitation config, pre-encoded relevant-set records, batched flow,
holdout evaluations inside the pass) on four small games of the local
imitation corpus, and is skipped when the corpus is absent."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import tools.supervised_train as st  # noqa: E402
from tools.signal_telemetry import read_signal_rows  # noqa: E402
from tools.unit_vocab import seed_vocab  # noqa: E402
import wesnoth_ai.train_perf as train_perf  # noqa: E402

# Every run wraps these, not a previous run's wrapper.
REAL_FLUSH, REAL_SAVE, REAL_ADAMW = st._flush_batch, st._save_checkpoint, train_perf.adamw

# --- The scripted harness ---------------------------------------------------

# Pairs per file, and whether the file ends in an error instead of its
# file_done: g2 fails after yielding pairs, as the serial stream does, g4
# before yielding any, as the pre-encoded and parallel streams do.
SCRIPT = {"g0.json.gz": (5, False), "g1.json.gz": (9, False), "g2.json.gz": (3, True),
          "g3.json.gz": (6, False), "g4.json.gz": (0, True), "g5.json.gz": (7, False),
          "g6.json.gz": (4, False)}
LOST_BATCH = ("g5.json.gz", 2)    # the batched flush holding this pair raises
LOST_PAIR = ("g3.json.gz", 4)     # the per-pair loss of this pair raises
# The seed orders the third epoch so that both file errors and the lost
# batch or pair come before a cut at SCRIPT_CUT of its pairs, in either
# flow; the test asserts it.
SCRIPT_SEED = 56
SCRIPT_CUT = 24


def _scripted_corpus(tmp_path: Path) -> Path:
    """The scripted corpus's files (empty: the stream is scripted) and a
    value index that selects about half of each game's pairs."""
    ds = tmp_path / "scripted"
    ds.mkdir()
    for name in SCRIPT:
        (ds / name).write_bytes(b"")
    (ds / "value_corpus_index.jsonl").write_text(
        "".join(json.dumps({"file": name, "winner": 1, "n_commands": 32}) + "\n" for name in SCRIPT),
        encoding="utf-8")
    return ds


def _train_scripted(ds: Path, ckpt: Path, monkeypatch, *, batched: bool, failures: bool = True,
                    max_pairs: int = 0, resume=None) -> SimpleNamespace:
    """One three-epoch run over the scripted corpus in batches of 4.
    Returns the pairs in the order their loss was computed, as (pair,
    value draw, learning rate); the count of those at each epoch's
    snapshot; and the events of the pass (epoch starts, file errors, the
    lost batch or pair)."""
    run = SimpleNamespace(trained=[], epoch_ends={}, events=[])
    optimizers = []

    def adamw(*args, **kwargs):
        optimizers.append(REAL_ADAMW(*args, **kwargs))
        return optimizers[-1]

    def stream(files, _preencoded):
        run.events.append("epoch")
        for f in files:
            n, fails = SCRIPT[f.name]
            for i in range(n):
                state = SimpleNamespace(global_info=SimpleNamespace(current_side=1 + i % 2))
                yield ("pair", state, SimpleNamespace(key=(f.name, i), target_off_subset=False), f.name)
            if fails:
                run.events.append(f.name)
                yield ("file_error", f.name, "scripted")
            else:
                yield ("file_done", f.name, n)

    def record(ai, value_z):
        run.trained.append((ai.key, value_z, optimizers[-1].param_groups[0]["lr"]))

    def flush(model, encoder, raws, ais, zws, *args, **kwargs):
        if failures and any(ai.key == LOST_BATCH for ai in ais):
            run.events.append("lost batch")
            raise RuntimeError("scripted flush failure")
        for ai, (value_z, _value_w, _policy_w) in zip(ais, zws):
            record(ai, value_z)
        return 0

    def loss_parts(encoder, model, state, ai, device, *, value_z=None, **kwargs):
        if failures and ai.key == LOST_PAIR:
            run.events.append("lost pair")
            raise RuntimeError("scripted loss failure")
        record(ai, value_z)
        zero = sum(p.sum() for p in model.parameters()) * 0.0
        return st.LossParts(zero, zero, zero, zero, zero, actor_fired=False, type_fired=False,
                            target_fired=False, weapon_fired=False)

    def save(path, *args, **kwargs):
        REAL_SAVE(path, *args, **kwargs)
        if "_epoch" in path.stem:
            run.epoch_ends[int(path.stem.rsplit("_epoch", 1)[1])] = len(run.trained)

    monkeypatch.setattr(train_perf, "adamw", adamw)
    monkeypatch.setattr(st, "_pair_stream_preencoded", stream)
    monkeypatch.setattr(st, "check_preencoded", lambda *args: None)
    monkeypatch.setattr(st, "_raw_one", lambda encoder, state: state)
    monkeypatch.setattr(st, "_flush_batch", flush)
    monkeypatch.setattr(st, "_loss_parts_for_pair", loss_parts)
    monkeypatch.setattr(st, "_save_checkpoint", save)
    st.train(ds, ckpt, epochs=3, batch_size=4, max_pairs=max_pairs, resume=resume,
             seed=SCRIPT_SEED, d_model=16, num_layers=1, num_heads=2, d_ff=32, competitive_only=False,
             max_replay_commands=0, holdout_games=0, batched_forward=batched,
             preencoded=ds / "encoded", log_every=1000, ckpt_every=1000, signal_every=8)
    return run


def test_a_failed_file_costs_only_its_own_pairs(tmp_path, monkeypatch):
    """A batch spans files in the batched flow; a file that fails drops
    its own pairs still waiting in the batch, never those of the files
    before it."""
    ds = _scripted_corpus(tmp_path)
    run = _train_scripted(ds, tmp_path / "run.pt", monkeypatch, batched=True, failures=False)
    completed = [(name, i) for name, (n, fails) in SCRIPT.items() if not fails for i in range(n)]
    trained = [key for key, _z, _lr in run.trained if not SCRIPT[key[0]][1]]
    assert sorted(trained) == sorted(completed * 3)
    # The signal telemetry probed through train() and failed on the
    # scripted pairs, which carry no encoding: rows record the failure
    # and training went on.
    probes = [r for r in read_signal_rows(tmp_path / "run_signal.jsonl") if r["kind"] == "probe"]
    assert len(probes) >= 3 and all("probe_error" in r for r in probes)

@pytest.mark.parametrize("batched", [True, False], ids=["batched", "per_pair"])
def test_a_run_cut_in_its_last_epoch_resumes_on_the_uncut_pass(tmp_path, monkeypatch, batched):
    """Cut in the third epoch after the epoch's file failures and a lost
    batch (or pair), then resumed: the pairs, their value draws and the
    learning rate are the uncut run's. The same from the second epoch's
    snapshot."""
    ds = _scripted_corpus(tmp_path)
    full = _train_scripted(ds, tmp_path / "full.pt", monkeypatch, batched=batched)
    assert len({lr for *_, lr in full.trained}) == 3, "each epoch must train at its own rate"
    two_epochs = torch.load(tmp_path / "full_epoch1.pt", map_location="cpu",
                            weights_only=False)["supervised_pairs"]
    cut = _train_scripted(ds, tmp_path / "cut.pt", monkeypatch, batched=batched,
                          max_pairs=two_epochs + SCRIPT_CUT)
    last_epoch = cut.events[len(cut.events) - cut.events[::-1].index("epoch"):]
    lost = "lost batch" if batched else "lost pair"
    assert {"g2.json.gz", "g4.json.gz", lost} <= set(last_epoch), "the cut must follow what it tests"
    assert len(cut.trained) < len(full.trained), "the run must be cut"
    assert cut.trained == full.trained[:len(cut.trained)]
    # The checkpoint keeps where the telemetry last probed, so the resumed
    # run probes where the uncut one does.
    saved = torch.load(tmp_path / "cut.pt", map_location="cpu", weights_only=False)
    probes = [r for r in read_signal_rows(tmp_path / "cut_signal.jsonl") if r["kind"] == "probe"]
    assert saved["supervised_resume"]["signal"]["last_row_pairs"] == probes[-1]["pairs"]
    rest = _train_scripted(ds, tmp_path / "cut.pt", monkeypatch, batched=batched,
                           resume=tmp_path / "cut.pt")
    assert cut.trained + rest.trained == full.trained
    shutil.copyfile(tmp_path / "full_epoch1.pt", tmp_path / "snapshot.pt")
    last = _train_scripted(ds, tmp_path / "snapshot.pt", monkeypatch, batched=batched,
                           resume=tmp_path / "snapshot.pt")
    assert last.trained == full.trained[full.epoch_ends[1]:]


# --- The corpus harness -----------------------------------------------------

CORPUS = ROOT / "replays_dataset_imitation"
needs_corpus = pytest.mark.skipif(
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
    seed_vocab(enc)
    vocab = root / "vocab.pt"
    torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
                "faction_to_id": dict(enc.faction_to_id)}, vocab)
    out = root / "encoded"
    assert preencode(["--dataset", str(ds), "--out", str(out), "--vocab-from", str(vocab),
                      "--relevant-set-hexes", "--fog-hides-enemy-villages", "--terrain-multi-hot",
                      "--workers", "1"]) == 0
    return ds, out


def _train(corpus, ckpt: Path, monkeypatch, *, epochs=1, max_pairs=0, resume=None):
    """One run of the recipe's path on the small corpus; the pairs it
    trained, in order, as (action, value draw, policy weight, digest of
    the encoded position, learning rate)."""
    ds, encoded = corpus
    trained = []

    def flush(model, encoder, batch_raws, batch_ais, batch_zw, opt, *args, **kwargs):
        lr = opt.param_groups[0]["lr"]
        for raw, ai, zw in zip(batch_raws, batch_ais, batch_zw):
            digest = hashlib.sha1(raw.global_feats.tobytes() + raw.unit_feats.tobytes()).hexdigest()[:12]
            trained.append((ai.action_type, ai.actor_idx, ai.target_idx, ai.weapon_idx, tuple(zw),
                            digest, lr))
        return REAL_FLUSH(model, encoder, batch_raws, batch_ais, batch_zw, opt, *args, **kwargs)

    monkeypatch.setattr(st, "_flush_batch", flush)
    st.train(ds, ckpt, epochs=epochs, batch_size=8, max_pairs=max_pairs, resume=resume, seed=7,
             d_model=16, num_layers=1, num_heads=2, d_ff=32, competitive_only=False,
             max_replay_commands=0, batched_forward=True, preencoded=encoded,
             relevant_set_hexes=True, terrain_multi_hot=True, fog_hides_enemy_villages=True,
             imitation_config=ROOT / "configs" / "imitation.json", value_states_per_game=40,
             eval_every=40, eval_pairs=16, log_every=1000, ckpt_every=1000, signal_every=16,
             signal_probe_pairs=3)
    return trained


def _signal_rows(*paths: Path):
    """The signal telemetry's probe readings by trained-pair count; the
    step norms are left out, since a resumed run's first row counts only
    the steps it took itself."""
    rows = {}
    for path in paths:
        for row in read_signal_rows(path):
            if row["kind"] == "probe":
                rows[row["pairs"]] = {key: row.get(key) for key in (
                    "probe_pairs", "fired", "gradient", "gradient_gram", "update", "update_gram")}
    return rows


def _weights(ckpt: Path):
    return torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]


@needs_corpus
def test_a_cut_and_resumed_run_trains_the_uncut_pass(corpus, tmp_path, monkeypatch, caplog):
    full = _train(corpus, tmp_path / "full.pt", monkeypatch)
    assert len(full) > 2 * CUT
    assert any(zw[0] is not None for *_, zw, _digest, _lr in full), \
        "no value state drawn: the replay is untested"
    cut = _train(corpus, tmp_path / "cut.pt", monkeypatch, max_pairs=CUT)
    assert cut == full[:CUT]
    shutil.copyfile(tmp_path / "cut.pt", tmp_path / "resumed.pt")
    with caplog.at_level("INFO", logger="supervised_train"):
        rest = _train(corpus, tmp_path / "resumed.pt", monkeypatch, resume=tmp_path / "resumed.pt")
    assert cut + rest == full
    assert "the replayed draws land on the checkpoint's state" in caplog.text
    a, b = _weights(tmp_path / "full.pt"), _weights(tmp_path / "resumed.pt")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    # The signal telemetry probes the same pairs at the same weights.
    uncut = _signal_rows(tmp_path / "full_signal.jsonl")
    assert len(uncut) >= 3 and min(uncut) < CUT < max(uncut)
    assert _signal_rows(tmp_path / "cut_signal.jsonl", tmp_path / "resumed_signal.jsonl") == uncut


@needs_corpus
def test_a_run_cut_in_its_second_epoch_trains_the_uncut_pass(corpus, tmp_path, monkeypatch, caplog):
    """The second epoch's order and learning rate are the uncut run's, and
    the weights end bit-identical."""
    full = _train(corpus, tmp_path / "full.pt", monkeypatch, epochs=2)
    first_epoch = torch.load(tmp_path / "full_epoch0.pt", map_location="cpu",
                             weights_only=False)["supervised_pairs"]
    assert len({lr for *_, lr in full}) == 2, "each epoch must train at its own rate"
    cut = _train(corpus, tmp_path / "cut.pt", monkeypatch, epochs=2, max_pairs=first_epoch + CUT)
    assert cut == full[:first_epoch + CUT]
    shutil.copyfile(tmp_path / "cut.pt", tmp_path / "resumed.pt")
    with caplog.at_level("INFO", logger="supervised_train"):
        rest = _train(corpus, tmp_path / "resumed.pt", monkeypatch, epochs=2,
                      resume=tmp_path / "resumed.pt")
    assert cut + rest == full
    assert "the replayed draws land on the checkpoint's state" in caplog.text
    a, b = _weights(tmp_path / "full.pt"), _weights(tmp_path / "resumed.pt")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@needs_corpus
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
