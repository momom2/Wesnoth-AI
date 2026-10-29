"""scripts/stage1_local_upload.py on a synthetic local stage 1 (two self-play
matches of five games, the second's last two held out, D0 and the memory
report), against a stub of HfApi: no network, no token.

The dry run lists every name with its size and destination and writes and
sends nothing; an incomplete or foreign run is refused before anything is
sent; the upload sends the box stage 1's names (plan.txt written, each games
directory a tarball under its own name, the server stats copied out, the
markers after their tarballs, ALL_DONE last) into a folder of its own; and
scripts/stage_c_box.sh's own restore reads the uploaded folder into the
directories its stages run on."""
from __future__ import annotations

import gzip
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.stage1_local_upload as up  # noqa: E402
from scripts.box import box_upload  # noqa: E402

PLAN = up.Plan(n_selfplay=10, chunk_games=5, heldout_from=8, seed_base=800000)
SHA = up.reference_sha()
HF_DIR = "tier-b/selfplay_t1_local_test"


class EntryNotFoundError(Exception):
    """Named as huggingface_hub names a file absent from the repo."""


class StubHf:
    """HfApi's calls the upload and the restore make, on a directory: each
    upload lands at root/<path in repo>, in the order of `sent`."""

    def __init__(self, root: Path, on_hf=()):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.sent: list[str] = []
        for name in on_hf:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("someone else's")

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id):
        assert repo_id == box_upload.REPO
        dst = self.root / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(path_or_fileobj).read_bytes())
        self.sent.append(path_in_repo.rsplit("/", 1)[1])

    def list_repo_files(self, repo_id):
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def hf_hub_download(self, repo_id, filename):
        p = self.root / filename
        if not p.is_file():
            raise EntryNotFoundError(filename)
        return str(p)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _result(seed: int, **over) -> dict:
    r = {"label_a": "own2a", "label_b": "own2b", "procedure_a": "raw:t1", "procedure_b": "raw:t1",
         "memory_a": 64, "memory_b": 64, "checkpoint_sha256_a": SHA, "checkpoint_sha256_b": SHA,
         "forced_faction": "none", "decisions_logged": True, "infer_bf16": False, "infer_fp16": True,
         "infer_packed_trunk": True, "code_version": "0.31.14", "side_a": 1, "seed": seed, "outcome_a": "win"}
    r.update(over)
    return r


def _games(d: Path, seeds, **over) -> None:
    for seed in seeds:
        stem = f"game_own2a_own2b_s1_{seed}"
        _write(d / f"{stem}.json", json.dumps(_result(seed, **over)))
        (d / f"{stem}.game.jsonl.gz").write_bytes(gzip.compress(json.dumps({"seed": seed}).encode()))
    _write(d / ".inference_server_0.json", '{"infer_fp16": true}')
    _write(d / ".worker_stages.json", "{}")
    _write(d / ".worker_1.log", "a worker's log")
    _write(d / "half_written.json.tmp", "partial")


def _run_dir(root: Path) -> Path:
    """A whole local stage 1 of PLAN, as scripts/selfplay_t1_local.py leaves it."""
    run = root / "selfplay_t1_local"
    rows, labels = [], []
    for k, name in enumerate(PLAN.chunks()):
        seeds = range(PLAN.seed_base + k * PLAN.chunk_games, PLAN.seed_base + (k + 1) * PLAN.chunk_games)
        _games(run / f"games_{name}", seeds)
        walk, chunk_rows = [], []
        for seed in seeds:
            game, index = f"game_own2a_own2b_s1_{seed}", seed - PLAN.seed_base
            walk.append(json.dumps({"game": game, "seed": seed}))
            for side in (1, 2):
                z = 1.0 if side == 1 else -1.0
                chunk_rows.append(json.dumps({"game": game, "index": index, "side": side, "z": z}))
                held = index >= PLAN.heldout_from
                labels.append(json.dumps({"game": game, "side": side, "split": index % 8, "heldout": held,
                                          "half": ("A" if index % 2 else "B") if held else None, "z": z,
                                          "capped": False, "cell": "m|1|a|b", "S": [0.0] * 7, "b": 0.0,
                                          "beta_S": 0.0, "R": z}))
        _write(run / f"walk_{name}.jsonl", "\n".join(walk) + "\n")
        _write(run / f"rows_{name}.jsonl", "\n".join(chunk_rows) + "\n")
        rows += chunk_rows
        _write(run / f"chunk_{name}.done", "2026-10-29 12:00:00\n")
        _write(run / f"wall_{name}.txt", "100\n")
    _write(run / "rows.jsonl", "\n".join(rows) + "\n")
    _write(run / "labels.jsonl", "\n".join(labels) + "\n")
    _write(run / "fits.json", '{"b": {}}')
    _write(run / "readout.json", json.dumps({k: {"pass": True} for k in up.runner.GATES}))
    _write(run / "readout.done", "2026-10-29 12:00:00\n")
    for name, base in (("d0", 830000), ("memory", 845000)):
        _games(run / f"games_{name}", range(base, base + 3), decisions_logged=False)
        _write(run / f"{name}.fit.json", '{"tables": {}}')
        _write(run / f"measure_{name}.done", "2026-10-29 12:00:00\n")
    _write(run / "matches.txt", "s00: games 5\n")
    _write(run / "runner.log", "stage 1 local\n")
    _write(run / "ALL_DONE", "2026-10-29 20:00:00\n")
    (run / "notes").mkdir()
    return run


def _no_api():
    raise AssertionError("a dry run or a refused upload never builds the client")


def _main(run: Path, *extra, api=_no_api, staging=None) -> int:
    staging = staging or run.with_name("staging")
    return up.main(["--run", str(run), "--hf-dir", HF_DIR, "--staging", str(staging), *extra],
                   api_factory=api, plan=PLAN)


EXPECTED = {"plan.txt", "local_provenance.json", "rows.jsonl", "labels.jsonl", "fits.json", "readout.json",
            "readout.done", "d0.fit.json", "memory.fit.json", "matches.txt", "runner.log",
            "games_d0.tar.gz", "games_memory.tar.gz", "measure_d0.done", "measure_memory.done", "ALL_DONE",
            "d0.inference_server_0.json", "d0.worker_stages.json", "memory.inference_server_0.json",
            "memory.worker_stages.json"} | {
    f"{kind}_{n}{ext}" for n in ("s00", "s01")
    for kind, ext in (("walk", ".jsonl"), ("rows", ".jsonl"), ("chunk", ".done"), ("wall", ".txt"),
                      ("games", ".tar.gz"))} | {
    f"{n}.{f}" for n in ("s00", "s01") for f in ("inference_server_0.json", "worker_stages.json")}


def test_the_dry_run_lists_every_name_and_sends_nothing(tmp_path, capsys):
    """The default: every name with its size and its destination, plan.txt's
    line, the census, READY, and the stage C settings; the run directory is
    only read, and no staging directory or client comes into being."""
    run = _run_dir(tmp_path)
    before = sorted((p.relative_to(run).as_posix(), p.stat().st_mtime_ns) for p in run.rglob("*"))
    assert _main(run) == 0
    out = capsys.readouterr().out
    plan = up.plan_upload(run, PLAN, SHA)
    assert {e.name for e in plan.entries} == EXPECTED and not plan.blockers
    for name in EXPECTED:
        assert f"  {name} " in out, name
    assert f"destination: {box_upload.REPO}:{HF_DIR}/" in out
    assert "plan.txt" in out and "written at upload: 10 8 unknown" in out
    assert "in 13 files" in out and "in 9 files" in out     # 5 (or 3) results and records, 3 hidden, no .tmp
    assert "READY: every check passed" in out
    assert "STAGE1_HF_DIR=tier-b/selfplay_t1_local_test SP_SEED_BASE=800000 CHUNK_GAMES=5" in out
    assert "notes/ (a directory the box does not send)" in out
    assert "dry run: nothing written, nothing sent" in out
    assert _main(run, "--dry-run") == 0
    with pytest.raises(SystemExit):
        _main(run, "--dry-run", "--upload")
    assert not (tmp_path / "staging").exists()
    assert sorted((p.relative_to(run).as_posix(), p.stat().st_mtime_ns) for p in run.rglob("*")) == before


@pytest.mark.parametrize("damage, blocker", [
    (lambda run: (run / "ALL_DONE").unlink(), "ALL_DONE: the runner has not ended"),
    (lambda run: _write(run / "STOP", "x"), "STOP: the run was asked to stop"),
    (lambda run: (run / "walk_s01.jsonl").unlink(), "walk_s01.jsonl: missing"),
    (lambda run: (run / "readout.done").unlink(), "readout.done: missing"),
    (lambda run: _write(run / "readout.json", json.dumps({k: {"pass": k != "dice_control"}
                                                          for k in up.runner.GATES})), "dice_control=FAIL"),
    (lambda run: _games(run / "games_s01", [800007], infer_fp16=False, infer_bf16=True),
     "1 results with served in bfloat16"),
    (lambda run: _games(run / "games_s00", [800001], memory_b=0), "1 results with memory [64, 0]"),
    (lambda run: _games(run / "games_s00", [800002], checkpoint_sha256_b="0" * 64), "1 results with checkpoint"),
    (lambda run: _write(run / "labels.jsonl", (run / "labels.jsonl").read_text().replace(
        '"heldout": true', '"heldout": false')), "held-out flag is not the plan's"),
])
def test_an_incomplete_or_foreign_run_is_refused_before_anything_is_sent(tmp_path, capsys, damage, blocker):
    run = _run_dir(tmp_path)
    damage(run)
    assert _main(run) == 0                                    # the dry run still lists
    assert "NOT READY" in capsys.readouterr().out
    assert _main(run, "--upload") == 2                        # and the upload sends nothing
    out = capsys.readouterr().out
    assert blocker in out and "REFUSED: nothing sent" in out
    assert not (tmp_path / "staging").exists()


def test_the_box_stage_1s_folder_and_a_staging_inside_the_run_are_refused(tmp_path):
    run = _run_dir(tmp_path)
    box = up.box_stage1_hf_dir()
    assert box == "tier-b/selfplay_t1_20261022"
    with pytest.raises(SystemExit):
        up.main(["--run", str(run), "--hf-dir", box], api_factory=_no_api, plan=PLAN)
    with pytest.raises(SystemExit):
        up.main(["--run", str(run), "--staging", str(run / "up")], api_factory=_no_api, plan=PLAN)


def test_a_folder_another_upload_filled_is_refused(tmp_path, capsys):
    run = _run_dir(tmp_path)
    hf = StubHf(tmp_path / "hf", on_hf=[f"{HF_DIR}/plan.txt"])
    assert _main(run, "--upload", api=lambda: hf) == 2
    assert "already holds 1 files this staging directory did not send (plan.txt)" in capsys.readouterr().out
    assert hf.sent == []


def _upload(tmp_path: Path):
    run = _run_dir(tmp_path)
    hf = StubHf(tmp_path / "hf")
    assert _main(run, "--upload", api=lambda: hf) == 0
    return run, hf


def test_the_upload_sends_the_box_layout_markers_after_their_data_and_all_done_last(tmp_path, capsys):
    run, hf = _upload(tmp_path)
    folder = hf.root / HF_DIR
    assert {p.name for p in folder.iterdir()} == EXPECTED | {box_upload.LOG_NAME}
    assert hf.sent[-1] == "ALL_DONE" and hf.sent.count("ALL_DONE") == 1
    for marker, data in (("chunk_s00.done", "games_s00.tar.gz"), ("chunk_s01.done", "games_s01.tar.gz"),
                         ("measure_d0.done", "games_d0.tar.gz")):
        assert hf.sent.index(data) < hf.sent.index(marker)
    assert (folder / "plan.txt").read_text() == "10 8 unknown\n"
    with tarfile.open(folder / "games_s00.tar.gz") as tf:
        names = sorted(tf.getnames())
    want = sorted(f"games_s00/{p.relative_to(run / 'games_s00').as_posix()}"
                  for p in (run / "games_s00").iterdir() if not p.name.endswith(".tmp"))
    assert names == want and "games_s00/.inference_server_0.json" in names
    provenance = json.loads((folder / "local_provenance.json").read_text())
    assert provenance["census"]["games_s00"]["kinds"][0]["served"] == "float16"
    assert (folder / "s01.inference_server_0.json").read_text() == '{"infer_fp16": true}'
    assert "UPLOADED" in capsys.readouterr().out
    # Again: nothing changed, so only the logs and ALL_DONE go up again.
    hf.sent.clear()
    assert _main(run, "--upload", api=lambda: hf) == 0
    assert sorted(hf.sent) == sorted([box_upload.LOG_NAME, box_upload.LOG_NAME, "ALL_DONE"])


def test_stage_cs_restore_reads_the_uploaded_folder(tmp_path):
    """box_upload.py's own --restore of the names scripts/stage_c_box.sh asks
    for, then its extraction: plan.txt names the plan, and each games
    directory comes back whole under its own name."""
    run, hf = _upload(tmp_path)
    s1 = tmp_path / "stage1"
    s1.mkdir()
    restore = box_upload.Round(s1, HF_DIR, hf)
    names = ["plan.txt", "labels.jsonl", "fits.json", "readout.json"]
    assert restore.restore(names)
    n, heldout_from, _ = (s1 / "plan.txt").read_text().split()
    chunks = [f"s{k:02d}" for k in range((int(n) + PLAN.chunk_games - 1) // PLAN.chunk_games)]
    assert chunks == PLAN.chunks() and int(heldout_from) == PLAN.heldout_from
    assert restore.restore([f"walk_{c}.jsonl" for c in chunks] + [f"games_{c}.tar.gz" for c in chunks])
    for c in chunks:
        with tarfile.open(s1 / f"games_{c}.tar.gz") as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(s1 / f".extract_{c}", filter="data")
            else:                                             # a Python before the extraction filters
                tf.extractall(s1 / f".extract_{c}")
        assert (s1 / f".extract_{c}" / f"games_{c}").is_dir()
        got = {p.name: p.read_bytes() for p in (s1 / f".extract_{c}" / f"games_{c}").iterdir()}
        assert got == {p.name: p.read_bytes() for p in (run / f"games_{c}").iterdir() if not p.name.endswith(".tmp")}
        assert (s1 / f"walk_{c}.jsonl").read_bytes() == (run / f"walk_{c}.jsonl").read_bytes()


RESTORE_FROM_HF = r'''
import os as _os, pathlib as _pathlib, shutil as _shutil, sys as _sys
if _sys.argv[1] == "restore stage1":
    _a = _sys.argv[2:]
    _out = _pathlib.Path(_a[_a.index("--out") + 1])
    _hf = _pathlib.Path(_os.environ["STUB_HF"]) / _a[_a.index("--hf-dir") + 1]
    for _n in _a[_a.index("--restore") + 1:]:
        if (_hf / _n).is_file() and not (_out / _n).exists():
            _shutil.copyfile(_hf / _n, _out / _n)
    _sys.exit(0)
'''


@pytest.mark.slow
def test_stage_c_box_runs_on_the_uploaded_folder(tmp_path, monkeypatch):
    """scripts/stage_c_box.sh's restore_stage1, by bash with the box library
    stubbed (tests/test_stage_c_box.py's harness), on the uploaded folder
    with STAGE1_HF_DIR set to it: plan.txt's games give the matches, their
    tarballs extract, the walks join, the readout's gates pass, and every
    stage runs on the restored games with the plan's held-out split."""
    import test_stage_c_box as harness
    from helpers.posix_bash import bash_path
    if harness.BASH is None:
        pytest.skip("no POSIX bash")
    run, hf = _upload(tmp_path / "up")
    monkeypatch.setattr(harness, "FAKE", RESTORE_FROM_HF + harness.FAKE)
    got = harness._run(tmp_path, STAGE1_HF_DIR=HF_DIR, STUB_HF=str(hf.root), CHUNK_GAMES=PLAN.chunk_games)
    assert got["finish"].startswith("STAGE_C_DONE"), (got["finish"], got["stderr"][-3000:])
    s1 = tmp_path / "stage1"
    for c in PLAN.chunks():
        assert sorted(p.name for p in (s1 / f"games_{c}").iterdir()) == \
            sorted(p.name for p in (run / f"games_{c}").iterdir() if not p.name.endswith(".tmp"))
    dirs = " ".join(f"{bash_path(s1)}/games_{c}" for c in PLAN.chunks())
    assert f"--games {dirs} --labels {bash_path(s1)}/labels.jsonl --train-limit 200" in harness._step(got, "c0 pass")
    assert f"--heldout-from {PLAN.heldout_from} " in harness._step(got, "plant labels")
    assert "--seed-base 800000 " in harness._step(got, "plant delta")
    assert (s1 / "walk.jsonl").read_text().count("\n") == PLAN.n_selfplay
