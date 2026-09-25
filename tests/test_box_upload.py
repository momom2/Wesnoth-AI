"""scripts/box/box_upload.py and box_stage.py against a stub of HfApi: no
network, no token. A round sends plain files smallest first, then
directories as tarballs, then held markers once their data has landed,
then upload.log, then (final only) ALL_DONE; one file's failure leaves the
others alone; an unchanged file is not sent again; a hung upload is
abandoned; exceptions are named by type only, whatever their message
holds; --clear and --restore handle the previous entry's state."""
from __future__ import annotations

import io
import os
import tarfile
import time
from pathlib import Path

import pytest

import scripts.box.box_stage as box_stage
import scripts.box.box_upload as box_upload

TOKEN = "hf_SENTINEL0token0never0written0anywhere"  # not-a-secret: a test sentinel
HF_DIR = "tier-b/upload_test"


class EntryNotFoundError(Exception):
    """Named as huggingface_hub names a file absent from the repo."""


class RemoteEntryNotFoundError(EntryNotFoundError):
    pass


class LocalEntryNotFoundError(EntryNotFoundError):
    """What huggingface_hub raises when the Hub cannot be reached and the
    file is not in the local cache."""


class StubHf:
    """Stands in for HfApi. `fail` maps a name to how many attempts fail
    before one lands (None: every one); `hang` names files whose upload
    never answers in time."""

    def __init__(self, fail=None, hang=(), hang_s=3.0, on_hf=None, raise_on_read=None):
        self.fail = dict(fail or {})
        self.hang, self.hang_s = set(hang), hang_s
        self.on_hf = dict(on_hf or {})          # path in repo -> bytes
        self.raise_on_read = raise_on_read
        self.sent: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id):
        name = path_in_repo.rsplit("/", 1)[1]
        if name in self.hang:
            time.sleep(self.hang_s)
        left = self.fail.get(name, 0)
        if left is None or left > 0:
            if left:
                self.fail[name] = left - 1
            raise ConnectionError(f"PUT https://huggingface.co/{repo_id}/{path_in_repo} "
                                  f"Authorization: Bearer {TOKEN}")
        data = Path(path_or_fileobj).read_bytes()
        self.sent.append((path_in_repo, data))
        self.on_hf[path_in_repo] = data

    def names(self) -> list[str]:
        return [path.rsplit("/", 1)[1] for path, _ in self.sent]

    def file_exists(self, repo_id, filename):
        if self.raise_on_read:
            raise self.raise_on_read
        return filename in self.on_hf

    def delete_file(self, path_in_repo, repo_id):
        self.deleted.append(path_in_repo)
        del self.on_hf[path_in_repo]

    def hf_hub_download(self, repo_id, filename):
        if self.raise_on_read:
            raise self.raise_on_read
        if filename not in self.on_hf:
            raise RemoteEntryNotFoundError(f"404 for {filename} with {TOKEN}")
        local = Path(os.environ["STUB_HF_CACHE"]) / filename.replace("/", "_")
        local.write_bytes(self.on_hf[filename])
        return str(local)


FAST = box_upload.Timing(base_s=2.0, min_rate=1e12, reserve_s=0.5, min_attempt_s=0.05,
                         retry_pause_s=0.0)


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A records directory holding a small file, a checkpoint, a marker that
    vouches for the checkpoint, a games directory (one game still being
    written) and ALL_DONE; an extra file lives outside it."""
    out = tmp_path / "out"
    write(out / "small.txt", b"s" * 10)
    write(out / "arm_epoch0.pt", b"c" * 5000)
    write(out / "DONE", b"stage=x\n")
    write(out / "ALL_DONE", b"RUN_DONE\n")
    write(out / "games_m" / "game_1.json", b"{}")
    write(out / "games_m" / "game_1.game.jsonl.gz", b"g" * 300)
    write(out / "games_m" / "game_2.game.jsonl.gz.tmp", b"half")
    extra = write(tmp_path / "onstart_script.log", b"o" * 100)
    spec = write(out / "tmp" / "upload_spec.tsv",
                 f"extra\t{extra}\ndir\tgames_m\t{out / 'games_m'}\nhold\tDONE\tarm_epoch0.pt\n".encode())
    monkeypatch.setenv("STUB_HF_CACHE", str(write(tmp_path / "cache" / ".keep", b"").parent))

    def make(api, final=False, timing=FAST, budget_s=60.0, spec_path=spec):
        rnd = box_upload.Round(out, HF_DIR, api, spec=box_upload.read_spec(str(spec_path)),
                               budget_s=budget_s, timing=timing)
        return rnd.run(final=final)

    make.out = out
    return make


def log_of(out: Path) -> str:
    return (out / "upload.log").read_text(encoding="utf-8")


def test_a_final_round_sends_in_order_and_all_done_last(run):
    api = StubHf()
    assert run(api, final=True) is True
    assert api.names() == ["small.txt", "onstart_script.log", "arm_epoch0.pt", "games_m.tar.gz",
                           "DONE", "upload.log", "ALL_DONE"]
    assert all(path.startswith(HF_DIR + "/") for path, _ in api.sent)
    tarball = dict(api.sent)[f"{HF_DIR}/games_m.tar.gz"]
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
        assert sorted(tf.getnames()) == ["games_m/game_1.game.jsonl.gz", "games_m/game_1.json"]


def test_a_periodic_round_leaves_all_done_on_the_box(run):
    api = StubHf()
    assert run(api) is True
    assert "ALL_DONE" not in api.names() and api.names()[-1] == "upload.log"


def test_one_failing_file_leaves_the_others_and_holds_its_marker(run):
    api = StubHf(fail={"arm_epoch0.pt": None})
    assert run(api, final=True) is False
    assert "arm_epoch0.pt" not in api.names() and "DONE" not in api.names()
    assert api.names()[-2:] == ["upload.log", "ALL_DONE"]
    assert {"small.txt", "onstart_script.log", "games_m.tar.gz"} <= set(api.names())
    log = log_of(run.out)
    assert "arm_epoch0.pt 5000 attempt 3 failed: ConnectionError" in log
    assert "DONE held back: arm_epoch0.pt has not landed" in log


def test_a_marker_goes_up_once_its_data_has_landed(run):
    api = StubHf(fail={"arm_epoch0.pt": 3})           # the first round's three attempts fail
    run(api)
    assert "DONE" not in api.names()
    run(api)                                          # the checkpoint lands, then its marker
    names = api.names()
    assert names.index("arm_epoch0.pt") < names.index("DONE")
    # A new version of the data that does not land holds the new marker.
    write(run.out / "arm_epoch0.pt", b"d" * 6000)
    write(run.out / "DONE", b"stage=y\n")
    api.fail["arm_epoch0.pt"] = None
    before = len(api.sent)
    run(api)
    assert "DONE" not in api.names()[before:]


def test_what_landed_unchanged_is_not_sent_again(run):
    api = StubHf()
    run(api)
    before = len(api.sent)
    run(api)
    assert api.names()[before:] == ["upload.log"]
    time.sleep(0.01)
    write(run.out / "small.txt", b"t" * 11)
    write(run.out / "games_m" / "game_3.json", b"{}")
    before = len(api.sent)
    run(api)
    assert api.names()[before:] == ["small.txt", "games_m.tar.gz", "upload.log"]


def test_no_error_message_reaches_the_log_or_the_console(run, capsys):
    api = StubHf(fail={"small.txt": None, "games_m.tar.gz": None})
    run(api, final=True)
    out, err = capsys.readouterr()
    log = log_of(run.out)
    assert "failed: ConnectionError" in log
    for shown in (log, out, err):
        assert TOKEN not in shown and "huggingface.co" not in shown


def test_a_hung_upload_is_abandoned_and_the_round_goes_on(run):
    api = StubHf(hang={"arm_epoch0.pt"}, hang_s=3.0)
    timing = box_upload.Timing(base_s=0.3, min_rate=1e12, reserve_s=0.1, min_attempt_s=0.05,
                               retry_pause_s=0.0)
    start = time.monotonic()
    assert run(api, final=True, timing=timing) is False
    assert time.monotonic() - start < 2.5
    assert "arm_epoch0.pt" not in api.names() and api.names()[-1] == "ALL_DONE"
    assert "arm_epoch0.pt 5000 attempt 1 failed: no answer" in log_of(run.out)


def test_a_spent_budget_still_sends_upload_log_and_all_done(run):
    api = StubHf(hang={"small.txt"}, hang_s=1.5)
    timing = box_upload.Timing(base_s=5.0, min_rate=1e12, reserve_s=1.0, min_attempt_s=0.05,
                               retry_pause_s=0.0)
    assert run(api, final=True, timing=timing, budget_s=1.5) is False
    assert api.names() == ["upload.log", "ALL_DONE"]
    assert "not sent: the round's time is spent" in log_of(run.out)


def test_clear_deletes_the_previous_entrys_markers_and_forgets_them(run):
    on_hf = {f"{HF_DIR}/ALL_DONE": b"old", f"{HF_DIR}/FAILED": b"old", f"{HF_DIR}/train.log": b"x"}
    api = StubHf(on_hf=on_hf)
    run(api)
    rnd = box_upload.Round(run.out, HF_DIR, api)
    assert rnd.clear(["ALL_DONE", "FAILED", "NOT_THERE"]) is True
    assert sorted(api.deleted) == [f"{HF_DIR}/ALL_DONE", f"{HF_DIR}/FAILED"]
    assert f"{HF_DIR}/train.log" in api.on_hf
    assert "DONE" in box_upload.load_state(run.out, HF_DIR)
    unreachable = box_upload.Round(run.out, HF_DIR, StubHf(raise_on_read=ConnectionError(TOKEN)))
    assert unreachable.clear(["ALL_DONE"]) is False
    assert "ALL_DONE not cleared from HF: ConnectionError" in log_of(run.out)
    assert TOKEN not in log_of(run.out)


def test_restore_fetches_what_is_absent_and_refuses_when_hf_cannot_answer(run):
    api = StubHf(on_hf={f"{HF_DIR}/train.log": b"line\n" * 3, f"{HF_DIR}/small.txt": b"remote"})
    rnd = box_upload.Round(run.out, HF_DIR, api, timing=FAST)
    assert rnd.restore(["train.log", "small.txt", "never_uploaded.json"]) is True
    assert (run.out / "train.log").read_bytes() == b"line\n" * 3
    assert (run.out / "small.txt").read_bytes() == b"s" * 10       # present here: kept
    assert not (run.out / "never_uploaded.json").exists()
    run(api)
    assert "train.log" not in api.names()                          # restored: landed already
    for unreachable in (TimeoutError(TOKEN), LocalEntryNotFoundError(TOKEN)):
        down = box_upload.Round(run.out, HF_DIR, StubHf(raise_on_read=unreachable), timing=FAST)
        assert down.restore(["arm.pt"]) is False
        assert f"arm.pt restore attempt 3 failed: {type(unreachable).__name__}" in log_of(run.out)
    assert TOKEN not in log_of(run.out)


def test_main_without_a_token_says_why_and_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert box_upload.main(["--out", str(tmp_path), "--hf-dir", HF_DIR]) == 1
    assert "HF_TOKEN is not set" in (tmp_path / "upload.log").read_text()


def test_main_names_an_unexpected_error_by_type_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    def broken(token):
        raise RuntimeError(f"cannot build a client for {token}")

    monkeypatch.setattr(box_upload, "make_api", broken)
    assert box_upload.main(["--out", str(tmp_path), "--hf-dir", HF_DIR, "--final"]) == 1
    log = (tmp_path / "upload.log").read_text()
    assert "stopped by an unexpected RuntimeError" in log and TOKEN not in log


def test_the_library_of_a_stage_sits_beside_it():
    assert box_stage.library_dir("tier-b/staging/stage_20260926a.tar.gz") == \
        "tier-b/staging/stage_20260926a.box"
    with pytest.raises(ValueError):
        box_stage.library_dir("tier-b/staging/stage_20260926a.zip")


def test_a_stage_member_that_would_land_outside_is_refused(tmp_path):
    tarball = tmp_path / "stage.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        for name, data in (("tools/ok.py", b"print(1)\n"), ("../escaped.py", b"print(2)\n")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    with pytest.raises(Exception):
        box_stage.extract(str(tarball), str(tmp_path / "into"))
    assert not (tmp_path / "escaped.py").exists()
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as tf:
        info = tarfile.TarInfo("tools/ok.py")
        info.size = 9
        tf.addfile(info, io.BytesIO(b"print(1)\n"))
    assert box_stage.extract(str(good), str(tmp_path / "fresh")) == 1
    with pytest.raises(FileExistsError):
        box_stage.extract(str(good), str(tmp_path / "fresh"))
