"""tools/stage_code.py: every payload carries the box library, and --upload
writes the stage, the library beside it and the run script in one commit,
the last two read from the tarball; a shell file with CR line endings is
refused. No test talks to HF: the commit goes to a stub."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

import tools.stage_code as stage_code
from scripts.box.box_stage import LIBRARY_FILES

DEST = "tier-b/staging/stage_t.tar.gz"


class StubApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, operations, *, commit_message):
        self.commits.append((repo_id, list(operations), commit_message))


def tarball_of(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "stage_t.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def stage_members() -> dict[str, bytes]:
    members = {f"scripts/box/{name}": f"the stage's {name}\n".encode() for name in LIBRARY_FILES}
    members["scripts/demo_box.sh"] = b"#!/usr/bin/env bash\necho demo\n"
    members["tools/other.py"] = b"x = 1\n"
    return members


def test_one_commit_holds_the_stage_its_library_and_the_script(tmp_path):
    pytest.importorskip("huggingface_hub")
    members = stage_members()
    tarball = tarball_of(tmp_path, members)
    api = StubApi()
    stage_code.upload_stage(api, tarball, DEST, "scripts/demo_box.sh")
    ((repo, operations, _message),) = api.commits
    assert repo == stage_code.REPO
    sent = {op.path_in_repo: op.path_or_fileobj for op in operations}
    assert sent.pop(DEST) == str(tarball)
    assert sent.pop("tier-b/staging/demo_box.sh") == members["scripts/demo_box.sh"]
    assert sent == {f"tier-b/staging/stage_t.box/{name}": members[f"scripts/box/{name}"]
                    for name in LIBRARY_FILES}


@pytest.mark.parametrize("change, reason", [
    ({"scripts/demo_box.sh": b"#!/usr/bin/env bash\r\necho demo\r\n"}, "CR line endings"),
    ({"scripts/box/boxlib.sh": b"# shellcheck shell=bash\r\n"}, "CR line endings"),
    ({"scripts/box/box_stop.py": None}, "scripts/box/box_stop.py is not in the payload"),
    ({"scripts/demo_box.sh": None}, "scripts/demo_box.sh is not in the payload"),
], ids=["crlf script", "crlf library", "library file missing", "script missing"])
def test_a_stage_that_would_break_on_the_box_is_not_uploaded(tmp_path, change, reason):
    members = stage_members()
    for name, data in change.items():
        if data is None:
            del members[name]
        else:
            members[name] = data
    api = StubApi()
    with pytest.raises(SystemExit, match=reason):
        stage_code.upload_stage(api, tarball_of(tmp_path, members), DEST, "scripts/demo_box.sh")
    assert api.commits == []


def test_a_payload_without_the_box_library_is_refused():
    with pytest.raises(SystemExit, match="scripts/box/boxlib.sh"):
        stage_code.main(["--dry-run", "--exclude-prefix", "scripts/box/"])
