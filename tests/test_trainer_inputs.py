"""scripts/trainer_inputs.sh through the input cache (scripts/box_inputs.sh,
tools/input_cache.py), run by bash with the box library stubbed and the
model host stood in by a local directory: a first box builds every input
and sends each to the cache; a second box restores them all and builds
nothing; a corrupted entry is refused, its input built again and the entry
replaced; a legacy tarball of own sequences is adopted, checked against the
key, and sent as the entry, while by default the corpus's sequences are
pre-encoded and kept out of the cache. The corpus build and the own sequences' builders
are stubbed; the pre-encoding, the corpus's columns and the cache run for
real, so the sequences' key and the pre-encoder's records must agree."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]
RAW_TAR = "tier-b/test/raw.tar"
SOURCE_TAR = "tier-b/test/games_m1.tar.gz"
BUILDERS = {"raw", "corpus", "corpus-check", "preencode", "candidates", "targets"}

# The library's functions trainer_inputs calls; box_bounded runs each step,
# the stubbed builders through FAKE.
STUBS = r"""
BOX_ASYNC_PIDS=()
box_marked_this_stage() { [ -f "$1" ] && [ "$(sed -n 's/^stage=\([^ ]*\).*/\1/p' "$1" | head -1)" = "$STAGE" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_gpu_ok() { :; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
box_log() { printf 'stamp %s\n' "$2" >> "$OUT/$1"; }
box_size() { stat -c %s "$1" 2>/dev/null || echo 0; }
box_bounded() {                  # box_bounded [--stall FILE MIN] NAME CUT_MIN LOG CMD...
    local name log
    while [ "${1#--}" != "$1" ]; do shift 3; done
    name=$1 log=$3
    shift 3
    echo "$name|$*" >> "$OUT/steps.txt"
    BOX_RC=0 BOX_WHY=ok
    case $name in
        corpus|corpus-check|candidates|targets|"columns own_sequences_r2")
            python "$FAKE" "$name" "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
        *) "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
    esac
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""

# The stubbed builders: the corpus is the trainer tests' three synthetic
# games at this code's corpus version; the own sequences are the own
# game-sides the sources hold, with the fields the own builders write.
FAKE = r'''
import json, shutil, sys, tempfile
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "tests")]       # run from BOX_REPO
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
if name == "corpus":
    from helpers.sequence_pass import synthetic_dataset
    from tools.build_imitation_dataset import CORPUS_VERSION
    built = synthetic_dataset(Path(tempfile.mkdtemp()))
    out = Path(opt("--out"))
    shutil.copytree(built, out)
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    (out / "manifest.jsonl").write_text("".join(json.dumps({**r, "corpus_version": CORPUS_VERSION}) + "\n"
                                                for r in rows))
    print("BUILD_DONE")
elif name == "corpus-check":
    Path(opt("--check")).write_text(json.dumps({"candidates": 3, "games": 3, "problems": []}))
elif name == "candidates":
    from tools.own_trajectories import select_own_sides
    dirs = [Path(a) for a in args[args.index("--games") + 1:args.index("--teacher")]]
    names = sorted(select_own_sides(dirs, opt("--player-sha256"), int(opt("--slots")))[0])
    Path(opt("--out")).mkdir(parents=True)
    (Path(opt("--out")) / "names.json").write_text(json.dumps({"names": names, "player": opt("--player-sha256")}))
    print("CANDIDATES_DONE")
elif name == "targets":
    import wesnoth_core
    from tools.eval_provenance import file_sha256
    from tools.preencode_corpus import vocab_from_checkpoint
    from tools.preencode_sequences import encoding_fingerprint
    from wesnoth_ai.own_loss import TOP_ACTIONS
    from wesnoth_ai.rules_version import rules_version_of
    cand = json.loads((Path(opt("--candidates")) / "names.json").read_text())
    teacher, out = Path(opt("--teacher")), Path(opt("--out"))
    out.mkdir(parents=True)
    offset = json.loads(Path("configs/reference_player.json").read_text())["decode"]["raw_end_turn_offset"]
    fp = encoding_fingerprint(*vocab_from_checkpoint(teacher), 0, rules_version_of(wesnoth_core.__phase__))
    games = {n: {"positions_side1": 3, "holdout": False, "n_commands": 5, "winner": 1} for n in cand["names"]}
    for n in games:
        (out / f"{n}.seq.zpkl").write_bytes(b"own records")
    (out / "sequence_manifest.json").write_text(json.dumps({
        "source": "own", "fingerprint": fp, "teacher_sha256": file_sha256(teacher), "player_sha256": cand["player"],
        "slots": 64, "top": TOP_ACTIONS, "reference_offset": offset, "n_games": len(games), "errors": {},
        "games": games}))
    print("OWN_TARGETS_DONE")
elif name == "columns own_sequences_r2":
    from wesnoth_ai.sequence_columns import COLUMNS_FORMAT
    d = Path(opt("--sequences"))
    man = json.loads((d / "sequence_manifest.json").read_text())
    for n in man["games"]:
        (d / f"{n}.seq.cols").write_bytes(b"own columns")
    (d / "columns_manifest.json").write_text(json.dumps({"format": COLUMNS_FORMAT, "fingerprint": man["fingerprint"],
                                                          "games": {n: 3 for n in man["games"]}}))
    print("COLUMNS_DONE")
'''


def _host_inputs(host: Path, work: Path) -> str:
    """The raw tarball, the source games and own1's stand-in on the host
    and the box; the stand-in's SHA-256."""
    import torch
    from tools.input_cache_kinds import file_sha256
    from tools.preencode_sequences import fresh_vocab
    (host / "tier-b" / "test").mkdir(parents=True)
    raw = work / "raw_src"
    (raw / "training" / "logs").mkdir(parents=True)
    (raw / "training" / "logs" / "replay_dispositions.jsonl.gz").write_bytes(gzip.compress(b""))
    with tarfile.open(host / RAW_TAR, "w") as tf:
        tf.add(raw / "training", arcname="training")
    games = work / "games_src" / "games_m1"
    games.mkdir(parents=True)
    teacher = work / "teacher.pt"
    types, factions = fresh_vocab()
    torch.save({"unit_type_to_id": types, "faction_to_id": factions}, teacher)
    sha = file_sha256(teacher)
    for stem, player in (("game_1", sha), ("game_2", sha), ("game_3", "q" * 64)):
        res = {"side_a": 1, "outcome_a": "win", "checkpoint_sha256_a": player, "checkpoint_sha256_b": player,
               "memory_a": 64, "memory_b": 64}
        (games / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")
        (games / f"{stem}.game.jsonl.gz").write_bytes(gzip.compress(stem.encode()))
    with tarfile.open(host / SOURCE_TAR, "w:gz") as tf:
        tf.add(games, arcname="games_m1")
    return sha


def _run(tmp: Path, host: Path, teacher: Path, sha: str, **env) -> dict:
    """trainer_inputs on a new box (WORKDIR tmp/box), the cache on `host`;
    the background sends waited for."""
    work, out = tmp / "box", tmp / "box" / "out"
    out.mkdir(parents=True)
    shutil.copyfile(teacher, work / "own1.pt")
    fake = tmp / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    exports = "\n".join(f"export {k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
WORKDIR="{bash_path(work)}" OUT="{bash_path(out)}" BOX_STATE="{bash_path(work / 'state')}"
BOX_REPO="{bash_path(ROOT)}" STAGE=stage_test WORKERS=1 FAKE="{bash_path(fake)}"
RAW_TAR={RAW_TAR} SOURCE_TARS={SOURCE_TAR} INIT_SHA256={sha} OWN_LEGACY=
{exports}
{STUBS}
cd "$BOX_REPO" || exit 9
. scripts/trainer_inputs.sh
trainer_inputs
for pid in ${{BOX_ASYNC_PIDS[@]+"${{BOX_ASYNC_PIDS[@]}}"}}; do wait "$pid"; done
echo done > "$OUT/DONE"
"""
    (tmp / "driver.sh").write_bytes(script.encode())
    environ = {**os.environ, "INPUT_CACHE_HOST": str(host)}
    done = subprocess.run([BASH, bash_path(tmp / "driver.sh")], cwd=ROOT, env=environ,
                          capture_output=True, text=True, timeout=900)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    read = lambda name: (out / name).read_text() if (out / name).exists() else ""  # noqa: E731
    return {"rc": done.returncode, "stderr": done.stderr, "out": out, "work": work, "done": (out / "DONE").exists(),
            "finish": read("FINISH"), "cache": read("cache.txt"), "names": [s.split("|")[0] for s in steps],
            "steps": steps}


def _ok(got: dict) -> str:
    return f"{got['finish']}\n{got['cache']}\n{got['stderr'][-3000:]}\n" + "\n".join(got["steps"])


@pytest.fixture(scope="module")
def first_box(tmp_path_factory):
    """A first box on an empty cache, the sequences through it too."""
    from wesnoth_ai import game_core as gc
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available: the pre-encoder runs for real")
    tmp = tmp_path_factory.mktemp("first")
    host = tmp / "host"
    sha = _host_inputs(host, tmp)
    got = _run(tmp, host, tmp / "teacher.pt", sha, CACHE_SEQUENCES=1)
    return {**got, "host": host, "teacher": tmp / "teacher.pt", "sha": sha}


def _copy_host(first: dict, tmp: Path) -> Path:
    host = tmp / "host"
    shutil.copytree(first["host"], host)
    return host


def _entries(host: Path, kind: str):
    return sorted((host / "tier-b" / "input_cache" / kind).glob("*/manifest.json"))


def test_a_first_box_builds_every_input_and_sends_each(first_box):
    got = first_box
    assert got["rc"] == 0 and got["done"] and not got["finish"], _ok(got)
    for step in ("raw", "corpus", "corpus-check", "preencode", "preencode-check", "candidates", "targets",
                 "columns sequences", "columns own_sequences_r2", "cache-send corpus", "cache-send sequences",
                 "cache-send own"):
        assert step in got["names"], (step, _ok(got))
    for kind in ("corpus", "sequences", "own"):
        assert len(_entries(got["host"], kind)) == 1, (kind, _ok(got))
        assert f"CACHE_ABSENT {kind}" in got["cache"] and f"CACHE_STORED {kind}" in got["cache"], _ok(got)
    summary = json.loads((got["out"] / "sequence_summary.json").read_text())
    assert summary["games"] == 3 and summary["errors"] == 0


def test_a_box_that_finds_every_entry_restores_them_and_builds_nothing(first_box, tmp_path):
    host = _copy_host(first_box, tmp_path)
    got = _run(tmp_path, host, first_box["teacher"], first_box["sha"], CACHE_SEQUENCES=1)
    assert got["rc"] == 0 and got["done"] and not got["finish"], _ok(got)
    assert not BUILDERS & set(got["names"]) and not any(n.startswith("cache-send") for n in got["names"]), _ok(got)
    for kind in ("corpus", "sequences", "own"):
        assert f"CACHE_RESTORED {kind}" in got["cache"], _ok(got)
    for rel in ("corpus/manifest.jsonl", "sequences/sequence_manifest.json", "sequences/columns_manifest.json",
                "own_sequences_r2/sequence_manifest.json"):
        assert (got["work"] / rel).read_bytes() == (first_box["work"] / rel).read_bytes(), rel
    assert json.loads((got["out"] / "corpus_summary.json").read_text())["problems"] == []


def test_a_corrupted_entry_is_refused_built_again_and_replaced(first_box, tmp_path):
    from tools import input_cache as ic
    host = _copy_host(first_box, tmp_path)
    [archive] = [p for p in _entries(host, "corpus")[0].parent.iterdir() if p.suffix == ".tar"]
    data = bytearray(archive.read_bytes())
    data[1000] ^= 0xFF
    archive.write_bytes(bytes(data))
    got = _run(tmp_path, host, first_box["teacher"], first_box["sha"], CACHE_SEQUENCES=1)
    assert got["rc"] == 0 and got["done"], _ok(got)
    assert "CACHE_REFUSED corpus" in got["cache"] and "CACHE_STORED corpus" in got["cache"], _ok(got)
    assert {"raw", "corpus", "corpus-check"} <= set(got["names"]), _ok(got)
    send = next(s for s in got["steps"] if s.startswith("cache-send corpus|"))
    assert "--replace" in send
    key = ic.read_key(got["work"] / "state" / "input_keys" / "corpus.json")
    assert ic.restore(ic.DirHost(host), key, tmp_path / "check") == 0


def test_a_legacy_tarball_is_adopted_checked_and_sent_as_the_entry(first_box, tmp_path):
    """With the default switches: the sequences are pre-encoded here and stay out of the cache."""
    host = _copy_host(first_box, tmp_path)
    shutil.rmtree(host / "tier-b" / "input_cache" / "own")
    legacy = host / "tier-b" / "old" / "own_sequences_r2.tar.gz"
    legacy.parent.mkdir(parents=True)
    with tarfile.open(legacy, "w:gz") as tf:
        tf.add(first_box["work"] / "own_sequences_r2", arcname="own_sequences_r2")
    got = _run(tmp_path, host, first_box["teacher"], first_box["sha"], OWN_LEGACY="tier-b/old/own_sequences_r2.tar.gz")
    assert got["rc"] == 0 and got["done"], _ok(got)
    assert "CACHE_ADOPTED own" in got["cache"] and "CACHE_STORED own" in got["cache"], _ok(got)
    assert not {"candidates", "targets"} & set(got["names"]) and len(_entries(host, "own")) == 1, _ok(got)
    assert (BUILDERS & set(got["names"])) == {"preencode"}, _ok(got)
    assert not [n for n in got["names"] if n.endswith(" sequences") and n.startswith("cache-")], _ok(got)
