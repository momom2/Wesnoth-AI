"""scripts/selfplay_t1_box.sh's plan, sides and stages, run by bash with the
box library stubbed (no box, GPU, core or HF) and the real reference tool
reading the live reference: the self-play matches play `own2` on both
seats at temperature 1 without an offset, at 64 slots, logged, with every
capped game kept, on consecutive seeds; each is walked into rows on the
run's seed base; the labels split at HELDOUT_FROM and the readout reads
every match; D0 puts `own2` at temperature 1 against the reference at its
decode, the memory report 64 against 0 slots; a match cut short is resumed
once, a failed self-play match stops the self-play but not the
measurements, a failed gate is named, and a re-entry plays nothing done."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "selfplay_t1_box.sh"
REFERENCE = json.loads((ROOT / "configs" / "reference_player.json").read_text(encoding="utf-8"))
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]

STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload_async() { :; }
box_upload_dir() { echo "upload $1" >> "$OUT/steps.txt"; }
box_restore() { echo "restore $*" >> "$OUT/steps.txt"; }
box_mark_landed() { :; }
box_gpu_ok() { :; }
box_now() { date +%s; }
box_stamp() { date; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
box_bounded() {                  # box_bounded [--stall FILE MIN] NAME CUT_MIN LOG CMD...
    local name log
    [ "$1" != --stall ] || shift 3
    name=$1 log=$3
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    BOX_RC=0 BOX_WHY=ok
    python "$FAKE" "$name" "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$?
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""

# Each step writes what the real tool would leave for the next. A match plays
# as VARIANT_<name> says, one word an attempt (the last repeats): complete
# (every slot decisive, exit 0), capped (a third of the games without a
# result, exit 4), cut (half the slots, exit 3), crash (no game, exit 1).
FAKE = r'''
import json, os, pathlib, sys
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
def opts(flag):
    i = args.index(flag) + 1
    out = []
    while i < len(args) and not args[i].startswith("--"):
        out.append(args[i]); i += 1
    return out
if name.startswith("match "):
    match = name.split()[1]
    counter = pathlib.Path(os.environ["STUB_DIR"]) / f"attempts_{match}"
    k = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(k + 1))
    modes = os.environ.get(f"VARIANT_{match}", "complete").split()
    mode = modes[min(k, len(modes) - 1)]
    out, games, base = pathlib.Path(opt("--outdir")), int(opt("--games")), int(opt("--seed-base"))
    out.mkdir(parents=True, exist_ok=True)
    want = {"complete": games, "capped": games, "cut": games // 2, "crash": 0}[mode]
    for i in range(want):
        side = 1 if i % 2 == 0 else 2
        outcome = "timeout" if mode == "capped" and i % 3 == 0 else ("win" if i % 2 else "loss")
        path = out / f"game_{opt('--label-a')}_{opt('--label-b')}_s{side}_{base + i}.json"
        path.write_text(json.dumps({"outcome_a": outcome}))
    sys.exit({"complete": 0, "capped": 4, "cut": 3, "crash": 1}[mode])
if name.startswith("rows "):
    pathlib.Path(opt("--walk")).write_text(json.dumps({"game": name}) + "\n")
    pathlib.Path(opt("--out")).write_text(json.dumps({"game": name}) + "\n")
elif name == "labels":
    pathlib.Path(opt("--out")).write_text("{}\n")
    pathlib.Path(opt("--fits")).write_text("{}")
elif name == "readout":
    gates = {k: {"pass": k not in os.environ.get("STUB_FAIL_GATES", "").split()}
             for k in ("i_sample", "legal_mass", "i_count", "dice_control", "i_v")}
    pathlib.Path(opt("--out")).write_text(json.dumps(gates))
    sys.exit(0 if all(g["pass"] for g in gates.values()) else 1)
elif name.startswith("fit "):
    games = list(pathlib.Path(args[args.index("tools/elo_collect.py") + 1]).glob("game_*.json"))
    if not games:
        sys.exit(1)
    label = {"fit d0": "own2t1", "fit memory": "own2m64"}[name]
    table = {opt("--anchor"): {"elo": 0.0, "se": 0.0}, label: {"elo": -12.3, "se": 3.1}}
    pathlib.Path(opt("--save-json")).write_text(json.dumps({"tables": {"PURE (decisive only, primary)": table}}))
'''


def _block(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _checkout(tmp_path: Path) -> Path:
    """The live reference config, the real tools/reference_player.py with the
    package module it imports, and a stand-in for the reference's file."""
    root = tmp_path / "checkout"
    for rel in ("tools/reference_player.py", "wesnoth_ai/__init__.py", "wesnoth_ai/paths.py",
                "configs/reference_player.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    (root / REFERENCE["checkpoint_local"]).parent.mkdir(parents=True, exist_ok=True)
    (root / REFERENCE["checkpoint_local"]).write_bytes(b"own2")
    return root


def _run(tmp_path: Path, **env) -> dict:
    checkout = tmp_path / "checkout"
    if not checkout.exists():
        _checkout(tmp_path)
    out = tmp_path / "out"
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    (out / "steps.txt").unlink(missing_ok=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    settings = "\n".join(f"export {k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
cd "{bash_path(checkout)}" || exit 9
OUT="{bash_path(out)}" BOX_OUT="{bash_path(out)}" BOX_REPO="{bash_path(checkout)}" STAGE=stage_test
FAKE="{bash_path(fake)}" WALK_JOBS=3
export STUB_DIR="{bash_path(tmp_path)}"
N_SELFPLAY=60 CHUNK_GAMES=20 HELDOUT_FROM=40 D0_GAMES=16 MEM_GAMES=8
WALK_CUT_MIN=30 WALK_STALL_MIN=10 LABELS_CUT_MIN=15 READOUT_CUT_MIN=60 FIT_CUT_MIN=15
{settings}
{_block("# ---- the configuration (begin)", "# ---- the configuration (end)")}
CHUNKS=()
CHUNK_SIZES=()
CHUNK_OFFSETS=()
RESTORE=()
SELFPLAY="not run"
READOUT="not read"
MEASURES=()
{STUBS}
{_block("# ---- the plan (begin)", "# ---- the plan (end)")}
{_block("# ---- the sides (begin)", "# ---- the sides (end)")}
{_block("# ---- the stages (begin)", "# ---- the stages (end)")}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.split()[0] in ("match", "rows", "fit", "upload") else s.split()[0]
                      for s in steps],
            "finish": (out / "FINISH").read_text().strip() if (out / "FINISH").exists() else "", "out": out,
            "checkout": checkout}


def _step(got: dict, name: str) -> str:
    return next(s for s in got["steps"] if s.startswith(name + " "))


def test_the_run_plays_logs_labels_reads_and_measures(tmp_path):
    got = _run(tmp_path)
    assert got["rc"] == 0 and got["finish"].startswith("SELFPLAY_T1_DONE"), (got["finish"], got["stderr"][-3000:])
    assert got["names"] == ["match s00", "rows s00", "upload games_s00", "match s01", "rows s01", "upload games_s01",
                            "match s02", "rows s02", "upload games_s02", "labels", "readout", "match d0", "fit d0",
                            "upload games_d0", "match memory", "fit memory", "upload games_memory"], got["names"]
    ref = f"{bash_path(got['checkout'])}/{REFERENCE['checkpoint_local']}"
    out = bash_path(got["out"])
    eo = REFERENCE["decode"]["raw_end_turn_offset"]
    for k, name in enumerate(("s00", "s01", "s02")):
        match = _step(got, f"match {name}")
        assert match.startswith(f"match {name} python tools/run_elo_batch.py --label-a own2a --spec-a {ref} "
                                f"--memory-a 64 --label-b own2b --spec-b {ref} --memory-b 64 --mcts-sims 0 "
                                f"--raw-temperature-a 1 --raw-temperature-b 1 --log-decisions "
                                f"--outdir {out}/games_{name} --games 20 --max-extra-games 0 "
                                f"--seed-base {800000 + 20 * k} "), match
        assert "--raw-end-turn-offset" not in match
        for flags in ("--persistent-workers --shared-inference --no-infer-compile --device cuda --jobs 120 "
                      "--worker-games 6 --min-free-mb 300",):
            assert f" {flags} " in match, flags
        assert f"--games {out}/games_{name} --walk {out}/walk_{name}.jsonl --out {out}/rows_{name}.jsonl " \
               f"--seed-base 800000 --jobs 3" in _step(got, f"rows {name}")
    assert "--heldout-from 40 " in _step(got, "labels")
    readout = _step(got, "readout")
    assert f"--games {out}/games_s00 {out}/games_s01 {out}/games_s02 --labels" in readout
    assert f"--walk {out}/walk_s00.jsonl {out}/walk_s01.jsonl {out}/walk_s02.jsonl --out" in readout
    d0 = _step(got, "match d0")
    assert d0.startswith(f"match d0 python tools/run_elo_batch.py --label-a own2t1 --spec-a {ref} --memory-a 64 "
                         f"--label-b {REFERENCE['label']} --spec-b {ref} --memory-b 64 --raw-end-turn-offset-b {eo} "
                         f"--mcts-sims 0 --raw-temperature-a 1 --raw-temperature-b 0 --outdir {out}/games_d0 "
                         f"--games 16 --max-extra-games 8 --seed-base 830000 "), d0
    assert "--log-decisions" not in d0 and "--raw-end-turn-offset-a" not in d0
    memory = _step(got, "match memory")
    assert memory.startswith(f"match memory python tools/run_elo_batch.py --label-a own2m64 --spec-a {ref} "
                             f"--memory-a 64 --label-b own2m0 --spec-b {ref} --memory-b 0 --mcts-sims 0 "
                             f"--raw-temperature-a 1 --raw-temperature-b 1 --outdir {out}/games_memory "
                             f"--games 8 --max-extra-games 4 --seed-base 845000 "), memory
    assert "--anchor own2 " in _step(got, "fit d0") and "--anchor own2m0 " in _step(got, "fit memory")
    assert "own2t1 -12.3 +- 3.1 Elo against own2" in got["finish"]
    assert "i_sample=pass legal_mass=pass i_count=pass dice_control=pass i_v=pass" in got["finish"]
    # A re-entry plays nothing done.
    again = _run(tmp_path)
    assert again["rc"] == 0 and again["steps"] == [], (again["steps"], again["stderr"][-2000:])
    assert again["finish"] == got["finish"]


def test_capped_games_complete_a_match_and_a_cut_one_is_resumed_once(tmp_path):
    got = _run(tmp_path, VARIANT_s00="capped", VARIANT_s01="cut complete")
    assert got["rc"] == 0 and got["finish"].startswith("SELFPLAY_T1_DONE"), (got["finish"], got["stderr"][-3000:])
    assert got["names"][:7] == ["match s00", "rows s00", "upload games_s00", "match s01", "match s01", "rows s01",
                                "upload games_s01"]


def test_a_failed_self_play_match_stops_the_self_play_but_not_the_measurements(tmp_path):
    got = _run(tmp_path, VARIANT_s01="crash")
    assert got["rc"] == 1, got["stderr"][-3000:]
    assert got["names"] == ["match s00", "rows s00", "upload games_s00", "match s01", "match d0", "fit d0",
                            "upload games_d0", "match memory", "fit memory", "upload games_memory"]
    assert got["finish"].startswith("SELFPLAY_T1_INCOMPLETE self-play: FAILED: s01, the match ended failed")
    assert (got["out"] / "chunk_s01.failed").exists()
    again = _run(tmp_path)
    assert again["rc"] == 1 and again["steps"] == [], again["steps"]
    assert "s01 failed at an earlier entry" in again["finish"]


def test_a_cut_measure_twice_fails_and_a_failed_gate_is_named(tmp_path):
    got = _run(tmp_path, VARIANT_memory="cut", STUB_FAIL_GATES="i_sample")
    assert got["rc"] == 1, got["stderr"][-3000:]
    assert got["names"].count("match memory") == 2 and "fit memory" not in got["names"]
    assert "readout: i_sample=FAIL legal_mass=pass" in got["finish"]
    assert "memory: FAILED, the match ended short" in got["finish"]
