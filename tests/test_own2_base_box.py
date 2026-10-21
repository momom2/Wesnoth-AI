"""scripts/own2_base_box.sh's sides and match, run by bash with the box
library stubbed (no box, GPU or HF) and the real reference tool reading
the reference the match was run against, `own1`: `own2` on side A and
`own1` on side B reach the match driver at 64 slots under the reference
decode, with the declared seed base and game count; a short attempt is
played once more, its fit cleared first; a match that fails or is cut
records why; an attempt's budget is what the dead-man's switch leaves; a
re-entry with the games complete only fits and reads them, and the reading
applies the pre-registered rule. Under any other reference the script
plays nothing."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "own2_base_box.sh"
REFERENCE = json.loads((ROOT / "configs" / "reference_player.json").read_text(encoding="utf-8"))
# The reference the match was pre-registered and run against (2026-10-20):
# `own1`, as configs/reference_player.json named it from 2026-10-11 to
# 2026-10-21.
OWN1_REFERENCE = {
    "label": "own1", "checkpoint_local": "training/checkpoints/own1.pt",
    "checkpoint_hf": "tier-b/memory_own_trajectories_20261011/own.pt",
    "checkpoint_sha256": "70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80",
    "hf_repo": "momom2/wesnoth-model-checkpoints", "memory_slots": 64,
    "decode": {"mcts_sims": 0, "raw_temperature": 0.0, "raw_end_turn": "joint", "raw_end_turn_offset": -1.5},
    "procedure_tag": "raw:t0+eo-1.5", "adopted": "2026-10-11"}
BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

STUBS = r"""
box_now() { date +%s; }
box_upload_dir() { :; }
box_upload_hold() { :; }
box_upload_async() { :; }
box_on_round() { :; }
box_gpu_ok() { BOX_RC=0 BOX_WHY=ok; }
box_clear() { echo "clear $*" >> "$OUT/steps.txt"; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
box_bounded() {                  # box_bounded NAME CUT_MIN LOG CMD...
    local name=$1 cut=$2 log=$3
    shift 3
    echo "$name cut=$cut $*" >> "$OUT/steps.txt"
    BOX_RC=0 BOX_WHY=ok
    python "$FAKE" "$name" "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$?
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""

# The match driver plays its attempts as STUB_MATCH lists them, one word an
# attempt: complete (every slot decisive, exit 0), short (10 decisive games
# and a failed game, exit 1), cut (10 decisive games, exit 3), crash (exit 1,
# no game). The fit writes own2 at STUB_ELO +- STUB_SE against own1 at 0,
# and fails with no game to fit.
FAKE = r'''
import json, os, pathlib, sys
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
if name.startswith("match "):
    counter = pathlib.Path(os.environ["STUB_COUNTER"])
    k = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(k + 1))
    mode = os.environ.get("STUB_MATCH", "complete").split()[k]
    out, games, base = pathlib.Path(opt("--outdir")), int(opt("--games")), int(opt("--seed-base"))
    out.mkdir(parents=True, exist_ok=True)
    want = {"complete": games, "short": 10, "cut": 10, "crash": 0}[mode]
    for i in range(want):
        side = 1 if i % 2 == 0 else 2
        path = out / f"game_{opt('--label-a')}_{opt('--label-b')}_s{side}_{base + i}.json"
        path.write_text(json.dumps({"outcome_a": "win" if i % 3 else "loss"}))
    if mode == "short":
        (out / f"failed_own2_own1_s1_{base + 10}.json").write_text("{}")
    sys.exit({"complete": 0, "short": 1, "cut": 3, "crash": 1}[mode])
if name.startswith("fit "):
    games = list(pathlib.Path(args[args.index("tools/elo_collect.py") + 1]).glob("game_*.json"))
    if not games:
        sys.exit(1)
    table = {"own1": {"elo": 0.0, "se": 0.0},
             "own2": {"elo": float(os.environ.get("STUB_ELO", "16.5")), "se": float(os.environ.get("STUB_SE", "4.34"))}}
    pathlib.Path(opt("--save-json")).write_text(json.dumps(
        {"n_games": len(games), "tables": {"PURE (decisive only, primary)": table}}))
'''


def _block(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _checkout(tmp_path: Path, reference: dict) -> Path:
    """A checkout whose configs/reference_player.json is `reference`: the
    config, and the real tools/reference_player.py with the package module
    it imports, which is all of the repository the blocks run here read."""
    root = tmp_path / "checkout"
    for rel in ("tools/reference_player.py", "wesnoth_ai/__init__.py", "wesnoth_ai/paths.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    (root / "configs").mkdir()
    (root / "configs" / "reference_player.json").write_text(json.dumps(reference), encoding="utf-8")
    return root


def _run(tmp_path: Path, games_done: int = 0, deadline_min: float | None = None,
         reference: dict = OWN1_REFERENCE, **env) -> dict:
    checkout = _checkout(tmp_path, reference)
    out = tmp_path / "out"
    (out / "tmp").mkdir(parents=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    if games_done:                           # an earlier entry's complete games
        games = out / "games_own2_vs_own1"
        games.mkdir()
        for i in range(games_done):
            (games / f"game_own2_own1_s1_{i}.json").write_text(json.dumps({"outcome_a": "loss" if i % 2 else "win"}))
    if deadline_min is not None:             # when the dead-man's switch fires
        (out / "tmp" / "deadman.at").write_text(str(int(time.time() + deadline_min * 60)))
    settings = "\n".join(f"{k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
cd "{bash_path(checkout)}" || exit 9
OUT="{bash_path(out)}" BOX_OUT="{bash_path(out)}" STAGE=stage_test FAKE="{bash_path(fake)}"
export STUB_COUNTER="{bash_path(tmp_path / 'attempts')}"
BOX_RC=0 BOX_WHY=ok
{settings}
export STUB_MATCH="${{STUB_MATCH:-complete}}" STUB_ELO="${{STUB_ELO:-16.5}}" STUB_SE="${{STUB_SE:-4.34}}"
{_block("# ---- the configuration (begin)", "# ---- the configuration (end)")}
NAME=own2_vs_own1
DIR=$OUT/games_$NAME
OWN2_PT=/workspace/own2.pt
{STUBS}
{_block("decisive_results() {", "notes() {")}
{_block("# ---- the sides (begin)", "# ---- the sides (end)")}
{_block("# ---- the match (begin)", "# ---- the match (end)")}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    reading = out / "reading.json"
    walls = out / "match.walls"
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) for s in steps],
            "finish": (out / "FINISH").read_text().strip() if (out / "FINISH").exists() else "",
            "reading": json.loads(reading.read_text()) if reading.exists() else None,
            "walls": walls.read_text() if walls.exists() else ""}


def _side(side: str, label: str, spec: str) -> str:
    return (f"--label-{side} {label} --spec-{side} {spec} --memory-{side} {OWN1_REFERENCE['memory_slots']} "
            f"--raw-end-turn-offset-{side} {OWN1_REFERENCE['decode']['raw_end_turn_offset']}")


def test_own2_plays_the_reference_at_its_decode_and_the_fit_is_read(tmp_path):
    got = _run(tmp_path, GAMES=40)
    assert got["rc"] == 0 and got["finish"].startswith("OWN2_BASE_DONE"), (got["finish"], got["stderr"][-2000:])
    assert got["names"] == ["match own2_vs_own1", "fit own2_vs_own1"]
    match, fit = got["steps"]
    assert match.startswith("match own2_vs_own1 cut=195 python tools/run_elo_batch.py "
                            f"{_side('a', 'own2', '/workspace/own2.pt')} "
                            f"{_side('b', 'own1', 'training/checkpoints/own1.pt')} "), match
    for flags in ("--games 40 --max-extra-games 750 --seed-base 300000",
                  "--mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0",
                  "--persistent-workers --shared-inference --no-infer-compile --device cuda --jobs 20",
                  "--time-budget-min 165"):
        assert f" {flags}" in match, flags
    assert "--anchor own1 --save-json" in fit
    reading = got["reading"]
    assert reading["base"] == "own2" and reading["agrees_with_record"] and reading["n_games"] == 40
    assert reading["lower"] == pytest.approx(16.5 - 2 * 4.34)
    assert reading["record_window"] == pytest.approx([16.5 - 26.09, 16.5 + 26.09], abs=0.01)
    assert "base own2" in got["finish"] and "MATCH_READ own2_vs_own1" in got["walls"]


def test_the_declared_seed_base_and_game_count_reach_the_driver_and_a_failed_match_records_why(tmp_path):
    got = _run(tmp_path, STUB_MATCH="crash crash")
    assert got["rc"] == 1 and got["reading"] is None, got["stderr"][-2000:]
    assert got["names"] == ["match own2_vs_own1", "fit own2_vs_own1", "clear own2_vs_own1.fit.json",
                            "match own2_vs_own1", "fit own2_vs_own1"]
    for step in (got["steps"][0], got["steps"][3]):
        assert " --games 6400 --max-extra-games 750 --seed-base 300000 " in step
    assert got["finish"] == ("OWN2_BASE_FAILED after 2 attempts this entry: MATCH_FAILED own2_vs_own1: rc=1, "
                             "0 of 6400 decisive games (own2_vs_own1.log, games_own2_vs_own1/failed_*.json)")
    assert got["walls"].count("rc=1 failed") == 2 and "MATCH_FAILED own2_vs_own1: rc=1" in got["walls"]


def test_a_short_attempt_is_played_once_more_after_its_fit_is_cleared(tmp_path):
    got = _run(tmp_path, GAMES=40, STUB_MATCH="short complete")
    assert got["rc"] == 0 and got["finish"].startswith("OWN2_BASE_DONE"), (got["finish"], got["stderr"][-2000:])
    assert got["names"] == ["match own2_vs_own1", "fit own2_vs_own1", "clear own2_vs_own1.fit.json",
                            "match own2_vs_own1", "fit own2_vs_own1"]
    assert "10 decisive, rc=1 failed" in got["walls"] and "40 decisive, rc=0 ok" in got["walls"]
    assert got["reading"]["n_games"] == 40


def test_an_attempt_takes_the_budget_the_switch_leaves_and_a_cut_match_says_so(tmp_path):
    got = _run(tmp_path, deadline_min=200.5, GAMES=40, STUB_MATCH="cut cut")
    assert got["rc"] == 1 and got["reading"] is None, got["stderr"][-2000:]
    matches = [s for s in got["steps"] if s.startswith("match ")]
    assert len(matches) == 2
    for step in matches:                 # 200 minutes left, less the grace (30) and the finish's reserve (15)
        assert step.startswith("match own2_vs_own1 cut=185 ") and step.endswith(" --time-budget-min 155"), step
    assert got["finish"] == ("OWN2_BASE_CUT after 2 attempts this entry: MATCH_CUT own2_vs_own1: "
                             "10 of 40 decisive games (rc=3); the fit is not read")


def test_no_attempt_starts_when_the_switch_leaves_too_little(tmp_path):
    got = _run(tmp_path, deadline_min=50, GAMES=40)
    assert got["rc"] == 1 and not got["steps"], (got["steps"], got["stderr"][-2000:])
    assert got["finish"].startswith("OWN2_BASE_CUT after 0 attempts this entry: MATCH_CUT own2_vs_own1: "
                                    "0 of 40 decisive games; the switch leaves "), got["finish"]
    assert got["finish"].endswith("minutes of budget, under 10; a re-entry with time resumes it")


@pytest.mark.parametrize("elo, base, inside", [(4.0, "own1", True), (45.0, "own2", False), (-10.0, "own1", False)])
def test_a_re_entry_with_complete_games_only_fits_and_applies_the_rule(tmp_path, elo, base, inside):
    got = _run(tmp_path, games_done=40, GAMES=40, STUB_ELO=elo)
    assert got["rc"] == 0 and got["names"] == ["fit own2_vs_own1"], (got["steps"], got["stderr"][-2000:])
    reading = got["reading"]
    assert reading["base"] == base and reading["agrees_with_record"] is inside
    assert f"base {base}" in got["finish"] and ("OUTSIDE" in got["finish"]) is not inside


def test_under_the_live_reference_the_record_plays_nothing(tmp_path):
    """The reference moved on from `own1` (configs/reference_player.json):
    the script refuses before any match, so a re-run cannot play `own2`
    against another checkpoint under this match's name."""
    assert REFERENCE["label"] != "own1"
    got = _run(tmp_path, reference=REFERENCE, GAMES=40)
    assert got["rc"] == 1 and not got["steps"], (got["steps"], got["stderr"][-2000:])
    assert got["finish"] == (f"REFERENCE_CHANGED: configs/reference_player.json names {REFERENCE['label']}; "
                             "this match measures own2 against own1")
