"""scripts/recruit_rule_l_box.sh's plan and stages, run by bash with the box
library stubbed (no box, GPU, core or HF): the matches are synthetic games
of match L with a known hooked effect (tests/helpers/rule_games.py) and the
rows and readouts are the real tool's. Every match plays the rule and no
coin, is read as it lands, and a re-entry plays nothing done; a match that
fails is left out and the next one runs, two in a row stop the run; a
readout that fails a check stops it; a plan it cannot play is refused
before any step."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "recruit_rule_l_box.sh"
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]

# A match writes synthetic games of match L into its outdir. VARIANT_<name> =
# crash (no games, exit 1), lose (a failed game), misplayed (every tenth game
# plays a0 at a switch).
FAKE = r'''
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))
import numpy as np
from helpers.rule_games import write_games
outdir, games, base, rule, variant = pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
data = json.loads(pathlib.Path(rule).read_text())
write_games(outdir, np.random.default_rng(base), data, games, base, 0.1, variant=variant if variant != "lose" else "ok")
if variant == "lose":
    (outdir / f"failed_own1a_own1b_s1_{base}.json").write_text("{}")
'''

STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_upload_async() { :; }
box_upload_dir() { echo "upload $1" >> "$OUT/steps.txt"; }
box_gpu_ok() { :; }
box_now() { date +%s; }
box_stamp() { date; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
arg_of() {                       # arg_of FLAG ARGS...: the value after the first FLAG
    local flag=$1
    shift
    while [ "$#" -gt 0 ]; do
        [ "$1" = "$flag" ] && { echo "$2"; return; }
        shift
    done
}
box_bounded() {                  # box_bounded [--stall FILE MIN] NAME CUT_MIN LOG CMD...
    local name log key
    [ "$1" != --stall ] || shift 3
    name=$1 log=$3
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    key=VARIANT_${name#match }
    BOX_RC=0 BOX_WHY=ok
    case $name in
        match\ *)
            if [ "${!key:-ok}" = crash ]; then
                BOX_RC=1 BOX_WHY=failed
            else
                python "$FAKE" "$TESTS" "$(arg_of --outdir "$@")" "$(arg_of --games "$@")" "$(arg_of --seed-base "$@")" \
                    "$(arg_of --recruit-rule "$@")" "${!key:-ok}" >> "$OUT/$log" 2>&1 || BOX_RC=$?
            fi ;;
        *) "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
    esac
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""


def _region(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


@pytest.fixture(scope="module")
def rule(tmp_path_factory) -> Path:
    from helpers.rule_games import RECRUITS, spread_rule, write_rule
    return write_rule(tmp_path_factory.mktemp("rule"), spread_rule(RECRUITS))


def _run(tmp_path: Path, rule: Path, **env) -> dict:
    out, work, state = tmp_path / "out", tmp_path / "work", tmp_path / "state"
    for d in (out, work, state):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("steps.txt", "FINISH"):
        (out / name).unlink(missing_ok=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    settings = {"N_L": 1500, "CHUNK_GAMES": 600, "L_SEED_BASE": 700000, **env}
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
cd "{bash_path(ROOT)}" || exit 9
OUT="{bash_path(out)}" WORKDIR="{bash_path(work)}" BOX_STATE="{bash_path(state)}" STAGE=stage_test
FAKE="{bash_path(fake)}" TESTS="{bash_path(ROOT / 'tests')}" RULE="{bash_path(rule)}"
JOBS=120 WORKER_GAMES=6 MIN_FREE_MB=300 GAMES_PER_H=22500 ROWS_CUT_MIN=30 READOUT_CUT_MIN=30
SIDE_A=(--label-a own1a --raw-end-turn-offset-a -1.5) SIDE_B=(--label-b own1b --memory-b 64)
CHUNKS=() CHUNK_SIZES=() CHUNK_BASES=() RESTORE=()
BOX_RC=0 BOX_WHY=ok
{chr(10).join(f"{k}={v}" for k, v in settings.items())}
{STUBS}
{_region("# ---- the readers (begin)", "# ---- the readers (end)")}
{_region("# ---- the plan (begin)", "# ---- the plan (end)")}
echo "restore ${{RESTORE[*]}}" >> "$OUT/steps.txt"
{_region("# ---- the stages (begin)", "# ---- the stages (end)")}
case $STAGE_L in
    DONE*) box_finish "RECRUIT_RULE_L_DONE $REASON" ;;
    *) box_finish "RECRUIT_RULE_L_INCOMPLETE $REASON" 1 ;;
esac
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    readout = out / "readout_l.json"
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.split()[0] in ("match", "upload") else " ".join(s.split()[:3])
                      for s in steps if not s.startswith("restore")],
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "l": json.loads(readout.read_text()) if readout.exists() else {},
            "failed": (out / "matches_failed.txt").read_text() if (out / "matches_failed.txt").exists() else ""}


def _steps(name):
    return [f"match {name}", f"rule rows {name}", f"upload games_{name}", "readout l python"]


def test_every_match_plays_the_rule_and_is_read_as_it_lands_and_a_re_entry_plays_nothing(tmp_path, rule):
    got = _run(tmp_path, rule)
    assert got["rc"] == 0 and got["finish"].startswith("RECRUIT_RULE_L_DONE DONE: 3 of 3 matches read, readout PASS"), \
        (got["finish"], got["stderr"][-3000:])
    assert got["names"] == _steps("l00") + _steps("l01") + _steps("l02")
    for name, games, base in (("l00", 600, 700000), ("l01", 600, 700600), ("l02", 300, 701200)):
        line = next(s for s in got["steps"] if s.startswith(f"match {name} "))
        assert f"--games {games} " in line and f"--seed-base {base} " in line, line
        assert f"--recruit-rule {bash_path(rule)}" in line and "--jobs 120 --worker-games 6" in line
        assert "--coin-band" not in line and "--attack-coin-offset" not in line and "--label-a own1a" in line
    readouts = [s for s in got["steps"] if s.startswith("readout l ")]
    assert "rows_l00.jsonl" in readouts[0] and "rows_l01.jsonl" not in readouts[0] and "rows_l02.jsonl" in readouts[2]
    assert got["l"]["games"]["decisive"] == 1500 and got["l"]["verdict"] == "PASS" and got["l"]["outcome"]
    assert {m: a["slots"]["seed_base"] for m, a in got["l"]["matches"].items()} == \
           {"l00": 700000, "l01": 700600, "l02": 701200}
    restore = next(s for s in got["steps"] if s.startswith("restore"))
    assert "rows_l02.jsonl" in restore and "match_l02.done" in restore and "readout_l.json" in restore
    again = _run(tmp_path, rule)
    assert again["rc"] == 0 and again["finish"].startswith("RECRUIT_RULE_L_DONE"), again["finish"]
    assert not any(n.startswith(("match", "rule", "readout")) for n in again["names"]), again["names"]


def test_a_failed_match_is_left_out_and_two_in_a_row_stop_the_run(tmp_path, rule):
    got = _run(tmp_path, rule, VARIANT_l01="lose")
    assert got["rc"] == 0 and "DONE: 2 of 3 matches read" in got["finish"], (got["finish"], got["stderr"][-3000:])
    assert "MATCH_FAILED l01" in got["failed"] and "match l02" in got["names"]
    assert got["l"]["games"]["read"] == 900 and set(got["l"]["matches"]) == {"l00", "l02"}
    stopped = _run(tmp_path / "two", rule, VARIANT_l01="crash", VARIANT_l02="crash", N_L=2400)
    assert stopped["rc"] == 1 and "STOPPED: two matches failed in a row" in stopped["finish"], stopped["finish"]
    assert "match l03" not in stopped["names"]


def test_a_readout_that_fails_a_check_stops_the_run(tmp_path, rule):
    got = _run(tmp_path, rule, VARIANT_l00="misplayed")
    assert got["rc"] == 1 and "CHECK_FAILED after l00: FAIL: i_cap" in got["finish"], (got["finish"], got["stderr"][-3000:])
    assert "match l01" not in got["names"]


@pytest.mark.parametrize("env, why", [({"N_L": 50000}, "N_L=50000"), ({"CHUNK_GAMES": "1e4"}, "CHUNK_GAMES=1e4"),
                                      ({"N_L": 0}, "N_L=0")])
def test_a_plan_it_cannot_play_is_refused_before_any_step(tmp_path, rule, env, why):
    got = _run(tmp_path, rule, **env)
    assert got["rc"] == 1 and got["finish"].startswith("BAD_PLAN") and why in got["finish"], got["finish"]
    assert not got["names"]
