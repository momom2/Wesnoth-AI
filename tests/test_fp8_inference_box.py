"""scripts/fp8_inference_box.sh's arms and play check, run by bash with the
box library stubbed (no box, GPU or HF) and the real readouts: every arm
runs in its order with its flags, an arm whose match crashes or loses a
game fails alone and the next arm runs, speed.json reads the fp8 runs
against the reference runs after every arm, and the play check plays side
A alone with fp8 and reads its verdict, a crashed one included."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "fp8_inference_box.sh"
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]

# The library's functions the block calls, and box_bounded answering for the
# steps: STUB_<step> = crash (exit 1, no record) or lose (a failed game);
# RATE_<step>: the decisions a second its server records (default 1000);
# PLAY_WINS: side A's wins in the play check; PLAY_ELO_B: side B's Elo in
# its fit (side A anchored at 0).
STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { mkdir -p "$(dirname "$1")" && echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_upload_dir() { :; }
box_upload_hold() { :; }
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
has() {                          # has FLAG ARGS...: FLAG is among ARGS
    local flag=$1
    shift
    [[ " $* " == *" $flag "* ]] && echo true || echo false
}
box_bounded() {                  # box_bounded NAME CUT_MIN LOG CMD...
    local name=$1 log=$3 key rate dir t n
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    key=STUB_${name//[ -]/_}
    rate=RATE_${name//[ -]/_}
    BOX_RC=0 BOX_WHY=ok
    case $name in
        arm\ *)
            dir=$(arg_of --outdir "$@")
            mkdir -p "$dir"
            if [ "${!key:-}" != crash ]; then
                echo '{}' > "$dir/game_A_B_s1_1.json"
                printf '{"requests": %d, "wall_s": 100.0, "batches": %d, "leaves": %d, "infer_s": 1.0, "gpu_ms": 500.0, "fp8": %s, "compile_packed": %s, "packed_compile_stats": {"active": true}}\n' \
                    "$(( ${!rate:-1000} * 100 ))" "$(( ${!rate:-1000} * 2 ))" "$(( ${!rate:-1000} * 100 ))" \
                    "$(has --fp8-inference-a "$@")" "$(has --compile-packed "$@")" > "$dir/.inference_server_0.json"
                for (( t = 5; t <= 100; t += 5 )); do
                    n=$(( t > 10 ? (t - 10) * ${!rate:-1000} : 0 ))
                    printf '{"wall_s": %d.0, "requests": %d, "batches": %d, "leaves": %d, "gpu_ms": %d.0, "idle_s": 0.0, "infer_s": 0.0}\n' \
                        "$t" "$n" "$(( n / 50 ))" "$n" "$(( n / 50 * 12 ))"
                done > "$dir/.inference_server_0.timeline.jsonl"
                [ "${!key:-}" != lose ] || echo '{}' > "$dir/failed_A_B_s1_3.json"
            fi
            [ "${!key:-}" != crash ] || { BOX_RC=1 BOX_WHY=failed; } ;;
        "match play")
            dir=$(arg_of --outdir "$@")
            mkdir -p "$dir"
            if [ "${!key:-}" != crash ]; then
                python - "$dir" "$(arg_of --games "$@")" "${PLAY_WINS:-20}" "$(has --fp8-inference-b "$@")" <<'EOF'
import json, pathlib, sys
out, games, wins, b = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "true"
for i in range(games):
    (out / f"game_own1fp8_own1_s1_{i}.json").write_text(json.dumps(
        {"outcome_a": "win" if i < wins else "loss", "infer_fp8_a": True, "infer_fp8_b": b,
         "forwards_a": 200, "forwards_b": 210}))
for tag, fp8 in (("0", True), ("1", b)):
    (out / f".inference_server_{tag}.json").write_text(json.dumps({"fp8": fp8, "requests": 1}))
EOF
            else
                BOX_RC=1 BOX_WHY=failed
            fi ;;
        "fit play")
            printf '{"tables": {"PURE (decisive only, primary)": {"own1fp8": {"elo": 0.0, "se": 0.0}, "own1": {"elo": %s, "se": 6.1}}}}\n' \
                "${PLAY_ELO_B:-0.0}" > "$(arg_of --save-json "$@")" ;;
        summary|"read play") "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
    esac
    return "$BOX_RC"
}
"""


def _region(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _run(tmp_path: Path, **stubs) -> dict:
    out, work, state = tmp_path / "out", tmp_path / "work", tmp_path / "state"
    for d in (out, work, state):
        d.mkdir()
    env = "\n".join(f"{k}={v}" for k, v in stubs.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
cd "{bash_path(ROOT)}" || exit 9
OUT="{bash_path(out)}" WORKDIR="{bash_path(work)}" BOX_STATE="{bash_path(state)}" STAGE=stage_test
JOBS=120 GAMES=400 GAMES_PER_JOB=5 SEED_BASE=130000 PLAY_SEED_BASE=140000 PLAY_GAMES=40 PLAY_EXTRA=1500
TIMELINE_S=5 PROFILE_BATCHES=40 PROFILE_AFTER=400 VALID_SPREAD=0.05 ARM_CUT_MIN=30 MATCH_BUDGET_MIN=22
PLAY_BUDGET_MIN=45 PLAY_CUT_MIN=70 REF_PT=own1.pt
SIDE_A=(--label-a own1a) SIDE_B=(--label-b own1b --memory-b 64)
PLAY_A=(--label-a own1fp8) PLAY_B=(--label-b own1 --memory-b 64)
BOX_RC=0 BOX_WHY=ok
{env}
{STUBS}
{_region("# ---- the arm list (begin)", "# ---- the arm list (end)")}
{_region("# ---- the arms (begin)", "# ---- the arms (end)")}
box_finish "FP8_INFERENCE_DONE $SUMMARY; play: ${{PLAY:-not read}}; failed: ${{FAILED_ARMS:-none}}"
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []

    def read(name):
        return json.loads((out / name).read_text()) if (out / name).exists() else {}
    return {"rc": done.returncode, "stderr": done.stderr,
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "arms": [s.split()[1] for s in steps if s.startswith("arm ")],
            "steps": steps, "speed": read("speed.json"), "play": read("play.json"),
            "failed": (out / "arms_failed.txt").read_text() if (out / "arms_failed.txt").exists() else ""}


ORDER = ["ref_1", "fp8c_1", "bf16e", "fp8e", "ref_2", "profile_ref", "profile_fp8c", "fp8c_2", "ref_3"]


def _step(got, name):
    return next(s for s in got["steps"] if s.startswith(f"{name} "))


def test_every_arm_runs_with_its_flags_and_the_play_check_plays_one_side_fp8(tmp_path):
    got = _run(tmp_path, RATE_arm_fp8c_1=1100, RATE_arm_fp8c_2=1100)
    assert got["rc"] == 0 and "FP8_INFERENCE_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["arms"] == ORDER and not got["failed"], got["failed"]
    for name in ORDER:
        line = _step(got, f"arm {name}")
        assert "--inference-timeline-s 5" in line and "--worker-games 6" in line and "--seed-base 130000" in line
        assert ("--fp8-inference-a --fp8-inference-b" in line) == ("fp8" in name), name
        assert ("--compile-packed" in line) == (name not in ("bf16e", "fp8e")), name
        assert ("--inference-profile-batches 40" in line) == name.startswith("profile"), name
    play = _step(got, "match play")
    assert "--fp8-inference-a" in play and "--fp8-inference-b" not in play and "--compile-packed --bf16-residual" in play
    assert "--seed-base 140000" in play and "--games 40" in play and "--max-extra-games 1500" in play
    reading = got["speed"]["reading"]
    assert reading["rates_from"] == "window" and reading["valid"] and reading["fp8c_gain_resolved"]
    assert reading["fp8c_against_ref"] == 1.1 and not got["speed"]["failed_arms"]
    assert got["play"]["verdict"] == "PASS" and got["play"]["decisive"] == 40 and got["play"]["rc"] == 0
    assert "play: PASS" in got["finish"]


def test_failing_arms_fail_alone_and_the_play_check_still_runs(tmp_path):
    got = _run(tmp_path, STUB_arm_ref_2="crash", STUB_arm_fp8c_1="lose", RATE_arm_fp8c_2=1020)
    assert got["rc"] == 0 and "FP8_INFERENCE_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["arms"] == ORDER
    for name in ("ref_2", "fp8c_1"):
        assert f"ARM_FAILED {name}" in got["failed"], name
    speed = got["speed"]
    assert {"ref_2", "fp8c_1"} <= set(speed["failed_arms"])
    assert speed["reading"]["fp8c_against_ref"] == 1.02 and not speed["reading"]["fp8c_gain_resolved"]
    assert got["play"]["verdict"] == "PASS"


def test_a_crashed_play_check_is_read_as_not_read_and_the_entry_finishes(tmp_path):
    got = _run(tmp_path, STUB_match_play="crash")
    assert got["rc"] == 0 and "FP8_INFERENCE_DONE" in got["finish"], (got["finish"], got["stderr"][-2000:])
    assert got["play"]["verdict"] == "NOT_READ" and got["play"]["decisive"] == 0 and got["play"]["rc"] == 1
