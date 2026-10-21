"""scripts/coin_batch_box.sh's plan and stages, run by bash with the box
library stubbed (no box, GPU, core or HF): the matches are synthetic games
with known coins (tests/helpers/coin_games.py) and the rows, plant and
readouts are the real tools'. Stage A plays the attack coin and gates stage
B; stage B plays the smoke, reads it as a gate, then its matches, each read
as it lands; a smoke that fails a check, or a stage A outside its window,
stops the run with the reason; a later match that fails is left out and the
next one runs, two in a row stop the stage; a re-entry plays nothing done."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "coin_batch_box.sh"
BASH = find_bash()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BASH is None, reason="no POSIX bash")]

# A match writes synthetic games into its outdir: attack-coin games when it
# carries --attack-coin-offset, near-tie coin games otherwise. VARIANT_<name>
# = crash (no games, exit 1), lose (a failed game), reversed (the hook helps
# the hooked player), biased (coins land 1 at 0.45 against a logged 0.25).
FAKE = r'''
import json, gzip, pathlib, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))
import numpy as np
from helpers.coin_games import attack_game, synthetic_game
mode = sys.argv[2]
if mode == "walk":
    out, name, games = pathlib.Path(sys.argv[3]), sys.argv[4], pathlib.Path(sys.argv[5])
    with open(out, "w") as fh:
        for line in (games / ".luck.jsonl").read_text().splitlines():
            fh.write(json.dumps({**json.loads(line), "match": name}) + "\n")
    sys.exit(0)
outdir, games, base, kind, variant = pathlib.Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
outdir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(base)
luck = []
for i in range(games):
    seed, side_a = base + i, 1 if i % 2 == 0 else 2
    game = f"game_own1a_own1b_s{side_a}_{seed}"
    if kind == "attack":
        res, rec, lr = attack_game(rng, i, -0.2247 if variant == "reversed" else 0.2247, game=game, side_a=side_a)
    else:
        res, rec, lr, _ = synthetic_game(rng, i, game=game, side_a=side_a)
        if variant == "biased":
            for e in rec["coins"]["decisions"]:
                e["d"] = int(rng.random() < 0.45)
    (outdir / f"{game}.json").write_text(json.dumps({**res, "seed": seed, "label_a": "own1a", "label_b": "own1b"}))
    (outdir / f"{game}.game.jsonl.gz").write_bytes(gzip.compress((json.dumps(rec) + "\n").encode()))
    luck.append(json.dumps(lr))
(outdir / ".luck.jsonl").write_text("\n".join(luck) + "\n")
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
    local name log key kind
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
                kind=coins
                [[ " $* " != *" --attack-coin-offset "* ]] || kind=attack
                python "$FAKE" "$TESTS" match "$(arg_of --outdir "$@")" "$(arg_of --games "$@")" \
                    "$(arg_of --seed-base "$@")" "$kind" "${!key:-ok}" >> "$OUT/$log" 2>&1 || BOX_RC=$?
            fi ;;
        walk\ *)
            local m
            m=$(arg_of --match "$@")
            python "$FAKE" "$TESTS" walk "$(arg_of --out "$@")" "${m%%=*}" "${m#*=}" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
        "elo a") echo '{}' > "$(arg_of --save-json "$@")" ;;
        *) "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$? ;;
    esac
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""


def _region(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _run(tmp_path: Path, **env) -> dict:
    out, work, state = tmp_path / "out", tmp_path / "work", tmp_path / "state"
    for d in (out, work, state):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("steps.txt", "FINISH"):
        (out / name).unlink(missing_ok=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    settings = {"N_A": 1500, "N_B": 2200, "SMOKE_GAMES": 1000, "CHUNK_GAMES": 600, **env}
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
cd "{bash_path(ROOT)}" || exit 9
OUT="{bash_path(out)}" WORKDIR="{bash_path(work)}" BOX_STATE="{bash_path(state)}" STAGE=stage_test
FAKE="{bash_path(fake)}" TESTS="{bash_path(ROOT / 'tests')}"
JOBS=120 WORKER_GAMES=6 MIN_FREE_MB=300 GAMES_PER_H=21966 A_SEED_BASE=500000 B_SEED_BASE=600000
ATTACK_OFFSET=1 GATE_B_ON_A=1 COIN_BAND=0.35 COIN_P=0.25 BETA=configs/luck_beta_20261019.json
WALK_CUT_MIN=60 WALK_STALL_MIN=10 ROWS_CUT_MIN=30 READOUT_CUT_MIN=30 WALK_JOBS=2 REF_PT=own1.pt
SIDE_A=(--label-a own1a --raw-end-turn-offset-a -1.5) SIDE_B=(--label-b own1b --memory-b 64)
CHUNKS=() CHUNK_SIZES=() CHUNK_BASES=() RESTORE=()
BOX_RC=0 BOX_WHY=ok
{chr(10).join(f"{k}={v}" for k, v in settings.items())}
{STUBS}
{_region("# ---- the readers (begin)", "# ---- the readers (end)")}
{_region("# ---- the plan (begin)", "# ---- the plan (end)")}
echo "restore ${{RESTORE[*]}}" >> "$OUT/steps.txt"
{_region("# ---- the stages (begin)", "# ---- the stages (end)")}
case $STAGE_B in
    BATCH_DONE*) box_finish "NEAR_TIE_COINS_DONE $REASON" ;;
    *) box_finish "NEAR_TIE_COINS_INCOMPLETE $REASON" 1 ;;
esac
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []

    def read(name):
        return json.loads((out / name).read_text()) if (out / name).exists() else {}
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.split()[0] in ("match", "walk", "readout", "upload", "elo")
                      else " ".join(s.split()[:3]) for s in steps if not s.startswith("restore")],
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "a": read("readout_a.json"), "smoke": read("readout_smoke.json"), "b": read("readout_b.json"),
            "failed": (out / "matches_failed.txt").read_text() if (out / "matches_failed.txt").exists() else ""}


def _step(got, name):
    return next(s for s in got["steps"] if s.startswith(f"{name} "))


# Stage B's tests: stage A small and not a gate.
B_ONLY = {"N_A": 200, "GATE_B_ON_A": 0}
A_STEPS = ["match a", "upload games_a", "walk a", "coin rows a", "elo a", "readout readout_a"]


def _b_steps(name):
    return [f"match {name}", f"walk {name}", f"coin rows {name}", f"plant rows {name}", f"upload games_{name}"]


def test_both_stages_run_each_match_read_as_it_lands_and_a_re_entry_plays_nothing(tmp_path):
    got = _run(tmp_path)
    assert got["rc"] == 0 and got["finish"].startswith("NEAR_TIE_COINS_DONE"), (got["finish"], got["stderr"][-3000:])
    assert "stage A: PASS" in got["finish"] and "BATCH_DONE: 3 of 3 matches read" in got["finish"], got["finish"]
    assert got["names"] == (A_STEPS + _b_steps("b00") + ["readout readout_smoke", "readout readout_b"]
                            + _b_steps("b01") + ["readout readout_b"] + _b_steps("b02") + ["readout readout_b"])
    a = _step(got, "match a")
    assert "--attack-coin-offset 1" in a and "--coin-band-a" not in a and "--seed-base 500000" in a
    assert "--games 1500" in a and "--jobs 120 --worker-games 6" in a and "--label-a own1a" in a
    for name, games, base in (("b00", 1000, 600000), ("b01", 600, 601000), ("b02", 600, 601600)):
        line = _step(got, f"match {name}")
        assert f"--games {games} " in line and f"--seed-base {base} " in line, line
        assert "--coin-band-a 0.35 --coin-p-a 0.25 --coin-band-b 0.35 --coin-p-b 0.25" in line
        assert "--attack-coin-offset" not in line
    assert "--plant" not in _step(got, "readout readout_a")
    assert got["a"]["attack"]["games"] == 1500 and got["a"]["verdict"] == "PASS", got["a"]["checks"]
    assert got["smoke"]["verdict"] == "PASS" and got["smoke"]["games"]["read"] == 1000
    assert got["b"]["verdict"] == "PASS" and got["b"]["games"]["read"] == 2200 and set(got["b"]["plant"]) == {"a", "b"}
    slots = {m: a["slots"] for m, a in got["b"]["matches"].items()}
    assert slots["b01"]["planned"] == 600 and slots["b01"]["seed_base"] == 601000
    assert not any(s["cut"] for s in slots.values()) and got["a"]["matches"]["a"]["slots"]["with_result"] == 1500
    restore = next(s for s in got["steps"] if s.startswith("restore"))
    assert "coins_b02.jsonl" in restore and "match_b02.done" in restore and "readout_a.json" in restore
    again = _run(tmp_path)
    assert again["rc"] == 0 and again["finish"].startswith("NEAR_TIE_COINS_DONE"), again["finish"]
    assert not any(n.startswith(("match", "walk", "readout")) for n in again["names"]), again["names"]


def test_a_stage_a_outside_its_window_holds_stage_b(tmp_path):
    got = _run(tmp_path, VARIANT_a="reversed")
    assert got["rc"] == 1 and "HELD: stage A reads 'FAIL: attack_elo attack_label'" in got["finish"], got["finish"]
    assert got["names"] == A_STEPS


def test_a_smoke_that_fails_a_check_stops_the_batch_and_says_which(tmp_path):
    got = _run(tmp_path, VARIANT_b00="biased", **B_ONLY)
    assert got["rc"] == 1 and "SMOKE_GATE_FAILED: FAIL:" in got["finish"] and "i_coin_overall" in got["finish"], \
        (got["finish"], got["stderr"][-3000:])
    assert got["names"][-1] == "readout readout_smoke" and "match b01" not in got["names"]


@pytest.mark.parametrize("env, why", [({"N_B": 400000}, "N_B=400000"), ({"CHUNK_GAMES": "1e4"}, "CHUNK_GAMES=1e4"),
                                      ({"N_A": 0}, "N_A=0")])
def test_a_plan_it_cannot_play_is_refused_before_any_step(tmp_path, env, why):
    got = _run(tmp_path, **env)
    assert got["rc"] == 1 and got["finish"].startswith("BAD_PLAN") and why in got["finish"], got["finish"]
    assert not got["names"]


def test_a_crashed_smoke_stops_the_batch(tmp_path):
    got = _run(tmp_path, VARIANT_b00="crash", **B_ONLY)
    assert got["rc"] == 1 and "SMOKE_FAILED: the match failed" in got["finish"], got["finish"]
    assert "match b01" not in got["names"] and "MATCH_FAILED b00" in got["failed"]


def test_a_later_match_that_fails_is_left_out_and_two_in_a_row_stop_the_stage(tmp_path):
    got = _run(tmp_path, VARIANT_b01="lose", **B_ONLY)
    assert got["rc"] == 0 and "BATCH_DONE: 2 of 3 matches read" in got["finish"], (got["finish"], got["stderr"][-3000:])
    assert "MATCH_FAILED b01" in got["failed"] and "match b02" in got["names"]
    assert got["b"]["games"]["read"] == 1600
    stopped = _run(tmp_path / "two", VARIANT_b01="crash", VARIANT_b02="crash", N_B=2800, **B_ONLY)
    assert stopped["rc"] == 1 and "STAGE_B_STOPPED" in stopped["finish"], stopped["finish"]
    assert "match b03" not in stopped["names"]
