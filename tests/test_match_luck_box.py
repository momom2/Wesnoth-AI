"""scripts/match_luck_box.sh's match list and matches, run by bash with the
box library stubbed (no box, core or HF): the list refuses an entry it
cannot read before any step; every match is fetched against its pin and
walked in a step of its own, and the real readout reads the matches walked
whole; a match that is not fetched or whose walk fails is left out while
the others go on, and a re-entry keeps a failed walk out and walks only
what is not done; the default list names 23 pinned matches."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.posix_bash import bash_path, find_bash

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "match_luck_box.sh"
BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

STUBS = r"""
box_marked_this_stage() { [ -f "$1" ]; }
box_mark() { echo "stage=$STAGE" > "$1"; }
box_upload() { :; }
box_upload_async() { :; }
box_on_round() { :; }
notes() { echo notes; }
box_finish() { printf '%s\n' "$1" > "$OUT/FINISH"; exit "${2:-0}"; }
box_bounded() {                  # box_bounded [--stall FILE MIN] NAME CUT_MIN LOG CMD...
    local name log
    [ "$1" != --stall ] || shift 3
    name=$1 log=$3
    shift 3
    echo "$name $*" >> "$OUT/steps.txt"
    BOX_RC=0 BOX_WHY=ok
    if [ "$name" = readout ]; then
        (cd "$REPO" && "$@") >> "$OUT/$log" 2>&1 || BOX_RC=$?
    else
        python "$FAKE" "$name" "$@" >> "$OUT/$log" 2>&1 || BOX_RC=$?
    fi
    [ "$BOX_RC" -eq 0 ] || BOX_WHY=failed
    return "$BOX_RC"
}
"""

# The fetch lists every match but those in STUB_UNFETCHED; a match's walk
# appends 40 rows (the games it lacks), or crashes when the match is in
# STUB_CRASH.
FAKE = r'''
import json, os, random, sys
name, args = sys.argv[1], sys.argv[2:]
def opt(flag):
    return args[args.index(flag) + 1]
if name == "fetch":
    skip = os.environ.get("STUB_UNFETCHED", "").split()
    with open(opt("--manifest"), "a") as fh:
        for spec in args[args.index("fetch") + 1:]:
            if spec.count("=") != 2 or spec.split("=")[0] in skip:
                continue
            match, path, _ = spec.split("=")
            fh.write(f"{match}\t/data/{match}.tar.gz\t100\t{path}\n")
    sys.exit(1 if skip else 0)
match = name.split()[1]
if match in os.environ.get("STUB_CRASH", "").split():
    sys.exit(3)
out = opt("--out")
done = {json.loads(line)["game"] for line in open(out) if line.strip()}
rng = random.Random(match)
with open(out, "a") as fh:
    for k in range(40):
        game = f"game_{match}_{k}"
        if game in done:
            continue
        S = {c: rng.gauss(0, 1) for c in ("hp", "material", "material_nonterminal", "terminal",
                                          "value_50", "value_100", "value_200")}
        z = 1 if S["material"] + rng.gauss(0, 1) > 0 else -1
        fh.write(json.dumps({"match": match, "game": game, "z": z, "decisive": True, "turns": 20,
                             "scenario_id": "map", "raw_temperature_a": 0.0, "raw_temperature_b": 0.0,
                             "S": S, "SS": {c: v * v for c, v in S.items()},
                             "counts": {"attacks": 3, "enumerated": 3}}) + "\n")
'''

PIN = "0123456789abcdef" * 4
MATCHES = " ".join(f"{m}=tier-b/run/games_{m}.tar.gz={PIN}" for m in ("m_a", "m_b", "m_c"))


def _block(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _default_matches() -> str:
    """The script's DEFAULT_MATCHES line through its MATCHES line."""
    return _block('DEFAULT_MATCHES="', "\nrows() {")


def _run(tmp_path: Path, out: Path, **env) -> dict:
    out.mkdir(exist_ok=True)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    for name in ("steps.txt", "FINISH"):
        (out / name).unlink(missing_ok=True)
    exports = "\n".join(f"export {k}='{v}'" for k, v in env.items())
    script = f"""set -uo pipefail
export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"
OUT="{bash_path(out)}" STAGE=stage_test FAKE="{bash_path(fake)}" REPO="{bash_path(ROOT)}"
DATA=/data JOBS=4 FETCH_CUT_MIN=20 WALK_CUT_MIN=20 WALK_STALL_MIN=10 READOUT_CUT_MIN=15
BOX_RC=0 BOX_WHY=ok
{exports}
{_default_matches()}
{STUBS}
{_block("# ---- the match list (begin)", "# ---- the match list (end)")}
{_block("# ---- the matches (begin)", "# ---- the matches (end)")}
"""
    path = tmp_path / "driver.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], cwd=tmp_path, capture_output=True, text=True, timeout=180)
    steps = (out / "steps.txt").read_text().splitlines() if (out / "steps.txt").exists() else []
    failed = out / "matches_failed.txt"
    return {"rc": done.returncode, "stderr": done.stderr, "steps": steps,
            "names": [" ".join(s.split()[:2]) if s.startswith("walk ") else s.split()[0] for s in steps],
            "finish": (out / "FINISH").read_text() if (out / "FINISH").exists() else "",
            "failed": failed.read_text() if failed.exists() else ""}


def test_every_match_is_fetched_against_its_pin_walked_alone_and_read(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out, MATCHES=MATCHES)
    assert got["rc"] == 0 and got["finish"].startswith("MATCH_LUCK_DONE"), (got["finish"], got["stderr"][-2000:])
    assert "walked 3 of 3; left out: none" in got["finish"]
    assert got["names"] == ["fetch", "walk m_a", "walk m_b", "walk m_c", "readout"]
    fetch = got["steps"][0]
    assert f"--data /data --manifest {bash_path(out)}/data.txt {MATCHES}" in fetch
    for match, step in zip(("m_a", "m_b", "m_c"), got["steps"][1:4]):
        assert step.endswith(f"--out {bash_path(out)}/rows.jsonl --jobs 4 --match {match}=/data/{match}.tar.gz")
    assert "--matches m_a m_b m_c --save-json" in got["steps"][-1]
    readout = json.loads((out / "readout.json").read_text())
    assert readout["pooled_matches"] == ["m_a", "m_b", "m_c"] and readout["pooled_facts"]["walked"] == 120


def test_a_match_that_fails_is_left_out_and_a_re_entry_keeps_its_failed_walk_out(tmp_path):
    out = tmp_path / "out"
    got = _run(tmp_path, out, MATCHES=MATCHES, STUB_CRASH="m_b", STUB_UNFETCHED="m_c")
    assert got["rc"] == 0 and got["finish"].startswith("MATCH_LUCK_DONE"), (got["finish"], got["stderr"][-2000:])
    assert "walked 1 of 3; left out: m_b m_c" in got["finish"]
    assert got["names"] == ["fetch", "walk m_a", "walk m_b", "readout"], "an unfetched match is not walked"
    assert "LEFT_OUT m_b: its walk failed, rc=3" in got["failed"] and "LEFT_OUT m_c: not fetched" in got["failed"]
    assert json.loads((out / "readout.json").read_text())["pooled_matches"] == ["m_a"]
    again = _run(tmp_path, out, MATCHES=MATCHES)
    assert again["rc"] == 0, (again["finish"], again["stderr"][-2000:])
    assert again["names"] == ["fetch", "walk m_c", "readout"], "only what is not done runs, a failed walk stays out"
    assert "walked 2 of 3; left out: m_b" in again["finish"]
    assert "--matches m_a m_c --save-json" in again["steps"][-1]
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 80 and {r["match"] for r in rows} == {"m_a", "m_c"}


@pytest.mark.parametrize("matches, why", [
    (f"m_a=tier-b/run/games_m_a.tar.gz={PIN} m_a=tier-b/run/games_m_b.tar.gz={PIN}", "m_a is named twice"),
    (f"m_a=tier-b/run/games_m_a.tar.gz={PIN[:-1]}", "is not NAME=HF_PATH.tar.gz=SHA256"),
    (f"m_a=tier-b/run/games_m_a.json={PIN}", "is not NAME=HF_PATH.tar.gz=SHA256"),
    ("tier-b/run/games_m_a.tar.gz", "is not NAME=HF_PATH.tar.gz=SHA256"),
])
def test_a_match_list_it_cannot_read_is_refused_before_any_step(tmp_path, matches, why):
    got = _run(tmp_path, tmp_path / "out", MATCHES=matches)
    assert got["rc"] == 1 and got["finish"].startswith("BAD_MATCHES") and why in got["finish"], got["finish"]
    assert not got["steps"]


def test_the_default_list_names_23_pinned_matches(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    script = f"""set -uo pipefail
OUT="{bash_path(out)}"
box_finish() {{ printf '%s\\n' "$1" > "$OUT/FINISH"; exit "${{2:-0}}"; }}
{_default_matches()}
{_block("# ---- the match list (begin)", "# ---- the match list (end)")}
printf '%s\\n' "${{MATCH_SPECS[@]}}"
"""
    path = tmp_path / "list.sh"
    path.write_bytes(script.encode())
    done = subprocess.run([BASH, bash_path(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and not (out / "FINISH").exists(), done.stderr
    specs = [line.split("=") for line in done.stdout.splitlines()]
    assert len(specs) == 23
    for column in range(3):
        assert len({s[column] for s in specs}) == 23, "names, paths and pins are each distinct"
