"""The window's end (docs/window_end_prereg_20261104.md): once the after-stop
line has ended, a fresh-seed confirmation of the best candidate its final
matches read, then that candidate's T=1 tail. Logs to
C:/Users/momom2/wesnoth_runs/window_end.log. Run detached on the repository's
.venv:

    python C:/Users/momom2/wesnoth_runs/tools/window_end.py

1. Wait until neither chain_local.py nor after_stop_line.py runs (5-min polls).
2. Go on only if after_stop.log's last LINE line is LINE_DONE and chain.log's
   last CHAIN_DONE is followed by `CHAIN_FINAL_DONE: match_local returned 0`,
   or reads `the rounds returned 0` and is followed by CHAIN_NO_FINAL (a line
   re-armed after a design stop whose rounds found no time: amendment 1, the
   2026-11-04 21:43 review's H1); otherwise WINDOW_END_HELD (the heartbeat
   looks).
3. The candidate: of the complete final matches at seed base 900,000
   (final_match/ and final_match_after_stop*/, paired on their seeds), the
   highest Elo; its checkpoint's SHA-256 must still be the one it was read with.
4. The confirmation: scripts/match_local.py on it, 12,800 decisive games from
   seed base 980,000, into confirm_match/.
5. If that returned 0 and the candidate is not round 3's (whose tail is
   tail_match/): its tail, match_local --t1, 12,800 games from 950,000 (round
   3's tail's seeds, so the two pair), into tail_match_window/.
6. Then the extras (amendment 1, the review's L3), one at a time, each only
   after the last returned 0 and while no window_end.STOP file exists in the
   runs directory: if the candidate is not round 3, round 3 on the
   confirmation's seeds (into confirm_match_round03/, so the candidate's gain
   over round 3 reads paired); then each adopted claim round after round 3
   that no final match read (6, 9, 12, ..., ascending), at the final matches'
   seeds (into final_match_roundKK/, paired with every final): the argmax
   learning curve.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

RUNS = Path("C:/Users/momom2/wesnoth_runs")
RUN_TREE = Path("C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-run")
LOG = RUNS / "window_end.log"
GPU_PYTHON = "C:/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe"
TOOLS_PYTHON = "C:/Users/momom2/Desktop/perso/projects/wesnoth_ai/.venv/Scripts/python.exe"
FINAL_SEED_BASE = 900000                # the final matches' seeds (chain_local FINAL_SEED_BASE)
CONFIRM_SEED_BASE = 980000              # fresh: 980,000-992,799 and its +g * 1M replacements (the prereg)
TAIL_SEED_BASE = 950000                 # round 3's tail's seeds (chain_local TAIL_SEED_BASE)
GAMES = 12800                           # each match's decisive games, as the final and tail matches
PROCEDURE = "raw:t0+eo-1.5 against raw:t0+eo-1.5"
CLAIM_EVERY = 3                         # rounds_local CLAIM_EVERY: the claim rounds are 6, 9, 12, ...


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def running(script: str) -> bool:
    """Whether a python process's command line names `script`; True when the
    process list cannot be read."""
    q = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         f"$_.CommandLine -like '*{script}*' }}).Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True, timeout=120)
        return r.returncode != 0 or not r.stdout.strip() or int(r.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return True


def line_done() -> Tuple[bool, str]:
    """(the line ended with LINE_DONE and its last final match returned 0, why not)."""
    path = RUNS / "after_stop.log"
    lines = [x for x in (path.read_text(encoding="utf-8").splitlines() if path.exists() else [])
             if " LINE" in x]
    last = lines[-1] if lines else ""
    if " LINE_DONE" not in last:
        return False, f"after_stop.log's last LINE line is not LINE_DONE: {last or 'none'}"
    text = (RUNS / "chain.log").read_text(encoding="utf-8") if (RUNS / "chain.log").exists() else ""
    tail = text[text.rfind("CHAIN_DONE"):] if "CHAIN_DONE" in text else ""
    no_time = tail.startswith("CHAIN_DONE: the rounds returned 0") and "CHAIN_NO_FINAL" in tail
    if "CHAIN_FINAL_DONE: match_local returned 0" not in tail and not no_time:
        return False, f"chain.log's last CHAIN_DONE has no final match that returned 0: {tail.strip()[:300]}"
    return True, ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finals() -> List[dict]:
    """The complete final matches at the final seeds, each result.json with its
    directory and round."""
    out = []
    for d in sorted([RUNS / "final_match", *RUNS.glob("final_match_after_stop*")]):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        m = re.fullmatch(r"final_round(\d+)", str(r.get("candidate", "")))
        if not (r.get("complete") and r.get("seed_base") == FINAL_SEED_BASE and m
                and r.get("procedures") == [PROCEDURE]):
            log(f"WINDOW_END skips {d.name}: not a complete final match at {FINAL_SEED_BASE} ({r.get('candidate')})")
            continue
        out.append(dict(r, dir=d.name, round=int(m.group(1))))
    return out


def adopted() -> List[Tuple[int, str, str]]:
    """(k, candidate path, its SHA-256 as round.json records it) for every
    round whose round.json names a candidate, as chain_local's last_adopted
    reads them."""
    out = []
    for rj in (RUNS / "rounds_local").glob("round_*/round.json"):
        cand = json.loads(rj.read_text(encoding="utf-8")).get("candidate") or {}
        if cand.get("path"):
            out.append((int(rj.parent.name.split("_")[1]), str(cand["path"]), str(cand.get("sha256"))))
    return sorted(out)


def extras(k: int, read: List[dict]) -> int:
    """Step 6: round 3 on the confirmation's seeds unless it is the candidate,
    then the unread claim rounds at the final seeds; 0, or the first failure."""
    rounds = {j: (pi, sha) for j, pi, sha in adopted()}
    plan = []
    if k != 3 and 3 in rounds:
        plan.append((3, "confirm_round3", RUNS / "confirm_match_round03", CONFIRM_SEED_BASE))
    finalled = {r["round"] for r in read}
    for j in sorted(rounds):
        if j > 3 and j % CLAIM_EVERY == 0 and j not in finalled:
            plan.append((j, f"final_round{j}", RUNS / f"final_match_round{j:02d}", FINAL_SEED_BASE))
    log(f"WINDOW_END_EXTRAS: {', '.join(lab for _, lab, _, _ in plan) or 'none'}")
    for j, label, out, seeds in plan:
        if (RUNS / "window_end.STOP").exists():
            log(f"WINDOW_END_EXTRAS_STOPPED before {label}: window_end.STOP exists")
            return 0
        pi, recorded = rounds[j]
        digest = sha256(Path(pi))
        if digest != recorded:
            log(f"WINDOW_END_HELD: {pi}'s SHA-256 is {digest}, not round {j}'s recorded {recorded}")
            return 1
        rc = match(label, pi, out, ["--games", str(GAMES), "--seed-base", str(seeds)])
        if rc != 0:
            log(f"WINDOW_END_HELD: {label} returned {rc}; no further extras")
            return 1
    log("WINDOW_END_EXTRAS_DONE")
    return 0


def match(label: str, pi: str, out: Path, extra: List[str]) -> int:
    """match_local from the run tree, as chain_local plays its matches."""
    cmd = [GPU_PYTHON, "-u", "scripts/match_local.py", "--label", label, "--checkpoint", pi, "--out", str(out),
           *extra, "--tools-python", TOOLS_PYTHON, "--keep-awake"]
    out.mkdir(parents=True, exist_ok=True)
    log(f"WINDOW_END_MATCH {label} into {out}: {' '.join(cmd)}")
    env = dict(os.environ, HIP_VISIBLE_DEVICES="1")
    with open(out / "console.log", "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=RUN_TREE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"WINDOW_END_MATCH_DONE {label}: match_local returned {rc}")
    return rc


def main() -> int:
    log("WINDOW_END waiting for the after-stop line to end (docs/window_end_prereg_20261104.md)")
    while running("chain_local.py") or running("after_stop_line.py"):
        time.sleep(300)
    ok, why = line_done()
    if not ok:
        log(f"WINDOW_END_HELD: {why}; the lead looks")
        return 1
    read = finals()
    if not read:
        log("WINDOW_END_HELD: no complete final match to choose from; the lead looks")
        return 1
    for r in read:
        log(f"WINDOW_END final {r['dir']}: round {r['round']} {r['elo']:+.1f} +- {r['se']:.1f}")
    best: Optional[dict] = max(read, key=lambda r: r["elo"])
    k, pi = best["round"], best["checkpoint"]
    digest = sha256(Path(pi))
    if digest != best.get("checkpoint_sha256"):
        log(f"WINDOW_END_HELD: {pi}'s SHA-256 is {digest}, not the {best.get('checkpoint_sha256')} its final read")
        return 1
    log(f"WINDOW_END candidate: round {k} ({best['dir']}, {best['elo']:+.1f} +- {best['se']:.1f}), {pi}")
    rc = match(f"confirm_round{k}", pi, RUNS / "confirm_match",
               ["--games", str(GAMES), "--seed-base", str(CONFIRM_SEED_BASE)])
    if rc != 0:
        log(f"WINDOW_END_HELD: the confirmation returned {rc}; no tail")
        return 1
    if k == 3:
        log("WINDOW_END_DONE: the candidate is round 3, whose tail is tail_match/")
        return extras(k, read)
    rc = match(f"window_round{k}t1", pi, RUNS / "tail_match_window",
               ["--t1", "--games", str(GAMES), "--seed-base", str(TAIL_SEED_BASE)])
    log(f"WINDOW_END_DONE: the tail returned {rc}")
    return extras(k, read) if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
