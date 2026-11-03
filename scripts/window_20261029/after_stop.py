"""Prereg departure 33's watcher (amended 2026-11-03 before round 3's claim
read): if a design stop ends the rounds, the GPU does not idle to the
window's end. Every step is logged to C:/Users/momom2/wesnoth_runs/after_stop.log.
Run detached on the repository's .venv:

    python C:/Users/momom2/wesnoth_runs/tools/after_stop.py --head <commit with chain_local --after-stop>

1. Wait for the chain (chain_local.py without --after-stop) to exit.
2. Act only on a design stop: rounds_local/STOP names "STOP after round K0"
   or "PAUSE after round K0", and the chain's log shows the final match (and
   the tail) done after it. Anything else is logged and left to the lead.
3. Stamp round K0's end in rounds_local/times.json at the STOP file's time:
   rounds_local returns 3 before it stamps the end, and the resume would
   stamp it hours late (the final and tail matches between), so the round's
   wall would bound --end-by too long and end the rounds early.
4. Re-arm the chain at --head with --after-stop K0 and the heads-only arm
   (arm_after_stop.ps1), when a round fits before --rounds-end-by.

The paired arm reading (a full-network pass at --window 10^7, then the
reduced-AdamW step) was withdrawn by the amendment: pg_pass's window of
10^7 time steps holds a whole split (about 270k decisions in one graph),
which does not fit the card's 8 GB, and the reduced-AdamW metric would be
read from 8 split-level squares, so the arm would change with the window.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS = Path("C:/Users/momom2/wesnoth_runs")
ROUNDS = RUNS / "rounds_local"
RUN_TREE = Path("C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-run")
LOG = RUNS / "after_stop.log"
FMT = "%Y-%m-%d %H:%M"
HEADS_ROUND_H = 12.0                     # a heads-only round's wall, its claim aside (rounds 2 and 3: 9.6 and 11.4 h)


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def chain_alive() -> bool:
    q = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         "$_.CommandLine -like '*chain_local.py*' -and $_.CommandLine -notlike '*--after-stop*' }).Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True, timeout=120)
        return r.returncode != 0 or not r.stdout.strip() or int(r.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return True


def design_stop() -> int:
    """K0 when rounds_local's STOP is a design stop, else 0."""
    stop = ROUNDS / "STOP"
    why = stop.read_text(encoding="utf-8") if stop.exists() else ""
    m = re.search(r"(?:STOP|PAUSE) after round (\d+)", why)
    return int(m.group(1)) if m else 0


def stamp_stop_end(k0: int) -> None:
    """Step 3: round K0's end at the STOP file's time, unless one is there."""
    path = ROUNDS / "times.json"
    times = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entry = times.setdefault(str(k0), {})
    if "end" in entry:
        log(f"round {k0}'s end is already stamped ({entry['end']})")
        return
    entry["end"] = (ROUNDS / "STOP").stat().st_mtime
    path.write_text(json.dumps(times, indent=1), encoding="utf-8")
    wall = (entry["end"] - entry["start"]) / 3600 if "start" in entry else float("nan")
    log(f"round {k0}'s end stamped at the STOP's time, {time.strftime(FMT, time.localtime(entry['end']))} "
        f"(a wall of {wall:.1f} h)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="the commit with chain_local --after-stop (prereg departure 33)")
    ap.add_argument("--rounds-end-by", default="2026-11-08 02:00")
    a = ap.parse_args()
    log(f"AFTER_STOP waiting for the chain to exit (head {a.head}, rounds end by {a.rounds_end_by}; "
        f"amended: heads_only, no paired reading)")
    while chain_alive():
        time.sleep(300)
    k0 = design_stop()
    chain_log = (RUNS / "chain.log").read_text(encoding="utf-8") if (RUNS / "chain.log").exists() else ""
    tail = chain_log[chain_log.rfind("CHAIN_DONE"):] if "CHAIN_DONE" in chain_log else ""
    log(f"the chain exited; design stop at round {k0 or 'none'}; its log ends: "
        f"{chain_log.strip().splitlines()[-1] if chain_log.strip() else '(empty)'}")
    if not k0:
        log("AFTER_STOP_IDLE: no design stop ended the rounds; nothing to do")
        return 0
    if "CHAIN_FINAL_DONE: match_local returned 0" not in tail or "CHAIN_TAIL_DONE: match_local returned 0" not in tail:
        log("AFTER_STOP_HELD: the final match or the tail did not return 0 after the stop; the lead looks")
        return 1
    left_h = (datetime.strptime(a.rounds_end_by, FMT).timestamp() - time.time()) / 3600
    if left_h < HEADS_ROUND_H:
        log(f"AFTER_STOP_IDLE: {left_h:.1f} h left before {a.rounds_end_by}: no round fits")
        return 0
    stamp_stop_end(k0)
    subprocess.run(["git", "checkout", "-q", "--detach", a.head], cwd=RUN_TREE, check=True)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(RUNS / "tools" / "arm_after_stop.ps1"), "-Head", a.head, "-K0", str(k0), "-Arm",
                        "heads_only", "-EndBy", a.rounds_end_by], capture_output=True, text=True)
    log(f"AFTER_STOP_ARMED at {a.head}, round {k0}, arm heads_only: "
        f"{r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr[-300:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
