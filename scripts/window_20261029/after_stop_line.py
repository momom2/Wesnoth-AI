"""Prereg departure 35's watcher for the after-stop line (scripts/chain_local.py
--after-stop K0, armed by arm_after_stop.ps1). Logs to
C:/Users/momom2/wesnoth_runs/after_stop.log beside after_stop.py's lines.
Run detached on the repository's .venv:

    python C:/Users/momom2/wesnoth_runs/tools/after_stop_line.py --head <commit> --k0 3 --held 5000

1. Wait for chain_local to exit.
2. Read rounds_local/STOP:
   - REDEPLOY (written by the lead; its text names "after round K0", so
     chain_local --after-stop K0 takes it and moves it aside): check out
     --head in the run tree and re-arm the line with its K0 and final-out.
     Back to 1.
   - a design stop at a later claim round K ("STOP after round K" or
     "PAUSE after round K", K > K0), with the line's final match on it
     returned 0 in chain.log: stamp round K's end at the STOP's time
     (rounds_local returns 3 before it stamps it), and re-arm the line
     rebased on K, its final match into final_match_after_stop_kKK.
     Departure 33's rule, applied again: the GPU does not idle to the
     window's end. Back to 1.
   - anything else: logged, and the watcher exits (the rounds ended or
     failed; the heartbeat looks).
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import after_stop as base  # noqa: E402

LEFT_H = 6.0                             # a round needs at least this long before --end-by (rounds 4+: 7.8 h)


def chain_alive() -> bool:
    q = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         "$_.CommandLine -like '*chain_local.py*' }).Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True, timeout=120)
        return r.returncode != 0 or not r.stdout.strip() or int(r.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return True


def arm(head: str, k0: int, held: int, end_by: str, final_out: str) -> str:
    subprocess.run(["git", "checkout", "-q", "--detach", head], cwd=base.RUN_TREE, check=True)
    return base.powershell("arm_after_stop.ps1", "-Head", head, "-K0", str(k0), "-Arm", "heads_only",
                           "-EndBy", end_by, "-Held", str(held), "-FinalOut", final_out)


def final_done() -> bool:
    """The line's final match after its last CHAIN_DONE returned 0."""
    path = base.RUNS / "chain.log"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    tail = text[text.rfind("CHAIN_DONE"):] if "CHAIN_DONE" in text else ""
    return "CHAIN_FINAL_DONE: match_local returned 0" in tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="the commit the line runs (prereg departure 35)")
    ap.add_argument("--k0", type=int, required=True, help="the running line's --after-stop")
    ap.add_argument("--held", type=int, required=True, help="the line's |H| (prereg departure 34)")
    ap.add_argument("--final-out", default="final_match_after_stop", help="the running line's final match, "
                    "under C:/Users/momom2/wesnoth_runs")
    ap.add_argument("--end-by", default="2026-11-08 02:00")
    a = ap.parse_args()
    k0, final_out = a.k0, a.final_out
    base.log(f"LINE watching the after-stop line (head {a.head}, K0 {k0}, |H| {a.held}, final {final_out}, "
             f"rounds end by {a.end_by}; prereg departure 35)")
    while True:
        while chain_alive():
            time.sleep(300)
        stop = base.ROUNDS / "STOP"
        why = stop.read_text(encoding="utf-8").strip() if stop.exists() else ""
        if why.startswith(base.REDEPLOY):
            if f"after round {k0}" not in why:
                base.log(f"LINE_HELD: the REDEPLOY stop does not name round {k0} ({why}); the lead looks")
                return 1
            first = arm(a.head, k0, a.held, a.end_by, final_out)
            base.log(f"LINE_REDEPLOYED at {a.head}, K0 {k0}, final {final_out}: {first}")
            if not first.startswith("rc=0 "):
                return 1
            time.sleep(300)
            continue
        m = re.search(r"(?:STOP|PAUSE) after round (\d+)", why)
        k = int(m.group(1)) if m else 0
        last = (base.RUNS / "chain.log").read_text(encoding="utf-8").strip().splitlines()[-1:]
        if k <= k0:
            base.log(f"LINE_DONE: the line exited with no later design stop (STOP: {why or 'none'}; chain.log "
                     f"ends: {last}); the heartbeat looks")
            return 0
        if not final_done():
            base.log(f"LINE_HELD: round {k}'s design stop, but its final match did not return 0 ({last}); "
                     f"the lead looks")
            return 1
        left_h = (datetime.strptime(a.end_by, base.FMT).timestamp() - time.time()) / 3600
        if left_h < LEFT_H:
            base.log(f"LINE_DONE: round {k}'s design stop with {left_h:.1f} h left before {a.end_by}: no round "
                     f"fits")
            return 0
        base.stamp_stop_end(k)
        k0, final_out = k, f"final_match_after_stop_k{k:02d}"
        first = arm(a.head, k0, a.held, a.end_by, final_out)
        base.log(f"LINE_REARMED after round {k0}'s design stop at {a.head}, final {final_out}: {first}")
        if not first.startswith("rc=0 "):
            return 1
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
