"""Stub test of after_stop_line.py: processes, git and PowerShell stubbed,
the runs directory a temporary copy."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "C:/Users/momom2/wesnoth_runs/tools")
import after_stop as base  # noqa: E402
import after_stop_line as line  # noqa: E402


def setup(stops, chain_logs, times):
    """stops: the STOP text the chain leaves at each exit (None: no STOP);
    chain_logs: chain.log's text at each exit."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "rounds_local").mkdir()
    base.RUNS, base.ROUNDS, base.LOG = tmp, tmp / "rounds_local", tmp / "after_stop.log"
    (base.ROUNDS / "times.json").write_text(json.dumps(times), encoding="utf-8")
    calls, exits = [], iter(range(len(stops)))
    state = {"i": -1}

    def alive():
        # each "exit": the chain leaves its STOP and log, then reads as dead once
        if state.get("dead"):
            state["dead"] = False
            return True
        i = next(exits, None)
        if i is None:
            raise SystemExit("ran out of exits")
        state["i"] = i
        stop = base.ROUNDS / "STOP"
        if stops[i] is None:
            stop.unlink(missing_ok=True)
        else:
            stop.write_text(stops[i] + "\n", encoding="utf-8")
        (tmp / "chain.log").write_text(chain_logs[i], encoding="utf-8")
        return False

    def arm(script, *args):
        calls.append((script, args))
        # chain_local --after-stop moves the STOP aside when it starts
        (base.ROUNDS / "STOP").unlink(missing_ok=True)
        return "rc=0 pid=123"

    class R:
        returncode = 0

    line.chain_alive = alive
    base.powershell = arm
    line.subprocess.run = lambda *a, **k: R()
    line.time.sleep = lambda s: None
    return tmp, calls


def argv(*extra):
    sys.argv = ["after_stop_line.py", "--head", "abc1234", "--k0", "3", "--held", "5000", *extra]


FINAL_OK = "x\nCHAIN_DONE: the rounds returned 3 (...)\nCHAIN_FINAL_DONE: match_local returned 0 (...)\n"
now = time.time()

# 1. REDEPLOY, then round 6's design stop with its final, then the rounds end
tmp, calls = setup(["REDEPLOY at abc1234 (prereg departure 35): the line rebased after round 3 resumes",
                    "STOP after round 6's claim: claim - round 3's claim reads ...",
                    None],
                   ["CHAIN_DONE: the rounds returned 2", FINAL_OK, FINAL_OK + "CHAIN_DONE: the rounds returned 0\n"],
                   {"6": {"start": now - 10 * 3600}})
argv("--end-by", time.strftime("%Y-%m-%d %H:%M", time.localtime(now + 48 * 3600)))
rc = line.main()
log = base.LOG.read_text(encoding="utf-8")
print(log)
assert rc == 0
assert [c[1][3] for c in calls] == ["3", "6"], calls                       # -K0
assert [c[1][-1] for c in calls] == ["final_match_after_stop", "final_match_after_stop_k06"], calls
assert "-Held" in calls[0][1] and calls[0][1][calls[0][1].index("-Held") + 1] == "5000"
assert "end" in json.loads((base.ROUNDS / "times.json").read_text(encoding="utf-8"))["6"]
assert "LINE_REDEPLOYED" in log and "LINE_REARMED after round 6" in log and "LINE_DONE" in log

# 2. a REDEPLOY that does not name the line's K0
tmp, calls = setup(["REDEPLOY of something else"], ["CHAIN_DONE: the rounds returned 2"], {})
argv()
assert line.main() == 1 and not calls and "LINE_HELD" in base.LOG.read_text(encoding="utf-8")

# 3. a design stop whose final match failed
tmp, calls = setup(["PAUSE after round 9: the last 3 ..."],
                   ["CHAIN_DONE: the rounds returned 3\nCHAIN_FINAL_DONE: match_local returned 1\n"], {})
argv()
assert line.main() == 1 and not calls and "LINE_HELD" in base.LOG.read_text(encoding="utf-8")

# 4. a design stop with no time left
tmp, calls = setup(["STOP after round 12's claim: ..."], [FINAL_OK], {"12": {"start": now - 3600}})
argv("--end-by", time.strftime("%Y-%m-%d %H:%M", time.localtime(now + 3 * 3600)))
assert line.main() == 0 and not calls and "no round fits" in base.LOG.read_text(encoding="utf-8")

# 5. the rounds' own STOP at an earlier round (the stop the line rebased on) is not a later stop
tmp, calls = setup(["STOP after round 3's claim: ..."], [FINAL_OK], {})
argv()
assert line.main() == 0 and not calls and "LINE_DONE" in base.LOG.read_text(encoding="utf-8")
print("all 5 cases pass")
