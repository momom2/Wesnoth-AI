"""Stub test of window_end.py: processes, matches and hashing stubbed, the
runs directory a temporary copy."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "C:/Users/momom2/wesnoth_runs/tools")
import window_end as we  # noqa: E402

PROC = "raw:t0+eo-1.5 against raw:t0+eo-1.5"
FINAL_OK = "x\nCHAIN_DONE: the rounds returned 0 (...)\nCHAIN_FINAL: ...\nCHAIN_FINAL_DONE: match_local returned 0 (...)\n"


def setup(line_log, chain_log, results, digests=None, rcs=None):
    tmp = Path(tempfile.mkdtemp())
    we.RUNS, we.LOG = tmp, tmp / "window_end.log"
    (tmp / "after_stop.log").write_text(line_log, encoding="utf-8")
    (tmp / "chain.log").write_text(chain_log, encoding="utf-8")
    for d, r in results.items():
        (tmp / d).mkdir()
        (tmp / d / "result.json").write_text(json.dumps(r), encoding="utf-8")
    calls, rcs = [], list(rcs or [])
    polls = iter([True, True, False, False])

    def running(script):
        return next(polls, False)

    def match(label, pi, out, extra):
        calls.append((label, pi, out.name, extra))
        return rcs.pop(0) if rcs else 0

    we.running = running
    we.match = match
    we.time.sleep = lambda s: None
    we.sha256 = lambda p: (digests or {}).get(Path(p).parent.name, "sha_" + Path(p).parent.name)
    return tmp, calls


def res(k, elo, ck=None, **kw):
    r = {"candidate": f"final_round{k}", "elo": elo, "se": 3.0, "complete": True, "seed_base": 900000,
         "procedures": [PROC], "checkpoint": ck or f"C:/r/round_{k:02d}/pi.pt", "checkpoint_sha256": f"sha_round_{k:02d}"}
    r.update(kw)
    return r


LINES_OK = "a LINE watching ...\nb AFTER_STOP x\nc LINE_REARMED after round 6\nd LINE_DONE: round 15's design stop ...\n"

# 1. a later round reads best: its confirmation, then its tail
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(6, 40.0),
                                        "final_match_after_stop_k06": res(15, 55.5)})
assert we.main() == 0, (tmp / "window_end.log").read_text()
assert [(c[0], c[2], c[3]) for c in calls] == [
    ("confirm_round15", "confirm_match", ["--games", "12800", "--seed-base", "980000"]),
    ("window_round15t1", "tail_match_window", ["--t1", "--games", "12800", "--seed-base", "950000"])], calls
assert calls[0][1] == "C:/r/round_15/pi.pt"

# 2. round 3 reads best: the confirmation alone
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(14, 20.0)})
assert we.main() == 0 and [c[0] for c in calls] == ["confirm_round3"], calls

# 3. the line did not end with LINE_DONE
tmp, calls = setup(LINES_OK + "e LINE_HELD: ...\n", FINAL_OK, {"final_match": res(3, 31.8)})
assert we.main() == 1 and not calls and "WINDOW_END_HELD" in (tmp / "window_end.log").read_text()

# 4. the last final match failed
tmp, calls = setup(LINES_OK, FINAL_OK + "CHAIN_DONE: the rounds returned 0\nCHAIN_FINAL_DONE: match_local returned 1\n",
                   {"final_match": res(3, 31.8)})
assert we.main() == 1 and not calls

# 5. the candidate's checkpoint changed since its final read it
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(9, 45.0)},
                   digests={"round_09": "other"})
assert we.main() == 1 and not calls and "SHA-256" in (tmp / "window_end.log").read_text()

# 6. an incomplete final and a final at other seeds are skipped
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8),
                                        "final_match_after_stop": res(9, 90.0, complete=False),
                                        "final_match_after_stop_k09": res(12, 80.0, seed_base=1)})
assert we.main() == 0 and [c[0] for c in calls] == ["confirm_round3"], calls

# 7. a confirmation that fails plays no tail
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(9, 45.0)}, rcs=[1])
assert we.main() == 1 and [c[0] for c in calls] == ["confirm_round9"], calls
print("all 7 cases pass")
