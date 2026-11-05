"""Stub test of window_end.py: processes, matches and hashing stubbed, the
runs directory a temporary copy. It tests the window_end.py beside it (the
tracked copy; the deployed one must be byte-equal), or the directory in
WINDOW_END_DIR."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ.get("WINDOW_END_DIR", str(Path(__file__).resolve().parent)))
import window_end as we  # noqa: E402

PROC = "raw:t0+eo-1.5 against raw:t0+eo-1.5"
FINAL_OK = "x\nCHAIN_DONE: the rounds returned 0 (...)\nCHAIN_FINAL: ...\nCHAIN_FINAL_DONE: match_local returned 0 (...)\n"


def setup(line_log, chain_log, results, digests=None, rcs=None, rounds=(), stop=False):
    tmp = Path(tempfile.mkdtemp())
    for k in rounds:
        (tmp / "rounds_local" / f"round_{k:02d}").mkdir(parents=True)
        (tmp / "rounds_local" / f"round_{k:02d}" / "round.json").write_text(json.dumps(
            {"candidate": {"path": f"C:/r/round_{k:02d}/pi.pt", "sha256": f"sha_round_{k:02d}"}}), encoding="utf-8")
    (tmp / "rounds_local" / "round_07").mkdir(parents=True, exist_ok=True)
    (tmp / "rounds_local" / "round_07" / "round.json").write_text(json.dumps({"candidate": None}), encoding="utf-8")
    if stop:
        (tmp / "window_end.STOP").write_text("x", encoding="utf-8")
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

NO_FINAL = FINAL_OK + "CHAIN_DONE: the rounds returned 0 (runner.log: ROUNDS_DONE through round 15: ...)\nCHAIN_NO_FINAL: no round after round 15 adopted a candidate\n"
# 8. a late re-arm whose rounds found no time: CHAIN_NO_FINAL after "returned 0" confirms
tmp, calls = setup(LINES_OK, NO_FINAL, {"final_match": res(3, 31.8), "final_match_after_stop_k06": res(15, 55.5)})
assert we.main() == 0 and [c[0] for c in calls] == ["confirm_round15", "window_round15t1"], calls

# 9. CHAIN_NO_FINAL after a failed rounds code holds
tmp, calls = setup(LINES_OK, FINAL_OK + "CHAIN_DONE: the rounds returned 1 (...)\nCHAIN_NO_FINAL: ...\n", {"final_match": res(3, 31.8)})
assert we.main() == 1 and not calls

R = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
# 10. the extras after the tail (amendment 2's order): the unread claim rounds latest first (12, 6; 9 has its
# final), then round 3 on the confirmation seeds
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(9, 45.0)}, rounds=R)
assert we.main() == 0, (tmp / "window_end.log").read_text()
assert [(c[0], c[1], c[2], c[3]) for c in calls] == [
    ("confirm_round9", "C:/r/round_09/pi.pt", "confirm_match", ["--games", "12800", "--seed-base", "980000"]),
    ("window_round9t1", "C:/r/round_09/pi.pt", "tail_match_window", ["--t1", "--games", "12800", "--seed-base", "950000"]),
    ("final_round12", "C:/r/round_12/pi.pt", "final_match_round12", ["--games", "12800", "--seed-base", "900000"]),
    ("final_round6", "C:/r/round_06/pi.pt", "final_match_round06", ["--games", "12800", "--seed-base", "900000"]),
    ("confirm_round3", "C:/r/round_03/pi.pt", "confirm_match_round03", ["--games", "12800", "--seed-base", "980000"])], calls

# 11. round 3 the candidate: no round-3 extra, the claim rounds 12, 9 and 6 at the final seeds
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8)}, rounds=R)
assert we.main() == 0 and [c[0] for c in calls] == ["confirm_round3", "final_round12", "final_round9", "final_round6"], calls

# 12. window_end.STOP: no extras
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(9, 45.0)}, rounds=R, stop=True)
assert we.main() == 0 and [c[0] for c in calls] == ["confirm_round9", "window_round9t1"], calls
assert "WINDOW_END_EXTRAS_STOPPED" in (tmp / "window_end.log").read_text()

# 13. an extra whose checkpoint changed holds
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(9, 45.0)}, rounds=R,
                   digests={"round_06": "other"})
assert we.main() == 1 and [c[0] for c in calls] == ["confirm_round9", "window_round9t1", "final_round12"], calls

# 14. a failed extra plays no further extras
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8), "final_match_after_stop": res(9, 45.0)}, rounds=R,
                   rcs=[0, 0, 0, 5])
assert we.main() == 1 and [c[0] for c in calls] == ["confirm_round9", "window_round9t1", "final_round12", "final_round6"], calls

# 15. amendment 2: chain_local's other CHAIN_NO_FINAL lines (a candidate that cannot be read, or whose SHA-256
# changed) after "returned 0" hold
for bad in ("CHAIN_NO_FINAL: C:/r/round_15/pi.pt's SHA-256 is aa, not round 15's bb\n",
            "CHAIN_NO_FINAL: round 15's candidate C:/r/round_15/pi.pt cannot be read: [Errno 2]\n"):
    tmp, calls = setup(LINES_OK, FINAL_OK + "CHAIN_DONE: the rounds returned 0 (...)\n" + bad, {"final_match": res(3, 31.8)})
    assert we.main() == 1 and not calls and "WINDOW_END_HELD" in (tmp / "window_end.log").read_text(), calls

# 16. amendment 2: the wait covers every script of the line, an orphan included
seen = []
tmp, calls = setup(LINES_OK, FINAL_OK, {"final_match": res(3, 31.8)})
we.running = lambda s: seen.append(s) or (s == "match_local.py" and seen.count(s) == 1)
assert we.main() == 0 and seen.count("match_local.py") == 2 and set(seen) == set(we.WAIT_FOR), seen
print("all 16 cases pass")
