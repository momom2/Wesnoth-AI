"""Where a decision's time goes (wesnoth_ai/stage_timing.py) and its sum
over a match's workers (tools/worker_stages.py)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_stages_go_to_the_current_timer_of_their_thread():
    import threading
    from wesnoth_ai.stage_timing import StageTimer, stage, timing
    outer, inner = StageTimer(), StageTimer()
    with stage("nowhere"):
        pass
    with timing(outer):
        with stage("encode"):
            time.sleep(0.002)
        with timing(inner):
            with stage("encode"):
                pass
        with stage("step"):
            pass

        def other_thread():
            with stage("encode"):
                pass
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert outer.entries == {"encode": 1, "step": 1} and outer.seconds["encode"] >= 0.002
    assert inner.entries == {"encode": 1}
    back = StageTimer.from_dict(json.loads(json.dumps(outer.as_dict())))
    back.merge(inner)
    assert back.entries == {"encode": 2, "step": 1}


def test_a_match_summary_reads_its_workers_per_decision(tmp_path):
    from tools.worker_stages import summarize
    from wesnoth_ai.stage_timing import StageTimer
    for pid, (n, rt, step) in {11: (100, 2.0, 0.5), 12: (300, 3.0, 1.5)}.items():
        t = StageTimer()
        t.decisions, t.decision_wall_s, t.decision_cpu_s = n, 0.03 * n, 0.01 * n
        t.seconds = {"round_trip": rt, "step": step, "decide": rt + 0.4}
        t.entries = {"round_trip": n, "step": n, "decide": n}
        (tmp_path / f".worker_stages_{pid}.json").write_text(json.dumps(t.as_dict()), encoding="utf-8")
    s = summarize(tmp_path)
    assert s["workers"] == 2 and s["decisions"] == 400
    assert abs(s["decision_wall_ms"] - 30.0) < 1e-6 and abs(s["decision_cpu_ms"] - 10.0) < 1e-6
    assert abs(s["stage_ms_per_decision"]["round_trip"] - 12.5) < 1e-6
    assert abs(s["own_wall_ms"] - 17.5) < 1e-6
    assert abs(s["decide_other_ms"] - 2.0) < 1e-6        # decide less its sub-stages
