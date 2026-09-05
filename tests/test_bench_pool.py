"""tools/bench_pool.py: the packed_compile row certifies every server's
compiled packed trunk, not the learner's alone."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def test_packed_compile_row_is_active_only_when_every_server_is():
    from tools.bench_pool import merged_packed_compile
    ok = {"active": True, "recompiles": 0, "fallback_reason": None, "warmup_seconds": 1.0}
    fell = {"active": False, "recompiles": 2, "fallback_reason": "guard failure"}
    both = merged_packed_compile([ok, dict(ok, warmup_seconds=2.0)])
    assert both["active"] is True and both["recompiles"] == 0
    assert both["fallback_reason"] is None and len(both["per_server"]) == 2
    one_fell = merged_packed_compile([ok, fell])
    assert one_fell["active"] is False and one_fell["recompiles"] == 2
    assert one_fell["fallback_reason"] == "guard failure"
    assert merged_packed_compile([ok, None])["active"] is False   # stats lost: uncertified
    assert merged_packed_compile([{"active": False}])["active"] is False
