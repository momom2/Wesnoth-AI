"""tools/analysis/turn_gap_verdict.py on synthetic screen and
confirmation records (docs/turn_gap_ref_prereg_20260921.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.analysis import turn_gap_verdict as tv  # noqa: E402


def _record(index, base, alt=None, gap=None):
    rec = {"index": index, "base": {"outcomes": base, "mean": sum(base) / len(base)},
           "alternatives": []}
    if alt is not None:
        rec["alternatives"] = [{"outcomes": alt, "mean": sum(alt) / len(alt)}]
    rec["gap"] = (rec["alternatives"][0]["mean"] - rec["base"]["mean"]) if gap is None and alt is not None else (gap or 0.0)
    return rec


def _screen(n, hits):
    return {"positions": [_record(i, [0.0] * 4, gap=(0.6 if i in hits else 0.0)) for i in range(n)],
            "summary": {"frac_large": len(hits) / n}}


def test_a_confirmed_position_needs_the_gap_and_its_lower_bound():
    wins, losses, mixed = [1.0] * 40, [-1.0] * 40, [1.0, -1.0] * 20
    records = [
        _record(0, losses, wins),        # gap 2.0, SE 0: confirmed
        _record(1, mixed, mixed),        # gap 0: not
        # gap 0.5 on 80 playouts a side, SE 0.15: lower 2-SE bound 0.20 >= 0.10
        _record(2, mixed * 2, [1.0] * 60 + [-1.0] * 20),
        _record(3, [0.0] * 2, [0.3, 0.3]),            # gap 0.3, SE 0 on constant outcomes
        _record(4, mixed),                            # no alternative: not
    ]
    assert tv.confirmed_positions(records) == [0, 2, 3]
    assert tv.confirmed_positions(records, margin=0.35) == [0]


def test_verdict_bars_and_partial_runs(tmp_path):
    screen = _screen(60, hits={1, 5, 9, 20})
    losses, wins = [-1.0] * 20, [1.0] * 20
    kill = {"positions": [_record(1, losses, wins), _record(5, wins, wins)]}
    assert tv.verdict(screen, kill)["status"] == "KILL"
    sparse = {"positions": [_record(i, losses, wins) for i in (1, 5, 9)]}
    v = tv.verdict(screen, sparse)
    assert v["status"] == "SPARSE" and v["confirmed"] == [1, 5, 9] and v["nominal"] == [1, 5, 9, 20]
    assert abs(v["confirmed_fraction"] - 0.05) < 1e-9
    rich = {"positions": [_record(i, losses, wins) for i in range(6)]}
    assert tv.verdict(screen, rich)["status"] == "RICH"
    assert tv.verdict(screen, None)["status"] == "SCREEN ONLY"
    assert tv.verdict(_screen(30, hits={1}), sparse)["status"] == "PARTIAL"
    # The CLI reads the files and reports the same.
    s, c, out = tmp_path / "s.json", tmp_path / "c.json", tmp_path / "v.json"
    s.write_text(json.dumps(screen))
    c.write_text(json.dumps(sparse))
    assert tv.main([str(s), str(c), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["status"] == "SPARSE"
    assert tv.main([str(s), str(tmp_path / "missing.json")]) == 0
