"""wesnoth_ai/pg_cross_play.py: the rung pg_step.decide leaves to a
cross-play match (design §5), the match's mean z read from its records, and
the decision it takes in half B's place."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai import pg_cross_play as xp  # noqa: E402
from wesnoth_ai import pg_step  # noqa: E402

SHA = "c" * 64


def _rung(multiplier: float, s: float, ess_b: bool) -> pg_step.Rung:
    reading = pg_step.Reading(0.1, 0.01, ("g0", "g1"), np.zeros(2), np.ones(2))
    halves = {"A": pg_step.HalfReading(reading, reading, 0.6, 0.25, True),
              "B": pg_step.HalfReading(reading, reading, 0.09 if not ess_b else 0.6, 0.5, ess_b)}
    return pg_step.Rung(multiplier, multiplier * 1e-3, s, multiplier * 1e-3, True, halves)


def _case():
    """decide's summary when the rung chosen on half A fails half B's check."""
    rungs = [_rung(0.5, 0.02, True), _rung(1.0, 0.03, True), _rung(2.0, 0.04, False)]
    rungs[2].halves["A"] = pg_step.HalfReading(pg_step.Reading(0.2, 0.01, ("g0", "g1"), np.zeros(2), np.ones(2)),
                                               pg_step.Reading(0.2, 0.01, ("g0", "g1"), np.zeros(2), np.ones(2)),
                                               0.35, 0.5, True)
    summary = pg_step.decide(rungs, 1e-3).summary()
    return summary, summary["rungs"]


def test_the_constants_are_pg_steps():
    assert xp.REVERT_SE == pg_step.REVERT_SE


def test_decides_failed_weight_check_is_the_cross_play_case_and_nothing_else_is():
    dec, rungs = _case()
    assert dec["adopt"] is False and dec["chosen"] == 2 and "cross-play" in dec["reason"]
    assert xp.cross_play_case(dec, rungs) == 2
    assert xp.cross_play_case({**dec, "adopt": True}, rungs) is None
    assert xp.cross_play_case({**dec, "chosen": None}, rungs) is None
    assert xp.cross_play_case({**dec, "chosen": 1}, rungs) is None            # its weights pass on half B
    assert xp.cross_play_case(dec, [{**r, "halves": {"A": r["halves"]["A"]}} for r in rungs]) is None
    assert xp.cross_play_case({"adopt": False, "chosen": 0}, [{"halves": {"B": {}}}]) is None   # no check read
    adopted = pg_step.decide([_rung(1.0, 0.03, True)], 1e-3)
    assert adopted.adopt and xp.cross_play_case(adopted.summary(), adopted.summary()["rungs"]) is None


def _game(games: Path, i: int, outcome: str, **over) -> None:
    r = {"label_a": "rung1", "label_b": "pi0", "checkpoint_sha256_a": SHA, "memory_a": 64, "procedure_a": "raw:t1",
         "procedure_b": "raw:t1", "outcome_a": outcome, **over}
    (games / f"game_rung1_pi0_s1_{i}.json").write_text(json.dumps(r), encoding="utf-8")
    (games / f"game_rung1_pi0_s1_{i}.game.jsonl.gz").write_bytes(b"")


def test_the_match_reads_wins_losses_and_caps_and_leaves_out_or_refuses_the_rest(tmp_path):
    for i, outcome in enumerate(["win"] * 6 + ["loss"] * 2 + ["draw", "timeout"]):
        _game(tmp_path, i, outcome)
    _game(tmp_path, 20, "timeout_kill")                                      # the wall clock: left out
    _game(tmp_path, 21, "win", checkpoint_sha256_a="d" * 64)                # another network: refused
    _game(tmp_path, 22, "win", procedure_b="raw:t0")                        # another decode: refused
    r = xp.mean_z(tmp_path, "rung1", SHA, 64)
    z = [1.0] * 6 + [-1.0] * 2 + [0.0, 0.0]
    assert r["games"] == 10 and r["games_left_out"] == 1 and r["games_refused"] == 2
    assert r["estimate"] == pytest.approx(0.4)
    assert r["se"] == pytest.approx(np.std(z, ddof=1) / math.sqrt(10))
    empty = tmp_path / "none"
    empty.mkdir()
    assert xp.mean_z(empty, "rung1", SHA, 64)["games"] == 0 and math.isnan(xp.mean_z(empty, "rung1", SHA, 64)["se"])


def test_the_match_takes_half_bs_place_in_decides_rule():
    dec, rungs = _case()
    up = xp.decide_by_match(dec, rungs, {"estimate": 0.15, "se": 0.01, "games": 10000})
    assert up["adopt"] is True and up["s"] == pytest.approx(rungs[2]["s"]) and up["next_eps"] == pytest.approx(2e-3)
    assert up["importance"] == {k: dec[k] for k in ("adopt", "s", "next_eps", "reason")}
    assert up["cross_play"]["estimate"] == 0.15 and up["chosen"] == 2 and up["rungs"] == rungs
    weak = xp.decide_by_match(dec, rungs, {"estimate": -0.015, "se": 0.01})        # above -2 SE: adopted
    assert weak["adopt"] is True
    for reading in ({"estimate": -0.025, "se": 0.01}, {"estimate": 0.2, "se": 0.0}, {"estimate": 0.2, "se": math.nan}):
        down = xp.decide_by_match(dec, rungs, reading)
        assert down["adopt"] is False and down["s"] == 0.0 and down["next_eps"] == pytest.approx(5e-4), reading
    with pytest.raises(ValueError, match="not the cross-play case"):
        xp.decide_by_match({**dec, "chosen": 1}, rungs, {"estimate": 0.1, "se": 0.01})


def test_a_rung_read_at_the_ranges_edge_sets_the_next_radius_to_the_edge():
    """A rung clipped to the range's edge (pg_step.ladder) is adopted at its
    target KL, not at its multiplier times the round's radius."""
    dec, rungs = _case()
    rungs[2] = {**rungs[2], "target_kl": 1.7e-3}
    up = xp.decide_by_match(dec, rungs, {"estimate": 0.15, "se": 0.01, "games": 10000})
    assert up["adopt"] is True and up["next_eps"] == pytest.approx(1.7e-3)
