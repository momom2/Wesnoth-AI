"""The interleaved bench (tools/bench_interleave.py): every arm against the
reference round by round, A/A pairs of identical flags judging the session,
rounds of other positions flagged; on the test pass, every arm runs the same
windows in rotated order, the fused ReLU's losses are the packed trunk's,
and an arm that fails, to build or in a window, leaves the others timed."""
from __future__ import annotations

import json

import pytest
from test_sequence_launches import QUIET

from tools import bench_interleave


def _rows(times, positions=None, flags=None):
    """Rows of an interleaved run: per arm its window times over the rounds."""
    flags = flags or {}
    rows = [{"kind": "config", "arm": arm, "flags": flags.get(arm, arm)} for arm in times]
    for arm, seconds in times.items():
        for i, s in enumerate(seconds):
            pos = (positions or {}).get(arm, [512] * len(seconds))[i]
            rows.append({"kind": "window", "arm": arm, "i": i, "seconds": s, "positions": pos,
                         "tokens_real": 1000 + i, "peak_gb": 10.0 + i, "kept_steps": 16})
    return rows


def test_the_summary_times_each_arm_against_the_reference_and_judges_the_session_by_its_aa_pairs():
    ref = [1.0, 1.1, 0.9, 1.3, 1.0]
    times = {"ref": ref, "ref2": [t * 1.01 for t in ref], "fast": [t * 0.8 for t in ref]}
    rows = _rows(times, flags={"ref": "--x", "ref2": "--x", "fast": "--x --y"})
    rows.append({"kind": "warmup", "arm": "fast", "seconds": [30.0, 0.85], "trunk_block": None})
    rows.append({"kind": "error", "arm": "fast", "i": 7, "error": "OutOfMemoryError", "message": "m"})
    out = bench_interleave.summarize(rows, spread=0.03)
    assert out["reference"] == "ref" and out["valid"] is True
    (pair,) = out["aa"]
    assert (pair["a"], pair["b"]) == ("ref", "ref2") and pair["window_time_ratio_median"] == pytest.approx(1.01)
    fast = out["arms"]["fast"]
    assert fast["against_reference"]["speedup_median"] == pytest.approx(1.25)
    assert fast["against_reference"]["same_positions"] and fast["against_reference"]["rounds"] == 5
    assert fast["positions_per_second"] == pytest.approx(5 * 512 / (0.8 * sum(ref)), rel=1e-3)
    assert fast["first_window_extra_seconds"] == pytest.approx(30.0 - 0.8, abs=0.01)
    assert fast["error"]["error"] == "OutOfMemoryError" and fast["peak_gb_max"] == 14.0
    assert "against_reference" not in out["arms"]["ref"]


def test_a_disagreeing_aa_pair_invalidates_the_session_and_other_positions_are_flagged():
    ref = [1.0, 1.1, 0.9, 1.3, 1.0]
    times = {"ref": ref, "ref2": [t * 1.05 for t in ref], "other": ref}
    out = bench_interleave.summarize(_rows(times, positions={"other": [512, 512, 500, 512, 512]},
                                           flags={"ref": "--x", "ref2": "--x", "other": "--z"}), spread=0.03)
    assert out["valid"] is False and not out["aa"][0]["agree"]
    assert out["arms"]["other"]["against_reference"]["same_positions"] is False
    alone = bench_interleave.summarize(_rows({"ref": ref, "other": ref}))
    assert alone["valid"] is None and alone["aa"] == []


def test_the_arms_are_named_once_with_their_flags():
    assert bench_interleave.parse_arms(["a=--packed-trunk  --x", "b="]) == [("a", "--packed-trunk --x"), ("b", "")]
    for bad in (["a=--x"], ["a=--x", "a=--y"], ["a b=--x", "c="], ["nameless"]):
        with pytest.raises(SystemExit):
            bench_interleave.parse_arms(bad)


@pytest.mark.slow
def test_an_interleaved_run_times_every_arm_on_the_same_windows(tmp_path, pass_inputs, monkeypatch):
    """Four arms on the test pass, one of which fails in its second round:
    the others run every round, the order rotated by one arm a round, on the
    same positions;
    the fused ReLU's losses are the packed trunk's; the summary pairs the two
    identical arms and leaves the failed arm's later rounds out."""
    from tools import bench_sequence_step as bench
    real = bench_interleave.timed_window
    calls = {"bad": 0}

    def flaky(trainer):
        if trainer.args.out.stem == "arm_bad":
            calls["bad"] += 1
            if calls["bad"] == 3:                   # 1 warm-up, round 0, then round 1 fails
                raise RuntimeError("injected")
        return real(trainer)
    monkeypatch.setattr(bench_interleave, "timed_window", flaky)
    out = tmp_path / "rows.jsonl"
    code = bench.main(["interleave", "--out", str(out), "--arm", "ref=--packed-trunk", "--arm", "ref2=--packed-trunk",
                       "--arm", "fused=--packed-trunk --fused-relu", "--arm", "bad=--packed-trunk", "--warmup", "1",
                       "--rounds", "3", "--profile-windows", "0", "--not-production", "--summary",
                       str(tmp_path / "speed.json"), "--", *pass_inputs, *QUIET, "--window", "1"])
    assert code == 0
    rows = bench_interleave.read_rows(out)
    assert [r["arm"] for r in rows if r["kind"] == "config"] == ["ref", "ref2", "fused", "bad"]
    windows = [r for r in rows if r["kind"] == "window"]
    assert [r["arm"] for r in windows if r["i"] == 0] == ["ref", "ref2", "fused", "bad"]
    assert [r["arm"] for r in windows if r["i"] == 1] == ["ref2", "fused", "ref"]
    assert [r["arm"] for r in windows if r["i"] == 2] == ["fused", "ref", "ref2"]
    by = {(r["arm"], r["i"]): r for r in windows}
    for i in range(3):
        assert by[("fused", i)]["loss"] == by[("ref", i)]["loss"]
        assert by[("fused", i)]["tokens_real"] == by[("ref", i)]["tokens_real"]
    (error,) = [r for r in rows if r["kind"] == "error"]
    assert (error["arm"], error["i"], error["message"]) == ("bad", 1, "injected")
    summary = json.loads((tmp_path / "speed.json").read_text(encoding="utf-8"))
    assert [(p["a"], p["b"]) for p in summary["aa"]] == [("ref", "ref2")]      # bad ran one round
    assert summary["arms"]["fused"]["against_reference"]["rounds"] == 3
    assert summary["arms"]["fused"]["against_reference"]["same_positions"]
    assert summary["arms"]["bad"]["windows"] == 1 and summary["arms"]["bad"]["against_reference"] is None
    assert rows[-1]["kind"] == "summary"


@pytest.mark.slow
def test_an_arm_that_fails_to_build_or_exits_leaves_the_others_timed(tmp_path, pass_inputs, monkeypatch):
    """The reference refused at construction (fp8 with the fused ReLU) and an
    arm that exits in its first round (a pass out of positions): both
    recorded with their stage, the two others timed every round and paired,
    with no ratio to the missing reference."""
    from tools import bench_sequence_step as bench
    real = bench_interleave.timed_window

    def quits(trainer):
        if trainer.args.out.stem == "arm_quits" and trainer.state["windows"] >= 1:
            raise SystemExit("the pass ran out of positions before the bench's last window")
        return real(trainer)
    monkeypatch.setattr(bench_interleave, "timed_window", quits)
    out = tmp_path / "rows.jsonl"
    code = bench.main(["interleave", "--out", str(out), "--arm", "refused=--packed-trunk --fused-relu --fp8-trunk",
                       "--arm", "ok=--packed-trunk", "--arm", "ok2=--packed-trunk", "--arm", "quits=--packed-trunk",
                       "--reference", "refused", "--warmup", "1", "--rounds", "2", "--profile-windows", "0",
                       "--not-production", "--summary", str(tmp_path / "speed.json"), "--", *pass_inputs, *QUIET,
                       "--window", "1"])
    assert code == 0
    rows = bench_interleave.read_rows(out)
    errors = {r["arm"]: r for r in rows if r["kind"] == "error"}
    assert errors["refused"]["stage"] == "build" and "not both" in errors["refused"]["message"]
    assert errors["quits"]["stage"] == "window" and errors["quits"]["i"] == 0
    windows = [r for r in rows if r["kind"] == "window"]
    assert sorted((r["arm"], r["i"]) for r in windows) == [("ok", 0), ("ok", 1), ("ok2", 0), ("ok2", 1)]
    summary = json.loads((tmp_path / "speed.json").read_text(encoding="utf-8"))
    assert summary["arms"]["refused"]["windows"] == 0 and summary["arms"]["refused"]["error"]["stage"] == "build"
    assert summary["arms"]["ok"]["against_reference"] is None and summary["arms"]["ok"]["windows"] == 2
    assert [(p["a"], p["b"]) for p in summary["aa"]] == [("ok", "ok2")]
