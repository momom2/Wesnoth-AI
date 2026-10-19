"""tools/fp8_inference_readout.py: an arm fails for its match, its servers,
a compiled loop that fell back to eager or an fp8 arm served without fp8;
the reading takes the steady windows' rates (the whole run's when an arm
lacks one), is valid when the reference runs agree, and resolves the fp8
runs' gain only above their spread and MIN_GAIN; the play check reads PASS,
KILL below -2 standard errors, and NOT_READ when short or when its records
do not carry the sides' switches."""
from __future__ import annotations

import json

import pytest

from tools import fp8_inference_readout as ro

REF = "--worker-games 6 --compile-packed --bf16-residual"
FP8 = REF + " --fp8-inference-a --fp8-inference-b"


def _timeline(rate, leaves_per_batch=50, device_ms=12.0):
    """Rows every 5 s for 100 s at `rate` decisions a second from 10 s."""
    rows, requests, batches = [], 0, 0
    for t in range(5, 105, 5):
        if t > 10:
            requests += 5 * rate
            batches += 5 * rate / leaves_per_batch
        rows.append({"wall_s": float(t), "requests": requests, "batches": batches, "leaves": requests,
                     "gpu_ms": batches * device_ms, "idle_s": 0.0, "infer_s": batches * device_ms / 1e3})
    return rows


def _arm(out, name, flags, rate, *, rc=0, failed_games=0, timeline=True, **server):
    games = 600
    (out / f"arm_{name}.json").write_text(json.dumps(
        {"arm": name, "jobs": 120, "flags": flags, "slots": games, "wall_s": 130, "games": games,
         "failed_games": failed_games, "rc": rc, "why": "ok"}), encoding="utf-8")
    compiled = "--compile-packed" in flags
    stats = {"requests": rate * 100, "batches": rate * 2, "leaves": rate * 100, "wall_s": 100.0,
             "gpu_ms": rate * 2 * 12.0, "infer_s": rate * 2 * 0.013, "failed_requests": 0, "fatal": None,
             "compile_packed": compiled, "fp8": "--fp8-inference" in flags,
             **({"packed_compile_stats": {"active": True, "fallback_reason": None}} if compiled else {}), **server}
    (out / f"arm_{name}.inference_server_0.json").write_text(json.dumps(stats), encoding="utf-8")
    if timeline:
        (out / f"arm_{name}.inference_server_0.timeline.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in _timeline(rate)), encoding="utf-8")


def test_an_arm_fails_for_its_match_its_servers_its_compile_or_its_switch(tmp_path):
    _arm(tmp_path, "ref_1", REF, 2400)
    _arm(tmp_path, "fp8c_1", FP8, 2600, fp8=False)
    _arm(tmp_path, "fp8c_2", FP8, 2600, packed_compile_stats={"active": False, "fallback_reason": "guard"})
    _arm(tmp_path, "ref_2", REF, 2400, rc=1)
    _arm(tmp_path, "ref_3", REF, 2400, failed_games=2)
    _arm(tmp_path, "bf16e", "--bf16-residual", 2300, failed_requests=3)
    _arm(tmp_path, "fp8e", "--bf16-residual --fp8-inference-a --fp8-inference-b", 2350)
    rows = {n: ro.arm_row(tmp_path, n) for n in ("ref_1", "fp8c_1", "fp8c_2", "ref_2", "ref_3", "bf16e", "fp8e")}
    assert {n for n, r in rows.items() if r["failed"]} == {"fp8c_1", "fp8c_2", "ref_2", "ref_3", "bf16e"}
    assert any("did not play fp8" in p for p in rows["fp8c_1"]["problems"])
    assert any("guard" in p for p in rows["fp8c_2"]["problems"])
    assert rows["ref_1"]["window"]["decisions_per_s"] == 2400.0 and rows["ref_1"]["decisions_per_s"] == 2400.0
    assert rows["ref_1"]["device_ms_per_leaf"] == 0.24 and rows["ref_1"]["decisions_per_game"] == 400.0


@pytest.mark.parametrize("fp8_rate,refs,resolved,valid", [
    (2600, (2400, 2450, 2350), True, True),          # +8.3% over a 2% spread
    (2450, (2400, 2450, 2350), False, True),         # +2.1%: under MIN_GAIN
    (2600, (2400, 2600, 2200), False, False),        # the reference runs disagree by 8%
])
def test_the_reading_resolves_a_gain_only_above_the_reference_spread(tmp_path, fp8_rate, refs, resolved, valid):
    for i, rate in enumerate(refs, 1):
        _arm(tmp_path, f"ref_{i}", REF, rate)
    _arm(tmp_path, "fp8c_1", FP8, fp8_rate)
    _arm(tmp_path, "fp8c_2", FP8, fp8_rate)
    _arm(tmp_path, "profile_fp8c", FP8, 100)
    assert ro.main(["speed", str(tmp_path), "--spread", "0.05"]) == 0
    reading = json.loads((tmp_path / "speed.json").read_text(encoding="utf-8"))["reading"]
    assert reading["rates_from"] == "window" and "profile_fp8c" not in reading["rates"]
    assert reading["valid"] is valid and reading["fp8c_gain_resolved"] is resolved
    assert reading["fp8c_against_ref"] == round(fp8_rate / (sum(refs) / 3), 4)


def test_an_arm_without_a_window_reads_every_arm_on_the_whole_run(tmp_path):
    for i in (1, 2):
        _arm(tmp_path, f"ref_{i}", REF, 2400)
    _arm(tmp_path, "fp8c_1", FP8, 2640, timeline=False)
    reading = ro.reading({n: ro.arm_row(tmp_path, n) for n in ("ref_1", "ref_2", "fp8c_1")}, 0.05)
    assert reading["rates_from"] == "whole_run" and reading["fp8c_against_ref"] == 1.1


def _games(n_decisive, wins, *, switches=True, draws=0):
    games = [{"outcome_a": "win" if i < wins else "loss", "infer_fp8_a": switches, "infer_fp8_b": False,
              "forwards_a": 230, "forwards_b": 240} for i in range(n_decisive)]
    return games + [{"outcome_a": "draw", "infer_fp8_a": switches, "infer_fp8_b": False}] * draws


def _fit(elo_b, se):
    return {"tables": {ro.PURE: {"own1fp8": {"elo": 0.0, "se": 0.0}, "own1": {"elo": elo_b, "se": se}}}}


SERVERS = [{"fp8": True}, {"fp8": False}]


@pytest.mark.parametrize("games,fit,servers,verdict", [
    (_games(3200, 1590, draws=20), _fit(6.0, 6.1), SERVERS, "PASS"),
    (_games(3200, 1530), _fit(13.0, 6.1), SERVERS, "KILL"),
    (_games(3100, 1550), _fit(0.0, 6.2), SERVERS, "NOT_READ"),
    (_games(3200, 1600, switches=False), _fit(0.0, 6.1), SERVERS, "NOT_READ"),
    (_games(3200, 1600), _fit(0.0, 6.1), [{"fp8": True}], "NOT_READ"),
    (_games(3200, 1600), None, SERVERS, "NOT_READ"),
], ids=["pass", "kill", "short", "switches", "one server", "no fit"])
def test_the_play_check_reads_its_verdict(games, fit, servers, verdict):
    got = ro.play_readout(games, fit, servers, 3200, "own1fp8", "own1", rc=0)
    assert got["verdict"] == verdict, got
    if fit is not None:
        assert got["elo_fp8"] == -fit["tables"][ro.PURE]["own1"]["elo"]
    if verdict == "PASS":
        assert got["decisive"] == 3200 and got["games"] == 3220 and got["decisions_a_per_game"] == 230.0
