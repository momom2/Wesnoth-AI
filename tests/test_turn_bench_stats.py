"""The step-1 readout's statistics (wesnoth_ai/turn_bench_stats.py): the
turn-value experiment's estimators reproduce its recorded figures from the
committed benchmark arrays, the paired versions agree with them, and the
readings fire as pre-registered."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import critic_bench as bench  # noqa: E402
from wesnoth_ai import turn_bench_stats as S  # noqa: E402

RECORDS = Path(__file__).resolve().parent.parent / "training/metrics/turn_value_20260925"


@pytest.fixture(scope="module")
def arrays():
    return bench.load_arrays()


def _positions(a):
    return np.unique(a["index"], return_inverse=True)[1]


def test_the_ported_measure_reproduces_the_verdicts_figures(arrays):
    a = arrays
    verdict = json.loads((RECORDS / "verdict.json").read_text(encoding="utf-8"))["validation"]
    adjusted = S.adjusted_outcomes(a["outcomes"], a["luck"], a["turn_luck"], a["beta"])
    for name in ("hp_margin_post", "value_reference"):
        got = S.corrected_correlation(a[name], adjusted, _positions(a), a["cluster"])
        assert got["corrected"] == pytest.approx(verdict[name]["corrected"], abs=1e-12)
        assert got["corrected_se"] == pytest.approx(verdict[name]["corrected_se"], abs=1e-12)
        raw = S.corrected_correlation(a[name], a["outcomes"], _positions(a), a["cluster"])
        assert raw["corrected"] == pytest.approx(verdict[name]["raw"]["corrected"], abs=1e-12)
    assert verdict["hp_margin_post"]["corrected"] == pytest.approx(0.397, abs=5e-4)


def test_the_ported_measure_reproduces_the_static_margin_of_the_reads_by_depth_record(arrays):
    """Read 0's HP margin against the raw truth of playouts 9 to 28, and its
    selection gain over the base turn (reads_by_depth_20261007.txt)."""
    a = arrays
    text = (RECORDS / "reads_by_depth_20261007.txt").read_text(encoding="utf-8")
    corr = re.search(r"^\s+0\s+0\.000\s+\S+ \+- \S+\s+-\s+(\S+) \+- (\S+)", text, re.M)
    gain = re.search(r"truth: raw outcomes\n.*\n\s+0\s+(\S+) \+- (\S+)", text)
    truth = a["outcomes"][:, S.TRUTH_FROM:]
    got = S.corrected_correlation(a["hp_margin_post"], truth, _positions(a), a["cluster"])
    assert f"{got['corrected']:.3f}" == corr.group(1) == "0.456"
    assert f"{got['corrected_se']:.3f}" == corr.group(2)
    gains = S.selection_gains(a["hp_margin_post"], S.truth_mean(truth), _positions(a), a["slot"])
    mean, se = S.mean_and_se(np.array(list(gains.values())))
    assert (f"{mean:+.3f}", f"{se:.3f}") == (gain.group(1), gain.group(2))


def test_a_grader_paired_with_itself_differs_by_nothing(arrays):
    a = arrays
    truth = a["outcomes"][:, S.TRUTH_FROM:]
    grades = {"hp": a["hp_margin_post"], "same": a["hp_margin_post"].copy(), "head": a["value_reference"]}
    pairs = [("same", "hp"), ("head", "hp")]
    corr = S.paired_correlations(grades, truth, _positions(a), a["cluster"], pairs)
    gains = S.paired_gains(grades, S.truth_mean(truth), _positions(a), a["slot"], pairs)
    assert corr["differences"]["same-hp"] == {"mean": 0.0, "se": 0.0}
    assert gains["differences"]["same-hp"] == {"mean": 0.0, "se": 0.0}
    alone = S.corrected_correlation(a["value_reference"], truth, _positions(a), a["cluster"])
    assert corr["graders"]["head"]["corrected"] == pytest.approx(alone["corrected"], abs=1e-12)
    assert corr["differences"]["head-hp"]["se"] > 0 and gains["differences"]["head-hp"]["se"] > 0


def _stats(gain_pre=(0.0, 0.02), gain_read0=(0.0, 0.02), size=(0.0, 0.05)):
    def by(gain):
        return {"gains": {"differences": {f"{c}-hp_margin": {"mean": 0.0, "se": 0.02} for c in ("T25", "T100")}
                          | {"TH-hp_margin": {"mean": gain[0], "se": gain[1]}}},
                "correlations": {"differences": {"T100-T25": {"mean": size[0], "se": size[1]}}}}
    return {"read0": by(gain_read0), "pre": by(gain_pre)}


def test_the_readings_fire_as_registered():
    critics = ("T25", "T100", "TH")
    passing = S.readings(_stats(gain_pre=(0.05, 0.02)), critics)
    assert passing["reading"] == "Pass" and passing["passing"][0]["critic"] == "TH"
    assert passing["passing"][0]["read"] == "pre"
    assert S.readings(_stats(gain_pre=(0.04, 0.02)), critics)["reading"] != "Pass", "2 SE is not above 2 SE"
    assert S.readings(_stats(gain_read0=(0.05, 0.02), size=(0.2, 0.05)), critics)["reading"] == "Pass"
    limited = S.readings(_stats(size=(0.11, 0.05)), critics)
    assert limited["reading"] == "Data-limited" and "T100 over T25" in limited["why"]
    killed = S.readings(_stats(size=(0.10, 0.05)), critics)
    assert killed["reading"] == "Kill" and "closest" in killed["why"]


def test_the_simulated_benchmark_reads_clear_effects_as_registered(arrays):
    from tools import critic_oc as oc
    cal = oc.calibrate(arrays)
    rng = np.random.default_rng(0)
    strong = {c: 0.45 for c in oc.CRITICS}
    assert oc.simulate(arrays, cal, strong, 0.5, rng, 200, 400)["reading"] == "Pass"
    size = {"T25": -0.37, "T50": -0.25, "T100": -0.1, "O100": -0.3, "TH": -0.3, "Tsmall": -0.3}
    assert oc.simulate(arrays, cal, size, 0.5, rng, 200, 400)["reading"] == "Data-limited"


def test_a_position_whose_base_turn_did_not_rebuild_is_read_by_no_grader():
    arrays = {"index": np.array([0, 0, 1, 1]), "slot": np.array([0, 1, 0, 1])}
    raws = {"pre": {}, "read0": {}}
    positions = [{"index": 0, "candidates": [{"slot": 0, "raws": raws, "hp_margin_pre": 5, "hp_margin_post": 4},
                                             {"slot": 1, "raws": raws, "hp_margin_pre": 7, "hp_margin_post": 6}]},
                 {"index": 1, "candidates": [{"slot": 0, "digest_ok": False},
                                             {"slot": 1, "raws": raws, "hp_margin_pre": 1, "hp_margin_post": 1}]}]
    reads = {"T25": {"reads": [{"index": i, "slot": s, "read": "pre", "value": 0.1 * (i + s)}
                               for i in (0, 1) for s in (0, 1)]}}
    grades = bench.grades_at(arrays, positions, reads, "pre")
    assert grades["hp_margin"][:2].tolist() == [5, 7] and np.isnan(grades["hp_margin"][2:]).all()
    assert grades["T25"][:2].tolist() == [0.0, 0.1] and np.isnan(grades["T25"][2:]).all()
    assert bench.grades_at(arrays, positions, reads, "read0")["hp_margin"][:2].tolist() == [4, 6]


def test_a_turn_applies_each_rejection_before_the_command_it_preceded():
    from types import SimpleNamespace as NS
    calls = []
    cs = NS(core=NS(add_recruit_rejected=lambda x, y: calls.append(("reject", x, y))),
            apply_command=lambda cmd: calls.append(tuple(cmd[:1])))
    bench.apply_turn(cs, [["recruit"], ["move"]], [[1, 4, 5], [2, 6, 7]])
    assert calls == [("recruit",), ("reject", 4, 5), ("move",), ("reject", 6, 7)]
