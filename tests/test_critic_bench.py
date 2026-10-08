"""The step-1 benchmark readout (tools/critic_bench.py) on real benchmark
positions: the rebuild reproduces them and gates the run on them, a cut
states file resumes cleanly, and the statistics refuse a reading that some
grader did not read whole."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import critic_bench as bench  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
FIXTURE = Path(__file__).parent / "fixtures" / "critic_bench"


def _vocab(tmp_path):
    from helpers.parity_games import FACTION_IDS
    from tools.preencode_sequences import fresh_vocab
    types, _ = fresh_vocab()
    path = tmp_path / "ref.pt"
    torch.save({"unit_type_to_id": types, "faction_to_id": FACTION_IDS}, path)
    return path


def _records(tmp_path, indices, tamper=None):
    data = json.loads((FIXTURE / "positions.json").read_text(encoding="utf-8"))
    data["positions"] = [p for p in data["positions"] if p["index"] in indices]
    if tamper is not None:
        data["positions"][0]["base"]["pre_end_turn"]["digest"] = tamper
    path = tmp_path / "records.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _rebuild(tmp_path, records, out):
    return bench.main(["rebuild", "--records", str(records), "--dataset", str(FIXTURE),
                       "--vocab-from", str(_vocab(tmp_path)), "--out", str(out), "--workers", "0"])


@needs_core
def test_real_benchmark_positions_rebuild_to_their_recorded_states(tmp_path):
    out = tmp_path / "bench"
    assert _rebuild(tmp_path, _records(tmp_path, (0, 2, 103)), out) == 0
    positions = {p["index"]: p for p in bench.read_states(out / "states.pkl")}
    assert sorted(positions) == [0, 2, 103]
    cands = [c for p in positions.values() for c in p["candidates"]]
    assert all(c["digest_ok"] for c in cands if "skipped" not in c)
    assert [c["skipped"] for c in cands if "skipped" in c] == ["terminal_in_turn"], "(103, 0) ends the game"
    assert all(c["to_move"] == {"pre": 2, "read0": 1} for c in cands if "raws" in c)
    basilisk = [c for c in positions[2]["candidates"] if c.get("margin_ok") is False]
    assert basilisk and all(c["hp_margin_post_recorded"] == c["hp_margin_pre"] for c in basilisk), \
        "the benchmark read the statues' turn start there"
    summary = json.loads((out / "rebuild_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["encoded"] == summary["expected"]["encoded"] == 12
    assert summary["gate"] == []


@needs_core
def test_a_rebuild_that_misses_a_recorded_state_stops_the_run(tmp_path):
    out = tmp_path / "bench"
    assert _rebuild(tmp_path, _records(tmp_path, (0,), tamper="0" * 16), out) != 0
    summary = json.loads((out / "rebuild_summary.json").read_text(encoding="utf-8"))
    assert any("digest" in problem for problem in summary["gate"])


@needs_core
def test_a_cut_states_file_resumes_after_its_last_whole_record(tmp_path):
    out = tmp_path / "bench"
    assert _rebuild(tmp_path, _records(tmp_path, (0,)), out) == 0
    with (out / "states.pkl").open("ab") as f:
        f.write(pickle.dumps({"index": 2, "candidates": []})[:20])        # a record cut by a kill
    assert _rebuild(tmp_path, _records(tmp_path, (0, 2)), out) == 0
    assert sorted(p["index"] for p in bench.read_states(out / "states.pkl")) == [0, 2]


def _synthetic_run(out, arrays, drop=None):
    """A states file and two critics' reads covering every benchmark row,
    the HP margin as their grade plus noise; `drop` leaves one read out."""
    rng = np.random.default_rng(0)
    out.mkdir(parents=True)
    by_position = {}
    for k, (i, s) in enumerate(zip(arrays["index"], arrays["slot"])):
        margin = float(arrays["hp_margin_post"][k])
        by_position.setdefault(int(i), []).append({"slot": int(s), "raws": {"pre": {}, "read0": {}},
                                                   "hp_margin_pre": margin, "hp_margin_post": margin})
    with (out / "states.pkl").open("wb") as f:
        for i, cands in by_position.items():
            pickle.dump({"index": i, "candidates": cands}, f)
    for name in ("T25", "T100"):
        reads = [{"index": i, "slot": c["slot"], "read": r, "value": c["hp_margin_pre"] / 300 + rng.normal(0, 0.05)}
                 for i, cands in by_position.items() for c in cands for r in ("pre", "read0")]
        if drop is not None and name == "T100":
            reads = reads[:drop] + reads[drop + 1:]
        (out / f"reads_{name}.json").write_text(json.dumps({"critic": name, "reads": reads}), encoding="utf-8")


def test_the_reading_is_refused_when_a_grader_misses_a_rebuilt_row(tmp_path, capsys):
    arrays = bench.load_arrays()
    _synthetic_run(tmp_path / "whole", arrays)
    assert bench.main(["stats", "--out", str(tmp_path / "whole")]) == 0
    assert "READING " in capsys.readouterr().out
    _synthetic_run(tmp_path / "holed", arrays, drop=7)
    assert bench.main(["stats", "--out", str(tmp_path / "holed")]) != 0
    printed = capsys.readouterr().out
    assert "READING_REFUSED" in printed and "READING " not in printed.replace("READING_REFUSED", "")
