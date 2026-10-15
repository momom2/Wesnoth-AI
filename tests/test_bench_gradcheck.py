"""The gradient check (tools/bench_gradcheck.py) checks each candidate on its
own: a candidate refused at construction and one that raises in its second
window are recorded as failed with their reasons and their windows so far,
and the candidate beside them is checked over every window and passes."""
from __future__ import annotations

import json

import pytest
from test_sequence_launches import QUIET


@pytest.mark.slow
def test_a_failing_candidate_leaves_the_others_checked(tmp_path, pass_inputs, monkeypatch):
    from tools import bench_sequence_step as bench
    from tools import sequence_train
    real = sequence_train.Trainer.train_window

    def breaks(self):
        if self.args.out.stem.endswith("_candidate1") and self.state["windows"] >= 1:
            raise RuntimeError("injected: out of memory")
        return real(self)
    monkeypatch.setattr(sequence_train.Trainer, "train_window", breaks)
    out = tmp_path / "check.json"
    code = bench.main(["gradcheck", "--out", str(out), "--windows", "3", "--reference=--packed-trunk",
                       "--candidate=--packed-trunk --fused-relu --fp8-trunk", "--candidate=--packed-trunk",
                       "--candidate=--packed-trunk --fused-relu", "--", *pass_inputs, *QUIET, "--window", "2",
                       "--out", str(tmp_path / "arm.pt")])
    assert code == 8
    refused, broken, fused = json.loads(out.read_text(encoding="utf-8"))["candidates"]
    assert not refused["pass"] and refused["error"].startswith("build: SystemExit") and "not both" in refused["error"]
    assert refused["windows"] == []
    assert not broken["pass"] and broken["error"] == "window 1: RuntimeError: injected: out of memory"
    assert [w["i"] for w in broken["windows"]] == [0]
    assert fused["pass"] and [w["i"] for w in fused["windows"]] == [0, 1, 2] and "error" not in fused
