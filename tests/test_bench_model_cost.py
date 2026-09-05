"""tools/bench_model_cost.py: the leaves/s a row reports is the rate
the ms column next to it implies (the same statistic, the median)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def test_leaves_per_s_derives_from_the_median_the_table_shows(monkeypatch):
    from tools import bench_model_cost as bmc
    monkeypatch.setattr(bmc, "time_batches", lambda *a, **k: {
        "n_batches": 3, "pad_ratio": 1.0, "ms_median": 20.0, "ms_mean": 40.0})
    encoded = [SimpleNamespace(hex_tokens=torch.zeros(1, 10, 1),
                               unit_tokens=torch.zeros(1, 2, 1),
                               recruit_tokens=torch.zeros(1, 1, 1))]
    row = bmc.measure_row("x", None, encoded, 16, lambda: None,
                          {"d_model": 32, "layers": 1, "d_ff": 64})
    assert row["leaves_per_s"] == 1000.0 * 16 / 20.0
    assert row["leaves_per_s_mean"] == 1000.0 * 16 / 40.0
    cells = [c.strip() for c in bmc.markdown_table([row]).splitlines()[-1].split("|")]
    assert cells[9] == "20.0" and cells[10] == "800"
