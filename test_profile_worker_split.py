#!/usr/bin/env python3
"""Worker-split profiler (2026-07-18): heartbeat parsing, per-device
rate math, and the split recommendation — measurements over
estimates for the cuda/cpu spool fleet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.profile_worker_split import (
    PEAK_SAFETY,
    HeartbeatSample,
    rates_by_device,
    read_heartbeats,
    recommend,
    trainer_peak_mb,
)


def _hb(worker, device, games, decisions, started, updated):
    return HeartbeatSample(worker=worker, device=device, games=games,
                           decisions=decisions, started=started,
                           updated=updated)


def test_read_heartbeats_skips_corrupt(tmp_path):
    (tmp_path / "w0.json").write_text(json.dumps(
        {"worker": 0, "device": "cuda", "games": 3, "decisions": 900,
         "started": 100.0, "updated": 700.0}), encoding="utf-8")
    (tmp_path / "w1.json").write_text("garbage", encoding="utf-8")
    hbs = read_heartbeats(tmp_path)
    assert list(hbs) == [0]
    assert hbs[0].device == "cuda" and hbs[0].decisions == 900


def test_windowed_rates_use_deltas_per_device():
    a = {0: _hb(0, "cuda", 2, 600, 0, 1000),
         1: _hb(1, "cpu", 1, 200, 0, 1000),
         2: _hb(2, "cpu", 1, 300, 0, 1000)}
    b = {0: _hb(0, "cuda", 4, 1800, 0, 2000),   # +1200 dec / 1000s
         1: _hb(1, "cpu", 2, 600, 0, 2000),     # +400
         2: _hb(2, "cpu", 2, 800, 0, 2000)}     # +500
    r = rates_by_device(a, b)
    assert r["cuda"]["workers"] == 1
    assert abs(r["cuda"]["decisions_per_s"] - 1.2) < 1e-9
    assert abs(r["cpu"]["decisions_per_s"] - 0.45) < 1e-9
    # games/h: cuda +2 games/1000s -> 7.2/h
    assert abs(r["cuda"]["games_per_h"] - 7.2) < 1e-9


def test_lifetime_rates_from_single_sample():
    r = rates_by_device({0: _hb(0, "cpu", 4, 1200, 0, 600)})
    assert abs(r["cpu"]["decisions_per_s"] - 2.0) < 1e-9


def test_recommend_fills_vram_cap_when_cuda_faster():
    rec = recommend(n_workers=48, r_cuda=1.5, r_cpu=0.5,
                    vram_total_mb=24000, peak_mb=12000,
                    per_worker_mb=600)
    assert rec["trainer_reserve_mb"] == int(12000 * PEAK_SAFETY)
    assert rec["k_cuda"] == (24000 - int(12000 * PEAK_SAFETY)) // 600
    assert 0 < rec["k_cuda"] < 48


def test_recommend_all_cpu_when_cuda_not_faster():
    rec = recommend(n_workers=48, r_cuda=0.5, r_cpu=0.6,
                    vram_total_mb=24000, peak_mb=8000,
                    per_worker_mb=600)
    assert rec["k_cuda"] == 0


def test_recommend_survives_missing_measurements():
    rec = recommend(n_workers=48, r_cuda=None, r_cpu=0.5,
                    vram_total_mb=None, peak_mb=None,
                    per_worker_mb=None)
    assert rec["k_cuda"] is None            # keep current config


def test_trainer_peak_prefers_peak_column(tmp_path):
    p = tmp_path / "hist.csv"
    p.write_text(
        "iter,gpu_mem_alloc_mb,gpu_mem_peak_mb\n"
        "0,7000,11000\n1,7200,12500\n2,7100,\n", encoding="utf-8")
    assert trainer_peak_mb(p) == 12500


def test_trainer_peak_falls_back_to_alloc(tmp_path):
    p = tmp_path / "hist.csv"
    p.write_text("iter,gpu_mem_alloc_mb\n0,7000\n1,7300\n",
                 encoding="utf-8")
    assert trainer_peak_mb(p) == 7300


def test_assign_devices_measured_cap_override():
    from tools.sim_self_play import _assign_spool_devices
    devices = _assign_spool_devices(
        48, "auto", cuda_available=True, total_vram=24 * 2**30,
        cuda_cap=13)
    assert devices.count("cuda") == 13
    assert devices.count("cpu") == 35
    # cap can't exceed the fleet
    assert _assign_spool_devices(
        4, "auto", cuda_available=True, total_vram=24 * 2**30,
        cuda_cap=99).count("cuda") == 4
