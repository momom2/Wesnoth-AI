#!/usr/bin/env python3
"""Spool-worker VRAM budgeting (2026-07-18 OOM incident).

56 all-cuda spool workers on a 24GB card starved the learner's
backward (318MB short) into a 3-death crash-loop. Worker devices are
now assigned by `_assign_spool_devices` under an explicit budget and
plumbed to each worker's argv; these tests pin the math and the
plumbing so a config change can't silently regress to all-cuda.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.sim_self_play import (
    SPOOL_WORKER_VRAM_BYTES,
    TRAINER_VRAM_RESERVE_BYTES,
    SpoolWorkers,
    _assign_spool_devices,
)


def test_auto_budget_on_24gb_card_caps_cuda_workers():
    """The incident configuration: 56 workers on a 23.52GiB 4090.
    auto must cap cuda workers under the trainer reserve -- never
    all-cuda. 13 = (23.52GiB - 15GiB reserve) // 640MiB (2026-07-20
    revision: reserve raised above the measured 12.6GiB trainer
    peak after the 19-cuda-worker OOM)."""
    total = int(23.52 * 2**30)
    devices = _assign_spool_devices(
        56, "auto", cuda_available=True, total_vram=total)
    n_cuda = devices.count("cuda")
    expected = (total - TRAINER_VRAM_RESERVE_BYTES) \
        // SPOOL_WORKER_VRAM_BYTES
    assert n_cuda == expected == 13
    assert devices.count("cpu") == 56 - n_cuda
    # cuda workers first (stable slot ordering for respawns).
    assert devices[:n_cuda] == ["cuda"] * n_cuda


def test_auto_with_few_workers_stays_all_cuda():
    devices = _assign_spool_devices(
        8, "auto", cuda_available=True, total_vram=24 * 2**30)
    assert devices == ["cuda"] * 8


def test_auto_without_cuda_is_all_cpu():
    assert _assign_spool_devices(8, "auto", cuda_available=False) \
        == ["cpu"] * 8


def test_cpu_mode_ignores_cuda_entirely():
    assert _assign_spool_devices(
        8, "cpu", cuda_available=True, total_vram=24 * 2**30) \
        == ["cpu"] * 8


def test_cuda_mode_is_unbudgeted_expert_override():
    assert _assign_spool_devices(
        56, "cuda", cuda_available=True, total_vram=24 * 2**30) \
        == ["cuda"] * 56


def test_tiny_card_yields_zero_cuda_workers():
    """A card smaller than the trainer reserve gives the trainer
    everything; all workers cpu."""
    assert _assign_spool_devices(
        8, "auto", cuda_available=True, total_vram=8 * 2**30) \
        == ["cpu"] * 8


def test_spool_workers_plumb_device_to_worker_argv(tmp_path,
                                                  monkeypatch):
    """SpoolWorkers must pass each worker its assigned --device, and
    a respawn must keep the dead worker's slot device."""
    spawned = []

    class _FakeProc:
        def poll(self):
            return None

    def fake_popen(argv, **kw):
        spawned.append(argv)
        return _FakeProc()

    import subprocess
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")
    args = SimpleNamespace(
        mcts_sims=2, mini_ratio=0.0, drill_ratio=0.0, max_turns=10,
        draw_tiebreak_cap=0.3, mcts_moves_left_utility=0.0,
        mcts_aux_value_bonus=0.0, fogless_ratio=0.0,
        midgame_ratio=0.0, ladder_ratio=1.0,
        midgame_dataset=Path("replays_dataset"),
        validate_export_every=0,
        validate_export_dir=tmp_path / "ve",
        train_draw_tiebreak=False, seed=0,
        spool_worker_device="cpu",   # deterministic on any machine
    )
    sw = SpoolWorkers(3, tmp_path / "spool", ckpt, args, "WARNING")
    assert len(spawned) == 3
    for argv in spawned:
        i = argv.index("--device")
        assert argv[i + 1] == "cpu"
    # Respawn keeps the slot's device.
    sw._procs[1] = None
    sw.ensure_alive()
    assert len(spawned) == 4
    i = spawned[-1].index("--device")
    assert spawned[-1][i + 1] == "cpu"
