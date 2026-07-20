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


def _make_spool(tmp_path, monkeypatch, n=4, device="auto",
                cuda_cap=3):
    """SpoolWorkers with fake Popen and a forced cuda/cpu split
    (cuda_cap slots cuda, rest cpu) for demotion tests."""
    spawned = []
    terminated = []

    class _FakeProc:
        def __init__(self, argv):
            self.argv = argv
            self._rc = None

        def poll(self):
            return self._rc

        def terminate(self):
            terminated.append(self.argv)
            self._rc = -15

        def kill(self):
            self._rc = -9

        def wait(self, timeout=None):
            return self._rc

    def fake_popen(argv, **kw):
        p = _FakeProc(argv)
        spawned.append(p)
        return p

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
        spool_worker_device=device, spool_cuda_workers=cuda_cap,
    )
    import tools.sim_self_play as ssp
    monkeypatch.setattr(
        ssp, "_assign_spool_devices",
        lambda n_, mode, **kw: ["cuda"] * min(cuda_cap, n_)
        + ["cpu"] * max(0, n_ - cuda_cap))
    sw = SpoolWorkers(n, tmp_path / "spool", ckpt, args, "WARNING")
    return sw, spawned, terminated


def test_graceful_demotion_writes_ctl_and_respawns_on_cpu(
        tmp_path, monkeypatch):
    """Graceful path: highest cuda slot flips to cpu, the ctl file
    appears, the process is NOT killed; when the worker exits on its
    own (rc=0), ensure_alive respawns the slot with --device cpu."""
    sw, spawned, terminated = _make_spool(tmp_path, monkeypatch)
    assert sw._devices == ["cuda", "cuda", "cuda", "cpu"]
    assert sw.demote_one_cuda_worker("test")
    assert sw._devices == ["cuda", "cuda", "cpu", "cpu"]
    ctl = tmp_path / "spool" / "ctl" / "w2.device"
    assert ctl.read_text() == "cpu"
    assert terminated == []                     # graceful: no kill
    # Worker sees the ctl between games and exits cleanly...
    sw._procs[2]._rc = 0
    n_before = len(spawned)
    sw.ensure_alive()
    assert len(spawned) == n_before + 1
    argv = spawned[-1].argv
    assert argv[argv.index("--device") + 1] == "cpu"


def test_hard_demotion_terminates_and_respawns_immediately(
        tmp_path, monkeypatch):
    sw, spawned, terminated = _make_spool(tmp_path, monkeypatch)
    n_before = len(spawned)
    assert sw.demote_one_cuda_worker("oom", hard=True)
    assert len(terminated) == 1                 # old proc killed now
    assert len(spawned) == n_before + 1         # respawned now
    argv = spawned[-1].argv
    assert argv[argv.index("--device") + 1] == "cpu"


def test_demotion_with_no_cuda_workers_is_a_noop(tmp_path,
                                                 monkeypatch):
    sw, _, _ = _make_spool(tmp_path, monkeypatch, cuda_cap=0)
    assert not sw.demote_one_cuda_worker("test")


def test_headroom_guard_demotes_only_under_margin(tmp_path,
                                                  monkeypatch):
    """24GiB card, 3 cuda workers (1.875GiB): trainer peak 12GiB
    leaves ~10GiB headroom -> no demotion; peak 21GiB leaves under
    the 2GiB margin -> exactly one demotion per call."""
    sw, _, _ = _make_spool(tmp_path, monkeypatch)
    total = 24 * 2**30
    assert not sw.maybe_demote_for_headroom(
        12 * 1024, cuda_available=True, total_vram=total)
    assert sw._devices.count("cuda") == 3
    assert sw.maybe_demote_for_headroom(
        21 * 1024, cuda_available=True, total_vram=total)
    assert sw._devices.count("cuda") == 2
    # None peak (off-CUDA trainer) never demotes.
    assert not sw.maybe_demote_for_headroom(
        None, cuda_available=True, total_vram=total)


def test_stale_ctl_files_cleared_at_startup(tmp_path, monkeypatch):
    """A previous run's demotions must not apply to a fresh budget:
    pre-existing ctl files are removed when SpoolWorkers starts."""
    ctl_dir = tmp_path / "spool" / "ctl"
    ctl_dir.mkdir(parents=True)
    (ctl_dir / "w0.device").write_text("cpu")
    _make_spool(tmp_path, monkeypatch)
    assert not (ctl_dir / "w0.device").exists()


def test_worker_ctl_exit_protocol(tmp_path):
    from tools.selfplay_worker import _ctl_wants_exit
    ctl = tmp_path / "w0.device"
    assert not _ctl_wants_exit(ctl, "cuda")       # no file
    ctl.write_text("cpu")
    assert _ctl_wants_exit(ctl, "cuda")           # demote signal
    assert not _ctl_wants_exit(ctl, "cpu")        # already there
    ctl.write_text("garbage\n")
    assert not _ctl_wants_exit(ctl, "cuda")       # torn read ignored


def test_max_turns_jitter_bounds_and_passthrough(tmp_path,
                                                 monkeypatch):
    """_roll_max_turns stays in [min, max] (anti-horizon-gaming,
    2026-07-20) and the spool parent forwards --max-turns-min."""
    import random
    from tools.sim_self_play import _roll_max_turns
    rng = random.Random(7)
    draws = {_roll_max_turns(rng, 100, 60) for _ in range(300)}
    assert min(draws) >= 60 and max(draws) <= 100
    assert len(draws) > 20                    # actually jitters
    assert _roll_max_turns(rng, 100, None) == 100     # off by default
    assert _roll_max_turns(rng, 100, 100) == 100      # degenerate
    sw, spawned, _ = _make_spool(tmp_path, monkeypatch)
    assert "--max-turns-min" not in spawned[0].argv   # unset -> absent
