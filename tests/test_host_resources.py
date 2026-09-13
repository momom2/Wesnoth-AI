"""Container-aware concurrency sizing (tools/host_resources.py).

Boxes are rented per leg, so --jobs is auto-derived per box. The
trap under test: host-wide readings (nproc, /proc/meminfo) lie on
shared hosts -- the cgroup files are authoritative, and the guard
must take the BINDING minimum.

The pids tests carry a second trap. The pids controller counts
TASKS, not processes, and exceeding it does not degrade gracefully:
a 2026-09-04 run at 38 actors produced ZERO leaves per second and
lost the rental. The clamp must therefore be derived from the file,
and must leave the count ALONE when no limit is readable -- a guessed
cap would cost throughput on every box that could take more.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import host_resources as hr  # noqa: E402


def _cg(tmp_path, **files) -> str:
    for rel, content in files.items():
        p = tmp_path / rel.replace("__", "/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="ascii")
    return str(tmp_path)


def test_effective_cores_v2_quota(tmp_path):
    root = _cg(tmp_path, **{"cpu.max": "1800000 100000"})
    assert hr.effective_cores(root) == min(
        18.0, float(__import__("os").cpu_count() or 1))


def test_effective_cores_unlimited_falls_back_to_host(tmp_path):
    import os
    root = _cg(tmp_path, **{"cpu.max": "max 100000"})
    assert hr.effective_cores(root) == float(os.cpu_count() or 1)


def test_cgroup_headroom_v2(tmp_path):
    gib = 1024 ** 3
    root = _cg(tmp_path, **{"memory.max": str(8 * gib),
                            "memory.current": str(6 * gib)})
    assert hr._cgroup_headroom_mb(root) == 2048.0


def test_cgroup_headroom_v1_unlimited_is_none(tmp_path):
    root = _cg(tmp_path, **{
        "memory__memory.limit_in_bytes": str(1 << 60),
        "memory__memory.usage_in_bytes": str(1024 ** 3)})
    assert hr._cgroup_headroom_mb(root) is None


def test_available_takes_binding_minimum(tmp_path, monkeypatch):
    """Host says 100GB free; OUR cgroup has 2GB headroom. The 2GB
    must win -- this is exactly the reading the old guard got
    wrong on shared Vast hosts."""
    gib = 1024 ** 3
    root = _cg(tmp_path, **{"memory.max": str(8 * gib),
                            "memory.current": str(6 * gib)})
    monkeypatch.setattr(hr, "_host_available_mb",
                        lambda: 100 * 1024.0)
    assert hr.available_mb(root) == 2048.0


def test_auto_jobs_min_of_constraints(tmp_path, monkeypatch):
    # 16 cores -> 8 by cpu; 9.5GB avail - 1.5GB reserve -> 4 by ram
    # at 2GB/job; vram 2.5GB -> 4 by vram at 600MB. Min = 4.
    gib = 1024 ** 3
    root = _cg(tmp_path, **{"cpu.max": "1600000 100000",
                            "memory.max": str(10 * gib),
                            "memory.current": str(gib // 2)})
    monkeypatch.setattr(hr, "_host_available_mb", lambda: 1e6)
    monkeypatch.setattr(hr, "vram_free_mb", lambda: 2500.0)
    jobs, how = hr.auto_jobs(per_job_mb=2000.0, per_job_vram_mb=600.0,
                             root=root)
    assert jobs == 4, how
    # CPU-only sizing (no vram constraint) on the same box: ram
    # still binds at 4.
    jobs2, _ = hr.auto_jobs(per_job_mb=2000.0, root=root)
    assert jobs2 == 4


def test_auto_jobs_never_zero(tmp_path, monkeypatch):
    """A cramped box degrades to 1 job, never 0 (the run must still
    make progress; the runtime floor guard handles true OOM risk)."""
    gib = 1024 ** 3
    root = _cg(tmp_path, **{"cpu.max": "100000 100000",
                            "memory.max": str(2 * gib),
                            "memory.current": str(gib)})
    monkeypatch.setattr(hr, "_host_available_mb", lambda: 1e6)
    jobs, _ = hr.auto_jobs(per_job_mb=4000.0, root=root)
    assert jobs == 1



# ---- pids: the limit that kills a run outright ----------------------

def test_pids_limit_and_headroom_v2(tmp_path):
    root = _cg(tmp_path, **{"pids.max": "512", "pids.current": "100"})
    assert hr.pids_limit(root) == 512
    assert hr.pids_current(root) == 100
    assert hr.pids_headroom(reserve=32, root=root) == 512 - 100 - 32


def test_pids_limit_v1_path(tmp_path):
    root = _cg(tmp_path, **{"pids__pids.max": "300", "pids__pids.current": "60"})
    assert hr.pids_limit(root) == 300
    assert hr.pids_current(root) == 60


def test_unlimited_pids_leaves_the_actor_count_alone(tmp_path):
    """No readable limit must NOT become a guessed cap."""
    root = _cg(tmp_path, **{"pids.max": "max", "pids.current": "10"})
    assert hr.pids_limit(root) is None
    assert hr.pids_headroom(root=root) is None
    fits, why = hr.max_actors(64, root=root)
    assert fits == 64 and "unclamped" in why

    missing = _cg(tmp_path / "empty")          # no cgroup files at all
    assert hr.max_actors(64, root=missing)[0] == 64


def test_max_actors_clamps_to_what_the_box_allows(tmp_path):
    # 200 tasks free after the reserve, 4 per actor -> 50 actors.
    root = _cg(tmp_path, **{"pids.max": "300", "pids.current": "68"})
    assert hr.pids_headroom(reserve=32, root=root) == 200
    fits, why = hr.max_actors(64, per_actor=4, reserve=32, root=root)
    assert fits == 50, why
    assert "ZERO leaves/s" in why, "the reason must say why this matters"

    # A request that fits is returned untouched.
    fits, why = hr.max_actors(12, per_actor=4, reserve=32, root=root)
    assert fits == 12 and "fits 12 actors" in why


def test_max_actors_never_returns_zero(tmp_path):
    """A cramped box should run one actor slowly, not none at all."""
    root = _cg(tmp_path, **{"pids.max": "40", "pids.current": "39"})
    fits, _ = hr.max_actors(64, per_actor=4, reserve=32, root=root)
    assert fits == 1


def test_pids_per_actor_measures_the_real_cost(tmp_path, monkeypatch):
    """The calibration az_loop logs, which is what stops
    PIDS_PER_ACTOR_ESTIMATE from being a permanent guess."""
    root = _cg(tmp_path, **{"pids.max": "512", "pids.current": "150"})
    assert hr.pids_per_actor(10, baseline=100, root=root) == 5.0
    assert hr.pids_per_actor(0, baseline=100, root=root) is None
    assert hr.pids_per_actor(10, baseline=100, root=str(tmp_path / "nope")) is None


def test_auto_jobs_is_bound_by_the_task_budget_too(tmp_path, monkeypatch):
    """A job count that fits CPU and RAM can still be refused by the
    pids controller, and that failure is not graceful."""
    root = _cg(tmp_path, **{"cpu.max": "3200000 100000",   # 32 cores
                            "memory.max": str(200 * 1024 ** 2 * 1024),
                            "memory.current": "0",
                            "pids.max": "80", "pids.current": "16"})
    monkeypatch.setattr(hr, "_host_available_mb", lambda: 200_000.0)
    jobs, why = hr.auto_jobs(per_job_mb=100, threads_per_job=2, root=root)
    # headroom 80 - 16 - 32 = 32 tasks, 3 per job -> 10.
    # effective_cores is min(quota, HOST cores), so on a small machine
    # the cpu term can bind first; the point is that pids is in the min.
    assert "pids 32 tasks->10" in why, why
    by_cpu = max(1, int(hr.effective_cores(root) / 2))
    assert jobs == min(by_cpu, 10), why

    # Tighten the budget until pids is unambiguously the binding term.
    tight = _cg(tmp_path / "tight", **{"cpu.max": "3200000 100000",
                                       "pids.max": "50", "pids.current": "15"})
    monkeypatch.setattr(hr, "_host_available_mb", lambda: 200_000.0)
    jobs3, why3 = hr.auto_jobs(per_job_mb=100, threads_per_job=2, root=tight)
    assert "pids 3 tasks->1" in why3 and jobs3 == 1, why3

    # With no pids limit the derivation says so and nothing is clamped.
    loose = _cg(tmp_path / "loose", **{"cpu.max": "400000 100000",
                                       "pids.max": "max"})
    monkeypatch.setattr(hr, "_host_available_mb", lambda: 200_000.0)
    jobs2, why2 = hr.auto_jobs(per_job_mb=100, threads_per_job=2, root=loose)
    assert "pids unlimited (skipped)" in why2 and jobs2 == 2, why2
