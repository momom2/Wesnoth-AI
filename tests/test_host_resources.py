"""Container-aware concurrency sizing (tools/host_resources.py).

Boxes are rented per leg, so --jobs is auto-derived per box. The
trap under test: host-wide readings (nproc, /proc/meminfo) lie on
shared hosts -- the cgroup files are authoritative, and the guard
must take the BINDING minimum.
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
