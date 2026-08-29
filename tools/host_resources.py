"""Container-aware resource caps -- stdlib-only, torch-free.

Boxes change under us constantly (rented per-leg), so concurrency
must be COMPUTED per box, never hand-tuned. The trap this module
encodes: on Vast, `os.cpu_count()` / `/proc/meminfo` / psutil all
describe the HOST, not our slice -- a container on a 72-core 256GB
host may own 18 cores and 64GB, and the cgroup files are the only
honest source (same lesson as the actor-pool sizing and the
HOST-wide /proc/loadavg memory note). Everything degrades
gracefully: no cgroup (Windows, bare metal) -> host numbers; no
reading at all -> None, and callers skip the constraint rather
than guess.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional, Tuple

# cgroup v1 reports "unlimited" as a page-rounded huge number, not
# a sentinel string; anything above 4 PiB is nobody's real limit.
_V1_UNLIMITED = 1 << 52


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


def effective_cores(root: str = "/sys/fs/cgroup") -> float:
    """CPU cores this process may actually use: cgroup quota when
    one is set, host count otherwise."""
    host = float(os.cpu_count() or 1)
    v2 = _read(f"{root}/cpu.max")
    if v2:
        q, _, p = v2.partition(" ")
        if q != "max":
            try:
                return min(host, int(q) / int(p))
            except (ValueError, ZeroDivisionError):
                pass
    q1 = _read(f"{root}/cpu/cpu.cfs_quota_us")
    p1 = _read(f"{root}/cpu/cpu.cfs_period_us")
    if q1 and p1:
        try:
            if int(q1) > 0:
                return min(host, int(q1) / int(p1))
        except (ValueError, ZeroDivisionError):
            pass
    return host


def _host_available_mb() -> Optional[float]:
    if sys.platform == "win32":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32),
                            ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            ms = _MS(dwLength=ctypes.sizeof(_MS))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullAvailPhys / (1024 ** 2)
        except Exception:  # noqa: BLE001
            return None
        return None
    txt = _read("/proc/meminfo")
    if txt:
        for line in txt.splitlines():
            if line.startswith("MemAvailable:"):
                try:
                    return int(line.split()[1]) / 1024.0
                except (ValueError, IndexError):
                    return None
    return None


def _cgroup_headroom_mb(root: str = "/sys/fs/cgroup") -> Optional[float]:
    """limit − current for OUR cgroup, or None when unlimited or
    unreadable."""
    lim = _read(f"{root}/memory.max")
    cur = _read(f"{root}/memory.current")
    if lim is None or cur is None:                       # try v1
        lim = _read(f"{root}/memory/memory.limit_in_bytes")
        cur = _read(f"{root}/memory/memory.usage_in_bytes")
    if lim is None or cur is None or lim == "max":
        return None
    try:
        lim_b, cur_b = int(lim), int(cur)
    except ValueError:
        return None
    if lim_b >= _V1_UNLIMITED:
        return None
    return (lim_b - cur_b) / (1024 ** 2)


def available_mb(root: str = "/sys/fs/cgroup") -> Optional[float]:
    """Memory this process can still allocate before it thrashes or
    the cgroup OOM-killer fires: the BINDING one of host-available
    and cgroup headroom. None only if neither is readable."""
    vals = [v for v in (_host_available_mb(), _cgroup_headroom_mb(root))
            if v is not None]
    return min(vals) if vals else None


def vram_free_mb() -> Optional[float]:
    """Free VRAM of GPU 0 via nvidia-smi; None when no GPU/driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError,
            subprocess.TimeoutExpired):
        pass
    return None


def auto_jobs(per_job_mb: float, threads_per_job: int = 2,
              per_job_vram_mb: Optional[float] = None,
              reserve_mb: float = 1500.0, cap: int = 32,
              root: str = "/sys/fs/cgroup") -> Tuple[int, str]:
    """Concurrency that fits THIS box: min over CPU quota, memory
    headroom, and (when per_job_vram_mb is given) free VRAM.
    Returns (jobs, human-readable derivation) so the choice is
    always in the log."""
    cores = effective_cores(root)
    by_cpu = max(1, int(cores // max(1, threads_per_job)))
    parts = [f"cpu {cores:.1f}->{by_cpu}"]
    jobs = by_cpu
    avail = available_mb(root)
    if avail is not None:
        by_ram = max(1, int((avail - reserve_mb) // per_job_mb))
        parts.append(f"ram {avail:.0f}MB->{by_ram}")
        jobs = min(jobs, by_ram)
    else:
        parts.append("ram unreadable (skipped)")
    if per_job_vram_mb is not None:
        vram = vram_free_mb()
        if vram is not None:
            by_vram = max(1, int(vram // per_job_vram_mb))
            parts.append(f"vram {vram:.0f}MB->{by_vram}")
            jobs = min(jobs, by_vram)
        else:
            parts.append("vram unreadable (skipped)")
    jobs = min(jobs, cap)
    return jobs, ", ".join(parts) + f" => jobs {jobs}"
