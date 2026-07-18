#!/usr/bin/env python3
"""Measure the optimal cuda/cpu spool-worker split from a LIVE fleet.

Motivation (2026-07-18): worker GPU placement is a VRAM allocation
decision (each cuda worker ~400-620MB; the trainer's backward peak
grows with play quality and OOM'd a 24GB card twice). The budget
constants in sim_self_play were set from one incident's numbers;
this tool replaces estimates with measurements taken from the
running campaign:

  1. per-worker throughput by device -- from the workers' heartbeat
     files (spool/stats/w*.json, cumulative decisions+games), read
     twice across an observation window;
  2. the trainer's TRUE backward peak -- the gpu_mem_peak_mb column
     the trainer-history CSV now logs per iteration (high-water
     mark, reset each iteration);
  3. per-cuda-worker VRAM -- live per-process usage via
     `nvidia-smi --query-compute-apps` (falls back to the budgeted
     constant when no cuda workers are running).

Recommendation logic: fleet size N is CPU-bound (workers are Python-
rollout-dominated on either device), so the only question is how
many of the N get cuda forwards. Throughput is linear in the split:
    rate(k) = k * r_cuda + (N - k) * r_cpu
so the optimum is k = k_vram_cap when r_cuda > r_cpu, else 0, with
    k_vram_cap = (vram_total - peak_seen * safety) // per_worker.

Run ON the box (or point --spool/--csv at synced copies):
    python tools/profile_worker_split.py --observe 1800
    python tools/profile_worker_split.py --report-only   # no wait:
        rates from each heartbeat's own lifetime counters

To APPLY a measured split, relaunch with
    SPOOL_CUDA_WORKERS=<k>   (onstart env; overrides the auto budget)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("profile_worker_split")

# Peak-seen safety multiplier: the backward peak GROWS as play
# quality raises per-state unit counts (7.1GB -> 12.2GB across
# 2026-07-18 alone), so reserving exactly the observed maximum
# re-OOMs on the next uptick. 1.3x the observed peak tracks the
# measurement while leaving growth headroom; revisit when unit
# counts saturate.
PEAK_SAFETY = 1.3


@dataclass
class HeartbeatSample:
    worker: int
    device: str
    games: int
    decisions: int
    started: float
    updated: float


def read_heartbeats(stats_dir: Path) -> Dict[int, HeartbeatSample]:
    out: Dict[int, HeartbeatSample] = {}
    for f in sorted(stats_dir.glob("w*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[int(d["worker"])] = HeartbeatSample(
                worker=int(d["worker"]), device=str(d["device"]),
                games=int(d["games"]), decisions=int(d["decisions"]),
                started=float(d["started"]), updated=float(d["updated"]))
        except (OSError, ValueError, KeyError) as e:
            log.warning(f"unreadable heartbeat {f.name}: {e}")
    return out


def rates_by_device(a: Dict[int, HeartbeatSample],
                    b: Optional[Dict[int, HeartbeatSample]] = None,
                    ) -> Dict[str, Dict[str, float]]:
    """Per-device mean per-worker rates.

    Windowed mode (two samples): rate = delta(decisions)/delta(time)
    per worker, averaged within each device class. Workers with no
    progress in the window still count (a stuck worker is real
    throughput loss), except those whose heartbeat vanished.

    Lifetime mode (one sample): rate = decisions/(updated-started).
    Cheaper but biased by warmup and checkpoint-reload pauses.
    """
    per_dev: Dict[str, List[float]] = {}
    per_dev_games: Dict[str, List[float]] = {}
    for wid, s in a.items():
        if b is not None:
            s2 = b.get(wid)
            if s2 is None:
                continue
            dt = s2.updated - s.updated
            if dt <= 0:
                dt = max(1.0, time.time() - s.updated)
                dd, dg = 0, 0
            else:
                dd = s2.decisions - s.decisions
                dg = s2.games - s.games
            dev = s2.device
        else:
            dt = max(1.0, s.updated - s.started)
            dd, dg = s.decisions, s.games
            dev = s.device
        per_dev.setdefault(dev, []).append(dd / dt)
        per_dev_games.setdefault(dev, []).append(dg / dt)
    return {
        dev: {
            "workers": len(v),
            "decisions_per_s": sum(v) / len(v),
            "games_per_h": 3600.0 * sum(per_dev_games[dev]) / len(v),
        }
        for dev, v in per_dev.items() if v
    }


def trainer_peak_mb(csv_path: Path) -> Optional[int]:
    """Max gpu_mem_peak_mb over the CSV (falls back to
    gpu_mem_alloc_mb with a warning when the column is absent/empty
    -- pre-instrumentation rows)."""
    import csv as _csv
    if not csv_path.exists():
        return None
    peaks, allocs = [], []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            for col, sink in (("gpu_mem_peak_mb", peaks),
                              ("gpu_mem_alloc_mb", allocs)):
                v = row.get(col)
                if v:
                    try:
                        sink.append(int(float(v)))
                    except ValueError:
                        pass
    if peaks:
        return max(peaks)
    if allocs:
        log.warning(
            "no gpu_mem_peak_mb rows (pre-instrumentation CSV); "
            "using max steady-state alloc %dMB -- the true backward "
            "peak is HIGHER; treat the recommendation as optimistic",
            max(allocs))
        return max(allocs)
    return None


def cuda_worker_mb() -> Optional[int]:
    """Max per-process VRAM (MB) among python compute processes that
    are NOT the biggest one (assumed to be the trainer). None when
    nvidia-smi is unavailable or shows <2 python processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    sizes = []
    for line in out.strip().splitlines():
        try:
            sizes.append(int(line.split(",")[1].strip()))
        except (IndexError, ValueError):
            continue
    if len(sizes) < 2:
        return None
    sizes.sort()
    return sizes[-2]        # largest excluding the trainer


def total_vram_mb() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def recommend(n_workers: int, r_cuda: Optional[float],
              r_cpu: Optional[float], vram_total_mb: Optional[int],
              peak_mb: Optional[int],
              per_worker_mb: Optional[int]) -> Dict:
    """The split decision. Returns a dict so tests can pin it and
    the CLI can print it."""
    from tools.sim_self_play import SPOOL_WORKER_VRAM_BYTES
    if per_worker_mb is None:
        per_worker_mb = SPOOL_WORKER_VRAM_BYTES // 2**20
    out = {"n_workers": n_workers, "per_worker_mb": per_worker_mb}
    if vram_total_mb is None or peak_mb is None:
        out["k_cuda"] = None
        out["why"] = "missing VRAM measurements; keep current config"
        return out
    reserve = int(peak_mb * PEAK_SAFETY)
    k_cap = max(0, (vram_total_mb - reserve) // per_worker_mb)
    out["trainer_reserve_mb"] = reserve
    out["k_vram_cap"] = k_cap
    if r_cuda is None or r_cpu is None:
        out["k_cuda"] = min(n_workers, k_cap)
        out["why"] = ("no per-device rate comparison available "
                      "(single-device fleet); recommending the VRAM "
                      "cap -- re-run with a mixed fleet to confirm "
                      "cuda workers actually outperform cpu ones")
    elif r_cuda <= r_cpu:
        out["k_cuda"] = 0
        out["why"] = (f"cuda workers are not faster "
                      f"({r_cuda:.2f} vs {r_cpu:.2f} decisions/s): "
                      f"all-cpu; trainer keeps the whole card")
    else:
        out["k_cuda"] = min(n_workers, k_cap)
        out["why"] = (f"cuda {r_cuda:.2f} > cpu {r_cpu:.2f} "
                      f"decisions/s per worker; fill the VRAM cap")
    return out


def main(argv) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spool", type=Path, default=Path("training/spool"))
    ap.add_argument("--csv", type=Path,
                    default=Path("training/logs/trainer_history_local.csv"))
    ap.add_argument("--observe", type=int, default=1800,
                    help="Window seconds between heartbeat samples "
                         "(>= a few game lengths; games run "
                         "10-30 min).")
    ap.add_argument("--report-only", action="store_true",
                    help="Single heartbeat sample, lifetime rates "
                         "(no waiting; warmup-biased).")
    args = ap.parse_args(argv[1:])

    stats_dir = args.spool / "stats"
    a = read_heartbeats(stats_dir)
    if not a:
        log.error(f"no heartbeats under {stats_dir} -- fleet not "
                  f"running, or workers predate the heartbeat patch")
        return 2
    if args.report_only:
        rates = rates_by_device(a)
    else:
        log.info(f"sampling {len(a)} workers over {args.observe}s ...")
        time.sleep(args.observe)
        rates = rates_by_device(a, read_heartbeats(stats_dir))

    for dev, r in sorted(rates.items()):
        log.info(f"  {dev}: {r['workers']} workers, "
                 f"{r['decisions_per_s']:.2f} decisions/s/worker, "
                 f"{r['games_per_h']:.2f} games/h/worker")

    peak = trainer_peak_mb(args.csv)
    vram = total_vram_mb()
    per_w = cuda_worker_mb()
    log.info(f"trainer peak: {peak}MB | card: {vram}MB | "
             f"per-cuda-worker: {per_w}MB")

    rec = recommend(
        n_workers=len(a),
        r_cuda=rates.get("cuda", {}).get("decisions_per_s"),
        r_cpu=rates.get("cpu", {}).get("decisions_per_s"),
        vram_total_mb=vram, peak_mb=peak, per_worker_mb=per_w)
    log.info("=== recommendation ===")
    for k, v in rec.items():
        log.info(f"  {k}: {v}")
    if rec.get("k_cuda") is not None:
        log.info(f"apply with: SPOOL_CUDA_WORKERS={rec['k_cuda']} "
                 f"(env.sh + reboot)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
