"""Profile a self-play iteration on the cluster (or locally).

Three layers of measurement, dumped to a single output directory:

  1. **Phase timing** (always on, ~0% overhead): wall-clock seconds
     spent in rollout vs train_step vs save, per iter. Surfaces the
     macro-level breakdown — "are we CPU-bound on rollout or GPU-
     bound on gradient?" — without the noise of finer-grained
     instrumentation.

  2. **cProfile** (always on, ~5-15% overhead): standard CPython
     deterministic profiler. Captures every Python function call's
     tottime / cumtime / ncalls. Headline output: top-30 by
     cumulative time and top-30 by self-time. The .prof file is
     also dumped so you can open it in snakeviz / py-spy / etc.
     for interactive exploration.

  3. **torch.profiler** (opt-in via --torch-profile, ~30-50%
     overhead): captures CUDA kernel timings, op-level autograd
     work, and CPU↔GPU memory transfers. Dumps a Chrome-trace
     JSON for `chrome://tracing` or `perfetto.dev`. Expensive
     enough that it's not the default — turn it on when you want
     to see *which kernels* dominate, not just "the model
     forward."

Plus background **nvidia-smi sampling** when CUDA is available
(samples every --nvidia-smi-interval seconds; writes a CSV with
SM%/mem%/W/MHz over time). Tells you if the GPU is actually
saturated or sitting idle waiting on CPU work — the single most
important diagnostic when scaling.

The script reuses `tools.sim_self_play.run_iteration` directly so
we profile the actual production code, not a model. Same setup,
same workers, same arch.

Output layout:

    training/profiles/<jobid_or_local>/
      summary.txt              -- human-readable headline numbers
      phase_timing.csv         -- per-iter rollout/train/save seconds
      cprofile.prof            -- raw pstats binary (snakeviz fodder)
      cprofile_cumulative.txt  -- top-N by cumtime
      cprofile_self.txt        -- top-N by tottime
      nvidia_smi.csv           -- GPU utilization timeline
      torch_trace.json         -- chrome-trace (only if --torch-profile)
      args.json                -- the exact invocation
      env.json                 -- python / torch / cuda versions

Usage:
    python tools/profile_selfplay.py
    python tools/profile_selfplay.py --iterations 3 --games-per-iter 8
    python tools/profile_selfplay.py --torch-profile --workers 6
    python tools/profile_selfplay.py --output-dir training/profiles/exp1
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import datetime as dt
import io
import json
import logging
import os
import pstats
import random
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.sim_self_play import (
    _recruit_cost_lookup,
    _TrainerHistoryCSV,
    run_iteration,
)
from wesnoth_ai.rewards import WeightedReward, load_reward_config
from wesnoth_ai.transformer_policy import TransformerPolicy
from wesnoth_sim import PvPDefaults


log = logging.getLogger("profile_selfplay")


# ---------------------------------------------------------------------
# Phase timing
# ---------------------------------------------------------------------

class PhaseTimer:
    """Records wall-clock seconds spent in named phases. One row per
    iter; the CSV writer dumps the full table at the end."""

    def __init__(self):
        self.rows: List[Dict[str, float]] = []
        self._current: Dict[str, float] = {}
        self._t0: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        """Time a code block; result accumulates into the current
        row (one row per iter via `next_iter()`)."""
        self._t0[name] = time.perf_counter()
        try:
            yield
        finally:
            self._current[name] = (
                self._current.get(name, 0.0)
                + (time.perf_counter() - self._t0[name])
            )

    def next_iter(self, iter_idx: int) -> None:
        """Close the current iter's row and start a fresh one."""
        if self._current:
            row = dict(self._current)
            row["iter"] = iter_idx
            self.rows.append(row)
        self._current = {}
        self._t0 = {}

    def dump_csv(self, path: Path) -> None:
        if not self.rows:
            return
        # Stable column order: iter first, then alphabetical phases.
        phase_names = sorted({k for r in self.rows for k in r
                              if k != "iter"})
        fieldnames = ["iter"] + phase_names
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------
# nvidia-smi sampler
# ---------------------------------------------------------------------

class NvidiaSmiSampler:
    """Background thread polling `nvidia-smi --query-gpu=...` every
    N seconds. Writes one CSV row per sample. Stops cleanly on
    `stop()` (signals the thread; one final sample at most).

    Why a separate thread + CSV (vs torch.profiler-only): the GPU's
    SM utilization between training events (e.g. between rollout
    workers' forwards) is exactly the signal you need to decide
    "is the GPU underused?" — and torch.profiler only sees its own
    profiled ops, not the wall-clock gaps. nvidia-smi catches the
    gaps.
    """

    QUERY = (
        "timestamp,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw,"
        "clocks.current.sm,clocks.current.memory"
    )

    def __init__(self, csv_path: Path, interval_s: float):
        self.csv_path = csv_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available: Optional[bool] = None

    def _probe_available(self) -> bool:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _loop(self) -> None:
        f = self.csv_path.open("w", encoding="utf-8", newline="",
                               buffering=1)
        try:
            # First row: the column header we asked nvidia-smi for.
            # Use the same names so downstream tools match.
            f.write(self.QUERY + "\n")
            while not self._stop.is_set():
                try:
                    r = subprocess.run(
                        ["nvidia-smi",
                         f"--query-gpu={self.QUERY}",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode == 0:
                        for line in r.stdout.strip().splitlines():
                            f.write(line + "\n")
                except (OSError, subprocess.TimeoutExpired) as e:
                    f.write(f"# sample failed: {e}\n")
                # Use stop-event.wait so we get prompt shutdown.
                self._stop.wait(self.interval_s)
        finally:
            f.close()

    def start(self) -> bool:
        if self._available is None:
            self._available = self._probe_available()
        if not self._available:
            return False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="nvidia-smi-sampler",
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


# ---------------------------------------------------------------------
# torch.profiler context (opt-in)
# ---------------------------------------------------------------------

@contextmanager
def _maybe_torch_profile(enabled: bool, out_dir: Path):
    """`torch.profiler.profile` context. No-op when `enabled=False`
    so the caller can use the same `with` block unconditionally.

    We use schedule=skip_first(1) + wait(0) + warmup(0) + active(1)
    so the FIRST profiled step is the second iter (after warmup),
    avoiding JIT / kernel-compile noise. Multiple iters all get
    captured in one trace; that's the schedule's default behavior.
    """
    if not enabled:
        yield None
        return
    import torch
    from torch.profiler import (ProfilerActivity, profile,
                                schedule as _schedule)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        schedule=_schedule(wait=0, warmup=1, active=99, repeat=1),
        record_shapes=False,         # too verbose; flip on if needed
        profile_memory=False,        # ditto
        with_stack=False,            # huge JSON otherwise
    ) as p:
        yield p
    # Export chrome-trace for chrome://tracing or perfetto.dev.
    trace_path = out_dir / "torch_trace.json"
    try:
        p.export_chrome_trace(str(trace_path))
        log.info(f"torch trace -> {trace_path}")
    except Exception as e:
        log.warning(f"failed to export torch trace: {e}")


# ---------------------------------------------------------------------
# Save: deduplicate the policy setup vs sim_self_play
# ---------------------------------------------------------------------

def _build_policy(
    ckpt_path: Optional[Path],
    device,
):
    """Same arch-peek + load logic as sim_self_play.main. Pulled out
    so the profile script doesn't have to duplicate it inline."""
    import torch
    arch_kwargs: Dict[str, int] = {}
    if ckpt_path and ckpt_path.exists():
        try:
            raw = torch.load(ckpt_path, map_location="cpu",
                             weights_only=False)
            for k in ("d_model", "num_layers", "num_heads", "d_ff"):
                v = (raw.get("arch") or {}).get(k)
                if v is not None:
                    arch_kwargs[k] = int(v)
            log.info(f"warm-start arch from checkpoint: {arch_kwargs}")
        except Exception as e:
            log.warning(f"couldn't peek arch from {ckpt_path}: {e!r}")
    policy = TransformerPolicy(device=device, **arch_kwargs)
    if ckpt_path and ckpt_path.exists():
        try:
            policy.load_checkpoint(ckpt_path)
        except RuntimeError as e:
            if "arch mismatch" in str(e).lower():
                log.warning(f"arch mismatch loading {ckpt_path}: {e}; "
                            f"using random init")
            else:
                raise
    return policy


# ---------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------

def _write_cprofile_text(prof: cProfile.Profile, out_dir: Path,
                        top_n: int = 30) -> None:
    """Dump human-readable top-N summaries by cumtime and tottime."""
    for sort_by, fname in (("cumulative", "cprofile_cumulative.txt"),
                           ("tottime",    "cprofile_self.txt")):
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf)
        stats.strip_dirs().sort_stats(sort_by).print_stats(top_n)
        (out_dir / fname).write_text(buf.getvalue(), encoding="utf-8")


def _write_summary(
    out_dir: Path, args: argparse.Namespace,
    phase_timer: PhaseTimer, n_iters: int,
    wall_s: float, ckpt: Optional[Path],
) -> None:
    """Headline text summary the operator reads first. Everything in
    here should be answerable from the dumped artifacts; this is
    the executive view."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"Self-play profile  ({n_iters} iters, "
                 f"{wall_s:.1f}s wall)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Checkpoint:     {ckpt}")
    lines.append(f"Workers:        {args.workers}")
    lines.append(f"Games per iter: {args.games_per_iter}")
    lines.append(f"Max turns:      {args.max_turns}")
    lines.append(f"MCTS:           {args.mcts} "
                 f"(sims={args.mcts_sims})" if args.mcts else
                 "MCTS:           off (REINFORCE)")
    lines.append(f"Device:         {args.device or '(default)'}")
    lines.append("")

    # Phase timing breakdown.
    if phase_timer.rows:
        lines.append("Phase wall-clock seconds per iter:")
        phase_names = sorted({k for r in phase_timer.rows
                              for k in r if k != "iter"})
        # Print mean / per-iter rows.
        means = {p: 0.0 for p in phase_names}
        for r in phase_timer.rows:
            for p in phase_names:
                means[p] += r.get(p, 0.0)
        for p in phase_names:
            means[p] /= max(1, len(phase_timer.rows))
        total = sum(means.values())
        lines.append(f"  {'phase':<20} {'mean_s':>10} {'pct':>8}")
        for p in sorted(phase_names, key=lambda k: -means[k]):
            pct = 100.0 * means[p] / max(total, 1e-9)
            lines.append(f"  {p:<20} {means[p]:>10.3f} {pct:>7.1f}%")
        lines.append(f"  {'TOTAL':<20} {total:>10.3f} {100.0:>7.1f}%")
        lines.append("")
        lines.append("Per-iter detail (see phase_timing.csv):")
        lines.append(f"  {'iter':>4} " +
                     " ".join(f"{p:>10}" for p in phase_names))
        for r in phase_timer.rows:
            lines.append(f"  {r['iter']:>4} " +
                         " ".join(f"{r.get(p, 0.0):>10.3f}"
                                  for p in phase_names))

    lines.append("")
    lines.append("Top callers (also see cprofile_cumulative.txt / "
                 "cprofile_self.txt):")
    lines.append("  cprofile.prof loads in snakeviz: "
                 "`pip install snakeviz && snakeviz cprofile.prof`")
    if (out_dir / "nvidia_smi.csv").exists():
        # Quick GPU utilization stats from the samples.
        import statistics as _stats
        utils: List[float] = []
        mem_used: List[float] = []
        with (out_dir / "nvidia_smi.csv").open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    utils.append(float(row.get(
                        "utilization.gpu", "").strip()))
                    mem_used.append(float(row.get(
                        "memory.used", "").strip()))
                except (ValueError, AttributeError):
                    continue
        if utils:
            lines.append("")
            lines.append(f"GPU utilization (over {len(utils)} samples):")
            lines.append(f"  mean   = {_stats.mean(utils):5.1f}%")
            lines.append(f"  median = {_stats.median(utils):5.1f}%")
            lines.append(f"  max    = {max(utils):5.1f}%")
            lines.append(f"  min    = {min(utils):5.1f}%")
            if mem_used:
                lines.append(f"GPU memory used (MiB):")
                lines.append(f"  mean   = {_stats.mean(mem_used):8.0f}")
                lines.append(f"  max    = {max(mem_used):8.0f}")

    lines.append("")
    lines.append("=" * 72)
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    # Echo to stdout so the SLURM log also captures the summary.
    print(text)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint to load. Default: "
                         "training/checkpoints/sim_selfplay.pt "
                         "(falls back to random init if absent).")
    ap.add_argument("--iterations", type=int, default=3,
                    help="Profiled iterations. Default 3 -- enough "
                         "samples for cProfile to be meaningful, "
                         "short enough to fit a 20-min walltime.")
    ap.add_argument("--warmup-iterations", type=int, default=1,
                    help="Iterations to run BEFORE profiling starts. "
                         "Default 1 -- absorbs torch JIT / kernel "
                         "compile / encoder vocab building overhead "
                         "that would skew iter 0's numbers.")
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--max-turns",      type=int, default=60,
                    help="Lower than the production 200 -- profile "
                         "runs want to see many iters in finite "
                         "time, not one huge game. 60 lets games "
                         "play out enough to exercise mid-game "
                         "code paths.")
    ap.add_argument("--workers", type=int, default=6,
                    help="Rollout worker threads (default 6 -- "
                         "matches the cluster sbatch).")
    ap.add_argument("--device", default=None,
                    help="Torch device. Default: TransformerPolicy "
                         "picks. Pass 'cuda' on the cluster.")
    ap.add_argument("--reward-config", type=Path, default=None)
    ap.add_argument("--forced-faction", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mcts", action="store_true",
                    help="Profile under MCTS instead of REINFORCE. "
                         "Much slower per iter (--mcts-sims forwards "
                         "per move) so the workload looks completely "
                         "different.")
    ap.add_argument("--mcts-sims", type=int, default=50)
    ap.add_argument("--mcts-c-puct", type=float, default=1.5)
    ap.add_argument("--torch-profile", action="store_true",
                    help="Enable torch.profiler (CUDA kernel + op "
                         "timings). ~30-50%% overhead and a multi-MB "
                         "trace file; off by default.")
    ap.add_argument("--nvidia-smi-interval", type=float, default=1.0,
                    help="Seconds between nvidia-smi samples "
                         "(default 1.0). Set to 0 to disable.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to dump artifacts. Default: "
                         "training/profiles/<SLURM_JOB_ID>/ on "
                         "cluster, training/profiles/local_<ts>/ "
                         "otherwise.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- 1. Resolve output dir --
    if args.output_dir is None:
        jobid = os.environ.get("SLURM_JOB_ID")
        if jobid:
            args.output_dir = Path("training/profiles") / jobid
        else:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = Path("training/profiles") / f"local_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"profile output: {args.output_dir}")

    # Dump invocation args + environment up front so a half-completed
    # run still produces a reproducible record.
    (args.output_dir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()},
                   indent=2), encoding="utf-8")

    try:
        import torch
        env_info = {
            "python":      sys.version.split()[0],
            "torch":       torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            "cuda_cap":    (str(torch.cuda.get_device_capability(0))
                            if torch.cuda.is_available() else None),
            "slurm_job":   os.environ.get("SLURM_JOB_ID"),
            "hostname":    os.uname().nodename if hasattr(os, "uname")
                           else os.environ.get("COMPUTERNAME"),
        }
    except Exception as e:
        env_info = {"error": repr(e)}
    (args.output_dir / "env.json").write_text(
        json.dumps(env_info, indent=2), encoding="utf-8")

    # -- 2. Resolve checkpoint --
    ckpt = args.checkpoint
    if ckpt is None:
        sp = Path("training/checkpoints/sim_selfplay.pt")
        if sp.exists():
            ckpt = sp

    # -- 3. Build policy + optional MCTS wrapper --
    import torch
    device = torch.device(args.device) if args.device else None
    policy = _build_policy(ckpt, device)
    if args.mcts:
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy
        cfg = MCTSConfig(n_simulations=args.mcts_sims,
                         c_puct=args.mcts_c_puct, batch_size=1)
        policy = MCTSPolicy(policy, cfg)
        log.info(f"MCTS mode: sims={args.mcts_sims} c_puct={args.mcts_c_puct}")

    # -- 4. Reward + cost lookup --
    if args.reward_config is not None:
        reward_fn = load_reward_config(args.reward_config)
        log.info(f"reward config: {args.reward_config}")
    else:
        reward_fn = WeightedReward()
    cost_lookup = _recruit_cost_lookup()
    pvp_defaults = PvPDefaults()

    # -- 5. Translate forced-faction --
    forced_faction_arg: object = ...
    if args.forced_faction is not None:
        if args.forced_faction.lower() == "none":
            forced_faction_arg = None
        else:
            forced_faction_arg = args.forced_faction

    # -- 6. Start nvidia-smi sampler (no-op if not on a GPU host) --
    sampler: Optional[NvidiaSmiSampler] = None
    if args.nvidia_smi_interval > 0:
        sampler = NvidiaSmiSampler(
            args.output_dir / "nvidia_smi.csv",
            args.nvidia_smi_interval,
        )
        if sampler.start():
            log.info(f"nvidia-smi sampler started @ "
                     f"{args.nvidia_smi_interval}s interval")
        else:
            log.info("nvidia-smi not available; skipping GPU sampling")
            sampler = None

    # -- 7. Warmup iterations (NOT profiled) --
    rng = random.Random(args.seed)
    if args.warmup_iterations > 0:
        log.info(f"running {args.warmup_iterations} warmup iter(s) "
                 f"(NOT profiled)")
        for it in range(args.warmup_iterations):
            run_iteration(
                policy, None, reward_fn, cost_lookup,
                iter_idx=-1 - it,
                games_per_iter=args.games_per_iter,
                max_turns=args.max_turns,
                rng=rng, pvp_defaults=pvp_defaults,
                workers=args.workers,
                forced_faction=forced_faction_arg,
            )

    # -- 8. Profiled iterations --
    log.info(f"profiling {args.iterations} iter(s)")
    phase_timer = PhaseTimer()
    prof = cProfile.Profile()
    t_start = time.perf_counter()

    with _maybe_torch_profile(args.torch_profile, args.output_dir) as tp:
        prof.enable()
        try:
            for it in range(args.iterations):
                # We can't easily split rollout vs train inside
                # run_iteration without modifying it. Instead, time
                # the whole iter and then time train_step explicitly
                # by passing train_at_end=False and calling
                # train_step ourselves. This gives us the rollout
                # vs train split cleanly.
                with phase_timer.phase("rollout_seconds"):
                    outcomes = run_iteration(
                        policy, None, reward_fn, cost_lookup,
                        iter_idx=it,
                        games_per_iter=args.games_per_iter,
                        max_turns=args.max_turns,
                        rng=rng, pvp_defaults=pvp_defaults,
                        workers=args.workers,
                        forced_faction=forced_faction_arg,
                        train_at_end=False,
                    )
                with phase_timer.phase("train_step_seconds"):
                    policy.train_step()
                phase_timer.next_iter(it)
                if tp is not None:
                    tp.step()       # advance torch.profiler schedule
                log.info(f"  iter {it} done; outcomes={len(outcomes)}")
        finally:
            prof.disable()
    wall_s = time.perf_counter() - t_start

    # -- 9. Stop sampler + dump artifacts --
    if sampler is not None:
        sampler.stop()

    # cProfile binary + text summaries.
    prof.dump_stats(str(args.output_dir / "cprofile.prof"))
    _write_cprofile_text(prof, args.output_dir, top_n=30)

    # Phase timing CSV.
    phase_timer.dump_csv(args.output_dir / "phase_timing.csv")

    # Headline summary.
    _write_summary(args.output_dir, args, phase_timer,
                   args.iterations, wall_s, ckpt)
    log.info(f"all artifacts in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
