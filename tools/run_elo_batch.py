"""Resumable, memory-guarded driver for a paired Elo run.

`elo_eval_game.py` is one game per process and already skips a result
file that exists, so a long ladder is really "loop over (side, seed)
until done". This driver is that loop plus the two things that made a
long run impossible to babysit here:

  * **A wall-clock budget.** It stops cleanly at `--time-budget-min`, so
    it can be run in short chunks that accumulate into one games dir.
    Re-running continues where it left off; nothing is recomputed.
  * **A memory guard.** Measured 2026-08-03: this laptop has 7.6 GB
    total, and with a browser open only ~0.6 GB was free. A torch
    process under that pressure page-thrashes rather than computes -- one
    eval game got ~1 s of CPU in 9 min of wall clock and produced
    nothing. Starting a game with no memory does not just run slowly, it
    wastes the whole slot and can take the machine down. So refuse.

Side assignment alternates so the pair is balanced: an odd game index
puts A on side 2. Seeds are derived from the index, so the same command
always schedules the same games and two chunks never collide.

Usage (raw-policy A/B -- `--mcts-sims 0` is what makes it RAW):
    python tools/run_elo_batch.py \\
        --label-a best  --spec-a training/checkpoints/campaign_live_20260730.pt \\
        --label-b anchor --spec-b training/checkpoints/selfplay_seed_20260718.pt \\
        --games 400 --outdir eval_games/tc_raw --mcts-sims 0 \\
        --time-budget-min 55

Then fit (draws are draws -- PURE is the headline; see elo_collect.py):
    python tools/elo_collect.py eval_games/tc_raw
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

log = logging.getLogger("run_elo_batch")

# Below this, a torch process thrashes instead of running (see module
# docstring). Generous on purpose: the cost of pausing is one idle slot,
# the cost of proceeding is a wasted slot or a hung machine.
DEFAULT_MIN_FREE_MB = 1800


def free_mb() -> Optional[float]:
    """Free physical memory in MB, or None if it cannot be determined
    (in which case the guard is skipped rather than guessed at)."""
    try:
        import psutil                                   # noqa: PLC0415
        return psutil.virtual_memory().available / (1024 ** 2)
    except Exception:                                   # noqa: BLE001
        pass
    if sys.platform == "win32":
        try:
            import ctypes                               # noqa: PLC0415

            class _S(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            s = _S()
            s.dwLength = ctypes.sizeof(_S)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                return s.ullAvailPhys / (1024 ** 2)
        except Exception:                               # noqa: BLE001
            pass
    return None


def result_name(label_a: str, label_b: str, side_a: int, seed: int) -> str:
    """Mirror elo_eval_game.py's output name so we can tell, without
    launching anything, whether this game is already done."""
    return f"game_{label_a}_{label_b}_s{side_a}_{seed}.json"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--spec-a", required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--spec-b", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--games", type=int, default=400,
                    help="TOTAL games wanted in outdir (not per chunk).")
    ap.add_argument("--mcts-sims", type=int, default=0,
                    help="0 = RAW policy (no search). 32 = training-matched.")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--time-budget-min", type=float, default=55.0,
                    help="Stop cleanly after this long. Re-run to continue.")
    ap.add_argument("--min-free-mb", type=float, default=DEFAULT_MIN_FREE_MB)
    ap.add_argument("--per-game-timeout-min", type=float, default=20.0,
                    help="Kill a single game that overruns; its slot is "
                         "skipped and the run continues.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    args.outdir.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.time_budget_min * 60.0
    done = played = failed = 0

    for i in range(args.games):
        side_a = 1 if i % 2 == 0 else 2
        seed = args.seed_base + i
        out = args.outdir / result_name(args.label_a, args.label_b,
                                        side_a, seed)
        if out.exists():
            done += 1
            continue
        if time.perf_counter() > deadline:
            log.info("time budget reached — stopping cleanly")
            break

        fm = free_mb()
        if fm is not None and fm < args.min_free_mb:
            log.error(
                "only %.0f MB free (need %.0f). A torch process below this "
                "thrashes instead of running. Close some applications and "
                "re-run; finished games are kept.", fm, args.min_free_mb)
            break

        cmd = [sys.executable, "-u", str(_THIS.parent / "elo_eval_game.py"),
               args.label_a, args.spec_a, args.label_b, args.spec_b,
               str(side_a), str(seed), str(args.outdir),
               "--mcts-sims", str(args.mcts_sims),
               "--max-turns", str(args.max_turns)]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.per_game_timeout_min * 60.0)
            ok = r.returncode == 0 and out.exists()
            if not ok:
                failed += 1
                log.warning("game %d (side %d, seed %d) failed rc=%s: %s",
                            i, side_a, seed, r.returncode,
                            (r.stderr or "").strip()[-200:])
            else:
                played += 1
        except subprocess.TimeoutExpired:
            failed += 1
            log.warning("game %d timed out after %.0f min; skipping slot",
                        i, args.per_game_timeout_min)
            continue
        log.info("game %d/%d done in %.1f min (played=%d failed=%d, "
                 "%.0f MB free)", i + 1, args.games,
                 (time.perf_counter() - t0) / 60.0, played, failed,
                 fm if fm is not None else -1)

    total = len(list(args.outdir.glob("game_*.json")))
    # Report as a fraction, never a percentage or an extrapolation.
    log.info("chunk end: %d/%d games present in %s (this chunk: %d played, "
             "%d failed, %d already done)",
             total, args.games, args.outdir, played, failed, done)
    if total < args.games:
        log.info("re-run the same command to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
