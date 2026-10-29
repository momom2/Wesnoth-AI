"""Stage 1 of the training design on the local machine (Windows, the RX
6600; docs/selfplay_t1_local_prereg_20261028.md): the steps of
scripts/selfplay_t1_box.sh, its games, seeds, splits, retries and
acceptance bounds, without the box library.

    the first self-play match, D0, the other self-play matches, the labels
    and the readouts, the memory report

What differs from the box script:
- The pool: JOBS games in flight on one server that autocasts to fp16
  (--infer-fp16: the packed trunk on the HIP kernels,
  wesnoth_ai/hip_kernels.py); the games record infer_fp16.
- No dead-man's switch, so no sizing of the self-play to a deadline: every
  planned match is played. Between steps the runner stops when the file
  STOP exists in --out, and a later start resumes from the markers
  (<step>.done, <step>.failed), as the box script's re-entry does.
- No stage C rates: they measure a box's trainer, and this GPU trains at
  about 3 positions a second in fp32 (training/metrics/local_gpu_20261028).
- No uploads: the records stay in --out; the small ones are copied into
  the repository by hand when the run ends.
- The runner, and the processes it starts, run at below-normal priority
  (Windows), so the machine's user keeps the CPU first.

    HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/selfplay_t1_local.py --out C:/Users/<you>/wesnoth_runs/selfplay_t1_local
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.local_machine import (  # noqa: E402
    below_normal_priority, default_tools_python, display_resets_since, keep_awake, log, reference_flags,
    require_visible_gpu, run_tool,
)

# The box script's configuration (scripts/selfplay_t1_box.sh, "the configuration").
SP_SEED_BASE = 800000
N_SELFPLAY = 30000
CHUNK_GAMES = 5000
HELDOUT_FROM = 20000
D0_SEED_BASE = 830000
D0_GAMES = 12800
MEM_SEED_BASE = 845000
MEM_GAMES = 3200
LOST_MAX = 10
MIN_FREE_MB = 300


class Run:
    def __init__(self, args):
        self.a = args
        self.out: Path = args.out
        self.py = args.python
        self.tools_py = args.tools_python

    # ---- markers and records
    def marked(self, step: str) -> bool:
        return (self.out / f"{step}.done").exists()

    def failed(self, step: str) -> Optional[str]:
        f = self.out / f"{step}.failed"
        return f.read_text(encoding="utf-8").strip() if f.exists() else None

    def mark(self, step: str, failed: Optional[str] = None) -> None:
        name = f"{step}.failed" if failed else f"{step}.done"
        (self.out / name).write_text((failed or time.strftime("%Y-%m-%d %H:%M:%S")) + "\n", encoding="utf-8")

    def note(self, line: str) -> None:
        with open(self.out / "matches.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def stop_requested(self) -> bool:
        return (self.out / "STOP").exists()

    def run(self, name: str, cmd: List[str], log_name: str, timeout_min: float, python: str) -> int:
        """One tool to its end or its cut, its output in log_name."""
        return run_tool(self.out, name, cmd, log_name, timeout_min, python)

    # ---- the sides (the box script's side_flags / reference_flags)
    def reference_flags(self, side: str) -> List[str]:
        return reference_flags(side, self.tools_py)

    def side_flags(self, side: str, label: str, slots: Optional[int] = None) -> List[str]:
        flags, out = self.reference_flags(side), []
        for i in range(0, len(flags) - 1, 2):
            k, v = flags[i], flags[i + 1]
            if k == f"--label-{side}":
                out += [k, label]
            elif k == f"--memory-{side}":
                out += [k, str(slots if slots is not None else v)]
            elif k == f"--raw-end-turn-offset-{side}":
                continue
            else:
                out += [k, v]
        return out

    # ---- a match (the box script's play / play_through)
    def count_games(self, d: Path) -> Tuple[int, int, int]:
        files = list(d.glob("game_*.json"))
        decisive = 0
        for p in files:
            try:
                decisive += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
            except Exception:  # noqa: BLE001 -- an unreadable result counts as no result
                pass
        lost = sum(not (d / ("game_" + p.name[len("failed_"):])).exists() for p in d.glob("failed_*.json"))
        return len(files), decisive, lost

    def play(self, name: str, games: int, base: int, extra: int, flags: List[str]) -> Tuple[int, int, int, int]:
        d = self.out / f"games_{name}"
        est = (games + extra / 4) * 60 / self.a.games_per_h + 1
        cmd = (["tools/run_elo_batch.py"] + flags
               + ["--outdir", str(d), "--games", str(games), "--max-extra-games", str(extra),
                  "--seed-base", str(base), "--persistent-workers", "--shared-inference", "--no-infer-compile",
                  "--device", "cuda", "--jobs", str(self.a.jobs), "--worker-games", str(self.a.worker_games),
                  "--min-free-mb", str(MIN_FREE_MB), "--infer-fp16", "--inference-timeline-s", "30",
                  "--time-budget-min", str(int(3 * est / 2 + 10))])
        t0 = time.time()
        rc = self.run(f"match {name}", cmd, f"match_{name}.log", 2 * est + 15, self.py)
        results, decisive, lost = self.count_games(d)
        resets = display_resets_since(t0)
        if resets:
            why = f"{resets} display-driver reset(s) (System event 4101) during match {name}"
            self.note(f"STOPPED: {why}")
            (self.out / "STOP").write_text(why + "\n", encoding="utf-8")
            log(self.out, f"STOPPED: {why}; the STOP file is written, delete it to resume")
            raise SystemExit(5)
        with open(self.out / f"wall_{name}.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.time() - t0:.0f}\n")
        self.note(f"{name}: games {games} from seed {base}, rc={rc}, {time.time() - t0:.0f} s, "
                  f"results {results}, decisive {decisive}, failed slots lost {lost}")
        return rc, results, decisive, lost

    def play_through(self, kind: str, name: str, games: int, base: int, extra: int,
                     flags: List[str]) -> Tuple[bool, int, int]:
        """The match, played again once when short; (accepted, rc, missing)."""
        rc, results, decisive, _ = self.play(name, games, base, extra, flags)
        if rc in (1, 3, 124):
            rc, results, decisive, _ = self.play(name, games, base, extra, flags)
        missing = games - (results if kind == "selfplay" else decisive)
        if rc in (0, 4):
            ok = kind == "measure" or missing <= LOST_MAX
        elif rc in (1, 3, 124):
            ok = missing <= LOST_MAX
        else:
            ok = False
        if ok and missing > 0:
            self.note(f"{name}: accepted with {missing} of {games} missing (at most {LOST_MAX})")
        return ok, rc, missing

    # ---- the steps
    def selfplay_match(self, k: int, sp_a: List[str], sp_b: List[str]) -> bool:
        name = f"s{k:02d}"
        step = f"chunk_{name}"
        if self.marked(step):
            return True
        if self.failed(step):
            log(self.out, f"{name}: failed at an earlier entry: {self.failed(step)}")
            return False
        size = min(CHUNK_GAMES, N_SELFPLAY - k * CHUNK_GAMES)
        flags = sp_a + sp_b + ["--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1",
                               "--log-decisions"]
        ok, rc, missing = self.play_through("selfplay", name, size, SP_SEED_BASE + k * CHUNK_GAMES, 0, flags)
        if not ok:
            self.mark(step, f"the match ended rc={rc} with {missing} of {size} games missing, past {LOST_MAX}")
            return False
        for f in (self.out / f"walk_{name}.jsonl", self.out / f"rows_{name}.jsonl"):
            f.unlink(missing_ok=True)
        rc = self.run(f"rows {name}", ["tools/selfplay_rows.py", "--games", str(self.out / f"games_{name}"),
                                       "--walk", str(self.out / f"walk_{name}.jsonl"),
                                       "--out", str(self.out / f"rows_{name}.jsonl"),
                                       "--seed-base", str(SP_SEED_BASE), "--jobs", str(self.a.walk_jobs),
                                       "--max-errors", str(LOST_MAX - max(missing, 0))],
                      "rows.log", 30, self.tools_py)
        if rc != 0:
            self.mark(step, f"its rows failed or left more than {LOST_MAX} games out, rc={rc} (rows.log)")
            return False
        self.mark(step)
        return True

    def read_selfplay(self) -> str:
        if self.marked("readout"):
            return "read"
        names = [f"s{k:02d}" for k in range(self.n_chunks())]
        with open(self.out / "rows.jsonl.tmp", "w", encoding="utf-8") as dst:
            for n in names:
                dst.write((self.out / f"rows_{n}.jsonl").read_text(encoding="utf-8"))
        os.replace(self.out / "rows.jsonl.tmp", self.out / "rows.jsonl")
        rc = self.run("labels", ["tools/pg_labels.py", str(self.out / "rows.jsonl"),
                                 "--heldout-from", str(HELDOUT_FROM), "--out", str(self.out / "labels.jsonl"),
                                 "--fits", str(self.out / "fits.json")], "labels.log", 30, self.tools_py)
        # pg_labels exits 3 when a game had no luck walk: both its sides are left out, named in fits.json.
        if rc not in (0, 3) or not (self.out / "labels.jsonl").exists():
            return f"FAILED: the labels, rc={rc} (labels.log)"
        rc = self.run("readout", ["tools/selfplay_readout.py", "--games"]
                      + [str(self.out / f"games_{n}") for n in names]
                      + ["--labels", str(self.out / "labels.jsonl"), "--fits", str(self.out / "fits.json"),
                         "--walk"] + [str(self.out / f"walk_{n}.jsonl") for n in names]
                      + ["--out", str(self.out / "readout.json")], "readout.log", 120, self.tools_py)
        # selfplay_readout exits 1 when a gate fails: readout.json says which.
        if rc > 1 or not (self.out / "readout.json").exists():
            return f"FAILED: the readout, rc={rc} (readout.log)"
        self.mark("readout")
        return "read"

    def measure(self, name: str, games: int, base: int, anchor: str, flags: List[str]) -> str:
        step = f"measure_{name}"
        if self.marked(step):
            return "done"
        if self.failed(step):
            return f"FAILED at an earlier entry: {self.failed(step)}"
        ok, rc, missing = self.play_through("measure", name, games, base, games // 2, flags)
        if not ok:
            why = f"the match ended rc={rc} with {games - missing} of {games} decisive games"
        elif self.run(f"fit {name}", ["tools/elo_collect.py", str(self.out / f"games_{name}"), "--no-catalog",
                                      "--anchor", anchor, "--save-json", str(self.out / f"{name}.fit.json")],
                      f"fit_{name}.log", 15, self.tools_py) != 0:
            why = f"its fit failed (fit_{name}.log)"
        else:
            self.mark(step)
            return "done"
        self.mark(step, why)
        return f"FAILED: {why}"

    def n_chunks(self) -> int:
        return (N_SELFPLAY + CHUNK_GAMES - 1) // CHUNK_GAMES

    def progress(self, what: str) -> None:
        done = sum(self.marked(f"chunk_s{k:02d}") for k in range(self.n_chunks()))
        (self.out / "progress.txt").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\nself-play: {done} of {self.n_chunks()} matches; {what}\n",
            encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=96, help="games in flight (the pilots: 96 kept the GPU busy)")
    ap.add_argument("--worker-games", type=int, default=8)
    ap.add_argument("--walk-jobs", type=int, default=6)
    ap.add_argument("--games-per-h", type=float, default=4300.0,
                    help="the pool's steady rate at temperature 1, for the steps' cuts (the 2026-10-28 pilot)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter of the matches (the ROCm venv)")
    ap.add_argument("--tools-python", default=None,
                    help="the interpreter of the CPU tools (default: the repository's .venv, whose CPU torch "
                         "is faster than the ROCm build's)")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--keep-awake", action="store_true",
                    help="ask Windows not to sleep while the runner lives (keep_awake); the machine's "
                         "user decides when a run may hold the machine up")
    ap.add_argument("--smoke", action="store_true",
                    help="the whole chain on a few games, on seeds outside stage 1's (873000 on): two "
                         "self-play matches of 24 (the second held out), D0 and the memory report of 24")
    args = ap.parse_args()
    if args.smoke:
        global SP_SEED_BASE, N_SELFPLAY, CHUNK_GAMES, HELDOUT_FROM, D0_SEED_BASE, D0_GAMES, MEM_SEED_BASE, MEM_GAMES
        SP_SEED_BASE, N_SELFPLAY, CHUNK_GAMES, HELDOUT_FROM = 873000, 48, 24, 24
        D0_SEED_BASE, D0_GAMES, MEM_SEED_BASE, MEM_GAMES = 873100, 24, 873200, 24
    if args.tools_python is None:
        args.tools_python = default_tools_python(REPO)
    require_visible_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.normal_priority:
        below_normal_priority()
    if args.keep_awake:
        keep_awake()
    r = Run(args)
    log(args.out, f"stage 1 local: out {args.out}, jobs {args.jobs}x{args.worker_games}, "
                  f"python {args.python}, tools {args.tools_python}")
    ref = r.reference_flags("b")
    ref_label = ref[ref.index("--label-b") + 1]
    if ref_label != "own2" or ref[ref.index("--memory-b") + 1] != "64":
        raise SystemExit(f"REFERENCE_CHANGED: {ref}; stage 1 plays own2 at 64 slots")
    sp_a, sp_b = r.side_flags("a", "own2a"), r.side_flags("b", "own2b")
    d0_a, d0_b = r.side_flags("a", "own2t1"), ref
    mem_a, mem_b = r.side_flags("a", "own2m64", 64), r.side_flags("b", "own2m0", 0)
    t1 = ["--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1"]

    def stop(at: str) -> int:
        log(args.out, f"STOP found before {at}: stopping; a new start resumes")
        r.progress(f"stopped before {at}")
        return 2

    if r.stop_requested():
        return stop("the first match")
    first_ok = r.selfplay_match(0, sp_a, sp_b)
    r.progress("first match " + ("done" if first_ok else "FAILED"))
    if r.stop_requested():
        return stop("D0")
    d0 = r.measure("d0", D0_GAMES, D0_SEED_BASE, ref_label,
                   d0_a + d0_b + ["--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "0"])
    log(args.out, f"D0: {d0}")
    selfplay = "not played" if not first_ok else "played"
    if first_ok:
        for k in range(1, r.n_chunks()):
            if r.stop_requested():
                return stop(f"self-play match s{k:02d}")
            if not r.selfplay_match(k, sp_a, sp_b):
                selfplay = f"FAILED at s{k:02d}"
                break
            r.progress(f"D0 {d0}")
    readout = "not read"
    if selfplay == "played":
        if r.stop_requested():
            return stop("the labels")
        readout = r.read_selfplay()
    log(args.out, f"self-play: {selfplay}; readout: {readout}")
    if r.stop_requested():
        return stop("the memory report")
    mem = r.measure("memory", MEM_GAMES, MEM_SEED_BASE, "own2m0", mem_a + mem_b + t1)
    log(args.out, f"memory report: {mem}")
    r.progress(f"done: self-play {selfplay}, readout {readout}, D0 {d0}, memory {mem}")
    (args.out / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    return 0 if (selfplay == "played" and readout == "read" and d0 == "done" and mem == "done") else 1


if __name__ == "__main__":
    sys.exit(main())
