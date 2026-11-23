"""The simple training loop on the local RX 6600
(docs/simple_loop_prereg_20261117.md; the rulings it follows are in
docs/training_dynamics_theory_20261111.md, "The user's decisions").

    set HIP_VISIBLE_DEVICES=1
    <rocm venv>/python scripts/simple_loop_local.py --out C:/Users/<you>/wesnoth_runs/simple_loop ^
        --start path/to/pi.pt --lr 1e-5 --lam 0.99 --keep-awake

One round is one optimizer step:

1. **Play.** The current network plays --games games at temperature 1 with
   no end_turn offset, its decisions logged (tools/run_elo_batch.py, the
   local pool of scripts/match_local.py): --past-games of them against one
   past self, drawn uniformly from the pool, the rest against itself. The
   pool is the start network and every checkpoint a match has read.
2. **Pass.** tools/pg_game_pass.py: the whole-game gradient over the
   current network's game-sides, with the value head trained on the
   result. At --lam 1 each decision is weighted by its side's result less
   the logged value estimate; under 1 the estimates logged later in the
   game stand in for part of the result, each later change of the estimate
   counting for --lam times the one before it (generalised advantage
   estimation, the pass's `lambda_weights`). Its exact checks (every logged
   decision trained, the replay identity) stop the run.
3. **Step.** tools/pg_adam_step.py: one Adam step at --lr, the optimizer's
   state carried from round to round. A gradient that is not finite stops
   the run.

Before the first round and after every --match-every rounds,
scripts/match_local.py reads the network as it trains (--sampled-candidate:
it samples at temperature 1, the reference plays its own decode) over
--match-games decisive games, always from the same seeds, and the matched
checkpoint joins the pool. --early-match adds one match after that round:
an early reading of a new learning rate. **The kill criterion:** the run stops when two
matches in a row read more than KILL_SE standard errors below the first
(the difference's standard error: the two matches' in quadrature).

**The blow-up warning** stops nothing: a line in warnings.log when a
round's gradient norm is more than BLOWUP_FACTOR times the median of the
rounds before it (from BLOWUP_MIN_ROUNDS on).

The run resumes from its own markers (`r0007_play.done`, `r0007_pass.done`,
`r0007_step.done`, `m0024_match.done`): the same command again goes on
where it stopped, a match from the games on disk. A STOP file in --out
ends it before the next tool; a display-driver reset during a GPU tool
writes one. --yield-gpu steps aside while the machine's user's programs
use the GPU (scripts/gpu_yield.py): a tool is stopped and run again once
they have been quiet.

In --out: config.json (a later start with another configuration is
refused; --lr, --lam and --max-rounds may change), round_0007/ (games_self/,
games_past/, grad.json, pi.pt, pi.step.json), matches/step_0024/ (the
match's own records), pool.json, matches.json, progress.jsonl (one line a
round), warnings.log, runner.log. Kept: every pool member's checkpoint and
the latest KEEP_LATEST rounds'; a round's grad.pt and the optimizer's
state go when the next round's step is done.

Exit 0 at --max-rounds, 2 on a STOP file, 5 on a display-driver reset or a
failed tool, 6 on the kill criterion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.gpu_yield import GpuYield  # noqa: E402
from scripts.local_machine import (  # noqa: E402
    MATCH_SHORT, Steps, Stopped, below_normal_priority, default_tools_python, display_resets_since, keep_awake,
    log, require_visible_gpu, reset_reason, run_match, run_tool,
)
from scripts.match_local import POOL  # noqa: E402
from tools.eval_provenance import file_sha256  # noqa: E402

GAMES = 128                 # a round's games (user ruling 2026-11-17, a number to revise)
PAST_GAMES = 26             # 20% of them against a past self (user ruling 2026-11-17)
SLOTS = 64                  # the memory both players play at (configs/reference_player.json's)
SEED_BASE = 100_000_000     # clear of the earlier rounds' seeds (10M + k * 1M, 39 rounds planned: under 50M)
ROUND_SEEDS = 1000          # round k's games take SEED_BASE + k * ROUND_SEEDS on, the past self's from +500
PAST_SEED_OFFSET = 500
MATCH_EVERY = 16            # rounds between matches: about four a day at the measured 23 min a round
MATCH_GAMES = 800           # decisive games a match (CLAUDE.md, "Strength claims")
KILL_SE = 3.0               # the kill criterion's standard errors
BLOWUP_FACTOR = 10.0        # the blow-up warning's factor over the median norm
BLOWUP_MIN_ROUNDS = 5       # rounds of norms before the warning reads
LOST_MAX = 2                # games a round may lack (a failed slot), each named in the log
KEEP_LATEST = 2             # rounds whose checkpoints are kept beside the pool's
JOBS = 96                   # games in flight (scripts/match_local.py's pool)
WORKER_GAMES = 8
EXIT_STOP, EXIT_FAILED, EXIT_KILL = 2, 5, 6
PASS_REFUSED = 3            # tools/pg_game_pass.py's and tools/pg_adam_step.py's exit on a refusal


class RunFailed(Exception):
    """A tool failed or refused: the run ends, its reason in STOP."""


def round_dir(out: Path, k: int) -> Path:
    return out / f"round_{k:04d}"


def checkpoint_of(out: Path, k: int) -> Path:
    """The network after round k (round 0: the start network's copy)."""
    return round_dir(out, k) / "pi.pt"


def round_seeds(k: int) -> Dict[str, int]:
    """Round k's seed bases: its self-play games' and its past-self games'."""
    base = SEED_BASE + k * ROUND_SEEDS
    return {"self": base, "past": base + PAST_SEED_OFFSET}


def draw_past(pool: List[dict], seed: int, k: int) -> dict:
    """Round k's past self: a uniform draw from the pool, a pure function
    of the run's seed and the round, so a resumed round draws the same."""
    if not pool:
        raise ValueError("the pool is empty")
    return random.Random(f"{seed}|past|{k}").choice(sorted(pool, key=lambda m: m["step"]))


def play_argv(label_a: str, spec_a: Path, label_b: str, spec_b: Path, outdir: Path, games: int, seed_base: int,
              jobs: int, worker_games: int) -> List[str]:
    """run_elo_batch's command line for a round's games: both players
    sampling at temperature 1 with no offset at SLOTS memory slots, their
    decisions logged, every slot played once (no replacement: a game
    without a winner is kept, and trains at z = 0)."""
    return (["tools/run_elo_batch.py", "--label-a", label_a, "--spec-a", str(spec_a), "--memory-a", str(SLOTS),
             "--label-b", label_b, "--spec-b", str(spec_b), "--memory-b", str(SLOTS),
             "--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1", "--log-decisions",
             "--outdir", str(outdir), "--games", str(games), "--max-extra-games", "0",
             "--seed-base", str(seed_base)] + POOL
            + ["--jobs", str(min(jobs, games)), "--worker-games", str(worker_games)])


def blowup(norm: float, previous: List[float]) -> Optional[str]:
    """The blow-up warning's text for a round's gradient norm against the
    norms of the rounds before it, or None."""
    if len(previous) < BLOWUP_MIN_ROUNDS:
        return None
    median = statistics.median(previous)
    if median > 0 and norm > BLOWUP_FACTOR * median:
        return (f"the gradient's norm {norm:.4g} is {norm / median:.1f} times the median {median:.4g} of the "
                f"{len(previous)} rounds before it")
    return None


def below_first(match: dict, first: dict) -> bool:
    """Whether a match reads more than KILL_SE standard errors below the
    run's first."""
    se = math.sqrt(match["se"] ** 2 + first["se"] ** 2)
    return match["elo"] - first["elo"] < -KILL_SE * se


def kill_reason(matches: List[dict]) -> Optional[str]:
    """The kill criterion on the run's matches in order (each {step, elo,
    se}): the reason when the last two both read below the first."""
    if len(matches) < 3:
        return None
    first, a, b = matches[0], matches[-2], matches[-1]
    if below_first(a, first) and below_first(b, first):
        return (f"the matches at steps {a['step']} and {b['step']} read {a['elo']:+.1f} +- {a['se']:.1f} and "
                f"{b['elo']:+.1f} +- {b['se']:.1f} Elo, both more than {KILL_SE:g} standard errors below the "
                f"first ({first['elo']:+.1f} +- {first['se']:.1f} at step {first['step']})")
    return None


def prunable(done: int, pool_steps: List[int]) -> List[int]:
    """The rounds whose checkpoints may go once round `done` is done: all
    but the pool's and the latest KEEP_LATEST."""
    keep = set(pool_steps) | set(range(done - KEEP_LATEST + 1, done + 1))
    return [k for k in range(done + 1) if k not in keep]


def count_results(d: Path) -> int:
    return len(list(d.glob("game_*.json")))


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Loop:
    """The run in `args.out` (module docstring). `runner` stands in for
    run_tool and `resets` for display_resets_since (the tests replace
    both)."""

    def __init__(self, args, runner=None, resets=None):
        self.a = args
        self.out: Path = args.out
        self.runner = runner or run_tool
        self.resets = resets or display_resets_since
        self.steps = Steps(self.out, args.python, args.tools_python, runner=self.runner, resets=self.resets)
        self.gpu_yield = (GpuYield(threshold=args.yield_gpu, quiet_minutes=args.yield_quiet_min)
                          if args.yield_gpu > 0 else None)

    # ---- the run's records --------------------------------------------------
    def pool(self) -> List[dict]:
        return _read_json(self.out / "pool.json", [])

    def matches(self) -> List[dict]:
        return _read_json(self.out / "matches.json", [])

    def progress(self) -> List[dict]:
        path = self.out / "progress.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def last_done(self) -> int:
        k = 0
        while self.steps.done(f"r{k + 1:04d}_step"):
            k += 1
        return k

    def warn(self, text: str) -> None:
        log(self.out, f"WARNING: {text}")
        log(self.out, text, name="warnings.log")

    # ---- tools ----------------------------------------------------------------
    def gpu_tool(self, name: str, cmd: List[str], log_name: str, cut_min: float) -> int:
        """A GPU tool to its end or its cut, stepping aside for the user's
        programs under --yield-gpu (it is then run again); a display reset
        during it writes STOP and ends the run."""
        self.steps.check_stop(name)
        t0 = time.time()
        rc = run_match(self.out, name, cmd, log_name, cut_min, self.a.python, gpu_yield=self.gpu_yield,
                       runner=self.runner)
        n = self.resets(t0)
        if n:
            why = reset_reason(n, name)
            (self.out / "STOP").write_text(why + "\n", encoding="utf-8")
            raise Stopped(why)
        return rc

    def fail(self, why: str) -> None:
        (self.out / "STOP").write_text(why + "\n", encoding="utf-8")
        raise RunFailed(why)

    # ---- a round ----------------------------------------------------------------
    def play(self, k: int, name: str, label_b: str, spec_b: Path, games: int, seed_base: int) -> None:
        """One of round k's two sets of games, played again once when short."""
        theta = checkpoint_of(self.out, k - 1)
        d = round_dir(self.out, k) / f"games_{name}"
        cmd = play_argv(f"s{k - 1:04d}", theta, label_b, spec_b, d, games, seed_base, self.a.jobs,
                        self.a.worker_games) + ["--time-budget-min", str(self.a.play_cut_min - 10)]
        tag = f"round {k} games ({name})"
        rc = self.gpu_tool(tag, cmd, f"round_{k:04d}.log", self.a.play_cut_min)
        if rc in MATCH_SHORT and count_results(d) < games:
            log(self.out, f"{tag}: ended rc={rc}, short: played again once from the games on disk")
            rc = self.gpu_tool(f"{tag} again", cmd, f"round_{k:04d}.log", self.a.play_cut_min)
        missing = games - count_results(d)
        if missing > LOST_MAX:
            self.fail(f"{tag}: rc={rc} with {missing} of {games} games missing, past {LOST_MAX}")
        if missing > 0:
            log(self.out, f"{tag}: accepted with {missing} of {games} games missing (at most {LOST_MAX})")

    def round(self, k: int) -> None:
        d = round_dir(self.out, k)
        d.mkdir(parents=True, exist_ok=True)
        theta = checkpoint_of(self.out, k - 1)
        seeds = round_seeds(k)
        past_games = self.a.past_games
        part = f"r{k:04d}_play"
        t_play = time.time()
        if not self.steps.done(part):
            self.play(k, "self", f"s{k - 1:04d}b", theta, self.a.games - past_games, seeds["self"])
            if past_games:
                past = draw_past(self.pool(), self.a.seed, k)
                _write_json(d / "past.json", past)
                self.play(k, "past", f"p{past['step']:04d}", Path(past["checkpoint"]), past_games, seeds["past"])
            self.steps.mark(part)
        play_s = time.time() - t_play

        part = f"r{k:04d}_pass"
        grad = d / "grad.pt"
        t_pass = time.time()
        if not self.steps.done(part):
            dirs = [str(d / "games_self")] + ([str(d / "games_past")] if past_games else [])
            cmd = (["tools/pg_game_pass.py", "--checkpoint", str(theta), "--games", *dirs, "--out", str(grad),
                    "--device", "cuda", "--fp16", "--packed-trunk", "--workers", str(self.a.pass_workers),
                    "--streams", str(self.a.streams), "--group-decisions", str(self.a.group_decisions),
                    "--lam", str(self.a.lam)])
            rc = self.gpu_tool(f"round {k} pass", cmd, f"round_{k:04d}.log", self.a.pass_cut_min)
            if rc != 0:
                self.fail(f"round {k}'s pass ended rc={rc}"
                          + (": an exact check failed (round log)" if rc == PASS_REFUSED else ""))
            self.steps.mark(part)
        pass_s = time.time() - t_pass
        reading = json.loads(grad.with_suffix(".json").read_text(encoding="utf-8"))

        part = f"r{k:04d}_step"
        if not self.steps.done(part):
            why = blowup(float(reading["norm"]), [float(p["grad_norm"]) for p in self.progress()])
            if why:
                self.warn(f"round {k}: {why}")
            if any(g["scale"] is not None and g["scale"] < 1.0 for g in reading["groups"]):
                self.warn(f"round {k}: a group's gradient ran at a loss scale under 1 "
                          f"({min(g['scale'] for g in reading['groups']):g}): fp16's range is short")
            state = round_dir(self.out, k - 1) / "adam.pt"
            cmd = (["tools/pg_adam_step.py", "--checkpoint", str(theta), "--grad", str(grad), "--lr", str(self.a.lr),
                    "--out", str(d / "pi.pt"), "--state-out", str(d / "adam.pt")]
                   + (["--state", str(state)] if k > 1 else []))
            self.steps.check_stop(f"round {k} step")
            rc = self.runner(self.out, f"round {k} step", cmd, f"round_{k:04d}.log", 30, self.a.tools_python)
            if rc != 0:
                self.fail(f"round {k}'s step ended rc={rc}"
                          + (": the step was refused (round log)" if rc == PASS_REFUSED else ""))
            step = json.loads((d / "pi.step.json").read_text(encoding="utf-8"))
            # The lambda is the pass's own record of it: a pass run before a stop keeps the one it used.
            weight = reading.get("weight") or {}
            line = {"round": k, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "lr": self.a.lr,
                    "lam": weight.get("lam", 1.0), "weight_mean_square": weight.get("mean_square"),
                    "weight_mean_square_at_1": weight.get("mean_square_at_1"),
                    "game_sides": reading["game_sides"], "skipped_sides": reading["skipped_sides"],
                    "decisions": reading["decisions"], "grad_norm": reading["norm"],
                    "update_norm": step["update_norm"], "replay_kl": reading["replay"]["kl"],
                    "value_cross_entropy": reading["value"]["cross_entropy"],
                    "value_squared_error": reading["value"]["squared_error"],
                    "pass_decisions_per_second": reading["rate"]["decisions_per_second"],
                    "peak_memory_mb": reading["peak_memory_mb"],
                    "overflows": (reading["loss_scale"] or {}).get("overflows"),
                    "play_seconds": round(play_s), "pass_seconds": round(pass_s), "warning": why}
            if not any(p["round"] == k for p in self.progress()):     # a step run again writes no second line
                with open(self.out / "progress.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
            self.steps.mark(part)
            log(self.out, f"round {k} done: {reading['game_sides']} game-sides, {reading['decisions']} decisions, "
                          f"gradient norm {reading['norm']:.4g}, update norm {step['update_norm']:.4g}, value CE "
                          f"{reading['value']['cross_entropy']:.4f}, play {play_s:.0f} s, pass {pass_s:.0f} s")
        self.prune(k)

    def prune(self, done: int) -> None:
        """What a finished round no longer needs: the round before's
        gradient and optimizer state, and the checkpoints `prunable` names."""
        for k in range(max(0, done - KEEP_LATEST - 1), done):
            for name in ("grad.pt", "adam.pt"):
                (round_dir(self.out, k) / name).unlink(missing_ok=True)
        (round_dir(self.out, done) / "grad.pt").unlink(missing_ok=True)
        for k in prunable(done, [m["step"] for m in self.pool()]):
            (round_dir(self.out, k) / "pi.pt").unlink(missing_ok=True)

    # ---- a match ------------------------------------------------------------------
    def match(self, k: int) -> None:
        """The network after round k against the reference; it joins the
        pool, and the kill criterion is read."""
        part = f"m{k:04d}_match"
        if self.steps.done(part):
            return
        d = self.out / "matches" / f"step_{k:04d}"
        spec = checkpoint_of(self.out, k)
        cmd = (["scripts/match_local.py", "--label", f"s{k:04d}", "--checkpoint", str(spec), "--out", str(d),
                "--games", str(self.a.match_games), "--sampled-candidate", "--python", self.a.python,
                "--tools-python", self.a.tools_python, "--jobs", str(self.a.jobs),
                "--worker-games", str(self.a.worker_games), "--normal-priority"]
               + (["--yield-gpu", str(self.a.yield_gpu), "--yield-quiet-min", str(self.a.yield_quiet_min)]
                  if self.a.yield_gpu > 0 else []))
        self.steps.check_stop(f"the match at step {k}")
        t0 = time.time()
        rc = self.runner(self.out, f"match at step {k}", cmd, "matches.log", self.a.match_cut_min, self.a.python)
        n = self.resets(t0)
        if n or rc == 5:
            why = reset_reason(n, f"the match at step {k}") if n else f"the match at step {k} stopped itself (rc 5)"
            (self.out / "STOP").write_text(why + "\n", encoding="utf-8")
            raise Stopped(why)
        result = _read_json(d / "result.json", None)
        if rc != 0 or result is None or result.get("elo") is None or not result.get("complete"):
            self.fail(f"the match at step {k} ended rc={rc} without a complete fit ({d})")
        matches = self.matches() + [{"step": k, "elo": float(result["elo"]), "se": float(result["se"]),
                                     "decisive": int(result["decisive"]), "wins": int(result["wins"]),
                                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}]
        _write_json(self.out / "matches.json", matches)
        pool = self.pool()
        if k not in [m["step"] for m in pool]:
            pool.append({"step": k, "checkpoint": str(spec), "sha256": file_sha256(spec)})
            _write_json(self.out / "pool.json", pool)
        self.steps.mark(part)
        log(self.out, f"match at step {k}: {result['elo']:+.1f} +- {result['se']:.1f} Elo against the reference "
                      f"({result['wins']} of {result['decisive']} decisive games)")
        why = kill_reason(matches)
        if why:
            (self.out / "STOP").write_text(f"kill criterion: {why}\n", encoding="utf-8")
            log(self.out, f"KILL CRITERION: {why}")
            raise SystemExit(EXIT_KILL)

    # ---- the run -------------------------------------------------------------------
    def start(self) -> None:
        """Round 0: the start network's copy, the first pool member."""
        d = round_dir(self.out, 0)
        if not self.steps.done("r0000_step"):
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "pi.pt.tmp"
            shutil.copyfile(self.a.start, tmp)
            os.replace(tmp, d / "pi.pt")
            _write_json(self.out / "pool.json", [{"step": 0, "checkpoint": str(d / "pi.pt"),
                                                  "sha256": file_sha256(d / "pi.pt")}])
            self.steps.mark("r0000_step")

    def run(self) -> int:
        self.start()
        self.match(0)
        k = self.last_done()
        if k and (k % self.a.match_every == 0 or k == self.a.early_match):
            self.match(k)                         # a run stopped between a round and its match
        while self.a.max_rounds <= 0 or k < self.a.max_rounds:
            self.steps.check_stop(f"round {k + 1}")
            k += 1
            self.round(k)
            if k % self.a.match_every == 0 or k == self.a.early_match:
                self.match(k)
        log(self.out, f"the run reached its {self.a.max_rounds} rounds")
        return 0


def check_config(out: Path, config: dict) -> None:
    """The run's configuration in out/config.json: written at the first
    start, and a later start with another one is refused."""
    path = out / "config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != config:
            diff = {k: (saved.get(k), config.get(k)) for k in sorted(set(saved) | set(config))
                    if saved.get(k) != config.get(k)}
            raise SystemExit(f"CONFIG_DIFFERS: {out} was started with another configuration: {diff}")
        return
    _write_json(path, config)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="the run's directory (resumed when it exists)")
    ap.add_argument("--start", type=Path, required=True, help="the network the run starts from")
    ap.add_argument("--lr", type=float, required=True, help="the learning rate (a change is a numbered departure)")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="the pass's weight: each later change of the value estimate counts for this much of the "
                         "one before it (1: the result less the estimate; a change is a numbered departure)")
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--past-games", type=int, default=PAST_GAMES)
    ap.add_argument("--match-every", type=int, default=MATCH_EVERY)
    ap.add_argument("--match-games", type=int, default=MATCH_GAMES)
    ap.add_argument("--early-match", type=int, default=0, help="one more match after this round (0: none)")
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this round (0: run until stopped)")
    ap.add_argument("--seed", type=int, default=20261117, help="the past selves' draws")
    ap.add_argument("--streams", type=int, default=32, help="the pass's game-sides a group")
    ap.add_argument("--group-decisions", type=int, default=8192, help="the pass's decisions a group")
    ap.add_argument("--pass-workers", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--worker-games", type=int, default=WORKER_GAMES)
    ap.add_argument("--play-cut-min", type=float, default=45.0)
    ap.add_argument("--pass-cut-min", type=float, default=90.0)
    ap.add_argument("--match-cut-min", type=float, default=120.0)
    ap.add_argument("--python", default=sys.executable, help="the interpreter of the GPU tools (the ROCm venv)")
    ap.add_argument("--tools-python", default=None, help="the interpreter of the CPU tools (default: .venv)")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--yield-gpu", type=float, default=0.0, metavar="PCT",
                    help="step aside while another process uses at least PCT%% of a GPU engine")
    ap.add_argument("--yield-quiet-min", type=float, default=10.0)
    ap.add_argument("--keep-awake", action="store_true")
    args = ap.parse_args(argv)
    if not 0 <= args.past_games < args.games or args.match_every < 1 or args.match_games < 1:
        ap.error("--past-games must lie in 0..games-1, --match-every and --match-games be positive")
    if args.games > PAST_SEED_OFFSET or args.past_games > ROUND_SEEDS - PAST_SEED_OFFSET:
        ap.error(f"a round's games must fit its seeds ({PAST_SEED_OFFSET} and {ROUND_SEEDS - PAST_SEED_OFFSET})")
    if not args.lr > 0:
        ap.error("--lr must be positive")
    if not 0.0 <= args.lam <= 1.0:
        ap.error("--lam must lie in 0..1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.tools_python is None:
        args.tools_python = default_tools_python(REPO)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "STOP").exists():
        why = (args.out / "STOP").read_text(encoding="utf-8").strip()
        log(args.out, f"STOP found ({why}): not starting; delete it to resume")
        return EXIT_STOP
    if not (args.out / "r0000_step.done").exists() and not args.start.is_file():
        raise SystemExit(f"--start {args.start}: no such file")
    require_visible_gpu()
    check_config(args.out, {"start_sha256": file_sha256(args.start) if args.start.is_file() else None,
                            "games": args.games, "past_games": args.past_games, "match_every": args.match_every,
                            "early_match": args.early_match,
                            "match_games": args.match_games, "seed": args.seed, "slots": SLOTS,
                            "seed_base": SEED_BASE, "streams": args.streams,
                            "group_decisions": args.group_decisions})
    if not args.normal_priority:
        below_normal_priority()
    if args.keep_awake:
        keep_awake()
    log(args.out, f"simple loop: out {args.out}, start {args.start}, lr {args.lr:g}, lambda {args.lam:g}, "
                  f"{args.games} games a round "
                  f"({args.past_games} against a past self), a match every {args.match_every} rounds, python "
                  f"{args.python}, tools {args.tools_python}")
    loop = Loop(args)
    try:
        return loop.run()
    except Stopped as e:
        log(args.out, f"STOPPED: {e}; delete the STOP file to resume")
        return EXIT_STOP if "STOP found" in str(e) else EXIT_FAILED
    except RunFailed as e:
        log(args.out, f"FAILED: {e}; the STOP file is written")
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
