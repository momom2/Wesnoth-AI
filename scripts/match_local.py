"""The standard strength match on the local machine (Windows, the RX 6600;
docs/new_machine.md step 8): a candidate checkpoint on side A against the
reference player on side B, N decisive games, then the PURE fit, the
reference anchored at 0 (CLAUDE.md, "Eval procedure").

    set HIP_VISIBLE_DEVICES=1
    <rocm venv>/python scripts/match_local.py --label cand --checkpoint path/to/cand.pt ^
        --out C:/Users/<you>/wesnoth_runs/match_cand

What it plays, as the box's 800-game matches do (scripts/own_confirm_box.sh,
scripts/parity3_baselines_box.sh, scripts/unit_vocab_retrain_box.sh):
- Side B: the reference, `tools/reference_player.py --flags b` (its label,
  checkpoint, memory slots and end_turn offset).
- Side A: the candidate at the reference's decode, its end_turn offset and
  its memory slots. A candidate that would play another decode or another
  memory (--end-turn-offset, --memory, a checkpoint with fewer slots or
  none) is refused unless --allow-decode-change: its Elo would then measure
  the weights and the decode together.
- Both sides raw players at argmax: --mcts-sims 0 --raw-temperature-a 0
  --raw-temperature-b 0, the procedure raw:t0+eo-1.5.
- With --t1, both sides sample at temperature 1 with no end_turn offset,
  the decode the rounds train at (scripts/selfplay_t1_local.py's
  player_flags): the procedure raw:t1, the reference relabelled <label>t1
  with its memory kept (docs/tail_local_prereg_20261102.md).
- N decisive games (800) from --seed-base. A capped game of slot i is
  replaced at seed + i + k * 1,000,000, up to --max-extra-games (1,500, the
  box's 800-game matches; they capped at most 7 of 807 games). A match short
  of its games (a failed game, a dead server, the time budget) is played
  again once, as the box scripts' `play` does.

The pool (training/metrics/local_gpu_20261028/README.md): 96 games in flight,
eight a worker process, one server autocasting to fp16 (--infer-fp16, the
HIP kernels) whose serve loop overlaps batches (--inference-pipelined, +9.2%
self-play on this GPU, training/metrics/pipelined_serve_20261029). The games
record infer_fp16, so a local match never pools with a box's bf16 games.

Seeds. The default base is 900000. On 2026-10-29 the seeds taken end below
it: no branch's scripts, documents or records name one from 900000 to
999999. Stage 1 holds 800000-848199, the recruit-rule match 700000-739999,
and the coin batches of exp/near-tie-coins up to 600000-899999 (their own
check keeps them below 900000); this machine's runs took 870000-871015
(the pilots), 873000-873299 (stage 1's smoke) and 874000-874015 (this
script's smoke). Every candidate measured from the default base plays the
same seeds (maps, factions, luck); pass another base for fresh ones, a
confirmation.

The process: below-normal priority (--normal-priority opts out),
--keep-awake holds the machine awake, and a display-driver reset (Windows
System event 4101) during the match kills it and writes --out/STOP, which
must be deleted before a new start. A re-run of the same command resumes:
run_elo_batch plays only the slots missing from --out/games, a complete
match is only fitted again, and a configuration other than the one in
--out/match.json is refused.

Records in --out: games/ (every game whole), match.json, matches.txt (one
line an attempt), match.log, fit.log, fit.json (tools/elo_collect.py's),
result.json (the candidate's PURE Elo and standard error), runner.log.
Exit 0 for a complete match and its fit, 1 for a short or failed one, 2 when
STOP is there, 5 on a display-driver reset.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.gpu_yield import GpuYield  # noqa: E402
from scripts.local_machine import (  # noqa: E402
    ToolStopped, below_normal_priority, default_tools_python, display_resets_since, keep_awake, log,
    reference_flags, require_visible_gpu, reset_reason, run_match, run_tool,
)
from tools.eval_provenance import effective_memory, file_sha256  # noqa: E402

GAMES = 800
SEED_BASE = 900000
MAX_EXTRA = 1500                 # the box's 800-game matches (scripts/own_confirm_box.sh `play`)
REPL_SEED_OFFSET = 1_000_000     # tools/run_elo_batch.py: a replacement's seed offset
JOBS = 96
WORKER_GAMES = 8
MIN_FREE_MB = 300                # scripts/selfplay_t1_local.py's
# 457 decisions a second at 96 games in flight (the 2026-10-28 pilot) over
# about 476 decisions a game at argmax (the census): 3,456 games an hour.
GAMES_PER_H = 3450.0
PURE = "PURE (decisive only, primary)"
# The procedure both sides play (CLAUDE.md, "Eval procedure").
DECODE = ["--mcts-sims", "0", "--raw-temperature-a", "0", "--raw-temperature-b", "0"]
# --t1: both sides sample, as the rounds' self-play does (raw:t1).
DECODE_T1 = ["--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1"]
POOL = ["--persistent-workers", "--shared-inference", "--no-infer-compile", "--device", "cuda", "--infer-fp16",
        "--inference-pipelined", "--min-free-mb", str(MIN_FREE_MB), "--inference-timeline-s", "30"]


@dataclass
class Reference:
    """The reference's side B as `tools/reference_player.py --flags b` gives it."""
    label: str
    spec: str
    memory: Optional[int]
    end_turn_offset: float
    flags: List[str]


def parse_reference(flags: List[str]) -> Reference:
    """The reference's flags, read; a flag this script does not know (a new
    option of the reference) is refused, since the candidate would not take
    it."""
    if len(flags) % 2:
        raise SystemExit(f"REFERENCE_FLAGS: not flag-value pairs: {flags}")
    d = dict(zip(flags[0::2], flags[1::2]))
    known = {"--label-b", "--spec-b", "--memory-b", "--raw-end-turn-offset-b"}
    if set(d) - known or not {"--label-b", "--spec-b"} <= set(d):
        raise SystemExit(f"REFERENCE_FLAGS: {flags}: this script knows {sorted(known)}, "
                         f"label and spec required")
    memory = int(d["--memory-b"]) if "--memory-b" in d else None
    return Reference(d["--label-b"], d["--spec-b"], memory, float(d.get("--raw-end-turn-offset-b", 0.0)),
                     list(flags))


def sampled(ref: Reference) -> Reference:
    """The reference as --t1 plays it: relabelled <label>t1, its end_turn
    offset dropped, its checkpoint and memory kept."""
    label, flags = f"{ref.label}t1", []
    for k, v in zip(ref.flags[0::2], ref.flags[1::2]):
        if k == "--raw-end-turn-offset-b":
            continue
        flags += [k, label if k == "--label-b" else v]
    return Reference(label, ref.spec, ref.memory, 0.0, flags)


@dataclass
class Candidate:
    """Side A: the checkpoint (an absolute path, its SHA-256), the --memory-a
    it is given (None: no flag), the memory slots it then plays (None: no
    memory), its end_turn offset, and how that decode differs from the
    reference's."""
    label: str
    spec: str
    sha256: str
    memory: Optional[int]
    plays: Optional[int]
    end_turn_offset: float
    diffs: List[str]

    def flags(self) -> List[str]:
        out = ["--label-a", self.label, "--spec-a", self.spec]
        if self.memory is not None:
            out += ["--memory-a", str(self.memory)]
        if self.end_turn_offset:
            out += ["--raw-end-turn-offset-a", str(self.end_turn_offset)]
        return out


def _slots(n: Optional[int]) -> str:
    return "no memory" if n is None else f"{n} memory slots"


def candidate_decode(ref: Reference, slots: int, memory: Optional[int], end_turn_offset: Optional[float],
                     allow: bool) -> Tuple[Optional[int], Optional[int], float, List[str]]:
    """(the --memory-a to pass, None for none; the slots the candidate then
    plays, None for no memory; the end_turn offset; how this decode differs
    from the reference's) for a checkpoint of `slots` memory slots under the
    --memory and --end-turn-offset given (None: the reference's). Without
    `allow`, any difference is refused."""
    eo = ref.end_turn_offset if end_turn_offset is None else float(end_turn_offset)
    if memory is None:
        # The reference's slots when the checkpoint has them; else all of its own.
        memory = ref.memory if (ref.memory is not None and slots >= ref.memory) else None
    try:
        plays = effective_memory(slots, memory)
    except ValueError as e:
        raise SystemExit(f"--memory: {e}") from e
    diffs = []
    if eo != ref.end_turn_offset:
        diffs.append(f"end_turn offset {eo:g}, the reference's {ref.end_turn_offset:g}")
    if plays != ref.memory:
        diffs.append(f"{_slots(plays)}, the reference's {_slots(ref.memory)}")
    if diffs and not allow:
        raise SystemExit("DECODE_DIFFERS: the candidate would play " + "; ".join(diffs)
                         + ". Pass --allow-decode-change to measure it so (the Elo then measures the "
                           "weights and the decode together).")
    return memory, plays, eo, diffs


_PEEK_SLOTS = (
    "import sys, torch; raw = torch.load(sys.argv[1], map_location='cpu', weights_only=True); "
    "print(int(raw.get('memory_slots') or 0) if isinstance(raw, dict) else 'not-a-checkpoint')")


def memory_slots_of(spec: Path, python: str) -> int:
    """The memory slots a checkpoint carries (the `memory_slots` field
    tools/eval_players.peek_checkpoint_arch reads; 0 without one), read in a
    child so the driver stays torch-free."""
    r = subprocess.run([python, "-c", _PEEK_SLOTS, str(spec)], capture_output=True, text=True, timeout=600)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode != 0 or not last.isdigit():
        raise SystemExit(f"could not read the memory slots of {spec}: {(r.stderr or last).strip()[-500:]}")
    return int(last)


def match_argv(cand: Candidate, ref: Reference, games_dir: Path, games: int, seed_base: int, max_extra: int,
               jobs: int, worker_games: int, budget_min: int, decode: List[str] = DECODE) -> List[str]:
    """run_elo_batch's command line for one attempt."""
    return (["tools/run_elo_batch.py"] + cand.flags() + ref.flags
            + ["--outdir", str(games_dir), "--games", str(games), "--max-extra-games", str(max_extra),
               "--seed-base", str(seed_base)] + decode + POOL
            + ["--jobs", str(jobs), "--worker-games", str(worker_games), "--time-budget-min", str(budget_min)])


def fit_argv(games_dir: Path, ref_label: str, fit_json: Path) -> List[str]:
    return ["tools/elo_collect.py", str(games_dir), "--no-catalog", "--anchor", ref_label,
            "--save-json", str(fit_json)]


def estimate_min(games: int, games_per_h: float) -> float:
    """The expected wall of a match: its games at the pool's rate, and 5
    minutes for the start and the last games' tail."""
    return games * 60.0 / games_per_h + 5.0


@dataclass
class Tally:
    results: int
    wins: int
    losses: int

    @property
    def decisive(self) -> int:
        return self.wins + self.losses


def tally(games_dir: Path) -> Tally:
    """The result files in games_dir and the candidate's (side A's) decisive
    results; an unreadable file counts as no result."""
    files = list(games_dir.glob("game_*.json")) if games_dir.is_dir() else []
    wins = losses = 0
    for p in files:
        try:
            out = json.loads(p.read_text(encoding="utf-8")).get("outcome_a")
        except Exception:  # noqa: BLE001 -- an unreadable result counts as no result
            continue
        wins += out == "win"
        losses += out == "loss"
    return Tally(len(files), wins, losses)


def check_config(out: Path, config: dict) -> None:
    """The match's configuration in out/match.json: written on the first
    start, and a later start with another one is refused (run_elo_batch
    would add games of other seeds or players to the directory)."""
    path = out / "match.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))["config"]
        changed = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        if changed:
            raise SystemExit(f"MATCH_CHANGED: {out} holds a match of another configuration "
                             f"(field: (recorded, now)) {changed}. Use a fresh --out.")
        return
    if tally(out / "games").results:
        raise SystemExit(f"{out / 'games'} holds games but {out} has no match.json: not this script's "
                         f"match. Use a fresh --out.")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"config": config, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def read_fit(fit_json: Path, cand: str, ref: str) -> Tuple[Optional[float], Optional[float]]:
    """The candidate's PURE Elo and standard error, the reference at 0."""
    fit = json.loads(fit_json.read_text(encoding="utf-8"))
    pure = fit["tables"][PURE]
    if pure.get(ref, {}).get("elo") not in (0.0, None):
        raise SystemExit(f"{fit_json}: the reference {ref} is not the anchor: {pure.get(ref)}")
    c = pure.get(cand) or {}
    return c.get("elo"), c.get("se")


def procedures(games_dir: Path) -> List[str]:
    """The (procedure_a, procedure_b) pairs the games record."""
    seen = set()
    for p in games_dir.glob("game_*.json"):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- unreadable: no procedure to read
            continue
        seen.add(f"{g.get('procedure_a')} against {g.get('procedure_b')}")
    return sorted(seen)


class DisplayReset(Exception):
    """A display-driver reset was logged during the match."""


class Match:
    def __init__(self, args, cand: Candidate, ref: Reference):
        self.a, self.cand, self.ref = args, cand, ref
        self.out: Path = args.out
        self.games_dir = self.out / "games"
        self.decode = DECODE_T1 if getattr(args, "t1", False) else DECODE
        self.gpu_yield = (GpuYield(threshold=args.yield_gpu, quiet_minutes=args.yield_quiet_min)
                          if getattr(args, "yield_gpu", 0) > 0 else None)

    def note(self, line: str) -> None:
        with open(self.out / "matches.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def attempt(self, k: int) -> int:
        """One run of run_elo_batch, resumed in games/; its exit status.
        Raises DisplayReset when a reset is logged while it runs."""
        est = estimate_min(self.a.games, self.a.games_per_h)
        argv = match_argv(self.cand, self.ref, self.games_dir, self.a.games, self.a.seed_base,
                          self.a.max_extra_games, self.a.jobs, self.a.worker_games, int(3 * est / 2 + 10),
                          self.decode)
        t0 = time.time()

        def resets() -> Optional[str]:
            n = display_resets_since(t0)
            return reset_reason(n, "the match") if n else None

        try:
            rc = run_match(self.out, f"match attempt {k}", argv, "match.log", 2 * est + 15, self.a.python,
                           poll=resets, gpu_yield=self.gpu_yield, runner=run_tool)
        except ToolStopped as e:
            raise DisplayReset(str(e)) from e
        why = resets()
        if why:
            raise DisplayReset(why)
        t = tally(self.games_dir)
        self.note(f"attempt {k}: {self.a.games} decisive games wanted from seed {self.a.seed_base}, rc={rc}, "
                  f"{time.time() - t0:.0f} s, results {t.results}, decisive {t.decisive}")
        return rc

    def play(self) -> Optional[int]:
        """The match to its decisive games, played again once when short (a
        failed game or server, the time budget, the cut); the last exit
        status, None when nothing was left to play."""
        if tally(self.games_dir).decisive >= self.a.games:
            log(self.out, f"the match holds its {self.a.games} decisive games: only the fit")
            return None
        require_visible_gpu()
        if run_tool(self.out, "reference", ["tools/reference_player.py", "--ensure"], "match.log", 20,
                    self.a.tools_python) != 0:
            raise SystemExit("REFERENCE_MISSING: tools/reference_player.py --ensure failed (match.log)")
        rc = self.attempt(1)
        if tally(self.games_dir).decisive < self.a.games and rc in (1, 3, 124):
            rc = self.attempt(2)
        return rc

    def fit(self) -> dict:
        """The fit of the games on disk, and the candidate's line of it."""
        t = tally(self.games_dir)
        fit_json = self.out / "fit.json"
        fit_json.unlink(missing_ok=True)
        result = {"candidate": self.cand.label, "reference": self.ref.label, "elo": None, "se": None,
                  "games_wanted": self.a.games, "decisive": t.decisive, "wins": t.wins, "losses": t.losses,
                  "no_result": t.results - t.decisive, "complete": t.decisive >= self.a.games,
                  "procedures": procedures(self.games_dir), "seed_base": self.a.seed_base,
                  "memory_a": self.cand.plays, "memory_b": self.ref.memory,
                  "end_turn_offset_a": self.cand.end_turn_offset, "end_turn_offset_b": self.ref.end_turn_offset,
                  "decode_differs": self.cand.diffs, "decode": self.decode, "infer_fp16": True,
                  "checkpoint": self.cand.spec, "checkpoint_sha256": self.cand.sha256,
                  "fit": None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if t.decisive:
            rc = run_tool(self.out, "fit", fit_argv(self.games_dir, self.ref.label, fit_json), "fit.log", 15,
                          self.a.tools_python)
            if rc != 0 or not fit_json.exists():
                raise SystemExit(f"FIT_FAILED: tools/elo_collect.py rc={rc} (fit.log)")
            result["elo"], result["se"] = read_fit(fit_json, self.cand.label, self.ref.label)
            result["fit"] = fit_json.name
        tmp = self.out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        os.replace(tmp, self.out / "result.json")
        return result


def summary(r: dict) -> str:
    """The match's line: the candidate's PURE Elo against the reference."""
    head = (f"{r['candidate']}: no decisive game" if r["elo"] is None else
            f"{r['candidate']}: {r['elo']:+.1f} +- {r['se']:.1f} Elo against {r['reference']} = 0 (PURE)")
    tail = (f"over {r['decisive']} decisive games ({r['wins']}-{r['losses']}), {r['no_result']} no-result "
            f"excluded; {', '.join(r['procedures']) or 'no procedure recorded'}")
    if not r["complete"]:
        tail += f"; SHORT: {r['decisive']} of {r['games_wanted']} decisive games, not the match asked for"
    return f"{head} {tail}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="the candidate's label (side A)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="the candidate's checkpoint")
    ap.add_argument("--out", type=Path, required=True, help="the match's directory (resumed when it exists)")
    ap.add_argument("--games", type=int, default=GAMES, help="decisive games")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--max-extra-games", type=int, default=MAX_EXTRA,
                    help="replacements for capped games (the box's 800-game matches: 1,500)")
    ap.add_argument("--memory", type=int, default=None,
                    help="the candidate's memory slots (default: the reference's)")
    ap.add_argument("--end-turn-offset", type=float, default=None,
                    help="the candidate's end_turn logit offset (default: the reference's)")
    ap.add_argument("--allow-decode-change", action="store_true",
                    help="measure a candidate whose decode or memory differs from the reference's")
    ap.add_argument("--t1", action="store_true",
                    help="both sides sample at temperature 1 with no end_turn offset (raw:t1), the "
                         "reference relabelled <label>t1")
    ap.add_argument("--jobs", type=int, default=JOBS, help="games in flight (the pilots: 96 kept the GPU busy)")
    ap.add_argument("--worker-games", type=int, default=WORKER_GAMES)
    ap.add_argument("--games-per-h", type=float, default=GAMES_PER_H,
                    help="the pool's rate at argmax, for the time budget and the cut")
    ap.add_argument("--python", default=sys.executable, help="the interpreter of the match (the ROCm venv)")
    ap.add_argument("--tools-python", default=None,
                    help="the interpreter of the CPU tools (default: the repository's .venv, whose CPU torch "
                         "is faster than the ROCm build's)")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--yield-gpu", type=float, default=0.0, metavar="PCT",
                    help="step aside while another process uses at least PCT%% of a GPU engine (the "
                         "machine's user's programs; scripts/gpu_yield.py), and resume after "
                         "--yield-quiet-min quiet minutes; 0 (the default) never yields")
    ap.add_argument("--yield-quiet-min", type=float, default=10.0)
    ap.add_argument("--keep-awake", action="store_true",
                    help="ask Windows not to sleep while the match runs; the machine's user decides when a "
                         "run may hold the machine up")
    args = ap.parse_args(argv)
    if args.games < 1 or args.max_extra_games < 0 or args.jobs < 1 or args.worker_games < 1:
        ap.error("--games, --jobs and --worker-games must be positive, --max-extra-games not negative")
    if args.seed_base < 0 or args.seed_base + args.games > REPL_SEED_OFFSET:
        ap.error(f"the seeds {args.seed_base}-{args.seed_base + args.games - 1} must lie in 0-"
                 f"{REPL_SEED_OFFSET - 1}: a replacement adds multiples of {REPL_SEED_OFFSET}")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.tools_python is None:
        args.tools_python = default_tools_python(REPO)
    if not args.checkpoint.is_file():
        raise SystemExit(f"--checkpoint {args.checkpoint}: no such file")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "STOP").exists():
        why = (args.out / "STOP").read_text(encoding="utf-8").strip()
        log(args.out, f"STOP found ({why}): not starting; delete it to resume")
        return 2
    if not args.normal_priority:
        below_normal_priority()
    if args.keep_awake:
        keep_awake()
    ref = parse_reference(reference_flags("b", args.tools_python))
    if args.t1:
        ref = sampled(ref)
    if args.label == ref.label:
        raise SystemExit(f"--label {args.label} is the reference's label; give the candidate its own")
    spec = args.checkpoint.resolve()
    memory, plays, eo, diffs = candidate_decode(ref, memory_slots_of(spec, args.tools_python), args.memory,
                                                args.end_turn_offset, args.allow_decode_change)
    cand = Candidate(args.label, str(spec), file_sha256(spec), memory, plays, eo, diffs)
    check_config(args.out, {"label_a": cand.label, "checkpoint_sha256_a": cand.sha256,
                            "memory_a": memory, "end_turn_offset_a": eo, "reference": ref.flags,
                            "games": args.games, "seed_base": args.seed_base,
                            "max_extra_games": args.max_extra_games,
                            "decode": DECODE_T1 if args.t1 else DECODE})
    log(args.out, f"match local: {cand.label} ({spec}) against {ref.label}, {args.games} decisive games from "
                  f"seed {args.seed_base}, jobs {args.jobs}x{args.worker_games}, python {args.python}, "
                  f"tools {args.tools_python}" + (f"; decode differs: {diffs}" if diffs else ""))
    m = Match(args, cand, ref)
    try:
        rc = m.play()
    except DisplayReset as e:
        (args.out / "STOP").write_text(f"{e}\n", encoding="utf-8")
        m.note(f"STOPPED: {e}")
        log(args.out, f"STOPPED: {e}; the STOP file is written, delete it to resume")
        return 5
    result = m.fit()
    line = summary(result)
    log(args.out, line)
    if rc not in (None, 0) and not result["complete"]:
        log(args.out, f"run_elo_batch ended rc={rc} (match.log; failed games in games/failed_*.json)")
    return 0 if result["complete"] and result["elo"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
