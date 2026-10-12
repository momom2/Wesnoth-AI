#!/usr/bin/env python3
"""Successive halving over the raw player's decode knobs, against the
reference (docs/decode_knobs_prereg_20261012.md).

Twenty-four knob vectors (`KNOB_VECTOR_KEYS`: the end_turn offset, the
attack and recruit offsets, the attack time-of-day slope): today's decode,
the reference's own, and 23 drawn by a Latin hypercube over the ranges
with a fixed seed, rounded to two decimals so each procedure tag names its
vector exactly. Each vector plays the reference's checkpoint at its memory
under that decode (side A, labelled kv00..kv23) against the reference
(side B, tools/reference_player.py), PURE, through tools/run_elo_batch.py,
one games directory per vector holding all its stages:

    stage 1: the 24 vectors x 100 decisive games, seed base 200000 + 1000 i
             (vector i); the 8 with the most wins go on
    stage 2: those 8 x 300 more, seed base 230000 + 1000 j (j their
             stage-1 rank); the 2 with the most wins over all their games
    stage 3: those 2 x 800 more, seed base 240000 + 1000 j; the one with
             the most wins over all its games is the winner
    confirmation: the winner x 800 decisive games, seed base 250000, in a
             directory of its own, fitted by tools/elo_collect.py

A base S plays S to S + games - 1 and replaces a capped game of slot i
with S + i + k x 1,000,000 (run_elo_batch), so no two stages share a seed.
Ties in wins go to the fewer losses, then to the lower vector index. The
null arm: today's decode in stage 1 must lie within two standard errors of
50% (at p = 0.5), or stage 2 refuses to start (`--ignore-null-arm`
overrides it).

Every result is written as produced, to ROOT/search.jsonl: the vectors
before any game, one row per vector per stage, each stage's ranking with
its survivors (and the null-arm check in stage 1), the confirmation and its
fit; ROOT/winner_decode.json holds the winner's knob vector in config form
(paste it into configs/reference_player.json's decode block, or play it
with `tools/reference_player.py --flags a --decode`). Counts are read from
the games directories every time, so a run cut anywhere resumes from what
is on disk:

    python tools/knob_search.py --root OUT --plan            # vectors and commands, no game
    python tools/knob_search.py --root OUT --stage 1         # stage 1, resumed where it stopped
    python tools/knob_search.py --root OUT --stage 1 --one   # ...one match, exit 10 while more remain
    python tools/knob_search.py --root OUT --stage 2
    python tools/knob_search.py --root OUT --stage 3
    python tools/knob_search.py --root OUT --confirm

Exit status: 0 done, 10 one match done and more remain (--one), 1 a match
failed, 5 the null arm is outside two standard errors, 2 a refusal.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import reference_player  # noqa: E402
from tools.eval_procedure import KNOB_VECTOR_KEYS, decode_flags, decode_procedure, knob_vector  # noqa: E402
from wesnoth_ai.paths import REPO_ROOT  # noqa: E402

# The pre-registration's ranges, in KNOB_VECTOR_KEYS order.
RANGES = {
    "raw_end_turn_offset": (-3.5, 0.0),
    "raw_attack_offset": (-1.0, 1.5),
    "raw_recruit_offset": (-1.0, 1.0),
    "raw_attack_tod_slope": (0.0, 4.0),
}
N_VECTORS = 24
LHS_SEED = 20261012
DECIMALS = 2


@dataclass(frozen=True)
class Stage:
    number: int
    games: int          # decisive games each vector adds in this stage
    keep: int           # vectors that go on
    seed_base: int      # the first slot's base; the next slots 1000 apart


STAGES = (Stage(1, 100, 8, 200_000), Stage(2, 300, 2, 230_000), Stage(3, 800, 1, 240_000))
CONFIRM_GAMES = 800
CONFIRM_SEED_BASE = 250_000
SEED_STRIDE = 1000
REPL_SEED_OFFSET = 1_000_000          # run_elo_batch.REPL_SEED_OFFSET
GAIN_ELO = 25.0                       # the prereg's "Gain" bar
PURE = "PURE (decisive only, primary)"

EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_NULL_ARM, EXIT_MORE = 0, 1, 2, 5, 10


def latin_hypercube(n: int, ranges: Dict[str, tuple], seed: int,
                    decimals: Optional[int] = DECIMALS) -> List[Dict[str, float]]:
    """`n` points, one per stratum of every range (a random permutation of
    the n strata per key, a uniform draw inside each), rounded to
    `decimals` (None: unrounded). Python's `random.Random`, whose stream is
    fixed across versions."""
    rng = random.Random(seed)
    columns = {}
    for key, (lo, hi) in ranges.items():
        strata = list(range(n))
        rng.shuffle(strata)
        draws = [lo + (hi - lo) * (s + rng.random()) / n for s in strata]
        columns[key] = draws if decimals is None else [round(x, decimals) + 0.0 for x in draws]
    return [{key: columns[key][i] for key in ranges} for i in range(n)]


def search_vectors(today: Dict[str, float]) -> List[Dict[str, float]]:
    """Today's decode, then N_VECTORS - 1 points of the hypercube."""
    return [dict(today)] + latin_hypercube(N_VECTORS - 1, RANGES, LHS_SEED)


def label_of(index: int) -> str:
    return f"kv{index:02d}"


def seed_base_of(stage: Stage, slot: int) -> int:
    return stage.seed_base + SEED_STRIDE * slot


# ---------------------------------------------------------------- records
class Log:
    """ROOT/search.jsonl: one JSON row per event, appended and flushed."""

    def __init__(self, path: Path):
        self.path = path

    def rows(self, event: str) -> List[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        return [r for r in rows if r.get("event") == event]

    def append(self, row: dict) -> None:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        print(json.dumps(row), flush=True)


_GAMES: Dict[Path, Tuple[Tuple[int, int], Optional[dict]]] = {}


def _game(path: Path) -> Optional[dict]:
    """A result file's (label_a, seed, outcome_a), parsed once per version."""
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _GAMES.get(path)
    if hit is None or hit[0] != key:
        try:
            g = json.loads(path.read_text(encoding="utf-8"))
            val = {k: g.get(k) for k in ("label_a", "seed", "outcome_a")}
        except (OSError, ValueError):
            val = None
        hit = (key, val)
        _GAMES[path] = hit
    return hit[1]


def count_games(outdir: Path, label: str, seed_base: int, games: int) -> Dict[str, int]:
    """Wins, losses and no-results of `label` (side A) among the games of
    the stage whose base is `seed_base` (seeds S..S+games-1 and their
    replacements), read from the result files."""
    out = {"wins": 0, "losses": 0, "no_result": 0}
    for path in (outdir.glob("game_*.json") if outdir.is_dir() else ()):
        g = _game(path)
        if g is None or g["label_a"] != label or g["seed"] is None:
            continue
        if not 0 <= int(g["seed"]) % REPL_SEED_OFFSET - seed_base < games:
            continue
        out[{"win": "wins", "loss": "losses"}.get(g["outcome_a"], "no_result")] += 1
    return {**out, "decisive": out["wins"] + out["losses"]}


def null_arm_check(wins: int, decisive: int) -> dict:
    """Today's decode against the reference: within two standard errors of
    50% at p = 0.5."""
    if decisive <= 0:
        return {"wins": wins, "decisive": decisive, "within_two_se": False}
    se = math.sqrt(0.25 / decisive)
    rate = wins / decisive
    return {"wins": wins, "decisive": decisive, "win_rate": rate, "se": se,
            "within_two_se": abs(rate - 0.5) <= 2.0 * se}


# ---------------------------------------------------------------- matches
@dataclass
class MatchOptions:
    device: str = "cuda"
    jobs: int = 20
    time_budget_min: float = 60.0
    shared_inference: bool = True
    extra_factor: float = 2.0       # replacement slots per decisive game wanted
    attempts: int = 3               # run_elo_batch chunks per match (exit 1 or 3: once more)


def batch_argv(ref: dict, vector: Dict[str, float], label: str, outdir: Path, games: int,
               seed_base: int, opts: MatchOptions) -> List[str]:
    """The run_elo_batch command of one vector's stage: side A the
    reference's checkpoint at its memory under the vector's decode, side B
    the reference."""
    decode = reference_player.with_decode(ref, vector)["decode"]
    side_a = ["--label-a", label, "--spec-a", ref["checkpoint_local"]]
    if ref.get("memory_slots"):
        side_a += ["--memory-a", str(int(ref["memory_slots"]))]
    side_a += decode_flags("a", decode)
    temp = str(float(decode["raw_temperature"]))
    argv = [sys.executable, "-u", str(REPO_ROOT / "tools" / "run_elo_batch.py"), *side_a,
            *reference_player.batch_flags("b", ref),
            "--outdir", str(outdir), "--games", str(games),
            "--max-extra-games", str(int(games * opts.extra_factor)), "--seed-base", str(seed_base),
            "--mcts-sims", "0", "--raw-temperature-a", temp, "--raw-temperature-b", temp,
            "--device", opts.device, "--jobs", str(opts.jobs),
            "--time-budget-min", str(opts.time_budget_min)]
    if opts.shared_inference:
        argv += ["--persistent-workers", "--shared-inference", "--no-infer-compile"]
    return argv


def subprocess_runner(argv: List[str], log_path: Path) -> int:
    """Run one run_elo_batch chunk, its output appended to `log_path`."""
    with log_path.open("ab") as log:
        log.write(f"\n$ {' '.join(argv)}\n".encode("utf-8"))
        log.flush()
        return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode


def elo_fit(games_dir: Path, out: Path, anchor: str) -> int:
    """tools/elo_collect.py on the confirmation, anchored on the reference."""
    return subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "elo_collect.py"), str(games_dir),
                           "--no-catalog", "--anchor", anchor, "--save-json", str(out)],
                          cwd=REPO_ROOT).returncode


class Search:
    """The search's state, read from ROOT. `runner(argv, log_path) -> rc`
    plays a chunk of games with run_elo_batch's exit codes; `fit(games_dir,
    out, anchor) -> rc` fits the confirmation. `stages` and
    `confirm_games` are the pre-registration's; the tests play fewer games
    on the same seed bases."""

    def __init__(self, root: Path, ref: Optional[dict] = None, opts: Optional[MatchOptions] = None,
                 runner: Callable[[List[str], Path], int] = subprocess_runner,
                 fit: Callable[[Path, Path, str], int] = elo_fit, ignore_null_arm: bool = False,
                 stages: Tuple[Stage, ...] = STAGES, confirm_games: int = CONFIRM_GAMES):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ref = ref or reference_player.load()
        self.opts = opts or MatchOptions()
        self.runner, self.fit = runner, fit
        self.ignore_null_arm = ignore_null_arm
        self.stages, self.confirm_games = tuple(stages), int(confirm_games)
        self.log = Log(self.root / "search.jsonl")
        self.vectors = search_vectors(knob_vector(self.ref["decode"]))
        self._survivors: Dict[int, List[int]] = {}

    def games_dir(self, index: int) -> Path:
        return self.root / f"games_{label_of(index)}"

    def decode_of(self, index: int) -> dict:
        return reference_player.with_decode(self.ref, self.vectors[index])["decode"]

    # ---- the vectors, recorded before any game
    def record_vectors(self) -> None:
        """The vectors row: written once, before any game; a later entry
        must find the same vectors and the same reference."""
        rows = [{"index": i, "label": label_of(i), "procedure": decode_procedure(self.decode_of(i)),
                 **vec} for i, vec in enumerate(self.vectors)]
        want = {"event": "vectors", "lhs_seed": LHS_SEED, "ranges": RANGES, "decimals": DECIMALS,
                "stages": [[s.number, s.games, s.keep, s.seed_base] for s in self.stages],
                "confirmation": [self.confirm_games, CONFIRM_SEED_BASE],
                "reference": {k: self.ref.get(k) for k in ("label", "checkpoint_sha256",
                                                           "memory_slots", "procedure_tag")},
                "vectors": rows}
        have = self.log.rows("vectors")
        if not have:
            self.log.append(want)
        elif json.loads(json.dumps({k: have[0].get(k) for k in want})) != json.loads(json.dumps(want)):
            raise SystemExit(f"{self.log.path} records other vectors or another reference than "
                             f"this search draws: use a fresh root, or the code that began it")

    # ---- one vector's stage
    def entrants(self, number: int) -> List[int]:
        """Stage `number`'s vectors, in slot order: all of them in stage 1,
        the previous stage's survivors, best first, after."""
        return list(range(N_VECTORS)) if number == 1 else self.survivors(number - 1)

    def stage_counts(self, index: int, stage: Stage) -> Dict[str, int]:
        slot = self.entrants(stage.number).index(index)
        return count_games(self.games_dir(index), label_of(index), seed_base_of(stage, slot),
                           stage.games)

    def cumulative(self, index: int, upto: int) -> Dict[str, int]:
        """Wins, losses, decisive games and no-results of vector `index`
        over stages 1..upto (the stages it entered)."""
        total = {"wins": 0, "losses": 0, "no_result": 0, "decisive": 0}
        for stage in self.stages[:upto]:
            if index in self.entrants(stage.number):
                for k, v in self.stage_counts(index, stage).items():
                    total[k] += v
        return total

    def stage_done(self, index: int, stage: Stage) -> bool:
        """The vector's stage holds its decisive games, or its last attempt
        spent the replacement guard (run_elo_batch exit 4) on exactly the
        games on disk now."""
        counts = self.stage_counts(index, stage)
        if counts["decisive"] >= stage.games:
            return True
        rows = [r for r in self.log.rows("match") if (r["stage"], r["vector"]) == (stage.number, index)]
        return bool(rows and rows[-1]["rc"] == 4
                    and (rows[-1]["decisive"], rows[-1]["no_result"])
                    == (counts["decisive"], counts["no_result"]))

    def _run(self, argv: List[str], log_name: str) -> Optional[int]:
        rc = None
        for _ in range(self.opts.attempts):
            rc = self.runner(argv, self.root / log_name)
            if rc not in (1, 3):            # 1: games failed; 3: the time budget
                break
        return rc

    def play(self, index: int, stage: Stage) -> Optional[int]:
        """Play vector `index`'s stage to its decisive games (resumed in its
        directory) and record its row; run_elo_batch's last exit code."""
        label = label_of(index)
        base = seed_base_of(stage, self.entrants(stage.number).index(index))
        t0 = time.time()
        rc = self._run(batch_argv(self.ref, self.vectors[index], label, self.games_dir(index),
                                  stage.games, base, self.opts), f"{label}.log")
        self.log.append({"event": "match", "stage": stage.number, "vector": index, "label": label,
                         "seed_base": base, "games_wanted": stage.games,
                         **self.stage_counts(index, stage), "rc": rc,
                         "wall_s": round(time.time() - t0, 1),
                         "cumulative": self.cumulative(index, stage.number)})
        return rc

    # ---- rankings
    def ranking(self, number: int) -> List[dict]:
        """Stage `number`'s vectors by wins over all their games so far
        (ties: fewer losses, then the lower index)."""
        table = []
        for i in self.entrants(number):
            row = {"vector": i, "label": label_of(i), **self.cumulative(i, number)}
            row["win_rate"] = row["wins"] / row["decisive"] if row["decisive"] else None
            table.append(row)
        return sorted(table, key=lambda r: (-r["wins"], r["losses"], r["vector"]))

    def survivors(self, number: int) -> List[int]:
        """The vectors stage `number` sends on, best first, from its recorded
        ranking, which must agree with the games on disk."""
        if number in self._survivors:
            return self._survivors[number]
        rows = [r for r in self.log.rows("ranking") if r["stage"] == number]
        if not rows:
            raise SystemExit(f"stage {number} has no ranking yet: run --stage {number} first")
        now = [r["vector"] for r in self.ranking(number)][:self.stages[number - 1].keep]
        if now != rows[-1]["survivors"]:
            raise SystemExit(f"stage {number}'s recorded survivors {rows[-1]['survivors']} differ "
                             f"from what its games on disk rank, {now}: the directories changed")
        self._survivors[number] = now
        return now

    def record_ranking(self, number: int) -> None:
        table = self.ranking(number)
        row = {"event": "ranking", "stage": number, "ranking": table,
               "survivors": [r["vector"] for r in table][:self.stages[number - 1].keep]}
        if number == 1:
            today = self.stage_counts(0, self.stages[0])
            row["null_arm"] = null_arm_check(today["wins"], today["decisive"])
        recorded = [r for r in self.log.rows("ranking") if r["stage"] == number]
        if not recorded or recorded[-1]["survivors"] != row["survivors"]:
            self.log.append(row)
        print(f"stage {number}: " + ", ".join(f"{r['label']} {r['wins']}-{r['losses']}" for r in table)
              + f"; survivors {row['survivors']}")
        if number == 1:
            print(f"null arm (today's decode, {label_of(0)}): {row['null_arm']}")

    def null_arm_refusal(self) -> Optional[str]:
        """Why stage 2 may not start, or None."""
        rows = [r for r in self.log.rows("ranking") if r["stage"] == 1]
        if self.ignore_null_arm or not rows or rows[-1]["null_arm"]["within_two_se"]:
            return None
        return (f"the null arm is outside two standard errors of 50%: {rows[-1]['null_arm']}; "
                f"the harness is suspect before anything is read (--ignore-null-arm goes on)")

    # ---- a stage, whole or one match at a time
    def run_stage(self, number: int, one: bool = False) -> int:
        self.record_vectors()
        if number == 2 and self.null_arm_refusal():
            print(self.null_arm_refusal())
            return EXIT_NULL_ARM
        stage = self.stages[number - 1]
        todo = [i for i in self.entrants(number) if not self.stage_done(i, stage)]
        for k, i in enumerate(todo):
            rc = self.play(i, stage)
            if not self.stage_done(i, stage):
                print(f"{label_of(i)} stage {number}: run_elo_batch exit {rc} left the stage short "
                      f"({self.root / (label_of(i) + '.log')})")
                return EXIT_FAILED
            if one and k + 1 < len(todo):
                return EXIT_MORE
        self.record_ranking(number)
        return EXIT_DONE

    # ---- the confirmation
    def confirm(self) -> int:
        self.record_vectors()
        winner = self.survivors(len(self.stages))[0]
        label, decode = label_of(winner), self.decode_of(winner)
        vector = {k: decode[k] for k in KNOB_VECTOR_KEYS}
        (self.root / "winner_decode.json").write_text(json.dumps(vector, indent=1) + "\n",
                                                      encoding="utf-8")
        outdir, fit_path = self.root / "games_confirm", self.root / "confirm.fit.json"
        counts = count_games(outdir, label, CONFIRM_SEED_BASE, self.confirm_games)
        rc = 0
        if counts["decisive"] < self.confirm_games:
            rc = self._run(batch_argv(self.ref, self.vectors[winner], label, outdir, self.confirm_games,
                                      CONFIRM_SEED_BASE, self.opts), "confirm.log")
            counts = count_games(outdir, label, CONFIRM_SEED_BASE, self.confirm_games)
            if counts["decisive"] < self.confirm_games and rc != 4:
                print(f"the confirmation is short: {counts}, run_elo_batch exit {rc}")
                return EXIT_FAILED
        if self.fit(outdir, fit_path, self.ref["label"]) != 0:
            print(f"the confirmation's fit failed ({fit_path})")
            return EXIT_FAILED
        fitted = json.loads(fit_path.read_text(encoding="utf-8"))["tables"][PURE][label]
        elo, se = fitted["elo"], fitted["se"]
        reading = ("gain" if elo >= GAIN_ELO else "flat" if abs(elo) <= 2.0 * se
                   else "neither gain nor flat")
        row = {"event": "confirmation", "vector": winner, "label": label,
               "seed_base": CONFIRM_SEED_BASE, **counts, "rc": rc, "decode": vector,
               "procedure": decode_procedure(decode), "elo_over_reference": elo, "se": se,
               "reading": reading}
        done = self.log.rows("confirmation")
        if not done or {k: done[-1].get(k) for k in row} != json.loads(json.dumps(row)):
            self.log.append(row)
        return EXIT_DONE


def plan(search: Search) -> None:
    """The vectors and stage 1's first commands, without a game."""
    for i, vec in enumerate(search.vectors):
        print(f"{label_of(i)}  {decode_procedure(search.decode_of(i)):<40} "
              + "  ".join(f"{k}={v:g}" for k, v in vec.items()))
    stage = STAGES[0]
    for i in (0, 1):
        print(" ".join(batch_argv(search.ref, search.vectors[i], label_of(i), search.games_dir(i),
                                  stage.games, seed_base_of(stage, i), search.opts)))


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the search's records and games")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--plan", action="store_true", help="print the vectors and commands, no game")
    what.add_argument("--stage", type=int, choices=[s.number for s in STAGES])
    what.add_argument("--confirm", action="store_true")
    ap.add_argument("--one", action="store_true", help="with --stage: one match, exit 10 while more remain")
    ap.add_argument("--ignore-null-arm", action="store_true",
                    help="start stage 2 although the null arm is outside two standard errors")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--time-budget-min", type=float, default=60.0)
    ap.add_argument("--no-shared-inference", action="store_true",
                    help="per-process inference (a CPU check); production shares it")
    args = ap.parse_args(argv)
    opts = MatchOptions(device=args.device, jobs=args.jobs, time_budget_min=args.time_budget_min,
                        shared_inference=not args.no_shared_inference)
    search = Search(args.root, opts=opts, ignore_null_arm=args.ignore_null_arm)
    if args.plan:
        plan(search)
        return EXIT_DONE
    if args.confirm:
        return search.confirm()
    return search.run_stage(args.stage, one=args.one)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit as e:
        if isinstance(e.code, str):         # a refusal: its reason, then exit 2
            print(e.code, file=sys.stderr)
            sys.exit(EXIT_REFUSED)
        raise
