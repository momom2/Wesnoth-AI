#!/usr/bin/env python3
"""Training data for the turn-ranking value function
(docs/turn_value_prereg_20260925.md).

Positions are the side-turn starts of recorded games (tools/game_record.py
logs, one game per file): a player side's position right after its
init_side, from turn --min-turn on, up to --per-game of them per game
drawn with a seed of the game's own. At each, tools/turn_gap.py proposes
the candidate turns (the reference's own, --alternatives sampled at
--temperature, --continue-edits) and plays each out --playouts times.
A position's record is turn_gap's record plus where it came from (the
game file, the command index of its init_side, the game's split), so
tools/turn_value.py can rebuild the position and each candidate's
pre-end_turn state.

Splits are by game, from a hash of the file name alone (so every run on
these games splits them alike): --proxy-games are the held-out proxy
set, --stop-games the fit's early-stopping set, the rest the fit set.

One task per game: a worker walks the record once and measures the
game's positions in order. Each finished game is appended to --out as
one gzip member (tools/game_record.py's log format: a header line, then
per game its position records and a `game_done` line), so a cut run
keeps every finished game whole, and a rerun with the same --out and
configuration skips the games already there.

Usage (box):
  python tools/turn_value_data.py --reference --games-dir DIR \\
      --device cuda --jobs 24 --shared-inference --out data.jsonl.gz
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import multiprocessing as mp
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import turn_gap as tg
from tools.game_record import GameRecordLog, read_records, turn_starts
from tools.raw_player import END_TURN_RULES

log = logging.getLogger("turn_value_data")

# A position's index is (its game's ordinal + 1) times this plus the
# command index of its init_side: unique, above the 0-based indices a
# manifest run gives its positions, and the playout salts derive from it.
INDEX_STRIDE = 100_000


@dataclass(frozen=True)
class GameTask:
    path: str
    name: str          # the file name: the game's identity in the splits and the log
    ordinal: int
    split: str         # "fit" | "stop" | "proxy"


@dataclass(frozen=True)
class Selection:
    per_game: int = 15
    min_turn: int = 2
    seed: int = 25


def game_files(games_dir: Path) -> List[Path]:
    files = sorted(Path(games_dir).glob("*.game.jsonl.gz"))
    if not files:
        raise SystemExit(f"no *.game.jsonl.gz under {games_dir}")
    return files


def assign_splits(names: Sequence[str], proxy: int, stop: int) -> Dict[str, str]:
    """Split by a hash of the name alone: the first `proxy` names in hash
    order are the proxy set, the next `stop` the early-stopping set."""
    if proxy + stop > len(names):
        raise ValueError(f"{proxy} proxy + {stop} stop games of {len(names)}")
    ranked = sorted(names, key=lambda n: hashlib.sha256(f"turn_value:{n}".encode()).hexdigest())
    return {n: ("proxy" if i < proxy else "stop" if i < proxy + stop else "fit")
            for i, n in enumerate(ranked)}


def tasks_for(games_dir: Path, proxy: int, stop: int) -> List[GameTask]:
    files = game_files(games_dir)
    splits = assign_splits([f.name for f in files], proxy, stop)
    return [GameTask(path=str(f), name=f.name, ordinal=i, split=splits[f.name])
            for i, f in enumerate(files)]


def pick_turn_starts(commands: Sequence[list], name: str,
                     sel: Selection) -> List[Tuple[int, int, int]]:
    """(command index, turn, side) of the turn starts to measure: the
    player sides' init_side commands from turn sel.min_turn on, the
    record's last command excluded (no turn follows it), up to
    sel.per_game drawn with a seed of the game's own. The turn counts
    side 1's init_sides, as a game played from its scenario start does;
    the walk checks it against the state."""
    starts = []
    turn = 0
    for k, cmd in enumerate(commands):
        if cmd[0] != "init_side":
            continue
        side = int(cmd[1])
        turn += side == 1
        if side in (1, 2) and turn >= sel.min_turn and k + 1 < len(commands):
            starts.append((k, turn, side))
    digest = hashlib.sha256(f"turn_value:{sel.seed}:{name}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if len(starts) > sel.per_game:
        starts = sorted(rng.sample(starts, sel.per_game))
    return starts


def measure_game(policy, task: GameTask, cfg: tg.GapConfig, sel: Selection) -> List[Dict]:
    """The records of one game's selected turn starts, in game order."""
    recs = list(read_records(Path(task.path)))
    if len(recs) != 1:
        raise RuntimeError(f"{task.name}: {len(recs)} records, one expected")
    rec = recs[0]
    picks = {k: (turn, side) for k, turn, side in pick_turn_starts(rec["commands"], task.name, sel)}
    out: List[Dict] = []
    if not picks:
        return out
    for k, gs in turn_starts(rec):
        if k not in picks:
            continue
        turn, side = picks.pop(k)
        gi = gs.global_info
        if (gi.turn_number, gi.current_side) != (turn, side):
            raise RuntimeError(f"{task.name}: command {k} starts turn {gi.turn_number} side "
                               f"{gi.current_side}, the selection expected turn {turn} side {side}")
        position = tg.BoundaryPosition(
            index=(task.ordinal + 1) * INDEX_STRIDE + k, gs=copy.deepcopy(gs),
            scenario_id=rec["scenario_id"],
            meta={"source": "game_record", "game": task.name, "command_index": k,
                  "split": task.split, "game_winner": int(rec["winner"])})
        out.append(tg.measure_position(policy, position, cfg))
        if not picks:
            break
    if picks:
        raise RuntimeError(f"{task.name}: turn starts {sorted(picks)} never reached")
    return out


@dataclass
class GameResult:
    name: str
    records: Optional[List[Dict]]      # None when the game failed
    secs: float
    error: Optional[str] = None


def measure_game_result(policy, task: GameTask, cfg: tg.GapConfig,
                        sel: Selection) -> GameResult:
    """measure_game, a failure reported instead of raised: one game that
    does not rebuild must not end a run of hundreds."""
    t0 = time.perf_counter()
    try:
        records = measure_game(policy, task, cfg, sel)
    except Exception as exc:                          # noqa: BLE001 - reported per game
        log.exception("game %s failed", task.name)
        return GameResult(task.name, None, time.perf_counter() - t0,
                          f"{type(exc).__name__}: {exc}")
    return GameResult(task.name, records, time.perf_counter() - t0)


def _game_worker(args) -> GameResult:
    task, cfg, sel = args
    if tg._WORKER_INIT_ERROR is not None:
        return GameResult(task.name, None, 0.0,
                          f"worker could not load the policy: {tg._WORKER_INIT_ERROR}")
    return measure_game_result(tg._WORKER_POLICY, task, cfg, sel)


def generate(tasks: Sequence[GameTask], cfg: tg.GapConfig, sel: Selection, on_game, *,
             policy=None, spec: Optional[tg.PolicySpec] = None, jobs: int = 1,
             log_level: str = "INFO") -> None:
    """Measure every game; `on_game(task, result)` receives each one as
    it finishes (in completion order when jobs > 1)."""
    by_name = {t.name: t for t in tasks}
    if jobs <= 1:
        if policy is None:
            policy = tg.load_reference_policy(spec)
        for task in tasks:
            on_game(task, measure_game_result(policy, task, cfg, sel))
        return
    if spec is None:
        raise ValueError("jobs > 1 needs a PolicySpec: each worker loads the policy")
    ctx = mp.get_context("spawn")
    with ctx.Pool(jobs, initializer=tg._worker_init, initargs=(spec, log_level)) as pool:
        for result in pool.imap_unordered(_game_worker, [(t, cfg, sel) for t in tasks]):
            on_game(by_name[result.name], result)


# ---------------------------------------------------------------------
# The output log
# ---------------------------------------------------------------------

def read_log(path: Path) -> Tuple[Optional[Dict], List[Dict], Set[str]]:
    """(header, position records, names of the finished games)."""
    header, positions, done = None, [], set()
    for obj in read_records(Path(path)):
        if "header" in obj:
            header = obj["header"]
        elif "game_done" in obj:
            done.add(obj["game_done"])
        else:
            positions.append(obj)
    return header, positions, done


def _same_run(old: Dict, new: Dict) -> bool:
    keys = ("config", "selection", "splits", "games")
    return all(old.get(k) == new.get(k) for k in keys)


def open_log(path: Path, header: Dict) -> Set[str]:
    """Start the log, or continue one written by the same run; returns
    the games already finished there."""
    if not Path(path).exists():
        GameRecordLog(path).write({"header": header})
        return set()
    old, _, done = read_log(path)
    if old is None or not _same_run(old, header):
        raise SystemExit(f"{path} holds another run's data (its header differs in "
                         f"config, selection, splits or games); use another --out")
    log.info("%s: continuing, %d games already finished", path, len(done))
    return done


def playout_count(record: Dict) -> int:
    return sum(tg.playouts_run(c) for c in [record["base"]] + record["alternatives"])


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-dir", type=Path, required=True,
                    help="Directory of recorded games (*.game.jsonl.gz, one game each).")
    ap.add_argument("--out", type=Path, required=True, help="The log (.jsonl.gz).")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--reference", action="store_true",
                    help="The reference player of configs/reference_player.json: its "
                         "checkpoint and decode.")
    ap.add_argument("--raw-end-turn", choices=END_TURN_RULES, default="joint")
    ap.add_argument("--raw-end-turn-offset", type=float, default=0.0)
    ap.add_argument("--per-game", type=int, default=Selection.per_game)
    ap.add_argument("--min-turn", type=int, default=Selection.min_turn)
    ap.add_argument("--proxy-games", type=int, default=100)
    ap.add_argument("--stop-games", type=int, default=70)
    ap.add_argument("--limit-games", type=int, default=0,
                    help="Measure only the first N games in file order (a smoke run).")
    ap.add_argument("--alternatives", type=int, default=2)
    ap.add_argument("--continue-edits", type=int, default=1)
    ap.add_argument("--playouts", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--playout-temperature", type=float, default=0.5)
    ap.add_argument("--cap-turns", type=int, default=30)
    ap.add_argument("--seed", type=int, default=Selection.seed)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--shared-inference", action="store_true")
    ap.add_argument("--inference-window-ms", type=float, default=1.5)
    ap.add_argument("--log-level", default="INFO")
    return ap


def main(argv) -> int:
    args = _parser().parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    reference = tg._apply_reference(args)
    if args.checkpoint is None:
        raise SystemExit("--checkpoint or --reference is required")
    import torch
    torch.set_num_threads(2)
    from wesnoth_ai import __version__

    cfg = tg.GapConfig(k_alternatives=args.alternatives, continue_edits=args.continue_edits,
                       playouts=args.playouts, temperature=args.temperature,
                       cap_turns=args.cap_turns, seed=args.seed,
                       playout_temperature=args.playout_temperature,
                       end_turn_rule=args.raw_end_turn,
                       end_turn_offset=args.raw_end_turn_offset)
    sel = Selection(per_game=args.per_game, min_turn=args.min_turn, seed=args.seed)
    tasks = tasks_for(args.games_dir, args.proxy_games, args.stop_games)
    header = {"tool": "turn_value_data", "config": asdict(cfg), "selection": asdict(sel),
              "splits": {"proxy": args.proxy_games, "stop": args.stop_games},
              "games": len(tasks), "games_dir": str(args.games_dir),
              "reference": reference, "checkpoint": str(args.checkpoint),
              "procedures": {"base": tg.procedure_tag(cfg, 0.0),
                             "alternatives": tg.procedure_tag(cfg, cfg.temperature),
                             "playouts": tg.procedure_tag(cfg, cfg.playout_temperature)},
              "code_version": __version__, "torch": torch.__version__,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    done = open_log(args.out, header)
    todo = [t for t in tasks if t.name not in done]
    if args.limit_games:
        todo = todo[:args.limit_games]
    spec = tg._resolve_inference(args)
    server = None
    if args.shared_inference:
        server, spec = tg.launch_shared_inference(spec, args.out.parent, args.jobs,
                                                  args.inference_window_ms, tag="turn_value")
    log.info("%d games to measure (%d finished before), per game %d from turn %d, K=%d "
             "continue=%d P=%d, playouts %s cap %d, jobs %d",
             len(todo), len(done), sel.per_game, sel.min_turn, cfg.k_alternatives,
             cfg.continue_edits, cfg.playouts, header["procedures"]["playouts"],
             cfg.cap_turns, args.jobs)
    t0 = time.time()
    totals = {"games": 0, "positions": 0, "playouts": 0}
    failed: List[Tuple[str, str]] = []

    def on_game(task: GameTask, result: GameResult) -> None:
        if result.records is None:
            failed.append((task.name, result.error))
            log.error("game %s failed (not written; a rerun retries it): %s",
                      task.name, result.error)
            return
        GameRecordLog(args.out).write_many(
            result.records + [{"game_done": task.name, "split": task.split,
                               "positions": len(result.records),
                               "secs": round(result.secs, 1)}])
        totals["games"] += 1
        totals["positions"] += len(result.records)
        totals["playouts"] += sum(playout_count(r) for r in result.records)
        wall = time.time() - t0
        log.info("game %s (%s): %d positions in %.0f s; %d/%d games, %d positions, "
                 "%d playouts, %.2f playouts/s", task.name, task.split, len(result.records),
                 result.secs, totals["games"], len(todo), totals["positions"],
                 totals["playouts"], totals["playouts"] / max(wall, 1e-9))

    try:
        generate(todo, cfg, sel, on_game, spec=spec, jobs=args.jobs, log_level=args.log_level)
    finally:
        if server is not None:
            log.info("inference server stats: %s", server.shutdown())
    log.info("done: %s in %.0f s; %d games failed%s", totals, time.time() - t0, len(failed),
             "".join(f"\n  {n}: {e}" for n, e in failed))
    return 0 if totals["games"] or not todo else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
