#!/usr/bin/env python3
"""Build the step-1 critics' training positions (docs/selfplay_program_20261008.md,
"Step 1"; what a position holds is wesnoth_ai/critic_data.py).

    python tools/critic_positions.py --out DIR --vocab-from training/checkpoints/parity3.pt \\
        --matches games_obs8a_vs_obs8b games_cand64_vs_ref ... \\
        --corpus replays_dataset_imitation --bench configs/bench_states.json --workers 30

Sources:
  M  the whole-game records of match directories (each an extracted
     games_<match>.tar.gz): every `*.game.jsonl.gz`, its fingerprints checked
     as it is walked. A game stopped at the turn cap (`ended_by` max_turns)
     has no label and is left out, as is one without a player's win; both
     are counted.
  H  the corpus at CORPUS_VERSION 5, the games of its manifest's training
     split. Every benchmark game the manifest holds must be on its holdout
     side, or the build stops before it starts (a leak); the benchmark's
     games the manifest lacks are counted; and no benchmark game is among
     H's games.

Each game is assigned to the 95/5 split and given its size draw
(`critic_data.split_of`) from its key: `<match>/<game label>` for M, the
corpus file for H.

Written as the build goes:
  DIR/<shard>.true.pkl, DIR/<shard>.obs.pkl   one game's positions in one
      view: {"key", "view", "positions": [...], "raws": [packed encoding]};
  DIR/manifest.jsonl   one row per game (its status, split, counts, seed);
  DIR/summary.json     the counts by source and status and the provenance,
      rewritten every PROGRESS_EVERY games; "BUILD_DONE" in the log at the end.
A game that already has a row is not built again, so a cut build continues.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai import critic_data as cd  # noqa: E402

log = logging.getLogger("critic_positions")

CORPUS_VERSION = 5
CAPPED = "max_turns"
PROGRESS_EVERY = 200


class HoldoutLeak(Exception):
    """A benchmark game sits where training could read it."""


# ---------------------------------------------------------------------
# What to build
# ---------------------------------------------------------------------

RECORD_SUFFIX = ".game.jsonl.gz"


def match_tasks(match_dirs: Sequence[Path]) -> List[Tuple]:
    """("M", game key, record path) for every game record of the match
    directories, in a stable order. The key is `<match>/<game label>`, the
    label read off the record's file name (and checked against the record)."""
    tasks = []
    for d in match_dirs:
        name = Path(d).name
        name = name[len("games_"):] if name.startswith("games_") else name
        for path in sorted(Path(d).glob("*" + RECORD_SUFFIX)):
            tasks.append(("M", f"{name}/{path.name[:-len(RECORD_SUFFIX)]}", str(path)))
    return tasks


def corpus_training_rows(manifest_rows: Sequence[Dict], bench_files: Sequence[str]) -> Tuple[List[Dict], Dict]:
    """The corpus's training rows and the benchmark check: every benchmark
    game in the manifest must be on its holdout side (HoldoutLeak
    otherwise), those the manifest lacks are counted, and the training rows
    hold none of them."""
    flags = {r["file"]: bool(r.get("holdout")) for r in manifest_rows}
    bench = set(bench_files)
    leaked = sorted(f for f in bench if f in flags and not flags[f])
    if leaked:
        raise HoldoutLeak(f"{len(leaked)} benchmark games are on the corpus's training side: {leaked[:5]}")
    train = [r for r in manifest_rows if not r.get("holdout")]
    if bench & {r["file"] for r in train}:
        raise HoldoutLeak("a benchmark game is among the corpus's training rows")
    check = {"bench_games": len(bench), "in_holdout": sum(f in flags for f in bench),
             "absent_from_manifest": sum(f not in flags for f in bench), "leaked": 0}
    return train, check


def corpus_tasks(rows: Sequence[Dict]) -> List[Tuple]:
    """("H", game key, corpus file, winner): the key is the file."""
    return [("H", r["file"], r["file"], int(r["winner_side"])) for r in rows]


# ---------------------------------------------------------------------
# One game (in a worker)
# ---------------------------------------------------------------------

_W: Dict = {}


def _init_worker(out: str, corpus: Optional[str], type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                 cap: int) -> None:
    _W.update(out=Path(out), corpus=None if corpus is None else Path(corpus), type_to_id=type_to_id,
              faction_to_id=faction_to_id, cap=cap)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def build_game(task: Tuple) -> Dict:
    """The manifest row of one game; its shards written when it has
    positions. Any failure is the row's status, never the build's end."""
    started = time.time()
    source, key = task[0], task[1]
    row: Dict = {"source": source, "key": key, "status": "ok", "positions": 0}
    row["split"], row["size_u"] = cd.split_of(key)
    try:
        if source == "M":
            _, _, path = task
            from tools.game_record import read_records
            match, label = key.split("/", 1)
            row.update(match=match, record=Path(path).name, shard=f"M/{key}")
            [rec] = list(read_records(Path(path)))
            if rec.get("game_label") != label:
                raise ValueError(f"{Path(path).name} holds game {rec.get('game_label')!r}")
            row.update(winner=rec.get("winner"), ended_by=rec.get("ended_by"), turns=rec.get("turns"),
                       seed=rec.get("seed"))
            if rec.get("ended_by") == CAPPED:
                row["status"] = "capped"
            elif rec.get("winner") not in (1, 2):
                row["status"] = "no_winner"
            else:
                game = cd.record_positions(rec, type_to_id=_W["type_to_id"], faction_to_id=_W["faction_to_id"],
                                           cap=_W["cap"])
        else:
            _, _, file, winner = task
            row.update(match=None, record=file, shard=f"H/{file}", winner=winner, ended_by=None)
            with gzip.open(_W["corpus"] / file, "rt", encoding="utf-8") as f:
                data = json.load(f)
            row["turns"] = data.get("n_turns")
            game = cd.corpus_positions(data, file, winner, type_to_id=_W["type_to_id"],
                                       faction_to_id=_W["faction_to_id"], cap=_W["cap"])
        if row["status"] == "ok":
            write_shards(_W["out"], row["shard"], key, game)
            row.update(seed=game.seed, eligible=game.eligible, positions=len(game.positions),
                       aux_missing=sum(1 for p in game.positions if p.aux != p.aux),
                       kinds=dict(Counter(p.kind for p in game.positions)))
    except Exception as e:  # noqa: BLE001 - one game must not end the build; the row says why
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    row["secs"] = round(time.time() - started, 2)
    return row


def write_shards(out: Path, shard: str, key: str, game: "cd.GamePositions") -> None:
    """One file per view, each written whole under a temporary name first."""
    base = out / shard
    base.parent.mkdir(parents=True, exist_ok=True)
    positions = [vars(p) for p in game.positions]
    for view, raws in game.raws.items():
        path = Path(f"{base}.{view}.pkl")
        tmp = Path(f"{path}.tmp")
        tmp.write_bytes(pickle.dumps({"key": key, "view": view, "positions": positions, "raws": raws},
                                     protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(tmp, path)


# ---------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------

def fingerprint(type_to_id: Dict[str, int], faction_to_id: Dict[str, int], core_phase: int, cap: int) -> str:
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    h = hashlib.sha1()
    h.update(json.dumps(sorted(type_to_id.items())).encode("utf-8"))
    h.update(json.dumps(sorted(faction_to_id.items())).encode("utf-8"))
    h.update(json.dumps(sorted(cd.ENCODING.items())).encode("utf-8"))
    h.update(f"|obs={OBSERVATION_EPOCH}|core={core_phase}|cap={cap}|views={cd.VIEWS}".encode("utf-8"))
    return h.hexdigest()


def manifest_rows(out: Path) -> Dict[str, Dict]:
    """The last row of each game in the manifest (a resumed build appends)."""
    man = out / "manifest.jsonl"
    if not man.exists():
        return {}
    rows = {}
    for line in man.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:          # a line cut by a killed build
                continue
            rows[row["key"]] = row
    return rows


def summarize(rows: Sequence[Dict]) -> Dict:
    by: Dict[str, Counter] = {}
    for r in rows:
        c = by.setdefault(r["source"], Counter())
        c[f"games_{r['status']}"] += 1
        c["positions"] += int(r.get("positions", 0))
        if r["status"] == "ok":
            c[f"games_ok_{r['split']}"] += 1
            c[f"positions_{r['split']}"] += int(r.get("positions", 0))
            c["aux_missing"] += int(r.get("aux_missing", 0))
    return {s: dict(c) for s, c in sorted(by.items())}


def write_summary(out: Path, rows: Iterable[Dict], meta: Dict, done: bool) -> None:
    rows = list(rows)
    summary = dict(meta, counts=summarize(rows), games=len(rows), done=done,
                   errors=[{"key": r["key"], "error": r.get("error")} for r in rows if r["status"] == "error"][:50])
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    os.replace(tmp, out / "summary.json")


@contextlib.contextmanager
def built_rows(tasks: Sequence[Tuple], workers: int, init: Tuple) -> Iterator[Iterator[Dict]]:
    """The rows of `tasks` as they finish: on `workers` spawned processes,
    or in this one when `workers` is 0."""
    if workers <= 0:
        _init_worker(*init)
        yield (build_game(t) for t in tasks)
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=init) as pool:
        yield pool.imap_unordered(build_game, tasks, chunksize=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-from", type=Path, required=True, help="the reference checkpoint (its vocabularies)")
    ap.add_argument("--matches", type=Path, nargs="*", default=[], help="extracted match game directories (M)")
    ap.add_argument("--corpus", type=Path, default=None, help="the imitation corpus (H)")
    ap.add_argument("--bench", type=Path, default=Path("configs/bench_states.json"))
    ap.add_argument("--cap", type=int, default=cd.MAX_POSITIONS_PER_GAME)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="0 builds in this process")
    ap.add_argument("--limit", type=int, default=None, help="first N games of each source (a smoke run)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import wesnoth_core
    from wesnoth_ai import __version__
    from wesnoth_ai.critic import file_sha256, reference_vocab
    type_to_id, faction_to_id = reference_vocab(args.vocab_from)
    tasks = match_tasks(args.matches)[:args.limit]
    bench_check = None
    if args.corpus is not None:
        from tools.replay_dataset import corpus_version_of
        version = corpus_version_of(args.corpus)
        if version != CORPUS_VERSION:
            raise SystemExit(f"{args.corpus} is corpus version {version}; the critics read {CORPUS_VERSION}")
        rows = [json.loads(line) for line in (args.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        bench = [s["file"] for s in json.loads(args.bench.read_text(encoding="utf-8"))["states"]]
        train_rows, bench_check = corpus_training_rows(rows, bench)
        tasks += corpus_tasks(train_rows)[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"fingerprint": fingerprint(type_to_id, faction_to_id, int(wesnoth_core.__phase__), args.cap),
            "code_version": __version__, "core_phase": int(wesnoth_core.__phase__),
            "vocab_from": str(args.vocab_from), "vocab_sha256": file_sha256(args.vocab_from),
            "unit_type_to_id": type_to_id, "faction_to_id": faction_to_id, "encoding": cd.ENCODING,
            "views": list(cd.VIEWS), "cap": args.cap, "holdout_fraction": cd.HOLDOUT_FRACTION,
            "matches": [str(m) for m in args.matches], "corpus": None if args.corpus is None else str(args.corpus),
            "corpus_version": CORPUS_VERSION if args.corpus is not None else None, "bench_check": bench_check}
    old = json.loads((args.out / "summary.json").read_text(encoding="utf-8")) if (args.out / "summary.json").exists() else {}
    if old and old.get("fingerprint") != meta["fingerprint"]:
        raise SystemExit(f"{args.out} holds positions of another encoding; use another --out")
    rows = manifest_rows(args.out)                  # a game that failed before is built again
    todo = [t for t in tasks if rows.get(t[1], {}).get("status", "error") == "error"]
    log.info("%d games (%d built before), %d workers, into %s; benchmark check %s", len(tasks),
             len(tasks) - len(todo), args.workers, args.out, bench_check)
    write_summary(args.out, rows.values(), meta, done=False)
    t0 = time.time()
    init = (str(args.out), None if args.corpus is None else str(args.corpus), type_to_id, faction_to_id, args.cap)
    with (args.out / "manifest.jsonl").open("a", encoding="utf-8") as man, built_rows(todo, args.workers, init) as results:
        for i, row in enumerate(results, 1):
            man.write(json.dumps(row) + "\n")
            man.flush()
            rows[row["key"]] = row
            if row["status"] == "error":
                log.warning("%s: %s", row["key"], row.get("error"))
            if i % PROGRESS_EVERY == 0 or i == len(todo):
                n = sum(int(r.get("positions", 0)) for r in rows.values())
                log.info("%d/%d games, %d positions, %s, %.0f s", i, len(todo), n,
                         dict(Counter(r["status"] for r in rows.values())), time.time() - t0)
                write_summary(args.out, rows.values(), meta, done=False)
    write_summary(args.out, rows.values(), meta, done=True)
    log.info("BUILD_DONE %d games, %s, %.0f s", len(rows), summarize(list(rows.values())), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
