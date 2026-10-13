"""Write the columnar records (wesnoth_ai/sequence_columns.py) of a sequence
directory beside its pickled ones: the corpus's (tools/preencode_sequences.py)
or the memory player's own (tools/own_targets.py). The sequence trainer reads
them with `--columns` (docs/trainer_speed_20261013.md, round 2).

Every game of the directory's manifest is read from its pickle, written as
columns, read back, and each position compared with the pickled one, field
by field and byte for byte (`sequence_columns.position_difference`), and its
game-sides' lengths with the manifest's; a game that differs anywhere is
reported and has no columns. Idempotent: the games the columns manifest
lists are skipped, so a killed pass continues; the manifest is written on
the run.

    DIR/<game>.seq.cols        one game's columns
    DIR/columns_manifest.json  the format, the sequence manifest's
                               fingerprint, the games written and the errors

    python tools/columnize_sequences.py --sequences replays_dataset_sequences --workers 30
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.preencode_sequences import load_manifest, read_record, record_path  # noqa: E402
from wesnoth_ai.sequence_columns import (COLUMNS_FORMAT, ColumnsError, columns_path,  # noqa: E402
                                         position_difference, read_sides, write_columns)

log = logging.getLogger("columnize_sequences")

COLUMNS_MANIFEST = "columns_manifest.json"


def load_columns_manifest(directory: Path) -> Optional[Dict]:
    p = Path(directory) / COLUMNS_MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def columns_problem(directory: Path, games) -> Optional[str]:
    """Why the columns of `directory` cannot stand for its pickled records
    of `games` (None when they can): no columns manifest, another format or
    another sequence manifest's fingerprint, or a game without columns."""
    directory = Path(directory)
    man, cols = load_manifest(directory), load_columns_manifest(directory)
    fix = f"run tools/columnize_sequences.py --sequences {directory}"
    if cols is None:
        return f"{directory} has no columnar records: {fix}"
    if cols.get("format") != COLUMNS_FORMAT:
        return f"{directory}'s columns are of format {cols.get('format')}, this trainer reads {COLUMNS_FORMAT}: {fix}"
    if man is None or cols.get("fingerprint") != man.get("fingerprint"):
        return f"{directory}'s columns were written from other records than its manifest's: {fix}"
    missing = [g for g in games if g not in cols["games"]]
    if missing:
        return f"{directory} has no columns for {len(missing)} games (first {missing[0]}): {fix}"
    return None


def convert_game(directory: Path, file: str, lengths: Dict[str, int]) -> int:
    """`file`'s columns written and checked against its pickle; its positions."""
    rec = read_record(record_path(directory, file))
    dst = columns_path(directory, file)
    write_columns(dst, rec)
    sides = read_sides(dst)
    n = 0
    for side, positions in rec.sides.items():
        want = lengths.get(f"positions_side{side}")
        if want is not None and int(want) != len(positions):
            raise ColumnsError(f"{file} side {side}: {len(positions)} positions, the manifest says {want}")
        got = sides.get(int(side))
        if got is None or len(got) != len(positions):
            raise ColumnsError(f"{file} side {side}: the columns hold {None if got is None else len(got)} "
                               f"positions of {len(positions)}")
        for i, original in enumerate(positions):
            why = position_difference(got.position(i), original)
            if why is not None:
                raise ColumnsError(f"{file} side {side} position {i}: {why} differs")
        n += len(positions)
    return n


_W: Dict = {}


def _worker_init(directory: str, games: Dict) -> None:
    _W.update(directory=Path(directory), games=games)


def _worker(file: str) -> Tuple[str, str, int]:
    try:
        return file, "ok", convert_game(_W["directory"], file, _W["games"][file])
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 - one bad game must not stop the pass
        columns_path(_W["directory"], file).unlink(missing_ok=True)
        return file, f"error: {type(e).__name__}: {e}"[:300], 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sequences", type=Path, required=True, help="a directory of pickled sequence records")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    man = load_manifest(args.sequences)
    if man is None:
        raise SystemExit(f"{args.sequences}: no sequence manifest")
    cols = load_columns_manifest(args.sequences) or {}
    if cols and (cols.get("format") != COLUMNS_FORMAT or cols.get("fingerprint") != man["fingerprint"]):
        log.info("the columns manifest is of another format or fingerprint: every game is written again")
        cols = {}
    done: Dict[str, int] = dict(cols.get("games", {}))
    errors: Dict[str, str] = {}
    todo = [f for f in man["games"] if f not in done]
    t0 = time.time()

    def flush() -> None:
        tmp = args.sequences / (COLUMNS_MANIFEST + ".tmp")
        tmp.write_text(json.dumps({"format": COLUMNS_FORMAT, "fingerprint": man["fingerprint"],
                                   "source": man.get("source", "corpus"), "n_games": len(done),
                                   "positions": sum(done.values()), "errors": errors, "games": done}),
                       encoding="utf-8")
        os.replace(tmp, args.sequences / COLUMNS_MANIFEST)

    log.info("columns for %d games of %d with %d workers in %s", len(todo), len(man["games"]), args.workers,
             args.sequences)
    games = {f: man["games"][f] for f in todo}
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                      initargs=(str(args.sequences), games)) as pool:
        for i, (file, status, n) in enumerate(pool.imap_unordered(_worker, todo, chunksize=4), 1):
            if status == "ok":
                done[file] = n
            else:
                errors[file] = status
                log.warning("%s: %s", file, status)
            if i % 500 == 0 or i == len(todo):
                log.info("%d/%d games, %d positions, %d errors, %.0f s", i, len(todo), sum(done.values()),
                         len(errors), time.time() - t0)
                flush()
    flush()
    log.info("COLUMNS_DONE %d games, %d positions, %d errors in %.0f s", len(done), sum(done.values()),
             len(errors), time.time() - t0)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
