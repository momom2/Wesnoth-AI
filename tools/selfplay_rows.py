"""Per-game-side rows of a self-play run (docs/training_design_20261019/
current_design.md §1, step 2): what the labels of tools/pg_labels.py and
the readouts of tools/selfplay_readout.py read.

The run's games directory holds run_elo_batch's result files beside their
records. Its luck walk (tools/match_luck.py: the seven attack-luck sums of
each game, from side A's view, written as each game is walked and resumed
past the games already there) is turned into two rows per game, one per
side, everything in that side's frame:

    game, index  the stem of the game's result and record files
                 (`<game>.json`, `<game>.game.jsonl.gz`), unique across a
                 run's matches since their seeds are; the game's slot index,
                 its seed less the run's seed base
    side         the side (1 or 2): its seat
    label        the player label on that side
    z            +1 the side won, -1 it lost, 0 no result: a game that
                 reached the turn cap (or ended without a winner) scores 0
                 and is kept (dropping it would condition on an outcome)
    capped       the game had no winner
    cell         map x factions x seat: "scenario|seat|own faction|opponent faction"
    S            the seven luck sums (match_luck.COLUMNS), signed for the side
    turns, ended_by
    error        present when the game's walk failed (S is then absent):
                 a game without a record (a game run_elo_batch killed on its
                 per-game timeout leaves a result, `timeout_kill`, and no
                 record) or one that does not rebuild. tools/pg_labels.py
                 leaves both of its sides out. The walk fails (exit 1) past
                 --max-errors such games, each named in the log.

A self-play run keeps its capped games only with run_elo_batch
--max-extra-games 0; it then exits 4 (EXIT_GUARD_SPENT) when every slot is
played and some capped, which is a complete run.

    python tools/selfplay_rows.py --games DIR --walk walk.jsonl --out rows.jsonl \\
        --seed-base 0 [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.match_luck import COLUMNS, result_z, walk_matches  # noqa: E402

log = logging.getLogger("selfplay_rows")

MATCH_NAME = "selfplay"


def cell_of(scenario: str, seat: int, faction: str, opponent: str) -> str:
    """The baseline's cell: map x factions x seat, in the side's frame."""
    return f"{scenario}|{int(seat)}|{faction}|{opponent}"


def side_rows(walk_row: Dict, seed_base: int) -> List[Dict]:
    """The two side rows of one game's walk row (tools/match_luck.py)."""
    side_a = int(walk_row["side_a"])
    z_a = result_z(walk_row.get("outcome_a"))
    index = int(walk_row["seed"]) - int(seed_base)
    if index < 0:
        raise ValueError(f"{walk_row['game']}: seed {walk_row['seed']} lies below the seed base {seed_base}")
    rows = []
    for side in (1, 2):
        sign = 1 if side == side_a else -1
        own, opp = ("a", "b") if sign == 1 else ("b", "a")
        row = {"game": walk_row["game"], "index": index, "side": side,
               "label": walk_row.get(f"label_{own}"), "z": sign * z_a, "capped": z_a == 0,
               "cell": cell_of(walk_row["scenario_id"], side, walk_row[f"faction_{own}"],
                               walk_row[f"faction_{opp}"]),
               "turns": walk_row.get("turns"), "ended_by": walk_row.get("ended_by")}
        if "error" in walk_row:
            row["error"] = walk_row["error"]
        else:
            row["S"] = [sign * float(walk_row["S"][c]) for c in COLUMNS]
        rows.append(row)
    return rows


def read_jsonl(path: Path) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def convert(walk_path: Path, out: Path, seed_base: int) -> Dict:
    """Every game of the walk as its two side rows into `out` (rewritten
    whole, atomically): {games, side_rows, errors, capped, unwalked (the
    games whose walk failed, by name)}."""
    counts: Dict = {"games": 0, "side_rows": 0, "errors": 0, "capped": 0, "unwalked": []}
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for walk_row in read_jsonl(walk_path):
            rows = side_rows(walk_row, seed_base)
            counts["games"] += 1
            if "error" in walk_row:
                counts["errors"] += 1
                counts["unwalked"].append(walk_row["game"])
            counts["capped"] += int(rows[0]["capped"])
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts["side_rows"] += 1
    tmp.replace(out)
    return counts


def load_rows(path: Path) -> List[Dict]:
    return list(read_jsonl(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True, help="the run's games directory or tarball")
    ap.add_argument("--walk", type=Path, required=True, help="the luck walk's rows (resumed when present)")
    ap.add_argument("--out", type=Path, required=True, help="the side rows")
    ap.add_argument("--seed-base", type=int, required=True, help="run_elo_batch's --seed-base of the run")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-verify", action="store_true", help="skip the records' fingerprint checks")
    ap.add_argument("--max-errors", type=int, default=0,
                    help="games whose walk may fail (left out of the labels) before the walk fails")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    n = walk_matches([(MATCH_NAME, args.games)], args.walk, jobs=args.jobs, verify=not args.no_verify)
    log.info("%d games walked into %s", n, args.walk)
    counts = convert(args.walk, args.out, args.seed_base)
    log.info("%s: %d games, %d side rows, %d capped, %d unwalked (at most %d)", args.out, counts["games"],
             counts["side_rows"], counts["capped"], counts["errors"], args.max_errors)
    for game in counts["unwalked"]:
        log.warning("unwalked, left out of the labels: %s", game)
    return 0 if counts["errors"] <= args.max_errors else 1


if __name__ == "__main__":
    sys.exit(main())
