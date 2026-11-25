"""The advancement choices in played games (docs/audit_fixes_prereg_20261125.md,
departure 1: "how many advancements the round's games held as choices, and
the share that took a type other than the first").

A choice is an `["advance", k]` command of a game record (format 8): the side
to move answered for its own unit, k the index of the type among the unit's
options, 0 the type Wesnoth lists first. Reads the records with the standard
library only, so it is light enough to run beside a training round.

    python -I -X utf8 advance_choices.py --out choices.json <games dir> [<games dir> ...]
"""
import argparse
import gzip
import json
from pathlib import Path


def records(path: Path):
    """Every game record of one .game.jsonl.gz file (a line each)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and "commands" in rec:
                yield rec


def count(directory: Path) -> dict:
    games = asked = choices = not_first = with_choice = 0
    by_index: dict = {}
    for path in sorted(directory.glob("*.game.jsonl.gz")):
        for rec in records(path):
            games += 1
            asked += int(bool(rec.get("ask_advance")))
            ks = [int(c[1]) for c in rec["commands"] if c and c[0] == "advance"]
            choices += len(ks)
            not_first += sum(k > 0 for k in ks)
            with_choice += int(bool(ks))
            for k in ks:
                by_index[str(k)] = by_index.get(str(k), 0) + 1
    return {
        "directory": str(directory),
        "games": games,
        "games_played_with_the_asking": asked,
        "games_with_a_choice": with_choice,
        "choices": choices,
        "choices_per_game": round(choices / games, 3) if games else None,
        "not_first": not_first,
        "share_not_first": round(not_first / choices, 3) if choices else None,
        "by_option_index": dict(sorted(by_index.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [count(d) for d in args.directories]
    total = {k: sum(r[k] for r in rows) for k in ("games", "choices", "not_first")}
    total["choices_per_game"] = round(total["choices"] / total["games"], 3) if total["games"] else None
    total["share_not_first"] = round(total["not_first"] / total["choices"], 3) if total["choices"] else None
    result = {"sets": rows, "total": total}
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
