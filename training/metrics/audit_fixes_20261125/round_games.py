"""What a round's games look like from their summaries (the .json beside each
record): outcomes, length, forced end of turns, unplayed turns. Read on both
sides of the switch of docs/audit_fixes_prereg_20261125.md to see whether the
new rules changed the games' shape. Standard library only.

    python -I -X utf8 round_games.py --out games.json <run dir> 605 606 607 608 609
"""
import argparse
import json
from pathlib import Path


def read(run: Path, k: int) -> dict:
    row = {"round": k}
    for name in ("games_self", "games_past"):
        games = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted((run / f"round_{k:04d}" / name).glob("game_*.json"))
                 if not p.name.endswith(".game.json")]
        n = len(games)
        if not n:
            continue
        outcomes: dict = {}
        for g in games:
            outcomes[str(g.get("outcome_a"))] = outcomes.get(str(g.get("outcome_a")), 0) + 1
        turns = sorted(int(g["turns"]) for g in games)
        row[name] = {
            "games": n,
            "outcome_a": dict(sorted(outcomes.items())),
            "turns_mean": round(sum(turns) / n, 1),
            "turns_median": turns[n // 2],
            "turns_max": turns[-1],
            "forced_end_turns": sum(int(g.get("forced_end_turns_a", 0)) + int(g.get("forced_end_turns_b", 0))
                                    for g in games),
            "unplayed_side_turns": sum(int(g.get("unplayed_side_turns", 0)) for g in games),
            "forwards_per_game": round(sum(int(g.get("forwards_a", 0)) + int(g.get("forwards_b", 0))
                                           for g in games) / n),
            "observation_epochs": sorted({int(g.get("observation_epoch", -1)) for g in games}),
        }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("rounds", nargs="+", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [read(args.run, k) for k in args.rounds]
    text = json.dumps(rows, indent=1)
    for r in rows:
        print(json.dumps(r))
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
