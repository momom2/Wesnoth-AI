"""The current network against its past selves, from the simple loop's own
rounds: 26 games a round against one past self drawn from the pool
(`past.json`), both sampling at temperature 1 under the round's rules. The
wins are grouped by where the past self comes from (before or after a given
round, here the switch of round 609) and by its age in rounds. A loop that
gains wins more than half against a younger self of its own, and more the
older the self. Standard library only.

    python -I -X utf8 past_self.py --out past.json <run dir> 609 752 --since 609
"""
import argparse
import json
from pathlib import Path

AGES = ((1, 32), (33, 64), (65, 96), (97, 160), (161, 320), (321, 10 ** 6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--since", type=int, default=0, help="split the past selves at this round")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cells: dict = {}
    rounds = []
    for k in range(args.first, args.last + 1):
        d = args.run / f"round_{k:04d}"
        past = int(json.loads((d / "past.json").read_text(encoding="utf-8"))["step"])
        res = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "games_past").glob("game_*.json"))
               if not p.name.endswith(".game.json")]
        wins = sum(r["outcome_a"] == "win" for r in res)
        decided = sum(r["outcome_a"] in ("win", "loss") for r in res)
        rounds.append({"round": k, "past": past, "wins": wins, "decided": decided})
        group = "from the switch on" if past >= args.since else "before the switch"
        age = k - past
        bucket = next(f"{a} to {b}" if b < 10 ** 6 else f"{a} or more" for a, b in AGES if a <= age <= b)
        for key in ((group, "all ages"), (group, bucket), ("all", "all ages")):
            c = cells.setdefault(key, {"wins": 0, "decided": 0, "rounds": 0})
            c["wins"] += wins
            c["decided"] += decided
            c["rounds"] += 1
    table = []
    for (group, bucket), c in cells.items():
        p = c["wins"] / c["decided"] if c["decided"] else float("nan")
        se = (p * (1 - p) / c["decided"]) ** 0.5 if c["decided"] else float("nan")
        table.append({"past_self": group, "age_in_rounds": bucket, **c, "share": p, "standard_error": se})
    order = {name: i for i, name in enumerate(["all ages"] + [f"{a} to {b}" if b < 10 ** 6 else f"{a} or more"
                                                              for a, b in AGES])}
    table.sort(key=lambda r: (r["past_self"], order[r["age_in_rounds"]]))
    for r in table:
        print(f"{r['past_self']:20s} {r['age_in_rounds']:12s} {r['wins']:5d} of {r['decided']:5d} = "
              f"{100 * r['share']:5.1f}% +- {100 * r['standard_error']:.1f}  ({r['rounds']} rounds)")
    if args.out:
        args.out.write_text(json.dumps({"rounds": [args.first, args.last], "since": args.since, "table": table,
                                        "by_round": rounds}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
