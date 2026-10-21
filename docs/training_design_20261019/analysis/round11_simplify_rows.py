"""Round 11, simplify stance: facts of the luck walk's game rows that the
proposal's costs and checks use (training/metrics/match_luck_20261019/rows.jsonl,
19,466 rows over the 23 argmax matches and their re-walks).

Per row: turns, the walk's CPU seconds (`secs`), attacks, recruits, how the
game ended. Reported over all rows and for the own1/own2 self-play matches:
mean turns a game, the walk's CPU seconds a game, attacks a game, the share
of games that ended at the turn cap. Standard library only.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"


def main():
    by = defaultdict(list)
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["match"]].append(r)
    groups = {"all": [r for rs in by.values() for r in rs],
              "self-play (gen_own1_selfplay, own64_selfplay)": by["gen_own1_selfplay"] + by["own64_selfplay"]}
    for name, rows in groups.items():
        ended = Counter(r.get("ended_by") for r in rows)
        turns = [r["turns"] for r in rows]
        secs = [r["secs"] for r in rows if r.get("secs") is not None]
        attacks = [r["counts"]["attacks"] for r in rows if "attacks" in r.get("counts", {})]
        print(f"== {name}: {len(rows)} rows")
        print(f"  turns a game: mean {statistics.mean(turns):.2f}, median {statistics.median(turns)}")
        print(f"  walk CPU seconds a game: mean {statistics.mean(secs):.3f}, median {statistics.median(secs):.3f}")
        print(f"  attacks a game: mean {statistics.mean(attacks):.1f}")
        print(f"  ended by: {dict(ended)}")


if __name__ == "__main__":
    main()
