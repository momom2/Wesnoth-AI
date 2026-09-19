#!/usr/bin/env python3
"""How many decisions a human side-turn holds in the imitation corpus.

The end_turn decode test (docs/endturn_rule_prereg_20260919.md) reads
the reference player at 6.0 decisions per side-turn and the same
player with the end_turn logit offset -1.5 at 9.25, where a decision
is one forward: every move, attack, recruit or recall, plus the
end_turn that closes the turn. This tool reads the corpus's own rate
in the same unit, so the offset's rate can be compared with the
players the imitation product learnt from.

    python tools/analysis/decisions_per_side_turn.py [--limit N] [--json OUT]

Per side-turn (init_side to the next init_side): the count of move,
attack, recruit and recall commands, plus one for the end_turn; means
overall, for the winner's and the loser's side-turns, and by turn
bucket (1-5, 6-10, 11-20, 21+). Games come from the corpus manifest
in order; --limit takes the first N.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
DECISIONS = ("move", "attack", "recruit", "recall")
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 10**6))


def _bucket(turn: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= turn <= hi:
            return f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
    return "?"


def side_turns(commands: List[list]):
    """(turn, side, decisions) per side-turn, the turn counted from the
    init_side sequence (two sides per turn)."""
    out = []
    turn, side, n, seen = 0, 0, 0, 0
    open_ = False
    for cmd in commands:
        kind = cmd[0] if cmd else "?"
        if kind == "init_side":
            if open_:
                out.append((turn, side, n))
            seen += 1
            side = int(cmd[1]) if len(cmd) > 1 else 0
            turn = (seen + 1) // 2
            n, open_ = 0, True
        elif kind in DECISIONS:
            n += 1
    if open_:
        out.append((turn, side, n))
    return out


def census(dataset: Path, limit: int | None) -> Dict[str, object]:
    rows = [json.loads(line) for line in
            (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:limit] if limit else rows
    per_turn: List[int] = []
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    winners: List[int] = []
    losers: List[int] = []
    games = 0
    for r in rows:
        path = dataset / r["file"]
        try:
            data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        except Exception:                    # noqa: BLE001 - one unreadable game is skipped, counted
            continue
        games += 1
        winner = r.get("winner_side")
        for turn, side, n in side_turns(data.get("commands", [])):
            d = n + 1                        # the end_turn is a decision too
            per_turn.append(d)
            by_bucket[_bucket(turn)].append(d)
            if winner in (1, 2):
                (winners if side == winner else losers).append(d)
    def summary(xs: List[int]) -> Dict[str, float]:
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
                "share_end_turn_only": sum(1 for x in xs if x == 1) / len(xs)}
    return {"games": games, "side_turns": summary(per_turn),
            "winner_side_turns": summary(winners), "loser_side_turns": summary(losers),
            "by_turn_bucket": {k: summary(v) for k, v in sorted(by_bucket.items(),
                                                              key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))}}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=ROOT / "replays_dataset_imitation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    out = census(args.dataset, args.limit)
    st = out["side_turns"]
    print(f"{out['games']} games, {st['n']} side-turns: decisions per side-turn "
          f"(end_turn included) mean {st['mean']:.2f}, median {st['median']:.0f}, "
          f"end_turn-only share {st['share_end_turn_only']:.3f}")
    for k in ("winner_side_turns", "loser_side_turns"):
        s = out[k]
        print(f"  {k}: mean {s['mean']:.2f} over {s['n']}")
    for k, s in out["by_turn_bucket"].items():
        print(f"  turns {k}: mean {s['mean']:.2f}, median {s['median']:.0f} over {s['n']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
