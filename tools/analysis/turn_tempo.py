#!/usr/bin/env python3
"""How players spend their turns, from match game records or the human
corpus (docs/memory_in_play_prereg_20261002.md).

Per player (a match label, or the human winner and loser):
  - actions per side-turn (moves, attacks, recruits; the end_turn not
    counted) by the side's turn number;
  - the end-of-turn hazard over turns 6 to 15: the chance a side-turn ends
    after n actions, given it reached n;
  - with --units (match records only, replayed on the Rust core): at each
    end_turn, the side's units, those that never moved this turn (full
    movement left), those standing next to a unit of the opposing player
    that did not attack, and the side's gold; over decided games.

    python tools/analysis/turn_tempo.py --games DIR [--units] [--json OUT]
    python tools/analysis/turn_tempo.py --corpus DATASET [--limit 2000] [--json OUT]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

ACTIONS = ("move", "attack", "recruit")
HAZARD_TURNS = (6, 15)
MAX_TURN_ROW = 15


def side_turns(commands: List) -> Iterator[Tuple[int, int, List]]:
    """(side, the side's turn number, the side-turn's commands) for every
    side-turn that ends with an end_turn."""
    count: Counter = Counter()
    side, current = None, []
    for cmd in list(commands) + [["init_side", None]]:
        if cmd[0] == "init_side":
            if side is not None and any(c[0] == "end_turn" for c in current):
                yield side, count[side], current
            side = cmd[1]
            if side is not None:
                count[side] += 1
            current = []
        else:
            current.append(cmd)


def n_actions(commands: List) -> int:
    return sum(c[0] in ACTIONS for c in commands)


class Tempo:
    """Actions per side-turn by turn number, and the turns 6-15 lengths."""

    def __init__(self):
        self.by_turn: Dict[int, List[int]] = defaultdict(list)
        self.lengths: Counter = Counter()

    def add(self, turn: int, n: int) -> None:
        self.by_turn[min(turn, MAX_TURN_ROW + 1)].append(n)
        if HAZARD_TURNS[0] <= turn <= HAZARD_TURNS[1]:
            self.lengths[n] += 1

    def summary(self) -> Dict:
        hazard = []
        for n in range(21):
            at_risk = sum(v for k, v in self.lengths.items() if k >= n)
            hazard.append(round(self.lengths[n] / at_risk, 4) if at_risk else None)
        total = sum(self.lengths.values())
        return {"actions_by_turn": {str(t) if t <= MAX_TURN_ROW else f"{MAX_TURN_ROW + 1}+":
                                    [round(sum(v) / len(v), 3), len(v)]
                                    for t, v in sorted(self.by_turn.items())},
                "turns_6_15": {"side_turns": total,
                               "mean_actions": round(sum(k * v for k, v in self.lengths.items()) / total, 3)
                               if total else None,
                               "hazard_by_n": hazard}}


def match_tempo(directory: Path) -> Dict[str, Tempo]:
    out: Dict[str, Tempo] = defaultdict(Tempo)
    for path in sorted(Path(directory).glob("*.game.jsonl.gz")):
        rec = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        labels = {p["side"]: p["label"] for p in rec["players"].values()}
        for side, turn, cmds in side_turns(rec["commands"]):
            if side in labels:
                out[labels[side]].add(turn, n_actions(cmds))
    return out


def corpus_tempo(dataset: Path, limit: int, seed: int = 0) -> Dict[str, Tempo]:
    entries = [json.loads(line) for line in open(Path(dataset) / "manifest.jsonl", encoding="utf-8")]
    random.Random(seed).shuffle(entries)
    out: Dict[str, Tempo] = defaultdict(Tempo)
    for entry in entries[:limit]:
        data = json.loads(gzip.open(Path(dataset) / entry["file"], "rt", encoding="utf-8").read())
        for side, turn, cmds in side_turns(data["commands"]):
            if side in (1, 2):
                out["winner" if side == entry["winner_side"] else "loser"].add(turn, n_actions(cmds))
    return out


# ---------------------------------------------------------------------------
# End-of-turn unit census (match records, on the Rust core)
# ---------------------------------------------------------------------------

def census_rows(rec: Dict) -> Iterator[Dict]:
    """At each end_turn of a player side: its units, those with full
    movement, those next to an opposing player's unit that did not
    attack, and its gold."""
    from tools.abilities import hex_neighbors
    from tools.game_record import _walk_core
    from wesnoth_ai.classes import UnitStatus
    labels = {p["side"]: p["label"] for p in rec["players"].values()}
    turns: Counter = Counter()
    for _, cs, cmd in _walk_core(rec, verify=True):
        if cmd[0] == "init_side":
            turns[int(cmd[1])] += 1
            continue
        if cmd[0] != "end_turn" or int(cs.core.current_side) not in labels:
            continue
        gs = cs.to_state()
        side = int(gs.global_info.current_side)
        units = list(gs.map.units)
        foes = {(u.position.x, u.position.y) for u in units
                if u.side == 3 - side and UnitStatus.PETRIFIED not in u.statuses}
        own = [u for u in units if u.side == side]
        yield {"label": labels[side], "turn": turns[side], "units": len(own),
               "unmoved": sum(u.current_moves == u.max_moves for u in own),
               "idle_next_to_foe": sum(not u.has_attacked and any(
                   h in foes for h in hex_neighbors(u.position.x, u.position.y)) for u in own),
               "gold": gs.sides[side - 1].current_gold}


def census_bucket(turn: int) -> str:
    if turn <= 10:
        return str(turn)
    return "11-15" if turn <= 15 else "16-25" if turn <= 25 else "26+"


def match_census(directory: Path) -> Dict[str, Dict[str, Dict]]:
    sums: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for path in sorted(Path(directory).glob("*.game.jsonl.gz")):
        rec = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        if rec.get("winner") not in (1, 2):
            continue
        for row in census_rows(rec):
            c = sums[row["label"]][census_bucket(row["turn"])]
            c["n"] += 1
            for key in ("units", "unmoved", "idle_next_to_foe", "gold"):
                c[key] += row[key]
    return {label: {b: {k: round(v / c["n"], 3) for k, v in c.items() if k != "n"} | {"n": c["n"]}
                    for b, c in buckets.items()}
            for label, buckets in sums.items()}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--games", type=Path, help="a match directory of game records")
    src.add_argument("--corpus", type=Path, help="an imitation dataset directory")
    ap.add_argument("--limit", type=int, default=2000, help="with --corpus: games sampled")
    ap.add_argument("--units", action="store_true", help="with --games: the end-of-turn unit census")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    tempo = match_tempo(args.games) if args.games else corpus_tempo(args.corpus, args.limit)
    result = {"source": str(args.games or args.corpus), "players": {k: v.summary() for k, v in tempo.items()}}
    if args.units and args.games:
        result["census"] = match_census(args.games)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
