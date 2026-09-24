#!/usr/bin/env python3
"""How often each rewrite class of docs/xod_dominance_design_20260924.md
would fire on played games, under each of the 36 combinations of
relaxations (tools/combat_dominance.py, tools/dominance_rewrites.py).

Sources: human games from the imitation corpus (training split, never
the holdout) and generated games from game-record files
(tools/game_record.py). Every player side-turn is read as played; per
source the tool counts decisions, attacks, the rewrites each class
finds, how many each combination admits, the smallest combinations that
admit each rewrite, and the candidates it could not resolve.

    python tools/analysis/dominance_count.py --corpus replays_dataset_imitation --games 200 \\
        [--records 'training/game_records/**/*.jsonl.gz'] [--jobs 8] \\
        --out counts.json [--outcomes-out outcomes.jsonl.gz]

`--outcomes-out` keeps every fight distribution the pass computed, one
gzip JSON line per game ({"game", "outcomes": {command index: {label:
distribution}}}), so a later pass can reuse them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.combat_dominance import COMBOS, admits, minimal_combos, side_turns  # noqa: E402
from tools.dominance_rewrites import Outcomes, side_turn_rewrites  # noqa: E402

CLASSES = ("W", "H", "A", "K", "Q", "F")
FIGHT_KINDS = ("binary", "hp", "xp")


def improves_fight(rw) -> bool:
    """A better fight (a unit's survival, hp, statuses, level-up or
    experience), as opposed to a rewrite better only in what the side
    sees, shows or holds, or a tier-O rewrite that banks an option."""
    return any(d.sym == ">" and d.kind in FIGHT_KINDS for d in rw.dims)


# ---------------------------------------------------------------------
# Sources: (source name, game id, loader) triples
# ---------------------------------------------------------------------
def corpus_games(corpus: Path, n: int, offset: int = 0) -> List[Tuple[str, str, str]]:
    rows = [json.loads(line) for line in (corpus / "manifest.jsonl").open(encoding="utf-8")]
    files = [r["file"] for r in sorted(rows, key=lambda r: r["file"]) if not r.get("holdout")]
    return [("human", f, str(corpus / f)) for f in files[offset:offset + n]]


def record_games(pattern: str) -> List[Tuple[str, str, str]]:
    from tools.game_record import read_records
    out = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        for k, rec in enumerate(read_records(Path(path))):
            out.append(("generated", rec.get("game_label") or f"{path}#{k}", f"{path}#{k}"))
    return out


def _load(where: str):
    """(start state, commands) of a corpus file or of record `path#k`."""
    if "#" in where:
        from tools.game_record import read_records, start_state
        path, k = where.rsplit("#", 1)
        rec = next(r for i, r in enumerate(read_records(Path(path))) if i == int(k))
        return start_state(rec), rec["commands"]
    from tools.replay_dataset import _build_initial_gamestate, _setup_scenario_events
    data = json.load(gzip.open(where, "rt", encoding="utf-8"))
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    return gs, data["commands"]


# ---------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------
def count_game(item: Tuple[str, str, str]) -> Dict:
    source, game_id, where = item
    logging.disable(logging.WARNING)
    tally = collections.Counter()
    admitted = collections.Counter()           # (class, combo name)
    minimal = collections.Counter()            # (class, combo name)
    gains = collections.defaultdict(float)     # (class, gain name) over rewrites admitted at R0
    examples: List[Dict] = []
    kept = Outcomes()
    t0 = time.time()
    try:
        start, commands = _load(where)
        for st in side_turns(game_id, start, commands):
            rewrites, counts = side_turn_rewrites(st, kept)
            tally["side_turns"] += 1
            tally.update(counts)
            for rw in rewrites:
                tally[f"found:{rw.klass}"] += 1
                fights = improves_fight(rw)
                for c in COMBOS:
                    if admits(rw.dims, rw.tier, c):
                        admitted[(rw.klass, c.name())] += 1
                        if fights:
                            admitted[(rw.klass + "*", c.name())] += 1
                mins = minimal_combos(rw.dims, rw.tier)
                for c in mins:
                    minimal[(rw.klass, c.name())] += 1
                if admits(rw.dims, rw.tier, COMBOS[0]):
                    for g, v in rw.gains.items():
                        gains[(rw.klass, g)] += v
                if len(examples) < 3:
                    examples.append({"class": rw.klass, "turn": rw.turn, "side": rw.side,
                                     "window": [st.first_index + w for w in rw.window],
                                     "detail": rw.detail, "minimal": [c.name() for c in mins],
                                     "vector": {d.name: d.sym for d in rw.dims if d.sym != "="}})
    except Exception as e:                          # noqa: BLE001
        tally["games_failed"] += 1
        return {"source": source, "game": game_id, "error": repr(e), "tally": dict(tally)}
    tally["games"] += 1
    return {"source": source, "game": game_id, "tally": dict(tally), "secs": time.time() - t0,
            "admitted": {f"{k}|{c}": v for (k, c), v in admitted.items()},
            "minimal": {f"{k}|{c}": v for (k, c), v in minimal.items()},
            "gains": {f"{k}|{g}": v for (k, g), v in gains.items()},
            "examples": examples, "outcomes": kept.by_command}


# ---------------------------------------------------------------------
# Totals and the report
# ---------------------------------------------------------------------
def merge(results: Iterator[Dict], outcomes_out: Optional[Path]) -> Dict[str, Dict]:
    totals: Dict[str, Dict] = {}
    sink = gzip.open(outcomes_out, "wt", encoding="utf-8") if outcomes_out else None
    try:
        for r in results:
            t = totals.setdefault(r["source"], {"tally": collections.Counter(),
                                                "admitted": collections.Counter(),
                                                "minimal": collections.Counter(),
                                                "gains": collections.Counter(),
                                                "examples": [], "errors": [], "slowest": []})
            t["tally"].update(r.get("tally", {}))
            if "error" in r:
                t["errors"].append({"game": r["game"], "error": r["error"]})
                continue
            t["slowest"] = sorted(t["slowest"] + [[round(r["secs"], 1), r["game"]]], reverse=True)[:5]
            t["admitted"].update(r["admitted"])
            t["minimal"].update(r["minimal"])
            t["gains"].update(r["gains"])
            if len(t["examples"]) < 40:
                t["examples"] += [dict(e, game=r["game"]) for e in r["examples"]]
            if sink is not None:
                sink.write(json.dumps({"game": r["game"], "source": r["source"],
                                       "outcomes": r["outcomes"]}, separators=(",", ":")) + "\n")
    finally:
        if sink is not None:
            sink.close()
    return {s: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in t.items()}
            for s, t in totals.items()}


REPORT_COMBOS = ("R0", "r1", "r2-0.05", "r2-0.15", "r3g", "r3l", "r4", "r3g+r4", "r3l+r4",
                 "r1+r2-0.15+r3g+r4", "r1+r2-0.15+r3l+r4")


def report(totals: Dict[str, Dict]) -> str:
    lines = []
    for source, t in totals.items():
        tl = t["tally"]
        dec = tl.get("decisions", 0) or 1
        lines.append(f"\n{source}: {tl.get('games', 0)} games ({tl.get('games_failed', 0)} failed), "
                     f"{tl.get('side_turns', 0)} side-turns, {tl.get('decisions', 0)} decisions, "
                     f"{tl.get('attacks', 0)} attacks, {tl.get('inconclusive', 0)} candidates unresolved")
        lines.append(f"slowest games (s): {t.get('slowest')}")
        lines.append("Each cell: rewrites with a better fight / all admitted rewrites.")
        lines.append("| combination | " + " | ".join(CLASSES) + " | all | per 1000 decisions |")
        lines.append("|---" * (len(CLASSES) + 3) + "|")
        for name in REPORT_COMBOS:
            both = [(t["admitted"].get(f"{k}*|{name}", 0), t["admitted"].get(f"{k}|{name}", 0))
                    for k in CLASSES]
            fight, every = sum(f for f, _ in both), sum(a for _, a in both)
            lines.append(f"| {name} | " + " | ".join(f"{f}/{a}" for f, a in both)
                         + f" | {fight}/{every} | {1000 * fight / dec:.1f}/{1000 * every / dec:.1f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--games", type=int, default=100, help="corpus games to read")
    ap.add_argument("--offset", type=int, default=0, help="skip this many corpus games first")
    ap.add_argument("--records", default=None, help="glob of game-record files")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--outcomes-out", type=Path, default=None)
    args = ap.parse_args(argv)
    items: List[Tuple[str, str, str]] = []
    if args.corpus is not None:
        items += corpus_games(args.corpus, args.games, args.offset)
    if args.records:
        items += record_games(args.records)
    if not items:
        ap.error("no games: give --corpus and/or --records")
    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            totals = merge(pool.imap_unordered(count_game, items, chunksize=1), args.outcomes_out)
    else:
        totals = merge(map(count_game, items), args.outcomes_out)
    print(report(totals))
    print(f"\n{len(items)} games, {time.time() - t0:.0f} s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"combos": [c.name() for c in COMBOS], "sources": totals},
                                       indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
