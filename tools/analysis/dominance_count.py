#!/usr/bin/env python3
"""How often each rewrite class of docs/xod_dominance_design_20260924.md
would fire on played games, under each of the 36 combinations of
relaxations (tools/combat_dominance.py, tools/dominance_rewrites.py).

Sources: human games from the imitation corpus (training split, never
the holdout) and generated games from game-record files
(tools/game_record.py). Every player side-turn is read as played; per
source the tool counts decisions, attacks, the rewrites each class
finds, how many each combination admits, the opportunities (attacks
with at least one admitted rewrite of a class; one attack is one
opportunity however many rewrites it has), the smallest combinations
that admit each rewrite, and the candidates set aside. A game that
fails to read counts as failed and adds nothing else.

    python tools/analysis/dominance_count.py --corpus replays_dataset_imitation --games 200 \\
        [--records 'training/game_records/**/*.jsonl.gz'] [--jobs 8] \\
        --out counts.json [--outcomes-out outcomes.jsonl.gz]

`--out` is rewritten every `--write-every` games and at the end, so a
cut run keeps what it counted. `--outcomes-out` keeps every fight
distribution the pass computed, one gzip member per game ({"game",
"source", "outcomes": {command index: {label: distribution}}}), so a
cut file still reads up to its last whole game.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.combat_dominance import COMBOS, admits, fight_gain, minimal_combos, side_turns  # noqa: E402
from tools.dominance_rewrites import Outcomes, side_turn_rewrites  # noqa: E402

CLASSES = ("W", "H", "A", "K", "Q", "F")


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
            label = rec.get("game_label") or ""
            out.append(("generated", f"{Path(path).name}#{k}:{label}", f"{path}#{k}"))
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
    """One game's counts. `admitted` counts rewrites and `opportunities`
    attacks, per (class, combination); a class name with `*` counts only
    those with a fight gain under that combination, `any` every class."""
    source, game_id, where = item
    logging.disable(logging.WARNING)
    tally = collections.Counter()
    admitted = collections.Counter()           # (class, combo name)
    anchors = collections.defaultdict(set)     # (class, combo name) -> attack indices
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
                for c in COMBOS:
                    if not admits(rw.dims, rw.tier, c):
                        continue
                    classes = [rw.klass, "any"]
                    if rw.tier == "D" and fight_gain(rw.dims, c):
                        classes += [rw.klass + "*", "any*"]
                    for k in classes:
                        admitted[(k, c.name())] += 1
                        anchors[(k, c.name())].add(rw.anchor)
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
        return {"source": source, "game": game_id, "error": repr(e), "tally": {"games_failed": 1}}
    tally["games"] += 1
    return {"source": source, "game": game_id, "tally": dict(tally), "secs": time.time() - t0,
            "admitted": {f"{k}|{c}": v for (k, c), v in admitted.items()},
            "opportunities": {f"{k}|{c}": len(v) for (k, c), v in anchors.items()},
            "minimal": {f"{k}|{c}": v for (k, c), v in minimal.items()},
            "gains": {f"{k}|{g}": v for (k, g), v in gains.items()},
            "examples": examples, "outcomes": kept.by_command}


# ---------------------------------------------------------------------
# Totals and the report
# ---------------------------------------------------------------------
def _plain(totals: Dict[str, Dict]) -> Dict[str, Dict]:
    return {s: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in t.items()}
            for s, t in totals.items()}


def write_counts(path: Path, totals: Dict[str, Dict], games_done: int, games_total: int) -> None:
    """The counts so far, replacing the file whole (write, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"combos": [c.name() for c in COMBOS], "games_done": games_done,
                               "games_total": games_total, "sources": _plain(totals)}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def merge(results: Iterator[Dict], outcomes_out: Optional[Path] = None, counts_out: Optional[Path] = None,
          write_every: int = 50, games_total: int = 0) -> Dict[str, Dict]:
    totals: Dict[str, Dict] = {}
    done = 0
    for r in results:
        t = totals.setdefault(r["source"], {"tally": collections.Counter(),
                                            "admitted": collections.Counter(),
                                            "opportunities": collections.Counter(),
                                            "minimal": collections.Counter(),
                                            "gains": collections.Counter(),
                                            "examples": [], "errors": [], "slowest": []})
        t["tally"].update(r.get("tally", {}))
        done += 1
        if "error" in r:
            t["errors"].append({"game": r["game"], "error": r["error"]})
        else:
            t["slowest"] = sorted(t["slowest"] + [[round(r["secs"], 1), r["game"]]], reverse=True)[:5]
            for key in ("admitted", "opportunities", "minimal", "gains"):
                t[key].update(r[key])
            if len(t["examples"]) < 40:
                t["examples"] += [dict(e, game=r["game"]) for e in r["examples"]]
            if outcomes_out is not None:
                line = json.dumps({"game": r["game"], "source": r["source"], "outcomes": r["outcomes"]},
                                  separators=(",", ":")) + "\n"
                with open(outcomes_out, "ab") as f:
                    f.write(gzip.compress(line.encode("utf-8")))
        if counts_out is not None and done % write_every == 0:
            write_counts(counts_out, totals, done, games_total)
    if counts_out is not None:
        write_counts(counts_out, totals, done, games_total)
    return _plain(totals)


REPORT_COMBOS = ("R0", "r1", "r2-0.05", "r2-0.15", "r3g", "r3l", "r4", "r3g+r4", "r3l+r4",
                 "r1+r2-0.15+r3g+r4", "r1+r2-0.15+r3l+r4")


SET_ASIDE = ("inconclusive", "h_fight_not_better", "h_not_isolated", "a_kill_in_reach", "a_not_isolated")


def report(totals: Dict[str, Dict]) -> str:
    lines = []
    for source, t in totals.items():
        tl, opp = t["tally"], t["opportunities"]
        dec = tl.get("decisions", 0)
        lines.append(f"\n{source}: {tl.get('games', 0)} games read, {tl.get('games_failed', 0)} failed; "
                     f"{tl.get('side_turns', 0)} side-turns, {dec} decisions, {tl.get('attacks', 0)} attacks")
        lines.append("set aside: " + ", ".join(f"{k} {tl.get(k, 0)}" for k in SET_ASIDE))
        lines.append(f"slowest games (s): {t.get('slowest')}")
        lines.append("Opportunities: attacks with an admitted rewrite of the class; "
                     "each cell reads (with a fight gain) / (all).")
        lines.append("| combination | " + " | ".join(CLASSES) + " | any class | any, of decisions | rewrites |")
        lines.append("|---" * (len(CLASSES) + 4) + "|")
        for name in REPORT_COMBOS:
            cells = [f"{opp.get(f'{k}*|{name}', 0)}/{opp.get(f'{k}|{name}', 0)}" for k in CLASSES]
            fight_any, any_ = opp.get(f"any*|{name}", 0), opp.get(f"any|{name}", 0)
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {fight_any}/{any_} | "
                         f"{fight_any}/{dec}, {any_}/{dec} | {t['admitted'].get(f'any|{name}', 0)} |")
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
    ap.add_argument("--write-every", type=int, default=50, help="rewrite --out every this many games")
    args = ap.parse_args(argv)
    items: List[Tuple[str, str, str]] = []
    if args.corpus is not None:
        items += corpus_games(args.corpus, args.games, args.offset)
    if args.records:
        items += record_games(args.records)
    if not items:
        ap.error("no games: give --corpus and/or --records")
    if args.outcomes_out is not None and args.outcomes_out.exists():
        args.outcomes_out.unlink()
    t0 = time.time()
    kw = dict(outcomes_out=args.outcomes_out, counts_out=args.out,
              write_every=args.write_every, games_total=len(items))
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            totals = merge(pool.imap_unordered(count_game, items, chunksize=1), **kw)
    else:
        totals = merge(map(count_game, items), **kw)
    print(report(totals))
    print(f"\n{len(items)} games, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
