#!/usr/bin/env python3
"""What the imitation corpus actually contains: era, map layout, board
size, factions and the host's rule settings, read from the raw replay
headers rather than assumed.

Written 2026-09-21 to answer four questions that the manifest cannot:

  * is the corpus one era?  29% of the games declare `era_dunefolk`
    rather than `era_default`, but that era is `{ERA_DEFAULT}` plus one
    extra faction file, and no game in the corpus fields a Dunefolk
    side, so the play is default-era throughout.
  * does one scenario name cover several layouts (a map picker, or a
    ladder variant of a mainline map)?  Each name resolves to exactly
    one layout hash, and every mainline-named map is byte-identical to
    the shipped 1.18.x map file.
  * which games were played under non-default rules?  The host can set
    the experience modifier, the village gold and a random time-of-day
    start; those cluster in the mini pack.
  * how are the games split between the mainline whitelist the
    evaluation pool uses and everything else?

    python tools/analysis/corpus_census.py --dataset replays_dataset_imitation \\
        --out training/metrics/corpus_census.json [--workers 8] [--limit N]

Writes one JSON row per game plus the summary the report prints, so a
later question about the corpus is a read rather than another scan.
"""
from __future__ import annotations

import argparse
import bz2
import collections
import gzip
import hashlib
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from wesnoth_ai.rules.scenario_pool import LADDER_SCENARIO_IDS  # noqa: E402

# The map name is the replay file name minus its date prefix and its
# turn/command suffix: "2024-03-22_2p__Hamlets_Turn_14_(335).json.gz".
NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_(.*)_Turn_\d+_\(\d+\)\.json\.gz$")
MAP_RE = re.compile(r'^\s*map_data\s*=\s*"(.*?)"\s*$', re.M | re.S)
FACTION_RE = re.compile(r'^\s*faction\s*=\s*"?([^"\r\n]*)"?\s*$', re.M)
MOD_ID_RE = re.compile(r'^\s*id\s*=\s*"?([^"\r\n]*)"?\s*$', re.M)
SETTING_KEYS = ("era_id", "experience_modifier", "random_start_time",
                "mp_village_gold", "mp_use_map_settings")
SETTING_RE = {k: re.compile(rf'^\s*{k}\s*=\s*"?([^"\r\n]*)"?\s*$', re.M)
              for k in SETTING_KEYS}
# The host's game-creation defaults for a 1v1; anything else is a
# non-default rule set, not an engine default (see scenario_pool).
DEFAULT_SETTINGS = {"experience_modifier": "70", "random_start_time": "no",
                    "mp_village_gold": "2", "mp_use_map_settings": "yes"}
# Mini and micro boards: tiny add-on maps whose games are 8-turn
# skirmishes at a third of the decision density of a mainline map.
MINI_MAPS = frozenset({"2p_mini_edited", "2p_-_Micro_Isar", "2p_mini", "around_mini",
                       "Modified_Close_Relation_(2p)", "2p_-_Mini_Fallenstar"})
HEAD_BYTES = 400_000          # the starting snapshot, never the command stream


def layout_hash(map_data: str) -> Tuple[str, int, int]:
    """(hash, rows, columns) of a map_data grid, whitespace-insensitive
    so the inline and on-disk spellings of one map agree."""
    rows = [r for r in (x.strip() for x in map_data.splitlines())
            if r and "=" not in r]
    digest = hashlib.sha1("\n".join(rows).replace(" ", "").encode()).hexdigest()
    return digest[:12], len(rows), (len(rows[0].split(",")) if rows else 0)


def pack_of(map_name: str) -> str:
    """Which pack a map belongs to: `ladder` (the whitelist the
    evaluation pool plays), `mainline` (a mainline map outside it),
    `mini`, or `other`."""
    if map_name in MINI_MAPS:
        return "mini"
    if not map_name.startswith("2p__"):
        return "other"
    return "ladder" if _ladder_names().get(map_name) else "mainline"


_LADDER_NAMES: Optional[Dict[str, bool]] = None


def _ladder_names() -> Dict[str, bool]:
    """Corpus map names that belong to the whitelist, matched on the
    scenario id's tail (`multiplayer_Basilisk` covers
    `2p__Caves_of_the_Basilisk`), so a whitelist edit reaches here."""
    global _LADDER_NAMES
    if _LADDER_NAMES is None:
        keys = set()
        for sid in LADDER_SCENARIO_IDS:
            keys.add(sid.removeprefix("multiplayer_").lower().replace("_", ""))
        _LADDER_NAMES = {}
        for name in _CORPUS_NAMES:
            flat = name.removeprefix("2p__").lower().replace("_", "")
            _LADDER_NAMES[name] = any(k in flat or flat in k for k in keys)
    return _LADDER_NAMES


_CORPUS_NAMES: Sequence[str] = ()


def scan_one(job: Tuple[str, str]) -> dict:
    """One replay's header: era, layout, factions and host settings."""
    rel, game_file = job
    out: dict = {"file": game_file, "map": NAME_RE.match(game_file).group(1)}
    try:
        with bz2.open(ROOT / rel.replace("\\", "/"), "rt",
                      encoding="utf-8", errors="replace") as fh:
            head = fh.read(HEAD_BYTES)
    except Exception as exc:                            # noqa: BLE001
        out["error"] = type(exc).__name__
        return out
    for key, rx in SETTING_RE.items():
        m = rx.search(head)
        out[key] = m.group(1) if m else None
    m = MAP_RE.search(head)
    if m:
        out["layout"], out["rows"], out["cols"] = layout_hash(m.group(1))
    out["factions"] = FACTION_RE.findall(head)[:2]
    out["schedule"] = read_schedule(head)
    out["mods"] = sorted({MOD_ID_RE.search(b.group(1)).group(1)
                          for b in re.finditer(r"\[modification\](.*?)\[/modification\]",
                                               head, re.S)
                          if MOD_ID_RE.search(b.group(1))})
    return out


# The board's time-of-day schedule, as ids and lawful bonuses. The sim
# reads the cycle off a hardcoded six-slot constant
# (`combat.TOD_DEFAULT_CYCLE`), so whether that is an assumption or a
# fact about the corpus is a question only the corpus can answer.
_TIME_BLOCK_RE = re.compile(r"\[time\](.*?)\[/time\]", re.S)
_TIME_ID_RE = re.compile(r"^\s*id\s*=\s*\"?([\w]+)", re.M)
_TIME_BONUS_RE = re.compile(r"^\s*lawful_bonus\s*=\s*\"?(-?\d+)", re.M)
DEFAULT_SCHEDULE = (("dawn", 0), ("morning", 25), ("afternoon", 25),
                    ("dusk", 0), ("first_watch", -25), ("second_watch", -25))


# A replay header can carry the scenario TWICE -- [replay_start] and
# [snapshot] both hold it on 50 of 1,200 sampled games -- so a search
# over the whole head sees each [time] block twice and every schedule
# reads as a 12-slot one. Reconstruction takes the first of
# replay_start / snapshot / scenario; so does this.
_CONTAINER_RE = re.compile(r"\[(?:replay_start|snapshot|scenario)\]")


def first_container(head: str) -> str:
    first = _CONTAINER_RE.search(head)
    if first is None:
        return head
    nxt = _CONTAINER_RE.search(head, first.end())
    return head[first.end():nxt.start() if nxt else len(head)]


def read_schedule(head: str):
    """The TOP-LEVEL [time] blocks of the first scenario container, as
    (id, lawful_bonus) pairs.

    [time_area] sub-schedules are excluded: they are a zone's cycle,
    not the board's, and the sim reads them separately. They are
    dropped by removing every [time_area] block before the search,
    which is what the engine's tod_manager ctor effectively does by
    reading its own child list.
    """
    board = re.sub(r"\[time_area\].*?\[/time_area\]", "",
                   first_container(head), flags=re.S)
    out = []
    for block in _TIME_BLOCK_RE.findall(board):
        ident = _TIME_ID_RE.search(block)
        bonus = _TIME_BONUS_RE.search(block)
        out.append((ident.group(1) if ident else "?",
                    int(bonus.group(1)) if bonus else 0))
    return out


def non_default(row: dict) -> List[str]:
    """The host settings this game changed from the 1v1 defaults."""
    return [k for k, want in DEFAULT_SETTINGS.items()
            if row.get(k) is not None and row[k] != want]


def summarize(rows: Sequence[dict]) -> dict:
    """Counts worth reading back later; the per-game rows carry the rest."""
    layouts: Dict[str, set] = collections.defaultdict(set)
    packs: collections.Counter = collections.Counter()
    settings: Dict[str, collections.Counter] = {k: collections.Counter()
                                                for k in SETTING_KEYS}
    factions: collections.Counter = collections.Counter()
    odd: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    schedules: collections.Counter = collections.Counter()
    odd_schedule_maps: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for r in rows:
        pack = pack_of(r["map"])
        packs[pack] += 1
        sched = tuple(tuple(t) for t in (r.get("schedule") or ()))
        schedules[sched] += 1
        if sched and sched != DEFAULT_SCHEDULE:
            odd_schedule_maps[pack][r["map"]] += 1
        if r.get("layout"):
            layouts[r["map"]].add(r["layout"])
        for k in SETTING_KEYS:
            settings[k][r.get(k)] += 1
        factions.update(r.get("factions") or [])
        changed = non_default(r)
        for k in changed:
            odd[pack][k] += 1
        if changed:
            odd[pack]["any"] += 1
    return {
        "games": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "packs": dict(packs),
        "maps": {name: {"games": sum(1 for r in rows if r["map"] == name),
                        "layouts": len(hs), "pack": pack_of(name)}
                 for name, hs in sorted(layouts.items())},
        "multi_layout_maps": {n: len(h) for n, h in layouts.items() if len(h) > 1},
        "settings": {k: dict(v.most_common()) for k, v in settings.items()},
        "factions": dict(factions.most_common()),
        "non_default_by_pack": {p: dict(c) for p, c in odd.items()},
        # The board schedule, keyed by its ids joined with "|" so the
        # summary stays JSON. `default` is the six-slot cycle the sim
        # hardcodes; anything else means a game whose time of day our
        # cycle cannot express.
        "schedules": {
            ("default" if k == DEFAULT_SCHEDULE else
             ("none" if not k else "|".join(f"{i}:{b}" for i, b in k))): n
            for k, n in schedules.most_common()},
        "non_default_schedule_maps": {p: dict(c)
                                      for p, c in odd_schedule_maps.items()},
    }


def render(s: dict) -> str:
    lines = [f"{s['games']} games, {s['errors']} unreadable",
             f"packs: {s['packs']}",
             f"maps with more than one layout: {s['multi_layout_maps'] or 'none'}",
             f"eras: {s['settings']['era_id']}",
             f"board schedules: "
             f"{ {k: v for k, v in list(s.get('schedules', {}).items())[:4]} }",
             f"factions: {s['factions']}",
             "",
             f"{'pack':>10}{'games':>8}{'non-default rules':>19}  which"]
    for pack, n in sorted(s["packs"].items(), key=lambda kv: -kv[1]):
        odd = s["non_default_by_pack"].get(pack, {})
        which = ", ".join(f"{k} {v}" for k, v in sorted(odd.items()) if k != "any")
        lines.append(f"{pack:>10}{n:>8}{odd.get('any', 0):>19}  {which or '-'}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "replays_dataset_imitation")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON with the summary (small enough to commit)")
    ap.add_argument("--rows", type=Path, default=None,
                    help="gzipped JSON with one row per game (about 6 MB "
                         "raw for the full corpus, so kept out of the "
                         "summary and out of git)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="an evenly strided sample of N games (0 = all)")
    args = ap.parse_args(argv)

    manifest = [json.loads(line) for line in
                open(args.dataset / "manifest.jsonl", encoding="utf-8")]
    if args.limit:
        manifest = manifest[::max(1, len(manifest) // args.limit)][:args.limit]
    global _CORPUS_NAMES, _LADDER_NAMES
    _CORPUS_NAMES = sorted({NAME_RE.match(r["file"]).group(1) for r in manifest})
    _LADDER_NAMES = None          # the pack map is derived from the names above
    jobs = [(r["source"], r["file"]) for r in manifest]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = pool.map(scan_one, jobs, chunksize=32)
    else:
        rows = [scan_one(j) for j in jobs]
    summary = summarize(rows)
    print(render(summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary}, indent=1),
                            encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.rows:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.rows, "wt", encoding="utf-8") as fh:
            json.dump({"summary": summary, "games": rows}, fh)
        print(f"wrote {args.rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
