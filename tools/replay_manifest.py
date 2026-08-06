"""Light manifest scan of a raw replay corpus (2026-08-06).

Stage 0 of the staged replay-bank filter (user plan 2026-08-06:
map whitelist -> default era -> mod split). Streams the first 400KB
of each .bz2 (all identity fields live in the save header, verified
on real files: mp_scenario=, mp_era=/era_id=, active_mods=,
[modification] addon_id=, version=, faction_name=) and emits one
JSONL row per file. The stage filters are then queries over the
manifest -- no 472k-file directory moves, and full extraction runs
only on survivors.

Usage:
    python tools/replay_manifest.py replays_raw manifest.jsonl.gz
        [--workers 10] [--limit N]

~5-10ms/file (bz2 partial decompress dominates); 472k files on 10
workers ~= 10-15 min.
"""
from __future__ import annotations

import bz2
import gzip
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_HEAD_BYTES = 400_000

_RE = {
    "scenario": re.compile(r'^\s*mp_scenario\s*=\s*"([^"]*)"', re.M),
    "scenario_alt": re.compile(r'^\s*\[replay_start\]\s*$.*?^\s*id\s*=\s*"([^"]*)"',
                               re.M | re.S),
    "era": re.compile(r'^\s*mp_era\s*=\s*"([^"]*)"', re.M),
    "era_alt": re.compile(r'^\s*era_id\s*=\s*"([^"]*)"', re.M),
    "version": re.compile(r'^\s*version\s*=\s*"([^"]*)"', re.M),
    "active_mods": re.compile(r'^\s*active_mods\s*=\s*"([^"]*)"', re.M),
    "addon_id": re.compile(r'addon_id\s*=\s*"([^"]+)"'),
    "faction": re.compile(r'^\s*faction_name\s*=\s*"([^"]*)"', re.M),
    "controller": re.compile(r'^\s*controller\s*=\s*"([^"]*)"', re.M),
}


def scan_one(path_str: str) -> dict:
    p = Path(path_str)
    row = {"path": path_str, "ok": False}
    try:
        with bz2.open(p, "rb") as f:
            head = f.read(_HEAD_BYTES).decode("utf-8", errors="replace")
    except Exception as e:                          # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:120]
        return row
    m = _RE["scenario"].search(head) or _RE["scenario_alt"].search(head)
    row["scenario"] = m.group(1) if m else ""
    m = _RE["era"].search(head) or _RE["era_alt"].search(head)
    row["era"] = m.group(1) if m else ""
    m = _RE["version"].search(head)
    row["version"] = m.group(1) if m else ""
    mods = []
    for am in _RE["active_mods"].finditer(head):
        for entry in am.group(1).split(","):
            entry = entry.strip().strip('"')
            if entry and entry not in mods:
                mods.append(entry)
    for ad in _RE["addon_id"].finditer(head):
        if ad.group(1) not in mods:
            mods.append(ad.group(1))
    row["mods"] = mods
    row["factions"] = _RE["faction"].findall(head)
    row["controllers"] = _RE["controller"].findall(head)
    row["ok"] = bool(row["scenario"] or row["era"])
    return row


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv[1:])

    files = sorted(str(p) for p in args.raw_root.rglob("*.bz2"))
    if args.limit:
        files = files[:args.limit]
    print(f"scanning {len(files)} files on {args.workers} workers",
          flush=True)
    t0 = time.time()
    n_ok = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out, \
            Pool(args.workers) as pool:
        for i, row in enumerate(
                pool.imap_unordered(scan_one, files, chunksize=200), 1):
            out.write(json.dumps(row) + "\n")
            n_ok += row["ok"]
            if i % 20000 == 0:
                rate = i / (time.time() - t0)
                print(f"  [{i}/{len(files)}] ok={n_ok} "
                      f"{rate:.0f}/s eta={int((len(files)-i)/rate)}s",
                      flush=True)
    print(f"Done. {n_ok}/{len(files)} parsed ok -> {args.out} "
          f"in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
