"""Bulk-download replays.wesnoth.org/1.18/ by date range.

Usage:
    python tools/download_replays.py 2026-04-17 2026-04-23  # inclusive dates
    python tools/download_replays.py START END --filter-maps configs/replay_download_maps.txt

--filter-maps FILE: pre-select at LISTING time by map name. FILE holds
one normalized token per line (lowercase alnum of the scenario title,
e.g. "2pdenofonis"); a listed .bz2 is fetched iff its normalized
filename starts with a token. Replay filenames are the GAME TITLE
(default = scenario title), so this keeps default-titled games on the
wanted maps and skips everything else BEFORE download -- custom-titled
or localized-title games on wanted maps are lost (recall trade-off,
accepted 2026-08-06; skip counts are printed per day so the loss is
visible). Mods are invisible at filename level: run
tools/sort_replays.py after download for the mod/era quarantine.

Output layout: replays_raw/YYYY-MM-DD/<filename>.bz2

Designed as a one-shot spike for the supervised-bootstrapping plan. Hits
the Apache directory listing, extracts .bz2 hrefs, downloads them in
parallel. Skips files already present so re-runs are cheap.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from datetime import date, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://replays.wesnoth.org/1.18"
USER_AGENT = "Wesnoth-AI-research-spike/0.1 (+local)"
OUT_DIR = Path(__file__).resolve().parents[1] / "replays_raw"


def list_day(d: date) -> list[str]:
    """Return list of .bz2 filenames for a given date."""
    url = f"{BASE}/{d:%Y/%m/%d}/"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError) as e:
        # ASCII only: a piped stdout on Windows encodes cp1252, and a
        # non-ASCII arrow here crashed the whole bulk run at the first
        # missing day (2026-07-07: the entire 2024 chunk died on the
        # pre-archive date 2024-03-18).
        print(f"[!] {url} -> {e}")
        return []
    # Apache listing: <a href="Foo_Turn_3_(123).bz2">Foo_Turn_3_(123).bz2</a>
    hrefs = re.findall(r'href="([^"?]+\.bz2)"', html)
    return sorted(set(hrefs))


def fetch_one(d: date, name: str) -> tuple[str, int, str]:
    """Download one file. Returns (name, bytes, status)."""
    target_dir = OUT_DIR / d.isoformat()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if target.exists() and target.stat().st_size > 0:
        return (name, target.stat().st_size, "skip")
    url = f"{BASE}/{d:%Y/%m/%d}/{name}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=60) as r:
            data = r.read()
        target.write_bytes(data)
        return (name, len(data), "ok")
    except (URLError, HTTPError) as e:
        return (name, 0, f"err: {e}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main() -> int:
    args = list(sys.argv[1:])
    tokens: list[str] = []
    if "--filter-maps" in args:
        i = args.index("--filter-maps")
        tokens = [_norm(line) for line in
                  Path(args[i + 1]).read_text(encoding="utf-8").split()
                  if line.strip()]
        del args[i:i + 2]
    if len(args) != 2:
        print("usage: download_replays.py START_YYYY-MM-DD END_YYYY-MM-DD"
              " [--filter-maps FILE]")
        return 2
    start = date.fromisoformat(args[0])
    end   = date.fromisoformat(args[1])
    if end < start:
        print("END must be >= START")
        return 2

    days: list[date] = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)

    print(f"Covering {len(days)} days: {start} .. {end}")
    all_jobs: list[tuple[date, str]] = []
    tot_listed = tot_kept = 0
    for d in days:
        names = list_day(d)
        if tokens:
            kept = [n for n in names
                    if any(_norm(n).startswith(tk) for tk in tokens)]
        else:
            kept = names
        tot_listed += len(names)
        tot_kept += len(kept)
        print(f"  {d}: {len(names)} listed, {len(kept)} kept")
        all_jobs.extend((d, n) for n in kept)
    if tokens:
        print(f"Filter recall: kept {tot_kept}/{tot_listed} listed files")

    print(f"Total files to fetch (pre-dedupe): {len(all_jobs)}")
    ok = skipped = err = total_bytes = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(fetch_one, d, n) for d, n in all_jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, size, status = f.result()
            if status == "ok":
                ok += 1
                total_bytes += size
            elif status == "skip":
                skipped += 1
                total_bytes += size
            else:
                err += 1
            if i % 200 == 0:
                print(f"  [{i}/{len(futs)}] ok={ok} skip={skipped} err={err}")

    print(f"\nDone. ok={ok} skipped={skipped} err={err} total={total_bytes/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
