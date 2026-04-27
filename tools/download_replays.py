"""Bulk-download replays.wesnoth.org/1.18/ by date range.

Usage:
    python tools/download_replays.py 2026-04-17 2026-04-23  # inclusive dates

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
        print(f"[!] {url} → {e}")
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


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download_replays.py START_YYYY-MM-DD END_YYYY-MM-DD")
        return 2
    start = date.fromisoformat(sys.argv[1])
    end   = date.fromisoformat(sys.argv[2])
    if end < start:
        print("END must be >= START"); return 2

    days: list[date] = []
    d = start
    while d <= end:
        days.append(d); d += timedelta(days=1)

    print(f"Covering {len(days)} days: {start} .. {end}")
    all_jobs: list[tuple[date, str]] = []
    for d in days:
        names = list_day(d)
        print(f"  {d}: {len(names)} replays listed")
        all_jobs.extend((d, n) for n in names)

    print(f"Total files to fetch (pre-dedupe): {len(all_jobs)}")
    ok = skipped = err = total_bytes = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(fetch_one, d, n) for d, n in all_jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, size, status = f.result()
            if status == "ok":
                ok += 1; total_bytes += size
            elif status == "skip":
                skipped += 1; total_bytes += size
            else:
                err += 1
            if i % 200 == 0:
                print(f"  [{i}/{len(futs)}] ok={ok} skip={skipped} err={err}")

    print(f"\nDone. ok={ok} skipped={skipped} err={err} total={total_bytes/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
