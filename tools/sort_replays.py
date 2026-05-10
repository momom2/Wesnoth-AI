"""Re-extract replays_raw/* and classify into competitive vs quarantine
buckets. One-shot tool used to refresh `replays_dataset/` after the
multi-block-concat fix (2026-05-08) plus the trait/scenario-event
fidelity sweep (2026-05-04).

Buckets (in priority order; first match wins):

  1. **modded/<mod_id>/**        — replay carries any `[modification]
                                    addon_id="..."` or `active_mods=`
                                    entry. Quarantined regardless of
                                    whether the mod is gameplay-
                                    affecting; cosmetic mods land in
                                    their own folders too. Multi-mod
                                    replays go under the FIRST listed
                                    mod (label is somewhat arbitrary
                                    in that case).
  2. **non_2p/**                  — fewer or more than 2 default-era
                                    player factions. A 3rd side with
                                    a non-player faction (Custom,
                                    scenery, etc.) is OK — the rule
                                    is `len(players) == 2 and
                                    len(non_players) <= 1`.
  3. **pve_2p/**                  — 2p but on a vanilla PvE scenario
                                    (Dark Forecast, Isle of Mists).
  4. **non_vanilla_map/**         — 2p but scenario_id isn't in any
                                    vanilla 2p .cfg (i.e. add-on map
                                    or unknown).
  5. **competitive_2p/**          — 2p competitive on a vanilla
                                    competitive map. The training
                                    corpus.

Output layout:
  <out_root>/competitive_2p/<hash>.json.gz
  <out_root>/competitive_2p/index.jsonl
  <out_root>/non_2p/...
  <out_root>/pve_2p/...
  <out_root>/non_vanilla_map/...
  <out_root>/modded/<mod_id>/...

Per-bucket index.jsonl carries the same fields as the existing
`replays_dataset/index.jsonl` plus `mods` (list of mod ids
detected at extraction time) and `bucket` (the assignment).

Parallelization: multiprocessing.Pool of N workers (default 4).
Each worker imports `tools.replay_extract` once (~200 MB), then
processes a chunk of bz2 paths. For ~46k inputs at ~0.08s each,
expect 8-15 min wall on 4 workers.

Dependencies: tools.replay_extract, tools.scenarios,
              tools.purge_mod_replays
Dependents: standalone CLI.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on sys.path so absolute imports work whether this
# is run as `python tools/sort_replays.py` or `python -m
# tools.sort_replays`.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))


# Default-era player factions. The 2p check is "exactly two of these
# present, plus at most one non-player faction (cosmetic 3rd side)."
PLAYER_FACTIONS: frozenset[str] = frozenset({
    "Drakes", "Knalgan Alliance", "Loyalists",
    "Northerners", "Rebels", "Undead",
})


_ADDON_RE = re.compile(r'addon_id\s*=\s*"([^"]+)"')
_ACTIVE_MODS_RE = re.compile(
    r'^\s*active_mods\s*=\s*"?([^\n\r"]+)', re.MULTILINE)


# Vanilla PvE scenarios shipped as 2p_*.cfg but objectively non-
# competitive. Bucket 'pve_2p' rather than 'competitive_2p'.
_PVE_2P_IDS: frozenset[str] = frozenset({
    "multiplayer_2p_Dark_Forecast",
    "multiplayer_2p_Isle_of_Mists",
})


log = logging.getLogger("sort_replays")


def _detect_mods(bz2_path: Path) -> List[str]:
    """List of mod ids found in the head of the file. Empty = vanilla.
    Reads only the first 500KB; mod blocks live near the top of the
    file before [replay]. Mirrors `tools/purge_mod_replays._detect_mods`."""
    try:
        with bz2.open(bz2_path, "rb") as f:
            head = f.read(500_000)
    except Exception:
        return []
    text = head.decode("utf-8", errors="replace")
    found: List[str] = []
    for m in _ADDON_RE.finditer(text):
        found.append(m.group(1))
    am = _ACTIVE_MODS_RE.search(text)
    if am:
        for entry in am.group(1).split(","):
            entry = entry.strip().strip('"')
            if entry:
                found.append(entry)
    # Dedup while preserving order (so the FIRST listed mod is the
    # primary bucket label).
    seen = set()
    out: List[str] = []
    for m in found:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _classify(
    rec: Optional[Dict],
    mods: List[str],
    competitive_set: frozenset,
) -> Tuple[str, str]:
    """Return (bucket, sub_label).

    Priority:
      1. If mods present → ('modded', mods[0])
      2. If extraction failed or rec missing scenario/factions →
         ('extract_failed', '')
      3. If not 2p (factions check) → ('non_2p', '')
      4. If scenario in PvE list → ('pve_2p', '')
      5. If scenario NOT in vanilla competitive list → ('non_vanilla_map', '')
      6. Else → ('competitive_2p', '')
    """
    if mods:
        return ("modded", mods[0])
    if rec is None:
        return ("extract_failed", "")
    factions = rec.get("factions", []) or []
    players = [f for f in factions if f in PLAYER_FACTIONS]
    non_players = [f for f in factions if f not in PLAYER_FACTIONS]
    if len(players) != 2 or len(non_players) > 1:
        return ("non_2p", "")
    sid = rec.get("scenario_id", "")
    if sid in _PVE_2P_IDS:
        return ("pve_2p", "")
    if sid not in competitive_set:
        return ("non_vanilla_map", "")
    return ("competitive_2p", "")


def _safe_subdir_name(name: str) -> str:
    """Filesystem-safe label for a mod id. Mod ids can contain spaces,
    parens, slashes, etc. Replace anything not alnum/dash/dot/underscore
    with `_` and collapse runs."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    if not out:
        out = "unnamed"
    return out[:80]   # cap length to keep paths reasonable


def _process_one(args: Tuple[str, str]) -> Dict:
    """Worker fn: extract one bz2 + classify. Returns a small dict
    with everything the main process needs to write the file +
    update the index. Intentionally returns no parsed game state to
    keep IPC small."""
    bz2_path_str, raw_root_str = args
    # Lazy imports inside the worker (so the parent doesn't pay for
    # all of them, and so the worker process loads them once at
    # first call rather than at fork time on Windows where workers
    # are spawn).
    from tools.replay_extract import extract_replay
    from tools.scenarios import COMPETITIVE_2P_SCENARIOS

    bz2_path = Path(bz2_path_str)
    raw_root = Path(raw_root_str)

    mods = _detect_mods(bz2_path)
    rec: Optional[Dict] = None
    err_msg = ""
    try:
        rec = extract_replay(bz2_path)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:160]}"

    bucket, sub = _classify(rec, mods, COMPETITIVE_2P_SCENARIOS)

    # Hash by stem (game_id) so the dataset filename is stable.
    game_id = bz2_path.stem
    hash_name = hashlib.sha1(game_id.encode("utf-8")).hexdigest()[:12]
    out_filename = f"{hash_name}.json.gz"

    # Prep the JSON record + meta. We DO write a .json.gz for
    # extract-failed inputs (empty body) so the index records them
    # — caller can decide later whether to re-try.
    payload: Optional[bytes] = None
    if rec is not None:
        # Stamp the detected mods onto the dict so downstream
        # consumers don't have to re-scan the bz2.
        rec["mods"] = mods
        try:
            payload = json.dumps(rec, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as e:
            err_msg = f"json: {type(e).__name__}: {str(e)[:160]}"
            payload = None

    return {
        "bz2_path":     str(bz2_path),
        "rel_to_raw":   str(bz2_path.relative_to(raw_root)),
        "game_id":      game_id,
        "hash_name":    hash_name,
        "out_filename": out_filename,
        "mods":         mods,
        "bucket":       bucket,
        "sub":          sub,
        "scenario_id":  (rec or {}).get("scenario_id", ""),
        "factions":     (rec or {}).get("factions", []),
        "n_commands":   len((rec or {}).get("commands", [])),
        "payload":      payload,
        "err":          err_msg,
    }


def _bucket_dir(out_root: Path, bucket: str, sub: str) -> Path:
    if bucket == "modded":
        return out_root / "modded" / _safe_subdir_name(sub or "unnamed")
    return out_root / bucket


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, action="append", required=False,
                    help="Root directory of source .bz2 replays "
                         "(repeatable). Defaults to ['replays_raw'].")
    ap.add_argument("--out", type=Path, default=Path("replays_dataset_new"),
                    help="Output root (created if absent).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Pool size. CPU-bound; 4-8 is reasonable.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap inputs (for testing).")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args(argv[1:])

    if not args.raw:
        args.raw = [Path("replays_raw")]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    # Walk each raw root, collect (bz2_path, raw_root_for_relative) pairs.
    inputs: List[Tuple[str, str]] = []
    for root in args.raw:
        if not root.exists():
            log.warning("skip missing root: %s", root)
            continue
        for p in sorted(root.glob("**/*.bz2")):
            inputs.append((str(p), str(root)))
    if args.limit:
        inputs = inputs[:args.limit]
    log.info("inputs: %d .bz2 files across %d roots",
             len(inputs), len(args.raw))
    if not inputs:
        log.error("no inputs found")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    # Per-bucket index records collected in the main process.
    bucket_records: Dict[Tuple[str, str], List[dict]] = {}
    bucket_counts: Counter = Counter()
    error_log: List[Tuple[str, str]] = []
    n_processed = 0
    n_written = 0
    t0 = time.perf_counter()

    # Pool. chunksize small enough to load-balance, large enough to
    # amortize IPC. ~50 per worker is a reasonable default.
    chunksize = max(1, len(inputs) // (args.workers * 50))
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_process_one, inputs, chunksize=chunksize):
            n_processed += 1
            bucket = r["bucket"]
            sub = r["sub"]
            bucket_counts[(bucket, sub)] += 1

            if r["err"]:
                error_log.append((r["bz2_path"], r["err"]))

            # Write the .json.gz if we have a payload (i.e. extract
            # didn't return None and didn't raise).
            if r["payload"] is not None:
                dst_dir = _bucket_dir(args.out, bucket, sub)
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / r["out_filename"]
                try:
                    with gzip.open(dst, "wb") as f:
                        f.write(r["payload"])
                    n_written += 1
                    rec_dict = {
                        "file":         r["out_filename"],
                        "game_id":      r["game_id"],
                        "scenario_id":  r["scenario_id"],
                        "factions":     r["factions"],
                        "n_commands":   r["n_commands"],
                        "mods":         r["mods"],
                        "bucket":       bucket,
                    }
                    if bucket == "modded":
                        rec_dict["mod_label"] = sub
                    bucket_records.setdefault((bucket, sub), []).append(rec_dict)
                except OSError as e:
                    error_log.append((r["bz2_path"], f"write: {e}"))

            if n_processed % args.log_every == 0:
                dt = time.perf_counter() - t0
                rate = n_processed / max(1e-3, dt)
                eta = (len(inputs) - n_processed) / max(1e-3, rate)
                log.info(
                    f"  {n_processed}/{len(inputs)} "
                    f"({rate:.1f}/s, ETA {eta:.0f}s, written {n_written})")

    # Write per-bucket index.jsonl.
    for (bucket, sub), recs in bucket_records.items():
        d = _bucket_dir(args.out, bucket, sub)
        idx = d / "index.jsonl"
        with idx.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    # And a top-level errors.txt if any.
    if error_log:
        with (args.out / "errors.txt").open("w", encoding="utf-8") as f:
            for path, msg in error_log:
                f.write(f"{path}\t{msg}\n")

    dt = time.perf_counter() - t0
    log.info("=" * 60)
    log.info(f"sort_replays: {n_processed} processed in {dt:.0f}s "
             f"({n_processed/dt:.1f}/s), {n_written} written, "
             f"{len(error_log)} errors")
    log.info("Bucket distribution:")
    for (bucket, sub), n in bucket_counts.most_common(50):
        label = f"{bucket}/{sub}" if sub else bucket
        log.info(f"  {n:>6d}  {label}")
    if len(bucket_counts) > 50:
        log.info(f"  ... ({len(bucket_counts) - 50} more buckets, "
                 f"see per-bucket index.jsonl)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
