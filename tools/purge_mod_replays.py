"""Triage Wesnoth replays by mod / add-on usage.

Mods change game rules in ways our sim doesn't model, making
combat math diverge from turn 1. They poison the corpus for
fidelity testing. We sort each replay into one of three buckets:

  - **keep**: vanilla, OR uses only cosmetic mods (no gameplay
    impact). Stays where it is.
  - **set_aside**: uses `plan_unit_advance` (UI-but-gameplay-
    adjacent: lets players preselect advancements that fire
    automatically when the unit levels up off-turn). Vanilla
    1.18 ships an option for this, so the replays are still
    valuable; we just can't reconstruct them until the sim
    learns the planning mechanic. Move to a quarantine dir.
  - **purge**: anything else (Biased RNG, Ageless Era,
    XP_Modification, etc.). Delete.

Files purged or set-aside, per replay:
  - `replays_raw/<date>/<game_id>.bz2`
  - `replays_dataset/<sha1(game_id)[:12]>.json.gz` (if present)
  - The corresponding row in `replays_dataset/index.jsonl`

Detection: a replay uses a mod if its first ~500KB contains either
  - `[modification]` block with `addon_id="..."`, OR
  - top-level `active_mods=...` line with at least one mod.

Usage:
    python tools/purge_mod_replays.py            # dry run, prints stats
    python tools/purge_mod_replays.py --apply    # actually delete/move

Dependencies: stdlib (bz2, hashlib, re, json, pathlib, shutil).
Dependents: standalone CLI.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


_ADDON_RE = re.compile(r'addon_id\s*=\s*"([^"]+)"')
_ACTIVE_MODS_RE = re.compile(
    r'^\s*active_mods\s*=\s*"?([^\n\r"]+)', re.MULTILINE)


# Mods with NO gameplay impact. Replays using only these are safe
# to keep in the main corpus -- combat/movement/recruit math is
# unchanged, our reconstruction still applies.
COSMETIC_MODS: frozenset[str] = frozenset({
    # Player team color modifications.
    "Color_Modification",
    "Rav_Color_Mod",
    # Damage splash / blood overlay (purely visual).
    "Bloody_Mod_PSR",
})


# Mods that change game rules in a small, known way that we may
# eventually emulate. For now, set these aside rather than
# deleting -- they'll become usable when the sim learns the
# corresponding mechanic.
#
# `plan_unit_advance`: when a unit with multiple AMLA/advancement
# options levels up outside its owner's turn, vanilla picks
# randomly; this mod makes it apply the player's pre-selected
# choice instead. Standard 1.18 quality-of-life mod. Future work:
# extend our action schema with a "set_planned_advancement" verb
# and replay it during reconstruction.
SET_ASIDE_MODS: frozenset[str] = frozenset({
    "plan_unit_advance",
})


def _detect_mods(bz2_path: Path) -> list[str]:
    """Return the list of mod ids found in the head of the file.

    Empty list = vanilla replay. Reads only the first 500KB; mod
    blocks live near the top of the file, before the `[replay]`
    command stream.
    """
    try:
        with bz2.open(bz2_path, "rb") as f:
            head = f.read(500_000)
    except Exception:
        return []
    text = head.decode("utf-8", errors="replace")
    found: list[str] = []
    for m in _ADDON_RE.finditer(text):
        found.append(m.group(1))
    am = _ACTIVE_MODS_RE.search(text)
    if am:
        for entry in am.group(1).split(","):
            entry = entry.strip().strip('"')
            if entry:
                found.append(entry)
    return found


def _classify(mods: list[str]) -> str:
    """Return 'keep', 'set_aside', or 'purge' for a replay."""
    if not mods:
        return "keep"
    non_cosmetic = [m for m in mods if m not in COSMETIC_MODS]
    if not non_cosmetic:
        return "keep"
    purgeable = [m for m in non_cosmetic if m not in SET_ASIDE_MODS]
    if not purgeable:
        return "set_aside"
    return "purge"


def _dataset_filename(game_id: str) -> str:
    """Mirror the hashing in `replay_extract.py` line 1052."""
    return hashlib.sha1(game_id.encode()).hexdigest()[:12] + ".json.gz"


def _move_into(src: Path, dst_dir: Path, raw_root: Path) -> None:
    """Move src to dst_dir/<relative-to-raw_root>/, creating parents."""
    rel = src.relative_to(raw_root)
    dst = dst_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("replays_raw"),
                    help="Directory of .bz2 replays.")
    ap.add_argument("--dataset", type=Path, default=Path("replays_dataset"),
                    help="Directory of extracted .json.gz files.")
    ap.add_argument("--set-aside-raw", type=Path,
                    default=Path("replays_raw_set_aside"),
                    help="Quarantine dir for raw .bz2 replays using "
                         "set-aside mods (e.g. plan_unit_advance).")
    ap.add_argument("--set-aside-dataset", type=Path,
                    default=Path("replays_dataset_set_aside"),
                    help="Quarantine dir for extracted .json.gz files "
                         "using set-aside mods.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete/move files (default: dry run).")
    args = ap.parse_args(argv[1:])

    if not args.raw.exists():
        print(f"raw dir not found: {args.raw}", file=sys.stderr)
        return 1

    bz2_files = sorted(args.raw.glob("**/*.bz2"))
    print(f"Scanning {len(bz2_files)} .bz2 files in {args.raw}...")

    mod_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    to_delete_raw: list[Path] = []
    to_delete_dataset: list[Path] = []
    purged_game_ids: set[str] = set()
    to_setaside_raw: list[Path] = []
    to_setaside_dataset: list[Path] = []
    setaside_game_ids: set[str] = set()

    for i, p in enumerate(bz2_files):
        mods = _detect_mods(p)
        for m in mods:
            mod_counter[m] += 1
        bucket = _classify(mods)
        bucket_counter[bucket] += 1

        if bucket == "keep":
            pass  # nothing to do
        elif bucket == "purge":
            to_delete_raw.append(p)
            gid = p.stem
            purged_game_ids.add(gid)
            ds = args.dataset / _dataset_filename(gid)
            if ds.exists():
                to_delete_dataset.append(ds)
        elif bucket == "set_aside":
            to_setaside_raw.append(p)
            gid = p.stem
            setaside_game_ids.add(gid)
            ds = args.dataset / _dataset_filename(gid)
            if ds.exists():
                to_setaside_dataset.append(ds)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(bz2_files)} scanned ...")

    total = len(bz2_files)
    print()
    print(f"Bucket totals (raw):")
    for b in ("keep", "set_aside", "purge"):
        n = bucket_counter[b]
        print(f"  {b:>10s}  {n:7d}  ({100*n/max(1,total):5.1f}%)")
    print()
    print(f"Dataset entries to delete: {len(to_delete_dataset)}")
    print(f"Dataset entries to set-aside: {len(to_setaside_dataset)}")
    print()
    print(f"Top 30 mods seen:")
    for k, v in mod_counter.most_common(30):
        print(f"  {v:7d}  {k}")

    # Index reconciliation: rows for purged/set-aside game_ids
    # leave the main index. Set-aside rows go to a separate index
    # in the quarantine dataset dir.
    idx_path = args.dataset / "index.jsonl"
    setaside_idx_path = args.set_aside_dataset / "index.jsonl"
    keep_idx: list[str] = []
    setaside_idx: list[str] = []
    dropped_idx = 0
    moved_idx = 0
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    keep_idx.append(line)
                    continue
                gid = rec.get("game_id")
                if gid in purged_game_ids:
                    dropped_idx += 1
                elif gid in setaside_game_ids:
                    setaside_idx.append(line)
                    moved_idx += 1
                else:
                    keep_idx.append(line)
        print(f"\nIndex rows to drop: {dropped_idx}")
        print(f"Index rows to move (set-aside): {moved_idx}")

    if not args.apply:
        print("\n[dry run] re-run with --apply to actually delete/move.")
        return 0

    # Delete purged.
    for p in to_delete_raw:
        try:
            p.unlink()
        except Exception as e:
            print(f"  delete failed: {p}: {e}", file=sys.stderr)
    for p in to_delete_dataset:
        try:
            p.unlink()
        except Exception as e:
            print(f"  delete failed: {p}: {e}", file=sys.stderr)

    # Move set-aside.
    args.set_aside_raw.mkdir(parents=True, exist_ok=True)
    args.set_aside_dataset.mkdir(parents=True, exist_ok=True)
    for p in to_setaside_raw:
        try:
            _move_into(p, args.set_aside_raw, args.raw)
        except Exception as e:
            print(f"  move failed: {p}: {e}", file=sys.stderr)
    for p in to_setaside_dataset:
        try:
            shutil.move(str(p), str(args.set_aside_dataset / p.name))
        except Exception as e:
            print(f"  move failed: {p}: {e}", file=sys.stderr)

    # Rewrite indexes.
    if idx_path.exists() and (dropped_idx or moved_idx):
        idx_path.write_text("\n".join(keep_idx) + ("\n" if keep_idx else ""),
                            encoding="utf-8")
    if setaside_idx:
        # Append-or-create the set-aside index.
        existing = ""
        if setaside_idx_path.exists():
            existing = setaside_idx_path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
        setaside_idx_path.write_text(
            existing + "\n".join(setaside_idx) + "\n", encoding="utf-8")

    # Drop now-empty date subdirectories under raw.
    for sub in sorted(args.raw.glob("*"), reverse=True):
        if sub.is_dir():
            try:
                if not any(sub.iterdir()):
                    sub.rmdir()
            except Exception:
                pass

    print(f"\nDeleted: {len(to_delete_raw)} raw, "
          f"{len(to_delete_dataset)} dataset, {dropped_idx} index rows.")
    print(f"Set-aside: {len(to_setaside_raw)} raw, "
          f"{len(to_setaside_dataset)} dataset, {moved_idx} index rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
