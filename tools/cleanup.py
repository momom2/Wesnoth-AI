"""Comprehensive training-tree cleanup.

Three jobs in one script:

  1. ARCHIVE the rolling self-play checkpoint into a timestamped
     snapshot whenever the freshest existing snapshot is older than
     `--archive-stride-hours`. This is how we get a history -- the
     training loop otherwise just overwrites `sim_selfplay.pt` every
     `--save-every` iters and prior states are lost.

  2. APPLY TIERED RETENTION to those snapshots. The retention curve
     is "denser near the present, sparser further back": hourly for
     the last day, daily for the last week, weekly for the last
     month, monthly thereafter. Within each bucket only the freshest
     file survives. The very newest snapshot is always kept (in case
     a tier-edge bug would otherwise drop it).

     Cost at steady state: ~30 + 7 + 4 + (months trained) snapshots,
     so ~50-80 files for a year of training. At 6MB per snapshot
     that's <500MB, well inside any reasonable quota.

  3. PRUNE the inert artifacts that aren't snapshots: legacy
     pre-pivot supervised step checkpoints (`checkpoint_<N>.pt`),
     `.bak` files, all-but-freshest `supervised_epoch*.pt` (the
     warm-start logic only ever reads the highest-N one),
     stale demo `.bz2` replays under `logs/` (default keep 5
     freshest), and old job log files under `training/logs/`
     (default keep 10 freshest per prefix).

Files we NEVER touch:
  - `training/checkpoints/sim_selfplay.pt` (the active rolling target)
  - `training/checkpoints/supervised.pt` (still warm-start fodder)
  - `training/checkpoints/supervised_epoch<MAX>.pt` (idem -- newest)
  - The most recent `slurm-*.log` and `selfplay-slurm-*.log`
  - The most recent N demo .bz2 (--keep-demos, default 5)
  - Anything outside the configured training tree
  - `replays_raw/` (the corpus, used by template extraction)
  - `replays_dataset/` (used by tests)

Default mode is DRY-RUN. Pass `--yes` to actually delete.

Usage:
    python tools/cleanup.py                         # dry-run, all rules
    python tools/cleanup.py --yes                   # commit deletions
    python tools/cleanup.py --no-archive --yes      # prune only, no new snapshot
    python tools/cleanup.py --archive-stride-hours 24 --yes
    python tools/cleanup.py --keep-logs 20 --keep-demos 10 --yes
    python tools/cleanup.py --only checkpoints --yes
    python tools/cleanup.py --only logs --yes

Run it after (or alongside) long training sessions to take archive
snapshots and prune old files. Idempotent -- safe to run any time,
including back-to-back.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent

log = logging.getLogger("cleanup")


# ---------------------------------------------------------------------
# Retention model
# ---------------------------------------------------------------------

@dataclass
class RetentionTier:
    """One tier of the periodic-snapshot retention curve.

    `max_age_s`: upper bound (exclusive) of the age range this tier
                 covers. Files older than this fall into the next tier.
    `bucket_s`:  bucket width in seconds. Files in the same bucket
                 (i.e., `floor(age / bucket_s)` is equal) are
                 considered "the same snapshot moment" for retention
                 purposes; only the freshest survives.
    `label`:     for log output.
    """
    max_age_s: float
    bucket_s:  float
    label:     str


# Default tier curve. The rationale:
#
#   - last 24h, hourly  : during a single cluster-day's chain-link
#                         flurry we keep granularity to compare states.
#   - 1-7d,    daily    : "yesterday vs today" comparisons stay cheap.
#   - 7-30d,   weekly   : trend over a month with one snapshot per week.
#   - 30d+,    monthly  : long-term archive, one per month.
#
# To tune: edit `_default_tiers()` or pass `--tier-spec`.
def _default_tiers() -> List[RetentionTier]:
    HOUR  = 3_600
    DAY   = 86_400
    WEEK  = 7 * DAY
    MONTH = 30 * DAY
    return [
        RetentionTier(max_age_s=DAY,            bucket_s=HOUR,  label="hourly"),
        RetentionTier(max_age_s=WEEK,           bucket_s=DAY,   label="daily"),
        RetentionTier(max_age_s=MONTH,          bucket_s=WEEK,  label="weekly"),
        RetentionTier(max_age_s=float("inf"),   bucket_s=MONTH, label="monthly"),
    ]


def _parse_tier_spec(spec: str) -> List[RetentionTier]:
    """Parse a comma-separated `max_age:bucket:label` tier spec. Each
    duration accepts a `h`/`d`/`w` suffix; integer = seconds. Example:

        --tier-spec '24h:1h:hourly,7d:1d:daily,30d:7d:weekly,inf:30d:monthly'

    Matches the DEFAULT_TIERS string-for-string when default.
    """
    def _parse_dur(s: str) -> float:
        s = s.strip().lower()
        if s in ("inf", "infinity", ""):
            return float("inf")
        mult = 1
        if s.endswith("h"):     mult, s = 3_600,         s[:-1]
        elif s.endswith("d"):   mult, s = 86_400,        s[:-1]
        elif s.endswith("w"):   mult, s = 7 * 86_400,    s[:-1]
        elif s.endswith("m"):   mult, s = 30 * 86_400,   s[:-1]
        return float(s) * mult
    out: List[RetentionTier] = []
    for part in spec.split(","):
        bits = part.split(":")
        if len(bits) != 3:
            raise ValueError(f"tier spec part must be max:bucket:label "
                             f"(got {part!r})")
        out.append(RetentionTier(
            max_age_s=_parse_dur(bits[0]),
            bucket_s =_parse_dur(bits[1]),
            label    =bits[2].strip() or f"tier{len(out)}",
        ))
    return out


def apply_tiered_retention(
    files: List[Path], tiers: List[RetentionTier],
    *, now: Optional[float] = None,
) -> Tuple[List[Path], List[Path]]:
    """Split `files` into (keep, drop) per the tiered policy.

    Algorithm:
      1. Sort newest-first.
      2. Always keep the newest file (belt-and-suspenders against a
         tier-edge bug that would otherwise drop it).
      3. For each file, find the first tier with `max_age_s > age`.
         Bucket the file by `floor(age / tier.bucket_s)`. The freshest
         file per (tier, bucket) wins; older bucketmates drop.

    Returns paths preserving newest-first ordering in each list.
    """
    if not files:
        return ([], [])
    if now is None:
        now = time.time()
    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    keep_set: set = {files_sorted[0]}                   # always keep newest
    bucket_winner: Dict[Tuple[int, int], Path] = {}

    for f in files_sorted:
        age = now - f.stat().st_mtime
        # Find first tier covering this age. Tiers are ordered young
        # -> old, so the first tier with max_age > age wins. Beyond
        # the last tier (shouldn't happen with `inf`-bounded last
        # tier), fall back to the last tier.
        tier_idx = next(
            (i for i, t in enumerate(tiers) if age < t.max_age_s),
            len(tiers) - 1,
        )
        bucket_idx = int(age // tiers[tier_idx].bucket_s)
        key = (tier_idx, bucket_idx)
        # Newest-first iteration => the first file in a bucket IS
        # the freshest. Record it; later files in the same bucket
        # are older and drop.
        if key not in bucket_winner:
            bucket_winner[key] = f
            keep_set.add(f)

    keep = [f for f in files_sorted if f in keep_set]
    drop = [f for f in files_sorted if f not in keep_set]
    return (keep, drop)


# ---------------------------------------------------------------------
# Archive snapshot
# ---------------------------------------------------------------------

# Snapshot filename pattern:
#   sim_selfplay_archive_<YYYYMMDD-HHMMSS>.pt
#
# Why ISO timestamp (not jobid): portable across clusters, sorts
# lexically by mtime, and decoupled from SLURM-specific env vars.
# The archive's mtime IS the snapshot's authoritative age for
# tiered-retention purposes -- we don't parse the filename.
_ARCHIVE_GLOB    = "sim_selfplay_archive_*.pt"
_ARCHIVE_PATTERN = re.compile(r"^sim_selfplay_archive_(\d{8}-\d{6})\.pt$")


def _gather_archives(ckpt_dir: Path) -> List[Path]:
    if not ckpt_dir.is_dir():
        return []
    return [p for p in ckpt_dir.glob(_ARCHIVE_GLOB) if p.is_file()]


def maybe_archive_rolling(
    ckpt_dir: Path, *, stride_s: float, dry_run: bool,
    now: Optional[float] = None,
) -> Optional[Path]:
    """If the freshest archive is older than `stride_s` (or there
    are no archives yet), copy `sim_selfplay.pt` to a new timestamped
    archive and return its path. Otherwise return None.

    Atomic-ish: writes via `shutil.copy2` to a `.tmp` sibling and
    renames into place. If `sim_selfplay.pt` doesn't exist (fresh
    tree), this is a no-op.
    """
    if now is None:
        now = time.time()
    rolling = ckpt_dir / "sim_selfplay.pt"
    if not rolling.is_file():
        log.info(f"[archive] no rolling checkpoint at {rolling}; "
                 f"nothing to archive")
        return None
    archives = _gather_archives(ckpt_dir)
    if archives:
        newest = max(archives, key=lambda p: p.stat().st_mtime)
        age = now - newest.stat().st_mtime
        if age < stride_s:
            log.info(f"[archive] newest archive {newest.name} is "
                     f"{age/3600:.1f}h old; stride is "
                     f"{stride_s/3600:.1f}h -- skipping")
            return None
    # Stamp from `now`, not from `rolling.stat().st_mtime`. Using
    # `now` makes back-to-back cleanup runs append cleanly; using
    # the rolling mtime would collide if the rolling file hadn't
    # changed since the last archive (and tiered retention would
    # then drop one of the two on the next pass).
    stamp = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
    target = ckpt_dir / f"sim_selfplay_archive_{stamp}.pt"
    if target.exists():
        # In the unlikely case two cleanups fire in the same second:
        # bail rather than overwrite an existing archive.
        log.warning(f"[archive] target {target.name} already exists; "
                    f"refusing to overwrite")
        return None
    size = rolling.stat().st_size
    log.info(f"[archive] {'DRY-RUN' if dry_run else 'COPY'}: "
             f"{rolling.name} -> {target.name} ({_human_bytes(size)})")
    if dry_run:
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(rolling, tmp)
    tmp.replace(target)
    return target


# ---------------------------------------------------------------------
# Prune categories
# ---------------------------------------------------------------------

# Pre-pivot supervised-training step checkpoints. The supervised loop
# wrote these every N steps before that path was deprecated. None of
# the current pipeline reads them.
_LEGACY_CKPT_PAT = re.compile(r"^checkpoint_\d+\.pt$")
# Backup files left by older recorpus rebuilds. `.bak` is unambiguous
# "prior version of an active file" -- droppable once we have a
# verified-good active version.
_BAK_PAT         = re.compile(r"\.bak$|\.bak\.")
# Demo replays from `tools/sim_demo_game.py`.
_DEMO_GLOB       = "sim_demo_*.bz2"


def _human_bytes(n: int) -> str:
    """B / KB / MB / GB. Pure presentation."""
    if n < 1024:                 return f"{n} B"
    if n < 1024 ** 2:            return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:            return f"{n / 1024 ** 2:.1f} MB"
    return                            f"{n / 1024 ** 3:.1f} GB"


def _commit(files: List[Path], *, dry_run: bool, label: str,
            root: Path) -> Tuple[int, int]:
    """Print + (optionally) delete a list of files. Returns
    (count, bytes_reclaimed)."""
    if not files:
        log.info(f"[{label}] nothing to do")
        return (0, 0)
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            pass
    log.info(f"[{label}] {'DRY-RUN' if dry_run else 'DELETE'}: "
             f"{len(files)} files, {_human_bytes(total)}")
    for f in files:
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = f
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        log.info(f"  {'-' if dry_run else 'X'} {rel}  "
                 f"({_human_bytes(size)})")
        if not dry_run:
            try:
                f.unlink()
            except OSError as e:
                log.warning(f"  failed to remove {rel}: {e}")
                total -= size
    return (len(files), total)


def prune_legacy_checkpoints(
    ckpt_dir: Path, *, dry_run: bool, root: Path,
) -> Tuple[int, int]:
    """Drop pre-pivot numbered checkpoints + .bak. Returns
    (count, bytes_reclaimed). Active files are NEVER touched
    (they don't match these patterns)."""
    if not ckpt_dir.is_dir():
        return (0, 0)
    victims: List[Path] = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file():
            continue
        if _LEGACY_CKPT_PAT.match(p.name) or _BAK_PAT.search(p.name):
            victims.append(p)
    return _commit(victims, dry_run=dry_run,
                   label="legacy-checkpoints", root=root)


def prune_supervised_epochs(
    ckpt_dir: Path, *, dry_run: bool, root: Path,
) -> Tuple[int, int]:
    """Keep only the freshest `supervised_epoch<N>.pt`. The cluster
    sbatch's warm-start logic does `ls supervised_epoch*.pt | sort
    -V | tail -n1` -- earlier-epoch files are never read.
    `supervised.pt` (the rolling mid-epoch latest) is untouched."""
    if not ckpt_dir.is_dir():
        return (0, 0)
    pat = re.compile(r"^supervised_epoch(\d+)\.pt$")
    candidates: List[Tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if len(candidates) <= 1:
        return (0, 0)
    # Keep the highest-N; drop the rest.
    candidates.sort(key=lambda kv: kv[0])
    victims = [p for _, p in candidates[:-1]]
    return _commit(victims, dry_run=dry_run,
                   label="old-supervised-epochs", root=root)


def prune_archive_snapshots(
    ckpt_dir: Path, *, tiers: List[RetentionTier],
    dry_run: bool, root: Path, now: Optional[float] = None,
) -> Tuple[int, int]:
    """Apply tiered retention to `sim_selfplay_archive_*.pt`."""
    if not ckpt_dir.is_dir():
        return (0, 0)
    archives = _gather_archives(ckpt_dir)
    keep, drop = apply_tiered_retention(archives, tiers, now=now)
    if archives:
        log.info(f"[archive-prune] {len(archives)} snapshot(s); "
                 f"keep {len(keep)}, drop {len(drop)}")
        for f in keep:
            try:    rel = f.relative_to(root)
            except ValueError: rel = f
            log.info(f"  K {rel}")
    return _commit(drop, dry_run=dry_run,
                   label="archive-snapshots", root=root)


def prune_demo_replays(
    logs_dir: Path, *, keep: int, dry_run: bool, root: Path,
) -> Tuple[int, int]:
    """Keep `keep` freshest `sim_demo_*.bz2` under `logs_dir`. The
    operator usually only watches one demo at a time; older ones
    are exhaust."""
    if not logs_dir.is_dir():
        return (0, 0)
    demos = sorted(logs_dir.glob(_DEMO_GLOB),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    victims = demos[keep:] if len(demos) > keep else []
    return _commit(victims, dry_run=dry_run,
                   label="demo-replays", root=root)


def prune_logs(
    logs_dir: Path, *, keep: int, prefixes: List[str],
    dry_run: bool, root: Path,
) -> Tuple[int, int]:
    """Per log prefix, keep `keep` newest files; drop the rest.
    Default prefixes cover both `slurm-*` (supervised job logs)
    and `selfplay-slurm-*` (self-play job logs). Files matching
    NEITHER prefix are not touched -- this is a per-stream prune,
    not a wipe."""
    if not logs_dir.is_dir():
        return (0, 0)
    victims: List[Path] = []
    for prefix in prefixes:
        candidates = sorted(logs_dir.glob(f"{prefix}*"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True)
        if len(candidates) > keep:
            victims.extend(candidates[keep:])
    return _commit(victims, dry_run=dry_run,
                   label="logs", root=root)


def prune_debug_artifacts(
    logs_dir: Path, *, dry_run: bool, root: Path,
) -> Tuple[int, int]:
    """Local-only: under `logs/` (NOT `training/logs/`), drop the
    one-off debugging artifacts that aren't part of the demo workflow.
    These are typically stray .txt / .log / orphan .pt files from
    iterating on the sim or trainer. Demo .bz2 files are handled
    separately via `prune_demo_replays` (keep-N semantics)."""
    if not logs_dir.is_dir():
        return (0, 0)
    victims: List[Path] = []
    for p in sorted(logs_dir.iterdir()):
        if not p.is_file():
            continue
        # Keep demo replays for prune_demo_replays to handle.
        if p.name.startswith("sim_demo_") and p.suffix == ".bz2":
            continue
        victims.append(p)
    return _commit(victims, dry_run=dry_run,
                   label="debug-artifacts", root=root)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

# What categories run by default for each location. "local" includes
# the developer's `logs/` (demos + ad-hoc debug crud); "cluster"
# replaces that with `training/logs/` SLURM streams.
_LOCAL_CATEGORIES   = ["archive", "archive-prune", "legacy-checkpoints",
                       "old-supervised-epochs", "demo-replays",
                       "debug-artifacts"]
_CLUSTER_CATEGORIES = ["archive", "archive-prune", "legacy-checkpoints",
                       "old-supervised-epochs", "logs"]


def main(argv: List[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--yes", action="store_true",
                    help="Actually delete + archive (default: dry-run).")
    ap.add_argument("--location", choices=("auto", "local", "cluster"),
                    default="auto",
                    help="Which category set to run. 'auto' picks "
                         "based on which dirs exist (cluster has "
                         "training/logs/, local has logs/).")
    ap.add_argument("--only", nargs="+", default=None,
                    metavar="CATEGORY",
                    help=f"Run only the listed categories. Choices: "
                         f"{', '.join(sorted(set(_LOCAL_CATEGORIES + _CLUSTER_CATEGORIES)))}.")
    ap.add_argument("--no-archive", action="store_true",
                    help="Skip the 'maybe make a new snapshot' step. "
                         "Useful when you want a pure prune.")
    ap.add_argument("--archive-stride-hours", type=float, default=6.0,
                    help="Minimum hours between snapshots. A run with "
                         "the freshest archive younger than this skips "
                         "the archive step. Default 6h -- roughly one "
                         "archive per cluster job (jobs are 4h walltime "
                         "with 2h gaps for queueing / operator review).")
    ap.add_argument("--tier-spec", default=None,
                    help="Override retention tiers. Format: "
                         "'max_age:bucket:label,...' where durations "
                         "use h/d/w/m suffixes or are seconds. Default "
                         "(see source): hourly / daily / weekly / "
                         "monthly buckets. Example: "
                         "'24h:1h:hourly,7d:1d:daily,inf:7d:weekly'.")
    ap.add_argument("--keep-demos", type=int, default=5,
                    help="Number of recent demo .bz2 to retain.")
    ap.add_argument("--keep-logs", type=int, default=10,
                    help="Per log prefix, keep the N most recent.")
    ap.add_argument("--log-prefixes", nargs="+",
                    default=["slurm-", "selfplay-slurm-"],
                    help="SLURM-log filename prefixes to prune. "
                         "Files not matching ANY prefix are left alone.")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=PROJECT_ROOT / "training" / "checkpoints",
                    help="Path to the training checkpoints directory.")
    ap.add_argument("--training-logs-dir", type=Path,
                    default=PROJECT_ROOT / "training" / "logs",
                    help="Path to the SLURM-log directory (cluster).")
    ap.add_argument("--local-logs-dir", type=Path,
                    default=PROJECT_ROOT / "logs",
                    help="Path to the local logs directory.")
    args = ap.parse_args(argv[1:])

    dry_run = not args.yes
    if dry_run:
        log.info("DRY-RUN mode: pass --yes to actually delete / "
                 "archive.")

    # -- 1. Pick categories --
    if args.location == "auto":
        # Heuristic: if `training/logs/` exists AND `logs/` doesn't,
        # we're on the cluster (or a cluster-shaped local clone).
        # Otherwise default to local. Cluster jobs always create
        # training/logs/ via the sbatch's `mkdir -p`.
        has_training_logs = args.training_logs_dir.is_dir()
        has_local_logs    = args.local_logs_dir.is_dir()
        if has_training_logs and not has_local_logs:
            location = "cluster"
        else:
            location = "local"
        log.info(f"auto-detected location: {location}")
    else:
        location = args.location

    cats = _CLUSTER_CATEGORIES if location == "cluster" else _LOCAL_CATEGORIES
    if args.only:
        unknown = set(args.only) - set(cats)
        if unknown:
            log.error(f"unknown category for location={location}: "
                      f"{sorted(unknown)}. Available: {cats}")
            return 1
        cats = [c for c in cats if c in args.only]
    if args.no_archive:
        cats = [c for c in cats if c != "archive"]
    log.info(f"running categories: {cats}")

    # -- 2. Tiers --
    if args.tier_spec:
        try:
            tiers = _parse_tier_spec(args.tier_spec)
        except ValueError as e:
            log.error(f"bad --tier-spec: {e}")
            return 1
    else:
        tiers = _default_tiers()
    log.info("retention tiers: " + ", ".join(
        f"{t.label}<{t.max_age_s/3600:.0f}h bucket={t.bucket_s/3600:.0f}h"
        for t in tiers
    ))

    # -- 3. Run each category --
    now = time.time()
    total_files = 0
    total_bytes = 0

    if "archive" in cats:
        maybe_archive_rolling(
            args.ckpt_dir,
            stride_s=args.archive_stride_hours * 3_600,
            dry_run=dry_run, now=now,
        )

    if "archive-prune" in cats:
        n, b = prune_archive_snapshots(
            args.ckpt_dir, tiers=tiers,
            dry_run=dry_run, root=PROJECT_ROOT, now=now,
        )
        total_files += n; total_bytes += b

    if "legacy-checkpoints" in cats:
        n, b = prune_legacy_checkpoints(
            args.ckpt_dir, dry_run=dry_run, root=PROJECT_ROOT,
        )
        total_files += n; total_bytes += b

    if "old-supervised-epochs" in cats:
        n, b = prune_supervised_epochs(
            args.ckpt_dir, dry_run=dry_run, root=PROJECT_ROOT,
        )
        total_files += n; total_bytes += b

    if "demo-replays" in cats:
        n, b = prune_demo_replays(
            args.local_logs_dir, keep=args.keep_demos,
            dry_run=dry_run, root=PROJECT_ROOT,
        )
        total_files += n; total_bytes += b

    if "debug-artifacts" in cats:
        n, b = prune_debug_artifacts(
            args.local_logs_dir, dry_run=dry_run, root=PROJECT_ROOT,
        )
        total_files += n; total_bytes += b

    if "logs" in cats:
        n, b = prune_logs(
            args.training_logs_dir, keep=args.keep_logs,
            prefixes=args.log_prefixes,
            dry_run=dry_run, root=PROJECT_ROOT,
        )
        total_files += n; total_bytes += b

    log.info(f"{'WOULD RECLAIM' if dry_run else 'RECLAIMED'}: "
             f"{total_files} files, {_human_bytes(total_bytes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
