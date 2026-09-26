#!/usr/bin/env python3
"""Build (and optionally upload) the code tarball a box run unpacks.

Every box script starts by downloading a `tier-b/staging/stage_*.tar.gz`
from the model host and extracting it. That tarball has been built by
hand each time, and the hand-built version has a known failure mode:
on 2026-09-13 a payload made from `git ls-files` silently omitted a
NEW, still-untracked tool, and the box run measured nothing because
the script it needed was not there.

This tool makes the payload a checked artifact:

  * `--require PATH ...` names files the run CANNOT work without. If
    any is missing from the payload, this refuses to write the tarball
    and says which. That is the whole point -- a box run that fails
    for a missing file costs a rental, and the check costs nothing.
  * untracked files are INCLUDED only when named with `--extra`, and
    listed in the output, so a new file is never silently absent and
    never silently present.
  * the payload takes WORKING-TREE contents, so uncommitted edits do
    ship. That is usually what you want while iterating, but it means
    the box can run code that is in no commit -- so every staged file
    that differs from HEAD is listed under "NOT IN ANY COMMIT".
  * every payload carries the box library (scripts/box/), and
    `--upload DEST` writes, in one commit, the tarball at DEST, the
    library beside it at `<DEST without .tar.gz>.box/` (a box needs it
    before its code stage: docs/box_runbook.md "Bring-up"), and with
    `--script` the run script under tier-b/staging/. The library and the
    script are read back from the tarball, so they are the stage's own
    bytes; a shell file with CR line endings is refused.

Quickstart
----------
    # what would ship, with the differences from HEAD
    python tools/stage_code.py --dry-run

    # build, requiring the scripts this particular run needs
    python tools/stage_code.py --out /tmp/stage_20260913d.tar.gz \\
        --require tools/diff_replay.py tools/diff_core.py tools/bench_pool.py

    # build and stage a run in one step (needs HF_TOKEN or a cached login)
    python tools/stage_code.py --out /tmp/stage_20260926a.tar.gz \\
        --script scripts/unit_vocab_retrain_box.sh \\
        --upload tier-b/staging/stage_20260926a.tar.gz

Dependencies: git, stdlib; huggingface_hub only for --upload.
Dependents:   every scripts/*_box.sh, via the tarball they download.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import List, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.box.box_stage import LIBRARY_FILES, library_dir  # noqa: E402

log = logging.getLogger("stage_code")
REPO = "momom2/wesnoth-model-checkpoints"
STAGING = "tier-b/staging/"
# The box library: in every payload, since every box script needs it.
LIBRARY_REQUIRED = tuple(f"scripts/box/{name}" for name in LIBRARY_FILES)

# Directories that are tracked but are DATA, not code. A box run
# downloads what it needs from the model host; shipping them in every
# tarball wastes upload time on a metered link.
DEFAULT_EXCLUDE_PREFIXES = (
    "replays_raw/",
    "eval_games/",
    "training/checkpoints/",
    "docs/archive/",
    "quarantine/",
)


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                         text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def tracked_files() -> List[str]:
    return [ln for ln in _git("ls-files").splitlines() if ln.strip()]


def differs_from_head(paths: Sequence[str]) -> Set[str]:
    """Staged paths whose working-tree content is not what HEAD has.

    Covers both modified-tracked and never-committed files, which are
    the two ways a box can end up running code no commit contains.
    """
    out: Set[str] = set()
    for line in _git("status", "--porcelain", "-uall").splitlines():
        if len(line) < 4:
            continue
        name = line[3:].strip().strip('"')
        # A rename reads "old -> new"; the new name is what ships.
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        out.add(name)
    return {p for p in paths if p in out}


def _relative(path: str) -> str:
    """A user-given path as the repo-relative, forward-slash name the
    payload uses. Refuses an absolute path (it would enter the tar
    with an absolute arcname). `str.lstrip("./")` strips CHARACTERS,
    which ate the dot of `.github/...` -- hence the loop."""
    rel = path.replace("\\", "/")
    if Path(rel).is_absolute():
        raise SystemExit(f"{path}: give paths relative to {ROOT}")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def build_payload(extra: Sequence[str],
                  exclude_prefixes: Sequence[str]) -> List[str]:
    paths = [p for p in tracked_files()
             if not any(p.startswith(x) for x in exclude_prefixes)]
    known = set(paths)
    for e in extra:
        rel = _relative(e)
        if rel in known:
            continue
        if not (ROOT / rel).is_file():
            raise SystemExit(f"--extra {e}: no such file under {ROOT}")
        paths.append(rel)
        known.add(rel)
    return sorted(paths)


def check_required(paths: Sequence[str], required: Sequence[str]) -> None:
    have = set(paths)
    missing = [r for r in (_relative(x) for x in required) if r not in have]
    if missing:
        raise SystemExit(
            "REFUSING to build the payload: "
            f"{len(missing)} required file(s) are not in it: {missing}. "
            "An untracked file must be named with --extra (or committed); "
            "a box run that fails for a missing file costs a rental.")


def write_tarball(out: Path, paths: Sequence[str]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tarfile.open(tmp, "w:gz") as tf:
        for rel in paths:
            tf.add(ROOT / rel, arcname=rel)
    tmp.replace(out)
    return out.stat().st_size


def staging_files(tarball: Path, dest: str, script: str | None) -> list[tuple[str, bytes]]:
    """(path in repo, bytes) of what goes up beside the tarball: the box
    library at library_dir(dest) and, when given, the run script under
    tier-b/staging/, each read from the tarball."""
    wanted = [(f"scripts/box/{name}", f"{library_dir(dest)}/{name}") for name in LIBRARY_FILES]
    if script:
        rel = _relative(script)
        wanted.append((rel, STAGING + rel.rsplit("/", 1)[-1]))
    files = []
    with tarfile.open(tarball, "r:gz") as tf:
        names = set(tf.getnames())
        for member, target in wanted:
            if member not in names:
                raise SystemExit(f"{member} is not in the payload; it cannot be staged")
            data = tf.extractfile(member).read()
            if member.endswith(".sh") and b"\r" in data:
                raise SystemExit(f"{member} has CR line endings: bash on the box fails on them")
            files.append((target, data))
    return files


def upload_stage(api, tarball: Path, dest: str, script: str | None) -> None:
    """One commit: the tarball at `dest`, its box library, the run script."""
    beside = staging_files(tarball, dest, script)
    from huggingface_hub import CommitOperationAdd
    operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(tarball))]
    operations += [CommitOperationAdd(path_in_repo=target, path_or_fileobj=data)
                   for target, data in beside]
    api.create_commit(repo_id=REPO, operations=operations,
                      commit_message=f"stage {dest}" + (f" for {script}" if script else ""))
    for op in operations:
        log.info("uploaded %s", op.path_in_repo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Quickstart")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    help="Tarball to write. Omit with --dry-run.")
    ap.add_argument("--require", nargs="*", default=[],
                    help="Files the run cannot work without. Missing any "
                         "refuses the build.")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Untracked files to include. Untracked files are "
                         "NEVER included implicitly.")
    ap.add_argument("--exclude-prefix", nargs="*",
                    default=list(DEFAULT_EXCLUDE_PREFIXES),
                    help="Tracked path prefixes to leave out (data, not code).")
    ap.add_argument("--upload", default=None,
                    help="path_in_repo of the tarball on momom2/wesnoth-model-checkpoints "
                         "(a .tar.gz); its box library goes beside it. Needs HF_TOKEN or "
                         "a cached login.")
    ap.add_argument("--script", default=None,
                    help="The run script (e.g. scripts/unit_vocab_retrain_box.sh), required "
                         "in the payload and uploaded with it to tier-b/staging/.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would ship and stop.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(message)s")

    required = [*LIBRARY_REQUIRED, *args.require, *([args.script] if args.script else [])]
    if args.upload and not args.upload.endswith(".tar.gz"):
        raise SystemExit(f"--upload {args.upload}: a code stage is a .tar.gz")
    paths = build_payload(args.extra, args.exclude_prefix)
    check_required(paths, required)

    dirty = differs_from_head(paths)
    log.info("payload: %d files", len(paths))
    if args.extra:
        log.info("  explicitly added: %s", ", ".join(sorted(args.extra)))
    if dirty:
        log.warning("  %d staged file(s) are NOT IN ANY COMMIT as staged; the "
                    "box will run code no commit contains:", len(dirty))
        for d in sorted(dirty):
            log.warning("    %s", d)
    if args.require:
        log.info("  required files present: %s", ", ".join(sorted(args.require)))

    if args.dry_run:
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --dry-run")

    size = write_tarball(args.out, paths)
    log.info("wrote %s (%.1f MB)", args.out, size / 1024 ** 2)

    # Read the tarball back and re-check. A payload that passed the
    # list check but lost a file to a tar error is the failure this
    # whole tool exists to prevent, so it is worth the second read.
    with tarfile.open(args.out, "r:gz") as tf:
        names = set(tf.getnames())
    check_required(sorted(names), required)
    if len(names) != len(paths):
        raise SystemExit(f"tarball holds {len(names)} entries, expected "
                         f"{len(paths)} -- do not ship it")
    log.info("verified: %d entries read back", len(names))

    if args.upload:
        from huggingface_hub import HfApi
        upload_stage(HfApi(), args.out, args.upload, args.script)
    elif args.script:
        staging_files(args.out, "unused.tar.gz", args.script)   # the same checks, nothing sent
    return 0


if __name__ == "__main__":
    sys.exit(main())
