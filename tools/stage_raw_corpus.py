"""Pack what an imitation-corpus rebuild reads -- the dispositions ledger
and the raw replay of every candidate it accepts -- into one tarball for
a box, and optionally upload it. The raw replays live only on the laptop.

The tarball holds `training/logs/replay_dispositions.jsonl.gz` and each
candidate at its ledger path (`replays_raw/<date>/<file>.bz2`), so
unpacking it in the repository root is all tools/build_imitation_dataset.py
needs. The 2026-09 ledger names 19,367 candidates, 0.23 GiB of bz2.

Usage (from the repository root):
    python tools/stage_raw_corpus.py --out /tmp/raw_corpus.tar \\
        --upload tier-b/corpus_v2/raw_corpus_20260926.tar
"""
from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_imitation_dataset import DISPOSITIONS, load_candidates, raw_path  # noqa: E402

log = logging.getLogger("stage_raw_corpus")
REPO = "momom2/wesnoth-model-checkpoints"


def write_tarball(out: Path, root: Path, dispositions: Path) -> int:
    """The ledger and every candidate's raw replay into `out` (bz2 files
    do not compress further, so the tar is plain); returns the number
    of replays, after reading the tarball back."""
    candidates = load_candidates(dispositions)
    missing = [p for p in candidates if not raw_path(p, root).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(candidates)} candidates have no raw replay "
                         f"under {root} (first: {missing[:3]})")
    with tarfile.open(out, "w") as tf:
        tf.add(dispositions, arcname=DISPOSITIONS.as_posix())
        for p in candidates:
            tf.add(raw_path(p, root), arcname=str(PurePosixPath(p.replace("\\", "/"))))
    with tarfile.open(out, "r") as tf:
        n = sum(1 for name in tf.getnames() if name.endswith(".bz2"))
    if n != len(candidates):
        raise SystemExit(f"{out} holds {n} replays, expected {len(candidates)}: do not ship it")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="directory the ledger's replays_raw/... paths resolve under")
    ap.add_argument("--dispositions", type=Path, default=DISPOSITIONS)
    ap.add_argument("--upload", default=None,
                    help=f"path_in_repo on {REPO}; needs HF_TOKEN or a cached login")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = write_tarball(args.out, args.root, args.dispositions)
    log.info(f"wrote {args.out}: the ledger and {n} replays, "
             f"{args.out.stat().st_size / 2 ** 20:.0f} MiB")
    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(args.out), path_in_repo=args.upload, repo_id=REPO)
        log.info(f"uploaded {args.upload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
