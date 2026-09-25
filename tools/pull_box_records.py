#!/usr/bin/env python3
"""Pull a box run's records from the model host into the repo.

    python tools/pull_box_records.py tier-b/graphed_default_20260918 \
        training/metrics/bench_pipeline/graphed_default_20260918

Every file under the HF prefix lands under the local directory with
the same basename. Files already present with the same size are
skipped, so the pull can run again while the box is still uploading.
A file with a secret-shaped string in it (tools/secret_scan.py: a token,
a keyed URL, a key in a traceback) is withheld and named, since records
go on to git and the repository is public. Prints one line per file and
the count at the end.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.secret_scan import scan_file  # noqa: E402

REPO = "momom2/wesnoth-model-checkpoints"
log = logging.getLogger("pull_box_records")


def pull(prefix: str, dest: Path, max_mb: float = 50.0) -> int:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    prefix = prefix.rstrip("/") + "/"
    dest.mkdir(parents=True, exist_ok=True)
    infos = [i for i in api.list_repo_tree(REPO, path_in_repo=prefix.rstrip("/"))
             if hasattr(i, "size")]
    n = withheld = 0
    for info in infos:
        name = info.path[len(prefix):]
        target = dest / name
        if target.exists() and target.stat().st_size == info.size:
            log.info("kept    %s", name)
            continue
        if name.endswith(".escrowed") or ".partial." in name or (name.startswith("phase_") and name.endswith(".json")):
            # Escrow markers, half-written partials and the per-state
            # value-head records stay on the host (see .gitignore).
            log.info("skipped %s (not a record the tree keeps)", name)
            continue
        if info.size > max_mb * 1e6:
            # Checkpoints stay on the model host; the metrics tree holds
            # records (a run's escrowed .pt files are 180 MB each).
            log.info("skipped %s (%d MB, over --max-mb %g)", name, info.size // 1_000_000, max_mb)
            continue
        src = hf_hub_download(REPO, info.path)
        hits = scan_file(Path(src))
        if hits:
            withheld += 1
            log.warning("WITHHELD %s: secret-shaped content at %s", name,
                        ", ".join(f"line {line} ({kind})" for line, kind in hits[:5]))
            continue
        shutil.copyfile(src, target)
        log.info("pulled  %s (%d bytes)", name, info.size)
        n += 1
    log.info("%d files pulled, %d withheld, %d present under %s", n, withheld, len(infos), dest)
    return 1 if withheld else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="path in the model host repo, e.g. tier-b/graphed_default_20260918")
    ap.add_argument("dest", type=Path, help="local directory to fill")
    ap.add_argument("--max-mb", type=float, default=50.0,
                    help="Skip files larger than this (checkpoints stay on the host).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    return pull(args.prefix, args.dest, max_mb=args.max_mb)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
