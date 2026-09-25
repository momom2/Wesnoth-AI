#!/usr/bin/env python3
"""The code stage a box run unpacks, and where its box library lives on HF.

    python box_stage.py fetch STAGE --into DIR

downloads the code stage STAGE (a `tier-b/staging/stage_*.tar.gz` built by
tools/stage_code.py) from the model host and extracts it into DIR, which
must not exist yet: boxlib.sh extracts into a temporary directory and swaps
it in whole. Three download attempts; a failure prints the exception's type
name only. Exit status 0 when the stage is extracted, 1 otherwise.

The box library (this directory: boxlib.sh and its helpers) is needed
before the code stage is on the box, so `tools/stage_code.py --upload`
writes a side copy of it next to the stage, in the same commit, from the
stage's own members: `library_dir(STAGE)`. The onstart of
scripts/rent_box.py fetches `box_onstart.sh` from there, and box_onstart.sh
fetches the rest (docs/box_runbook.md "Bring-up").

Standard library only, apart from huggingface_hub for the download.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time

REPO = "momom2/wesnoth-model-checkpoints"
# The library files, in the order box_onstart.sh fetches them: the stop
# helper first, so that a bring-up that fails later can still stop the box.
LIBRARY_FILES = ("box_stop.py", "box_onstart.sh", "boxlib.sh", "box_upload.py", "box_stage.py")
STAGE_SUFFIX = ".tar.gz"
ATTEMPTS = 3
RETRY_PAUSE_S = 20


def library_dir(stage: str) -> str:
    """The HF folder holding the box library of the code stage `stage`: the
    stage's path with `.tar.gz` replaced by `.box`."""
    if not stage.endswith(STAGE_SUFFIX):
        raise ValueError(f"a code stage is a {STAGE_SUFFIX} file: {stage!r}")
    return stage[: -len(STAGE_SUFFIX)] + ".box"


def extract(tarball: str, into: str) -> int:
    """Extract `tarball` into the new directory `into`; the number of members.
    The `data` filter (Python 3.11.4 and later) refuses members that would
    land outside `into`."""
    os.makedirs(into)
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        if hasattr(tarfile, "data_filter"):
            tf.extractall(into, filter="data")
        else:
            for member in members:
                target = os.path.realpath(os.path.join(into, member.name))
                if not target.startswith(os.path.realpath(into) + os.sep):
                    raise ValueError("a member would land outside the target directory")
            tf.extractall(into)
    return len(members)


def download(stage: str, sleep=time.sleep) -> str | None:
    """The local path of `stage` from the model host, or None after
    ATTEMPTS failures."""
    from huggingface_hub import hf_hub_download
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return hf_hub_download(REPO, stage)
        except Exception as exc:  # noqa: BLE001 -- the type name only
            print(f"stage {stage}: attempt {attempt} {type(exc).__name__}", flush=True)
            if attempt < ATTEMPTS:
                sleep(RETRY_PAUSE_S)
    return None


def fetch(stage: str, into: str) -> int:
    if os.path.exists(into):
        print(f"stage {stage}: {into} exists; extract into a new directory", flush=True)
        return 1
    path = download(stage)
    if path is None:
        return 1
    try:
        n = extract(path, into)
    except Exception as exc:  # noqa: BLE001 -- the type name only
        print(f"stage {stage}: extraction failed {type(exc).__name__}", flush=True)
        return 1
    print(f"stage {stage}: {n} members extracted into {into}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch a code stage (see the module docstring).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download a code stage and extract it into a new directory")
    f.add_argument("stage", help="path of the stage tarball on the model host")
    f.add_argument("--into", required=True, help="directory to create and fill")
    args = ap.parse_args(argv)
    return fetch(args.stage, args.into)


if __name__ == "__main__":
    sys.exit(main())
