#!/usr/bin/env python3
"""Find secret-shaped strings in text: tracked files, box records, logs.

    python tools/secret_scan.py            # every tracked text file; exit 1 on a hit

Reports the file, the line and the kind of match, never the value. The
repository is public and box records flow from the box to Hugging Face to
the laptop to git, so a key in a traceback or a log line would travel all
the way (2026-09-25: the Vast account key and a Hugging Face token had
reached local session logs through exception messages). CI runs this on
every push; `tools/pull_box_records.py` withholds a pulled file that
matches.

A line that must hold such a string on purpose (a test fixture) carries
the marker `not-a-secret`.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ALLOW_MARKER = "not-a-secret"
MAX_BYTES = 2_000_000
PATTERNS = {
    "huggingface token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "api_key parameter": re.compile(r"api_key=[A-Za-z0-9]{20,}"),
    "bearer credential": re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{20,}"),
    "container api key": re.compile(r"CONTAINER_API_KEY['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9]{20,}"),
    "hex key": re.compile(r"(?i:api|secret|access)[_-]?(?i:key)['\"]?\s*[:=]\s*['\"]?[0-9a-f]{40,}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "kaggle token": re.compile(r"KAGGLE_(?:API_TOKEN|KEY)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
}


def scan_text(text: str) -> List[Tuple[int, str]]:
    """(line number, kind) of each secret-shaped match, marker lines excepted."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        hits.extend((number, kind) for kind, pattern in PATTERNS.items() if pattern.search(line))
    return hits


def scan_file(path: Path) -> List[Tuple[int, str]]:
    """Hits in one file; binary files and files over MAX_BYTES are not read."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        data = path.read_bytes()
    except OSError:
        return []
    if b"\0" in data[:8192]:
        return []
    return scan_text(data.decode("utf-8", errors="replace"))


def tracked_files(root: Path) -> Iterable[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return (root / name for name in out.stdout.decode("utf-8").split("\0") if name)


def main(argv: List[str]) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent
    found = 0
    for path in tracked_files(root):
        for number, kind in scan_file(path):
            found += 1
            print(f"{path.relative_to(root)}:{number}: {kind}")
    print(f"{found} secret-shaped string(s) in tracked files")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
