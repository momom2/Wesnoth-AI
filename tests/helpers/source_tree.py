"""The project's source trees, for the tests that read source text.

Each walk is recursive, so a module that moves into a subpackage stays in
the scan, and a directory with no matching file fails the test instead of
shrinking the scan to nothing.
"""
from pathlib import Path
from typing import List

from wesnoth_ai.paths import REPO_ROOT


def source_files(*dirs: str, pattern: str = "*.py") -> List[Path]:
    """Every file matching `pattern` under each of `dirs` (relative to the
    repo root), recursively, sorted, caches left out."""
    files: List[Path] = []
    for rel in dirs:
        base = REPO_ROOT / rel
        found = sorted(p for p in base.rglob(pattern) if "__pycache__" not in p.parts)
        assert found, f"no {pattern} file under {base}: the scan would read nothing there"
        files.extend(found)
    return files
