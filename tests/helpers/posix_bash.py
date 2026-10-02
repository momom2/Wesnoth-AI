"""A POSIX bash for tests that drive shell scripts, and paths as it sees them."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_bash() -> str | None:
    """A POSIX bash: Git's on Windows, never the WSL launcher."""
    if sys.platform == "win32":
        for candidate in (r"C:\Program Files\Git\usr\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe"):
            if os.path.exists(candidate):
                return candidate
        found = shutil.which("bash")
        if found and "system32" not in found.lower() and "windowsapps" not in found.lower():
            return found
        return None
    return shutil.which("bash")


def bash_path(path: Path) -> str:
    """`path` as bash sees it (Git's bash wants /c/... on Windows)."""
    path = Path(path).resolve()
    if sys.platform == "win32" and path.drive:
        return f"/{path.drive[0].lower()}{path.as_posix()[2:]}"
    return str(path)
