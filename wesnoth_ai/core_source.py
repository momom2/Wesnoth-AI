"""The hash of the Rust core's source, which the wheel carries as
`__source_hash__` (rust/wesnoth_core/build.rs computes it at build time).
Two changes have landed within one phase number before, so the phase alone
does not say that a wheel was built from the source in front of it; the
hash does (tools/kernel_status.py, `game_core.game_core_class`).

The hash: SHA-256 over Cargo.toml, build.rs and every .rs file under src/,
in the order of their paths relative to the crate's directory ('/'
separated, compared bytewise); each contributes its path, a newline, its
length in bytes, a newline and its content, with CRLF line ends read as LF.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from wesnoth_ai.paths import RUST_CORE_SRC_DIR

# The crate's directory: Cargo.toml, build.rs and src/.
RUST_CORE_DIR = RUST_CORE_SRC_DIR.parent


def source_hash(crate_dir: Path = RUST_CORE_DIR) -> Optional[str]:
    """The hash of the crate's source in `crate_dir` (the module
    docstring), or None when the crate is not there."""
    crate_dir = Path(crate_dir)
    if not (crate_dir / "Cargo.toml").is_file() or not (crate_dir / "src").is_dir():
        return None
    files = ["Cargo.toml", "build.rs"] + [p.relative_to(crate_dir).as_posix()
                                          for p in (crate_dir / "src").rglob("*.rs")]
    h = hashlib.sha256()
    for rel in sorted(files, key=lambda r: r.encode("utf-8")):
        try:
            data = (crate_dir / rel).read_bytes().replace(b"\r\n", b"\n")
        except OSError:
            return None
        h.update(rel.encode("utf-8") + b"\n" + str(len(data)).encode("ascii") + b"\n" + data)
    return h.hexdigest()


def wheel_source_hash() -> Optional[str]:
    """`__source_hash__` of the installed wheel; None when it is absent or
    was built before the wheel carried one (phase 34 and earlier)."""
    try:
        import wesnoth_core
    except ImportError:
        return None
    return getattr(wesnoth_core, "__source_hash__", None)
