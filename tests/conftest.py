"""Path bootstrap for the test suite.

The tests moved out of the repo root into tests/ (2026-07-23). Core
library code now lives in the `wesnoth_ai` package at the repo root and
the scripts under `tools/`; both are imported as top-level names by the
tests (`from wesnoth_ai.model import ...`, `import mcts`). Put the repo
root, `tools/`, and this tests/ dir on sys.path so those imports resolve
no matter how pytest is invoked. The per-file `sys.path.insert` lines
remain for running a test file directly as a script; this makes them
redundant under pytest, not required.

The repo-root conftest.py stays responsible for `collect_ignore_glob`
(vendored trees that must not be collected).
"""
import os
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

for _p in (_ROOT, _ROOT / "tools", _TESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


@pytest.fixture(autouse=True, scope="session")
def _game_records_in_tmp(tmp_path_factory):
    """Self-play entry points record every game under
    training/game_records by default (tools/game_record.py); the suite's
    games go to a temporary directory instead, subprocesses included."""
    old = os.environ.get("WESNOTH_GAME_RECORD_DIR")
    os.environ["WESNOTH_GAME_RECORD_DIR"] = str(tmp_path_factory.mktemp("game_records"))
    yield
    if old is None:
        os.environ.pop("WESNOTH_GAME_RECORD_DIR", None)
    else:
        os.environ["WESNOTH_GAME_RECORD_DIR"] = old


def _source_phase() -> int:
    """`__phase__` as declared in the Rust source, or 0."""
    import re
    src = _ROOT / "rust" / "wesnoth_core" / "src" / "lib.rs"
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    m = re.search(r'__phase__[^0-9]{0,40}?(\d+)', text)
    return int(m.group(1)) if m else 0


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say out loud when the installed Rust wheel is behind the source.

    Every `tests/test_rust_*.py` and `tests/test_game_core.py` SKIPS
    when the kernel it needs is missing, and a skip is quiet. That
    turns "1,050 passed" into a statement about the Python paths only,
    while reading like full coverage -- the same silence that let a
    hand-rolled terrain table go wrong for months. This makes the hole
    impossible to miss without failing a run that is legitimately
    green for what it covers.
    """
    try:
        import wesnoth_core
        have = int(getattr(wesnoth_core, "__phase__", 0))
    except Exception:                        # noqa: BLE001 -- absence is the point
        have, exports = 0, ()
    else:
        exports = tuple(sorted(n for n in dir(wesnoth_core)
                               if not n.startswith("_")))
    want = _source_phase()
    if want and have >= want:
        return
    w = terminalreporter.write_line
    w("")
    if not want:
        # The banner's own parser failed. Going quiet here would make
        # the guard against silence silent, which is the one thing it
        # must not do.
        w("!! could not read __phase__ from rust/wesnoth_core/src/lib.rs, so "
          "the Rust-wheel check did not run.", yellow=True, bold=True)
        w(f"   installed wheel reports phase {have}. Check it by hand.")
        return
    if not have:
        w("!! wesnoth_core is NOT INSTALLED: every Rust-path test skipped.",
          yellow=True, bold=True)
    else:
        w(f"!! wesnoth_core wheel is phase {have}; the source declares {want}.",
          yellow=True, bold=True)
        w(f"   installed exports: {', '.join(exports) or 'none'}")
    w("   So tests/test_game_core.py and parts of tests/test_rust_*.py were")
    w("   SKIPPED, and this run says nothing about the Rust paths. Certify")
    w("   Rust changes on a box (CLAUDE.md, Testing).")
