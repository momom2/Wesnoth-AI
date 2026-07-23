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
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

for _p in (_ROOT, _ROOT / "tools", _TESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
