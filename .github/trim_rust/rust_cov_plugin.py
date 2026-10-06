"""Per-test LLVM profiles of the Rust core, for the suite trim (CI only).

Needs the coverage build of wesnoth_core (patch_lib.py, built with
`-C instrument-coverage`) and $TRIM_RUST_DIR. The counters gathered before
the first test go to `collect.profraw`; each test's go to
`t<index>-main.profraw`. A child process started during a test inherits
LLVM_PROFILE_FILE=`t<index>-%p.profraw` and writes there when it exits.
`tests.tsv` maps each index to its node id.
"""
import os

import pytest

_DIR = os.environ.get("TRIM_RUST_DIR", "")
_OUTSIDE = os.path.join(_DIR, "outside-%p.profraw")
_state = {"core": None, "index": 0}


def _core():
    """The coverage build, after writing what collection gathered."""
    if _state["core"] is None:
        import wesnoth_core
        _state["core"] = wesnoth_core
        wesnoth_core.trim_coverage_dump(os.path.join(_DIR, "collect.profraw"), _OUTSIDE)
    return _state["core"]


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if not _DIR:
        yield
        return
    core = _core()
    _state["index"] += 1
    stem = f"t{_state['index']:05d}"
    with open(os.path.join(_DIR, "tests.tsv"), "a", encoding="utf-8") as fh:
        fh.write(f"{stem}\t{item.nodeid}\n")
    os.environ["LLVM_PROFILE_FILE"] = os.path.join(_DIR, stem + "-%p.profraw")
    try:
        yield
    finally:
        os.environ["LLVM_PROFILE_FILE"] = _OUTSIDE
        core.trim_coverage_dump(os.path.join(_DIR, stem + "-main.profraw"), _OUTSIDE)
