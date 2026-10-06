"""Per-test timing and coverage contexts for the suite-trim measurement.

Loaded on CI only (`-p trim_plugin`, with this directory on PYTHONPATH).
For every test it records the setup, call and teardown durations, the
outcome, the `slow` marker and the non-function fixtures it uses, and
writes them as JSON to $TRIM_OUT at the end of the session.

When coverage.py runs in this process (started by a .pth hook reading
COVERAGE_PROCESS_START), each test's lines go to a dynamic context named
by its node id. The node id is also exported as COV_TRIM_CONTEXT for the
duration of the test, so a Python child process the test starts reads it
as its static context (coveragerc: `context = ${COV_TRIM_CONTEXT}`) and
its lines are credited to the same test.
"""
import json
import os

import pytest

_RECORDS: dict[str, dict] = {}


def _running_coverage():
    try:
        import coverage
    except ImportError:
        return None
    return coverage.Coverage.current()


def _shared_fixtures(item) -> dict[str, str]:
    """Fixtures of a scope wider than the function, by name."""
    info = getattr(item, "_fixtureinfo", None)
    if info is None:
        return {}
    shared = {}
    for name, defs in info.name2fixturedefs.items():
        if not defs:
            continue
        scope = str(getattr(defs[-1], "scope", "function"))
        if scope != "function":
            shared[name] = scope
    return shared


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    _RECORDS[item.nodeid] = {
        "slow": item.get_closest_marker("slow") is not None,
        "shared_fixtures": _shared_fixtures(item),
        "setup": 0.0, "call": 0.0, "teardown": 0.0,
        "outcome": None,
    }
    cov = _running_coverage()
    os.environ["COV_TRIM_CONTEXT"] = item.nodeid
    if cov is not None:
        cov.switch_context(item.nodeid)
    try:
        yield
    finally:
        if cov is not None:
            cov.switch_context("")
        os.environ.pop("COV_TRIM_CONTEXT", None)


def pytest_runtest_logreport(report):
    record = _RECORDS.get(report.nodeid)
    if record is None:
        return
    record[report.when] = report.duration
    if report.when == "call" or report.outcome != "passed":
        if record["outcome"] in (None, "passed"):
            record["outcome"] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("TRIM_OUT")
    if not out:
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_RECORDS, fh, indent=0)
