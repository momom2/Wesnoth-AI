"""Bare-clone [time_area] regression guard (2026-08-04).

A bare `git clone` carries only the tracked wesnoth_src subset. The
scenario [time_area] blocks invoke core ToD macros ({FIRST_WATCH},
{DUSK}, ...) whose definitions live in data/core/macros/schedules.cfg.
When that file was untracked, boxes parsed Kesorak's darkened hex
(WML 19,12) as a cycle of [0] (always NEUTRAL) instead of [-25]
(always night): a strong Spearman there recorded 10 dmg vs the
engine's 7 -- an engine-verified OOS caught by the 2026-08-04 export
sweep, and silently-wrong training ToD on 2 of the 21 ladder maps.

The test reconstructs a scratch wesnoth_src tree from the GIT INDEX
(git show HEAD:...), i.e. exactly what a bare clone sees, and asserts
the parsed cycles. It FAILS if schedules.cfg is ever dropped from
tracking, regardless of what the local Steam-robocopy tree contains.
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_TRACKED_NEEDED = [
    "wesnoth_src/data/multiplayer/scenarios/2p_Tombs_of_Kesorak.cfg",
    "wesnoth_src/data/core/macros/schedules.cfg",
]


def _git_show(relpath: str) -> str:
    # `:path` = the INDEX (staged content) -- equals HEAD after commit,
    # and lets the fix be validated at staging time.
    out = subprocess.run(
        ["git", "show", f":{relpath}"],
        cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.fail(
            f"{relpath} is not tracked (bare clones won't have it): "
            f"{out.stderr.strip()[:200]}")
    return out.stdout


def test_kesorak_time_areas_parse_from_tracked_files_only(tmp_path,
                                                          monkeypatch):
    # Materialize ONLY the committed files into a scratch tree.
    for rel in _TRACKED_NEEDED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(_git_show(rel), encoding="utf-8")

    import tools.scenario_events as se
    monkeypatch.setattr(se, "WESNOTH_SRC", tmp_path / "wesnoth_src")
    monkeypatch.setattr(se, "_CORE_MACROS_CACHE", None)
    # load_scenario_wml caches parsed roots; clear anything keyed on
    # the real tree so the scratch tree is actually consulted.
    for cache_attr in ("_SCENARIO_WML_CACHE", "_WML_CACHE"):
        if hasattr(se, cache_attr):
            getattr(se, cache_attr).clear()

    root = se.load_scenario_wml("multiplayer_Tombs_of_Kesorak")
    assert root is not None, "scenario cfg not found in scratch tree"

    cycles = {}
    def walk(n):
        for c in n.children:
            if c.tag == "time_area":
                cycles[c.attrs.get("x", "")] = se._parse_time_cycle(c)
            walk(c)
    walk(root)

    # Zone 3: the single darkened hex (WML 19,12) -- ALWAYS night.
    assert cycles.get("19") == [-25], cycles
    # Zone 1 (dark corners) and zone 2 (bright zone): full 6-cycles.
    assert cycles.get("9,10,28,29") == [-25, 0, 0, -25, -25, -25], cycles
    assert cycles.get("17,15,23,21") == [25, 25, 25, 25, 0, 0], cycles
