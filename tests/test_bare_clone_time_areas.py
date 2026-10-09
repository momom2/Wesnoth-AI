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
(git show HEAD:...), i.e. exactly what a bare clone sees, sets the
scenario up on the core from it, and asserts the areas' cycles. It FAILS if schedules.cfg is ever dropped from
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
    # The assertion is about the DEV repo's git index; a tarball
    # deployment (git-archive box trees) has no .git and cannot
    # regress it — skip there instead of failing the box gauntlet.
    probe = subprocess.run(["git", "rev-parse", "--git-dir"],
                           cwd=REPO, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("not a git checkout (tarball deployment)")
    # Materialize ONLY the committed files into a scratch tree.
    for rel in _TRACKED_NEEDED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(_git_show(rel), encoding="utf-8")

    from wesnoth_ai import game_core as gc
    from wesnoth_ai.rules import scenario_cfg
    from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    sid = "multiplayer_Tombs_of_Kesorak"
    gs = build_scenario_gamestate(ScenarioSetup(
        scenario_id=sid, faction1="Rebels", leader1="Elvish Captain",
        faction2="Loyalists", leader2="Lieutenant"))
    gs.global_info._time_areas = {}
    monkeypatch.setattr(gc, "_WML_TUPLES", {})
    monkeypatch.setattr(scenario_cfg, "WESNOTH_SRC", tmp_path / "wesnoth_src")
    monkeypatch.setattr(scenario_cfg, "_CORE_MACROS_CACHE", None)
    # load_scenario_wml caches parsed roots; clear anything keyed on
    # the real tree so the scratch tree is actually consulted.
    for cache_attr in ("_SCENARIO_WML_CACHE", "_WML_CACHE"):
        if hasattr(scenario_cfg, cache_attr):
            getattr(scenario_cfg, cache_attr).clear()

    root = scenario_cfg.load_scenario_wml(sid)
    assert root is not None, "scenario cfg not found in scratch tree"

    cs = gc.CoreState.from_state(gs)
    cs.setup_scenario(sid)
    cycles = {(x + 1, y + 1): list(c) for x, y, c in cs.core.time_areas_export()}

    # Zone 3: the single darkened hex (WML 19,12) -- ALWAYS night.
    assert cycles.get((19, 12)) == [-25], cycles
    # Zone 1 (dark corners) and zone 2 (bright zone): full 6-cycles.
    for wml in ((9, 4), (10, 3), (28, 20), (29, 20)):
        assert cycles.get(wml) == [-25, 0, 0, -25, -25, -25], (wml, cycles)
    for wml in ((17, 2), (15, 7), (23, 17), (21, 22)):
        assert cycles.get(wml) == [25, 25, 25, 25, 0, 0], (wml, cycles)
