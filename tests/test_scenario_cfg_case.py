#!/usr/bin/env python3
"""Every ladder-map scenario id must resolve to its .cfg with EXACT
on-disk casing.

Regression (2026-07-02, observed on a Linux training node): the two
mainline maps with lowercase WML ids (elensefar_courtyard,
thousand_stings_garrison) live in CamelCase files, so the naive
`2p_<id>.cfg` probe missed them on a case-sensitive filesystem and
they were silently dropped from the training pool. Windows' case-
insensitive filesystem masks the miss, so this test compares the
RESOLVED path's name against the directory's real entries -- it fails
on Windows too if a lookup only works by case-accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pytest

from tools.scenario_events import SCENARIO_DIR, find_scenario_cfg_path
from tools.scenario_pool import LADDER_SCENARIO_IDS

pytestmark = pytest.mark.skipif(
    not SCENARIO_DIR.exists(),
    reason="wesnoth_src scenario dir not present",
)


def test_all_ladder_ids_resolve_with_exact_casing():
    real_names = {p.name for p in SCENARIO_DIR.glob("*.cfg")}
    missing, miscased = [], []
    for sid in sorted(LADDER_SCENARIO_IDS):
        p = find_scenario_cfg_path(sid)
        if p is None:
            missing.append(sid)
        elif p.parent == SCENARIO_DIR and p.name not in real_names:
            # Resolved only via a case-insensitive filesystem: the
            # constructed name doesn't exist as an exact dir entry,
            # so a Linux node would get None here.
            miscased.append((sid, p.name))
    assert not missing, f"ladder ids with no scenario .cfg: {missing}"
    assert not miscased, (
        f"ladder ids resolving only by case-accident (would be "
        f"SILENTLY SKIPPED on Linux): {miscased}")


def test_the_scenario_id_is_the_scenario_tags_own_not_a_macro_bodys():
    from tools.scenario_events import scenario_id_of_cfg
    text = ("#define STATUE X Y\n"
            "    [unit]\n        [modifications]\n            [trait]\n"
            "                id=remove_hp\n            [/trait]\n"
            "        [/modifications]\n    [/unit]\n#enddef\n"
            "[multiplayer]\n    [side]\n        id=1\n    [/side]\n"
            "    id=\"My_Map\"\n[/multiplayer]\n")
    assert scenario_id_of_cfg(text) == "My_Map"


def test_every_addon_scenario_id_resolves_to_a_file_declaring_it():
    """WL_Troll_Toll (5 corpus games) resolved to no file, so its
    replays were rebuilt without the scenario: the first `id=` of its
    .cfg sits inside a #define."""
    from tools.scenario_events import scenario_id_of_cfg
    addons = SCENARIO_DIR.parent.parent / "add-ons"
    files = sorted(addons.glob("*/scenarios/*.cfg"))
    assert files, f"no add-on scenario under {addons}"
    unresolved = []
    for path in files:
        sid = scenario_id_of_cfg(path.read_text(encoding="utf-8", errors="ignore"))
        found = find_scenario_cfg_path(sid) if sid else None
        declared = (scenario_id_of_cfg(found.read_text(encoding="utf-8", errors="ignore"))
                    if found else None)
        if sid is None or declared != sid:
            unresolved.append((path.name, sid, found))
    assert unresolved == []
