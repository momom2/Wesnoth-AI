"""The template builder stays importable, and its inputs stay real
(2026-09-22).

`tools/build_scenario_templates.py` imported `DRILL_SCENARIO_IDS` from
`scenario_pool`, which stopped existing when the user ruled the drill
scenarios out. The module was therefore unimportable for six weeks and
nothing noticed, because the committed templates it produces are read
by everything and the builder itself is read by nothing.

A tool that regenerates committed data has to at least import, and its
scenario lists have to name scenarios whose sources exist -- otherwise
a regeneration reports "NOT FOUND" for ids no one can act on.

These tests do NOT run Wesnoth's preprocessor; that needs the install
and is the builder's own acceptance run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import wesnoth_ai.rules.build_scenario_templates as builder  # noqa: E402
from wesnoth_ai.rules.scenario_pool import (LADDER_SCENARIO_IDS,  # noqa: E402
                                            MINI_MAP_SCENARIO_IDS)

ROOT = Path(__file__).parent.parent
TEMPLATES = ROOT / "tools" / "templates" / "scenarios"


def test_the_builder_imports():
    """The regression this file exists for."""
    assert callable(builder.transform)
    assert callable(builder.run_preprocessor)


def test_every_scenario_the_builder_would_build_has_a_source():
    """The builder's default set must be buildable. Resolved with the
    production resolver, because a scenario's id is NOT its filename:
    `2p_mini_edited` lives in `2p_mini_1.cfg` and
    `Modified_Tiny_Close_Relation` in `Modified_Close_Relation.cfg`."""
    from wesnoth_ai.rules.scenario_cfg import find_scenario_cfg_path

    for scenario_id in sorted(set(LADDER_SCENARIO_IDS)
                              | set(builder.MINI_TEMPLATE_IDS)):
        assert find_scenario_cfg_path(scenario_id) is not None, scenario_id


def test_the_committed_templates_are_exactly_the_buildable_ones():
    """Every committed template can be regenerated, and every scenario
    the builder would build has one.

    A template whose source is gone is stale data pretending to be
    derived: nothing can check it and nothing can rebuild it. Three
    were in that state until 2026-09-22 -- the drill templates,
    orphaned when the drill scenarios were deleted in fba0513 -- and
    the user ruled them out. The set is empty now and this keeps it
    empty."""
    on_disk = {p.stem for p in TEMPLATES.glob("*.wml")}
    buildable = set(LADDER_SCENARIO_IDS) | set(builder.MINI_TEMPLATE_IDS)
    unbuildable = sorted(on_disk - buildable)
    assert not unbuildable, (
        f"these templates have no source and cannot be regenerated or "
        f"checked: {unbuildable}")
    missing = sorted(buildable - on_disk)
    assert not missing, f"the builder would build these and they are not "\
                        f"committed: {missing}"


def test_around_mini_keeps_its_template_and_its_source():
    """It left the training pool but not the repo: a test builds it,
    so retiring the template would break that test. Named in
    EXTRA_MINI_TEMPLATE_IDS so the builder knows which tree to
    preprocess for it."""
    assert "around_mini" not in MINI_MAP_SCENARIO_IDS
    assert "around_mini" in builder.MINI_TEMPLATE_IDS
    assert (TEMPLATES / "around_mini.wml").is_file()

    from wesnoth_ai.rules.scenario_cfg import find_scenario_cfg_path
    assert find_scenario_cfg_path("around_mini") is not None
