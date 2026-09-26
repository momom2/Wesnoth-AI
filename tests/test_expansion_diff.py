"""Our macro expansion against the game's own (2026-09-22).

The generation path expands a scenario `.cfg` with our expander; the
committed templates under `tools/templates/scenarios/` hold the same
scenarios as Wesnoth's preprocessor expanded them. Two renderings of
one file, and until this landed nothing compared them.

That gap is not hypothetical. `{DEFAULT_SCHEDULE}` sat on the expander's
cosmetic list, so our expansion emitted zero `[time]` blocks where the
game's emits six, and the board's time-of-day cycle came off a
hardcoded constant rather than the scenario. A census over our own
output could not have found it -- a classifier is blind to what the
expander already deleted -- which is why this compares against the
engine's rendering instead.

The expectation file records every accepted divergence WITH its reason.
A new one fails here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules.expansion_diff import (POOL, clusters,  # noqa: E402
                                             compare, expected_clusters)

ACCEPTED = {"SUBSTITUTED", "IGNORED"}


@pytest.fixture(scope="module")
def found():
    return clusters(POOL)


def test_no_unexpected_divergence(found):
    """The whole point. Refresh with
    `python tools/analysis/expansion_diff.py --write-expected` only
    after reading what changed."""
    expected = expected_clusters()
    new = sorted(set(found) - set(expected))
    gone = sorted(set(expected) - set(found))
    assert not new, (
        "our expansion newly disagrees with the game's:\n  "
        + "\n  ".join(f"{k} on {found[k]['scenarios']}" for k in new))
    assert not gone, (
        "these divergences are fixed; drop them from the expectation:\n  "
        + "\n  ".join(gone))


def test_the_affected_scenarios_are_the_recorded_ones(found):
    expected = expected_clusters()
    for key, row in sorted(found.items()):
        assert row["scenarios"] == expected[key]["scenarios"], key


def test_every_accepted_divergence_says_why(found):
    """An entry with no reason is an unreviewed bug wearing a green
    test. `stands_in` has to name real code for a SUBSTITUTED one."""
    expected = expected_clusters()
    root = Path(__file__).parent.parent
    for key in sorted(found):
        row = expected[key]
        assert row["classification"] in ACCEPTED, f"{key}: unclassified"
        assert row["why"].strip(), f"{key}: no reason recorded"
        if row["classification"] == "SUBSTITUTED":
            target = row["stands_in"].split()[0]
            assert (root / target).is_file(), f"{key}: {target} is gone"


def test_the_board_schedule_is_read_from_the_scenario():
    """`{DEFAULT_SCHEDULE}`'s six `[time]` blocks reach our expansion.
    Pinned separately from the cluster comparison because this is the
    bug that motivated the check, and it should fail loudly and by
    name if the macro is ever muted again."""
    from wesnoth_ai.rules.expansion_diff import _scenario_block
    from wesnoth_ai.rules.scenario_cfg import load_scenario_wml

    block = _scenario_block(load_scenario_wml("multiplayer_Hamlets"))
    times = block.all("time")
    assert [t.attrs.get("id") for t in times] == [
        "dawn", "morning", "afternoon", "dusk", "first_watch", "second_watch"]


def test_a_single_scenario_can_be_compared():
    """`compare` is the unit the tool and this test share. Hamlets
    carries nothing of its own, so its only divergence is the era's
    substituted lua -- the floor every scenario sits at."""
    assert compare("multiplayer_Hamlets") == [
        ("event count", "prestart: ours 0, the game's 1")]
