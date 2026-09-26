"""Everything a pool scenario declares is classified, and MODELLED
means a reader that exists (2026-09-22).

Detector 2 of docs/scenario_build_plan_20260922.md. The expansion diff
answers "did we expand this scenario the way the game does"; this
answers "of what the expansion contains, what do we actually READ?"

The binding is the point. Every defect this plan was written for --
village gold, per-side fog, `[hides] id=`, `ai_special=guardian` --
was an ORDINARY attribute with an ORDINARY value sitting in the tree
with nothing reading it. A classifier over names and values calls all
four fine. A classifier that demands a named reader calls all four
UNKNOWN, which is why the manifest is built that way.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules import scenario_surface as ss  # noqa: E402


@pytest.fixture(scope="module")
def found():
    return ss.surface()


@pytest.fixture(scope="module")
def manifest():
    return ss.load_manifest()


def test_nothing_in_the_pool_is_unclassified(found, manifest):
    bad = ss.unknowns(found, manifest)
    assert not bad, (
        "these declarations have no classification; decide each one "
        "(list them with `python tools/analysis/scenario_surface.py`):\n  "
        + "\n  ".join(f"{p}.{a} on {len(s)} scenarios" for p, a, s in bad))


def test_every_modelled_entry_names_a_reader_that_reads_it(manifest):
    """The reader must NAME the attribute in its own source, or say in
    `generic` why the read happens elsewhere. Existence alone was the
    original check, and it let 22 of 72 entries through bound to
    functions that never read them."""
    assert not ss.missing_readers(manifest)


@pytest.mark.parametrize("pair, wrong_reader", [
    # The 2026-09-23 audit found these bound to real functions that do
    # not read them. Rebinding each to its old reader must be caught.
    ("scenario/side.gold", "wesnoth_ai/rules/wml_state.py:read_tod"),
    ("scenario/event/unit.variation", "wesnoth_ai/sim/traits.py:roll_traits"),
    ("scenario/event/switch.variable", "tools/scenario_events.py:_lua_action"),
])
def test_a_reader_that_does_not_read_its_attribute_is_caught(manifest, pair,
                                                             wrong_reader):
    bound_wrong = copy.deepcopy(manifest)
    bound_wrong["pairs"][pair] = {"classification": "MODELLED",
                                  "reader": wrong_reader, "why": ""}
    assert any(pair in line for line in ss.missing_readers(bound_wrong))


def test_generic_is_an_explicit_reason_not_an_escape_hatch(manifest):
    """`generic` excuses a reader that gets its attribute from a table
    or from its caller. Every use has to carry that reason."""
    for key, entry in {**manifest["paths"], **manifest["pairs"]}.items():
        if "generic" in entry:
            assert entry["classification"] == "MODELLED", key
            assert len(entry["generic"].strip()) > 20, key


def test_control_flow_does_not_hide_an_action(found):
    """A [unit] inside [switch][case] is dispatched by the same handler
    as a bare one, so it is classified at the same position. Before
    this, Hornshark Island's preplaced heroes were an unclassified path
    of their own -- and one of them had carried an EMPTY [heals]."""
    assert not any("/switch/" in p or "/case/" in p or "/then/" in p
                   for p in found)
    assert "multiplayer_Hornshark_Island" in found[
        "scenario/event/unit/abilities/heals"]["value"]


def test_the_corpus_scenarios_are_covered_not_only_the_pool(found):
    """Reconstruction loads corpus maps outside the training pool
    through the same expander; the manifest has to see them."""
    covered = set().union(*(s for attrs in found.values()
                            for s in attrs.values()))
    for scenario_id in ("multiplayer_Hornshark_Island",
                        "multiplayer_Cynsaun_Battlefield"):
        assert scenario_id in covered, scenario_id


def test_no_path_default_masks_a_load_bearing_attribute(found, manifest):
    assert not ss.masking_path_defaults(found, manifest)


def test_the_manifest_has_no_dead_entries(found, manifest):
    """A classification for something the pool no longer declares is
    a claim nobody can check, and it would silently cover a different
    shape if one came back."""
    assert not ss.stale(found, manifest)


def test_every_entry_carries_a_reason(manifest):
    for table in ("paths", "pairs"):
        for key, entry in manifest[table].items():
            cls = entry.get("classification")
            assert cls in ("MODELLED", "IGNORED", "SUBSTITUTED"), f"{key}: {cls}"
            if cls == "IGNORED":
                assert entry.get("why", "").strip(), key
            elif cls == "SUBSTITUTED":
                assert entry.get("stands_in", "").strip(), key


# --- the detector has to fire, or the manifest is decoration --------
#
# Each case replays a defect this project actually shipped, by
# removing the entry that records the fix and checking the detector
# calls the attribute out again.

@pytest.mark.parametrize("pair", [
    # 2026-09-21: the pool hardcoded 2 gold per village and never read
    # the scenario's own.
    "scenario/side.village_gold",
    # 2026-09-08: fog was a global, so per-side declarations were unread.
    "scenario/side.fog",
    # 2026-09-13: [effect] members were keyed by TAG, so `id=submerge`
    # was dropped and Silverhead's Tentacle was visible where Wesnoth
    # hides it.
    "scenario/event/object/effect/abilities/hides.id",
    # 2026-09-13, same bug: `id=magical` was dropped, so the Tentacle
    # counter-attacked at terrain chance-to-hit instead of a flat 70%.
    "scenario/event/object/effect/specials/chance_to_hit.id",
    # 2026-09-22: ai_special=guardian was read by nothing, while the
    # neutral AI depended on it.
    "scenario/event/unit.ai_special",
])
def test_the_detector_fires_on_a_defect_we_actually_shipped(found, manifest,
                                                            pair):
    stripped = copy.deepcopy(manifest)
    del stripped["pairs"][pair]
    flagged = [f"{p}.{a}" for p, a, _ in ss.unknowns(found, stripped)]
    assert flagged == [pair], (
        f"removing the record of {pair} did not make it UNKNOWN; the "
        f"manifest would not have caught that defect")


def test_a_reader_that_is_deleted_is_caught(manifest):
    """The other half of the binding: the entry stays, the code goes."""
    broken = copy.deepcopy(manifest)
    broken["pairs"]["scenario/side.gold"]["reader"] = \
        "wesnoth_ai/rules/wml_state.py:read_side_that_does_not_exist"
    assert ss.missing_readers(broken)


def test_an_injected_unknown_attribute_is_caught(found, manifest):
    """A scenario growing a new attribute must fail rather than be
    quietly built without it."""
    grown = {p: dict(v) for p, v in found.items()}
    grown["scenario/side"] = dict(grown["scenario/side"])
    grown["scenario/side"]["teleport_on_turn_3"] = {"multiplayer_Hamlets"}
    flagged = [f"{p}.{a}" for p, a, _ in ss.unknowns(grown, manifest)]
    assert flagged == ["scenario/side.teleport_on_turn_3"]


def test_the_manifest_is_committed_json(manifest):
    raw = ss.MANIFEST.read_text(encoding="utf-8")
    assert json.loads(raw)["_what"].strip()
