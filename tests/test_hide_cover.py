"""Hide cover is the engine's terrain filter, not a defense lookup.

`[hides]` in wesnoth_src/data/core/macros/abilities.cfg:280-382 filters
on the hex's terrain CODE:

    ambush       terrain=*^F*,*^Qhhf,*^Qhuf
    concealment  terrain=*^V*
    submerge     terrain=Wo*^*
    nightstalk   time_of_day=chaotic   (no terrain condition)

Until 2026-09-13 the sim answered "is this forest / village / deep
water?" by looking up the hex's DEFENSE keys in a hand-rolled overlay
table (`replay_dataset._OVERLAY_DEFENSE_KEYS`). Every code the table
did not list resolved to plain flat, so the ability was silently
INACTIVE there: over the 21 Ladder-pool maps, 30.4% of forest-overlay
hexes gave no ambush cover and 27.7% of village-overlay hexes gave no
concealment. A unit that Wesnoth hides was visible to the fog gate, to
the legality mask and to the ambush stop in `walk_move_path`.
docs/wesnoth_rules.md carries the full census, in both directions.

These tests pin the rule to the engine's globs and keep a sample of the
codes that used to fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.terrain_resolver import hides_cover  # noqa: E402

# Codes that appear on the shipped maps and gave NO cover before the fix.
REGRESSED_FOREST = ["Gs^Fms", "Hh^Fms", "Gg^Fms", "Gs^Fmw", "Gs^Ftd",
                    "Re^Fms", "Gs^Fdw", "Rb^Fdw", "Ss^Fdw", "Gll^Fmw"]
REGRESSED_VILLAGE = ["Gg^Ve", "Gs^Vht", "Aa^Vha", "Aa^Vea", "Gg^Vl",
                     "Gd^Vhr", "Rr^Vhcr", "Gs^Vl", "Rd^Voa", "Ss^Vhr"]


@pytest.mark.parametrize("code", REGRESSED_FOREST)
def test_every_forest_overlay_gives_ambush_cover(code):
    assert hides_cover(code, "ambush"), f"{code} is forest to the engine (*^F*)"
    assert not hides_cover(code, "concealment")


@pytest.mark.parametrize("code", REGRESSED_VILLAGE)
def test_every_village_overlay_gives_concealment_cover(code):
    assert hides_cover(code, "concealment"), f"{code} is a village to the engine (*^V*)"
    assert not hides_cover(code, "ambush")


def test_the_engine_globs_verbatim():
    # *^F* plus the two named fungus overlays
    assert hides_cover("Gg^Fp", "ambush") and hides_cover("Uu^Qhhf", "ambush")
    assert hides_cover("Uu^Qhuf", "ambush")
    assert not hides_cover("Gg", "ambush"), "a bare base has no overlay to match"
    assert not hides_cover("Uu^Qhh", "ambush"), "only the *f fungus variants are listed"
    # *^V*
    assert hides_cover("Gg^Vh", "concealment")
    assert not hides_cover("Gg", "concealment")
    # Wo*^* -- the base decides, with or without an overlay
    assert hides_cover("Wo", "submerge") and hides_cover("Wo^Bw|", "submerge")
    assert not hides_cover("Ww", "submerge"), "shallow water is not deep water"
    assert not hides_cover("Wwf", "submerge")
    # an unknown ability never takes cover
    assert not hides_cover("Gg^Fp", "nightstalk")


def test_a_starting_position_label_is_stripped():
    """Map cells carry '1 Gg^Fp' for a start hex; the code is the rest.
    The label is any text before a space, not one digit -- the rule and
    its engine citation live in tests/test_start_positions.py. Reading
    only a single digit left '10 Wo' as a base terrain named '10 Wo',
    which matched no glob and gave no cover."""
    assert hides_cover("1 Gs^Fms", "ambush")
    assert hides_cover("2 Gg^Vh", "concealment")
    assert hides_cover("10 Wo", "submerge")
    assert hides_cover("12 Gs^Fp", "ambush")
    assert hides_cover("11 Gg^Vh", "concealment")
    assert hides_cover("book_start Gg^Fp", "ambush")
    assert hides_cover("P1_Burner Wo", "submerge")


def test_no_code_means_no_terrain_cover():
    for ability in ("ambush", "concealment", "submerge"):
        assert not hides_cover("", ability)
        assert not hides_cover(None, ability)


def test_hide_cover_active_uses_the_engine_rule():
    """End to end through the predicate the fog gate and the ambush stop
    both consume."""
    from sim_test_helpers import fresh_scenario_sim
    from tools.replay_dataset import _rebuild_unit
    from wesnoth_ai.visibility import _hide_cover_active

    sim = fresh_scenario_sim(0, max_turns=6, use_core=False)
    gs = sim.gs
    codes = getattr(gs.global_info, "_terrain_codes", None) or {}
    assert codes, "the scenario must carry terrain codes"
    unit = next(iter(sorted(gs.map.units, key=lambda u: u.id)))
    hidden = _rebuild_unit(unit, abilities={"ambush"})

    # Put the code of a previously-regressed forest under the unit.
    codes[(hidden.position.x, hidden.position.y)] = "Gs^Fms"
    assert _hide_cover_active(gs, hidden), \
        "ambush on a ^Fms forest must hide (it did not before 2026-09-13)"

    codes[(hidden.position.x, hidden.position.y)] = "Gg"
    assert not _hide_cover_active(gs, hidden), "open grass is not cover"

    concealed = _rebuild_unit(unit, abilities={"concealment"})
    codes[(concealed.position.x, concealed.position.y)] = "Gg^Ve"
    assert _hide_cover_active(gs, concealed), \
        "concealment in a ^Ve village must hide (it did not before 2026-09-13)"


def test_farmland_is_not_a_village():
    """`^Gvs` is Farmland, an embellishment with `aliasof=_bas`
    (wesnoth_src/data/core/terrain.cfg:399-405) -- not a village. The
    hand-rolled overlay table listed it as one, so concealment (and the
    village DEFENSE those keys also feed) applied on open farmland:
    302 playable hexes of the Ladder pool and 640 of all 114 tracked
    maps. The engine's `*^V*` does not match it."""
    for code in ("Rb^Gvs", "Re^Gvs", "Gs^Gvs", "Gg^Gvs", "Hhd^Gvs", "Dd^Gvs"):
        assert not hides_cover(code, "concealment"), f"{code} is farmland, not a village"
        assert not hides_cover(code, "ambush")


def test_tropical_deep_water_grants_submerge():
    """`Wot` is deep_water_tropical (terrain.cfg:44-47). The old base
    list omitted it, so submerge did not apply on 153 playable hexes
    of the Ladder pool (all on Ruphus Isle) and 332 of all tracked
    maps; the engine's `Wo*^*` matches any Wo base."""
    for code in ("Wot", "Wot^_fme", "Wog"):
        assert hides_cover(code, "submerge"), f"{code} is a deep-water base"
