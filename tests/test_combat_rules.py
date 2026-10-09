#!/usr/bin/env python3
"""Regression tests for Wesnoth combat-rule edge cases that diverged
from upstream during the 2026-05-08 100%-clean push, checked on the
Rust core:

  - weapon accuracy feeds the chance to hit, and marksman floors it
  - illuminate lights every unit's hex in the 7-hex area, enemies
    included, and a petrified unit projects no adjacency ability
  - AMLA grants +3 max_hp AND +20% max_experience AND clears
    poisoned/slowed
  - a petrifying hit ends the fight; one that kills is a death
  - Walking Corpse:mounted preserves the parent unit's `[resistance]
    arcane=140` override after movetype switch

Each builds a small hand-made game (tests/helpers/parity_games.py) and
asks the core, so a scrape regression or a core change catches the
same bugs immediately.

Dependencies: wesnoth_ai.game_core, tools.abilities, tools.replay_dataset
Dependents:   pytest only
"""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from helpers.parity_games import core_of, record, state_of, unit_id_at  # noqa: E402
from tools.abilities import hex_neighbors  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")

ATTACKER = (3, 3)


def _east_of(pos):
    """The neighbour of `pos` in the next column on its row."""
    return next(p for p in hex_neighbors(*pos) if p[0] == pos[0] + 1 and p[1] == pos[1])


def _duel(a_type, d_type, *, special=None, others=(), **kw):
    """The core of a game with `a_type` (side 1) next to `d_type`
    (side 2) on grass, `special` terrain codes by hex."""
    d_pos = _east_of(ATTACKER)
    units = [(a_type, 1, *ATTACKER, False), (d_type, 2, *d_pos, False), *others]
    return core_of(record(units, special=special, fog=False, **kw)), d_pos


def _attacker_cth(a_type, a_weapon, d_type, *, special=None):
    cs, d_pos = _duel(a_type, d_type, special=special)
    stats = cs.core.fight_stats(*ATTACKER, *d_pos, a_weapon, -1)
    return int(stats[0]["cth"])


# ---------------------------------------------------------------------
# accuracy and marksman -- the Elvish Champion's sword is the only
# default-era weapon with accuracy (no default-era weapon has parry)
# ---------------------------------------------------------------------

def test_accuracy_adds_to_cth():
    """Champion sword (accuracy=10) against a target hit 60% of the time
    plain: cth 70, not 60. attack.cpp:168-169."""
    plain = _attacker_cth("Elvish Fighter", 0, "Spearman")
    assert plain == 60
    assert _attacker_cth("Elvish Champion", 0, "Spearman") == plain + 10


def test_marksman_floors_cth_and_accuracy_is_not_floored():
    """marksman floors an attacker's cth at 60; accuracy alone is added
    without a floor. A target in forest, hit 30% of the time plain."""
    forest = {_east_of(ATTACKER): "Gg^Fds"}
    plain = _attacker_cth("Elvish Fighter", 1, "Elvish Fighter", special=forest)
    assert plain < 50, "the fixture needs a target the floor lifts"
    assert _attacker_cth("Elvish Marksman", 1, "Elvish Fighter", special=forest) == 60
    assert _attacker_cth("Elvish Champion", 0, "Elvish Fighter", special=forest) == plain + 10


def test_champion_sword_accuracy_in_scrape():
    """unit_stats.json must record Elvish Champion sword accuracy=10.
    Without this scraped attr the live combat reads accuracy=0 and
    silently runs Champion combats at 10% lower hit rate."""
    db = json.loads(
        (Path(__file__).parent.parent / "unit_stats.json").read_text(encoding="utf-8")
    )
    champ = db["units"]["Elvish Champion"]
    sword = next(a for a in champ["attacks"] if a["name"] == "sword")
    assert sword.get("accuracy") == 10, (
        f"Elvish Champion sword should have accuracy=10, "
        f"got {sword.get('accuracy')!r}"
    )


# ---------------------------------------------------------------------
# Illuminate -- every unit in the 7-hex area, not allies only; a
# petrified unit projects no adjacency ability
# ---------------------------------------------------------------------

def _spear_damage_at_night(mage_status=None, with_mage=True):
    """A side-2 Spearman's spear damage against an Elvish Fighter (no
    pierce resistance) at first watch, a side-1 Mage of Light next to
    the Spearman when `with_mage`."""
    spear_pos = (6, 3)
    target = _east_of(spear_pos)
    mage = next(p for p in hex_neighbors(*spear_pos) if p != target and p not in hex_neighbors(*target))
    units = [("Elvish Fighter", 1, *target, False), ("Spearman", 2, *spear_pos, False)]
    if with_mage:
        units.append(("Mage of Light", 1, *mage, False))
    cs = core_of(record(units, fog=False, tod_start_index=4))
    if mage_status:
        cs.core.update_unit(unit_id_at(cs, *mage), {"statuses": [mage_status]})
    spear = cs.core.unit_export(unit_id_at(cs, *spear_pos))["attacks"][0][2]
    stats = cs.core.fight_stats(*spear_pos, *target, 0, -1)
    return int(stats[0]["damage"]), spear


def test_illuminate_lights_enemy_too():
    """A side-1 Mage of Light next to a side-2 Spearman lights the
    Spearman's hex at night: tod_manager.cpp:237-262 scans all 7 hexes
    regardless of side."""
    lit, spear = _spear_damage_at_night()
    dark, _ = _spear_damage_at_night(with_mage=False)
    assert lit == spear, "first watch lit by the enemy mage reads as neutral"
    assert dark < lit


def test_petrified_source_projects_no_adjacency_abilities():
    """A petrified/incapacitated unit projects NO adjacency ability:
    get_abilities skips adjacent units where it->incapacitated()
    (abilities.cpp; illuminate via tod_manager.cpp:443). Verified vs
    the 1.18.4 tag 2026-07-01. Covers illuminate, heals and cures."""
    stoned, _ = _spear_damage_at_night("petrified")
    dark, _ = _spear_damage_at_night(with_mage=False)
    assert stoned == dark

    def patient_after_turn_start(healer_status, patient_status):
        healer, patient = (5, 3), _east_of((5, 3))
        gs = state_of(record([("White Mage", 1, *healer, False), ("Spearman", 1, *patient, False),
                              ("Spearman", 2, 15, 3, True)], fog=False))
        gs.global_info.turn_number = 2
        cs = gc.CoreState.from_state(gs)
        pid = unit_id_at(cs, *patient)
        cs.core.update_unit(pid, {"current_hp": 10, "statuses": [patient_status] if patient_status else []})
        if healer_status:
            cs.core.update_unit(unit_id_at(cs, *healer), {"statuses": [healer_status]})
        assert cs.apply_command(["init_side", 1]) == "rust"
        after = cs.core.unit_export(pid)
        return after["current_hp"], set(after["statuses"])

    hp_ok, _ = patient_after_turn_start(None, None)
    hp_stone, _ = patient_after_turn_start("petrified", None)
    assert hp_ok - hp_stone == 8, "a petrified White Mage heals nobody"
    hp_cured, cured = patient_after_turn_start(None, "poisoned")
    hp_sick, sick = patient_after_turn_start("petrified", "poisoned")
    assert "poisoned" not in cured and "poisoned" in sick
    assert hp_sick < hp_cured


# ---------------------------------------------------------------------
# AMLA -- +3 max_hp, +20% max_exp, clear poisoned/slowed
# ---------------------------------------------------------------------

@pytest.mark.parametrize("status", ["poisoned", "slowed"])
def test_amla_grows_the_unit_and_clears_its_status(status):
    """After each AMLA, max_experience grows by div100rounded(max*20),
    matching apply_modifier with `increase=20%` (string_utils.cpp:401-403,
    math.hpp:39-41); max_hp grows by 3 with a full heal, and the
    [effect][status] remove= entries of AMLA_DEFAULT clear poisoned and
    slowed (amla.cfg:22-24)."""
    cs, _ = _duel("Elvish Sharpshooter", "Spearman")
    uid = unit_id_at(cs, *ATTACKER)
    before = cs.core.unit_export(uid)
    cs.core.update_unit(uid, {"current_exp": before["max_exp"], "current_hp": 20, "statuses": [status]})
    assert cs.core.advance_unit_id(uid)
    after = cs.core.unit_export(uid)
    assert after["name"] == "Elvish Sharpshooter"
    assert after["max_hp"] == before["max_hp"] + 3
    assert after["current_hp"] == after["max_hp"], "AMLA heals full"
    assert after["max_exp"] == before["max_exp"] + (before["max_exp"] * 20 + 50) // 100
    assert after["current_exp"] == 0
    assert status not in after["statuses"]


# ---------------------------------------------------------------------
# petrify (turned to stone)
# ---------------------------------------------------------------------

def _petrifying_duel(defender_hp=None):
    """A Spearman whose javelin petrifies next to an Elvish Fighter."""
    cs, d_pos = _duel("Spearman", "Elvish Fighter")
    view = cs.to_state()
    attacker = next(u for u in view.map.units if u.side == 1)
    attacker.attacks[1].weapon_specials = {"petrifies"}
    cs = gc.CoreState.from_state(view)
    if defender_hp is not None:
        cs.core.update_unit(unit_id_at(cs, *d_pos), {"current_hp": defender_hp})
    return cs, d_pos


def test_petrify_stones_surviving_defender_and_ends_fight():
    """A surviving petrifying hit petrifies the defender and forfeits the
    rest of the fight (the attacker's other strikes and the defender's
    counter), and awards COMBAT xp, not kill xp. Mirrors attack.cpp's
    STATE_PETRIFIED + n_attacks 0/-1 (verified vs the 1.18.4 source)."""
    cs, d_pos = _petrifying_duel()
    aid, did = unit_id_at(cs, *ATTACKER), unit_id_at(cs, *d_pos)
    a0, d0 = cs.core.unit_export(aid), cs.core.unit_export(did)
    hit = int(cs.core.fight_stats(*ATTACKER, *d_pos, 1, 1)[0]["damage"])
    cs.core.apply_attack_scripted(*ATTACKER, *d_pos, 1, 1, [True], [])
    a1, d1 = cs.core.unit_export(aid), cs.core.unit_export(did)
    assert "petrified" in d1["statuses"]
    assert d1["current_hp"] == d0["current_hp"] - hit, "ONE hit, then stop"
    assert a1["current_hp"] == a0["current_hp"], "the defender never counters"
    assert "petrified" not in a1["statuses"]
    assert a1["current_exp"] == a0["current_exp"] + 1, "combat xp of a level-1 defender"


def test_petrify_that_kills_is_a_death_not_a_stone():
    """A petrifying blow that drops the target to 0 hp is a death, not a
    petrify -- the petrify branch is survive-only."""
    cs, d_pos = _petrifying_duel(defender_hp=1)
    aid, did = unit_id_at(cs, *ATTACKER), unit_id_at(cs, *d_pos)
    a0 = cs.core.unit_export(aid)
    cs.core.apply_attack_scripted(*ATTACKER, *d_pos, 1, 1, [True], [])
    assert did not in cs.core.unit_ids()
    assert cs.core.unit_export(aid)["current_exp"] == a0["current_exp"] + 8, "kill xp of a level-1 defender"


def test_outcome_dp_enumerates_petrified_states():
    """The outcome DP models petrify exactly (no None bail): a
    petrifying weapon yields d_petrified outcomes with both units alive,
    and the mass sums to 1."""
    from tools import combat_outcomes as co
    from wesnoth_ai.classes import Position
    cs, d_pos = _petrifying_duel()
    dist = co.enumerate_attack_outcomes(gc.view_of(cs), {
        "type": "attack", "start_hex": Position(*ATTACKER),
        "target_hex": Position(*d_pos), "attack_index": 1})
    assert dist is not None
    petrified = [k for k in dist.probs if k[7]]          # d_petrified
    assert petrified, "petrifying weapon must produce petrified outcomes"
    for k in petrified:
        assert k[1] > 0    # d_hp > 0 (petrify is survive-only)
        assert k[0] > 0    # attacker alive
    assert abs(sum(dist.probs.values()) - 1.0) < 1e-9   # mass conserved


# ---------------------------------------------------------------------
# Walking Corpse:mounted preserves parent's [resistance] arcane=140
# ---------------------------------------------------------------------

def test_wc_mounted_preserves_arcane_140():
    """The mounted variant inherits movetype=mounted (which has
    arcane=90 by default), but the Walking Corpse base unit's
    explicit `[resistance] arcane=140` must carry over.
    `tools/scrape_unit_stats.py::extract_variations` re-applies
    parent overrides on top of the new movetype's defaults."""
    db = json.loads(
        (Path(__file__).parent.parent / "unit_stats.json").read_text(encoding="utf-8")
    )
    base = db["units"]["Walking Corpse"]
    mounted = db["units"]["Walking Corpse:mounted"]
    assert base["resistance"].get("arcane") == 140
    assert mounted["resistance"].get("arcane") == 140, (
        "mounted variant should inherit WC's arcane=140 even though "
        "the mounted movetype defaults arcane to 90"
    )
    # And the variation correctly took the new movetype's other
    # resistances (should NOT be smallfoot's 100% across the board).
    assert mounted["resistance"].get("blade") == 80
    assert mounted["resistance"].get("pierce") == 120
    assert mounted["resistance"].get("impact") == 70


def test_wc_scorpion_variation_overrides_win():
    """The scorpion variant has its OWN [resistance] block that
    fully overrides the layered (movetype default + parent override).
    Variation overrides must take precedence."""
    db = json.loads(
        (Path(__file__).parent.parent / "unit_stats.json").read_text(encoding="utf-8")
    )
    scorpion = db["units"]["Walking Corpse:scorpion"]
    # Scorpion's own [resistance] block sets blade=90 pierce=80
    # impact=110 fire=90 cold=110 arcane=80. None of these should
    # leak from WC's arcane=140 override.
    assert scorpion["resistance"]["blade"] == 90
    assert scorpion["resistance"]["pierce"] == 80
    assert scorpion["resistance"]["impact"] == 110
    assert scorpion["resistance"]["arcane"] == 80


def test_every_recruitable_unit_has_real_stats():
    """`_stats_for` falls back to a generic 33 HP level-1 for a type it
    does not know, which silently corrupts combat, the value head's
    material reading and the combat oracle. Nothing may hit that path
    in the games we actually play.

    This is a DATA test: it fails when the pinned 1.18.4 scrape and the
    default era drift apart, which is the drift CLAUDE.md warns about
    (1.19's Ghoul gained a resistance override that overdamaged units).
    """
    from tools.replay_dataset import _stats_for, unknown_unit_types
    from wesnoth_ai.rules.scenario_pool import load_factions

    factions = load_factions()
    assert factions, "the default era must parse"

    wanted = set()
    for f in factions.values():
        wanted.update(f.recruit or ())
        wanted.update(f.leader_pool or ())
        wanted.update(f.random_leader_pool or ())
    wanted.discard("random")
    assert len(wanted) > 30, f"only {len(wanted)} types collected; the era parse is broken"

    before = set(unknown_unit_types())
    for name in sorted(wanted):
        _stats_for(name)
    fell_back = set(unknown_unit_types()) - before
    assert not fell_back, (
        f"{len(fell_back)} default-era unit types are missing from "
        f"unit_stats.json and silently got generic stats: {sorted(fell_back)}")


def test_the_unknown_type_fallback_is_reported():
    """The fallback itself must stay visible -- a quiet one is how a
    scrape mismatch would hide from the test above."""
    from tools.replay_dataset import _FALLBACK_STATS, _stats_for, unknown_unit_types

    name = "Not A Wesnoth Unit (test)"
    assert _stats_for(name) is _FALLBACK_STATS
    assert name in unknown_unit_types()
