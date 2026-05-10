#!/usr/bin/env python3
"""Regression tests for Wesnoth combat-rule edge cases that diverged
from upstream during the 2026-05-08 100%-clean push:

  - accuracy / parry weapon attrs feed the CTH formula
  - illuminate affects ALL units in the 7-hex area, not allies only
  - AMLA grants +3 max_hp AND +20% max_experience AND clears
    poisoned/slowed
  - Walking Corpse:mounted preserves the parent unit's `[resistance]
    arcane=140` override after movetype switch

These don't run a full replay — they exercise small surfaces in
combat.py / tools.replay_dataset / tools.abilities / unit_stats.json
so a future scrape regression or refactor catches the same bugs
immediately.

Dependencies: combat, tools.abilities, tools.replay_dataset, classes
Dependents:   pytest only
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import combat as cb


# ---------------------------------------------------------------------
# accuracy / parry — Elvish Champion sword (the only default-era user)
# ---------------------------------------------------------------------

def _mkunit(weapons, *, defense_pct=60, alignment=cb.Alignment.NEUTRAL,
            hp=40, level=1, abilities=None) -> cb.CombatUnit:
    return cb.CombatUnit(
        side=1, hp=hp, max_hp=hp, level=level,
        experience=0, max_experience=50, alignment=alignment,
        weapons=weapons,
        resistance={k: 100 for k in cb.DAMAGE_TYPES},
        defense_pct=defense_pct,
        abilities=list(abilities or []),
    )


def test_accuracy_adds_to_cth():
    """Champion sword (accuracy=10) attacking a 60%-to-be-hit target
    should give cth=70, not 60. attack.cpp:168-169."""
    sword = cb.Weapon("sword", damage=8, number=5, range="melee",
                      type="blade", accuracy=10)
    plain = cb.Weapon("staff", damage=5, number=2, range="melee",
                      type="impact")
    attacker = _mkunit([sword])
    defender = _mkunit([plain], defense_pct=60)

    stats = cb._compute_battle_stats(
        attacker, defender, 0, 0, 0, 0, is_attacker=True,
    )
    assert stats.cth == 70, f"expected 70, got {stats.cth}"


def test_parry_subtracts_from_cth():
    """A defender weapon with parry=10 should reduce attacker's cth
    by 10 (only when defender retaliates with that weapon)."""
    sword_no_acc = cb.Weapon("sword", damage=8, number=5, range="melee",
                             type="blade")
    parrying = cb.Weapon("rapier", damage=5, number=3, range="melee",
                         type="blade", parry=10)
    attacker = _mkunit([sword_no_acc])
    defender = _mkunit([parrying], defense_pct=60)

    stats = cb._compute_battle_stats(
        attacker, defender, 0, 0, 0, 0, is_attacker=True,
    )
    assert stats.cth == 50, f"expected 50, got {stats.cth}"


def test_accuracy_then_marksman_floor():
    """marksman special floors cth at 60 AFTER accuracy is applied;
    a sword with accuracy=10 vs 30%-to-hit terrain should end at 40
    (not floored), but the same with marksman should floor up to 60."""
    plain_acc = cb.Weapon("sword", damage=8, number=5, range="melee",
                          type="blade", accuracy=10)
    marksman_acc = cb.Weapon("longbow", damage=12, number=5,
                             range="ranged", type="pierce",
                             specials=["marksman"], accuracy=10)
    target = _mkunit([cb.Weapon("none", 1, 1, "melee", "blade")],
                     defense_pct=30)
    a_plain = _mkunit([plain_acc])
    a_mark = _mkunit([marksman_acc])

    s_plain = cb._compute_battle_stats(
        a_plain, target, 0, 0, 0, 0, is_attacker=True,
    )
    s_mark = cb._compute_battle_stats(
        a_mark, target, 0, 0, 0, 0, is_attacker=True,
    )
    assert s_plain.cth == 40, f"plain expected 40, got {s_plain.cth}"
    assert s_mark.cth == 60, f"marksman floored to 60, got {s_mark.cth}"


def test_champion_sword_accuracy_in_scrape():
    """unit_stats.json must record Elvish Champion sword accuracy=10.
    Without this scraped attr the live combat reads accuracy=0 and
    silently runs Champion combats at 10% lower hit rate."""
    db = json.loads(
        (Path(__file__).parent / "unit_stats.json").read_text(encoding="utf-8")
    )
    champ = db["units"]["Elvish Champion"]
    sword = next(a for a in champ["attacks"] if a["name"] == "sword")
    assert sword.get("accuracy") == 10, (
        f"Elvish Champion sword should have accuracy=10, "
        f"got {sword.get('accuracy')!r}"
    )


# ---------------------------------------------------------------------
# Illuminate — affects all units in 7-hex area, not allies only
# ---------------------------------------------------------------------

def test_illuminate_lights_enemy_too():
    """Mage of Light at (1,1) illuminates the area; a side-2 enemy
    at adjacent (2,1) should also count as illuminated. Per
    tod_manager.cpp:237-262 the scan iterates all 7 hexes regardless
    of side."""
    from classes import Position, Unit
    from tools.abilities import illuminate_step

    illuminator = Unit(
        id="u1", name="Mage of Light", name_id=0, side=1,
        is_leader=False, position=Position(1, 1),
        max_hp=27, max_moves=5, max_exp=80, cost=44,
        alignment=None, levelup_names=[],
        current_hp=27, current_moves=5, current_exp=0,
        has_attacked=False, attacks=[],
        resistances=[1.0]*6, defenses=[60]*14, movement_costs=[1]*14,
        abilities={"illuminates"}, traits=set(), statuses=set(),
    )
    enemy = Unit(
        id="u2", name="Orcish Grunt", name_id=0, side=2,
        is_leader=False, position=Position(2, 1),
        max_hp=38, max_moves=5, max_exp=42, cost=12,
        alignment=None, levelup_names=[],
        current_hp=38, current_moves=5, current_exp=0,
        has_attacked=False, attacks=[],
        resistances=[1.0]*6, defenses=[60]*14, movement_costs=[1]*14,
        abilities=set(), traits=set(), statuses=set(),
    )
    units = {illuminator, enemy}
    # Self-illumination always works
    assert illuminate_step(illuminator, units) == 1
    # Enemy adjacent to illuminator: also illuminated
    # (the bug was filtering by ally.side == unit.side)
    assert illuminate_step(enemy, units) == 1, (
        "Enemy adjacent to a Mage of Light must be illuminated "
        "(illumination is a terrain-light modifier, not an ally aura)"
    )


# ---------------------------------------------------------------------
# AMLA — +3 max_hp, +20% max_exp, clear poisoned/slowed
# ---------------------------------------------------------------------

def test_amla_increases_max_exp_20pct():
    """After each AMLA, max_experience grows by div100rounded(max*20),
    matching apply_modifier with `increase=20%`. Compounds across
    AMLAs (string_utils.cpp:401-403, math.hpp:39-41)."""
    from classes import Position, Unit
    from tools.replay_dataset import _maybe_advance_unit
    from classes import GameState, GlobalInfo, Map, SideInfo

    sharpshooter = Unit(
        id="u1", name="Elvish Sharpshooter", name_id=0, side=1,
        is_leader=False, position=Position(0, 0),
        max_hp=57, max_moves=6, max_exp=36, cost=62,
        alignment=None, levelup_names=[],
        current_hp=40, current_moves=6, current_exp=36,  # at threshold
        has_attacked=False, attacks=[],
        resistances=[1.0]*6, defenses=[50]*14, movement_costs=[1]*14,
        abilities=set(), traits={"resilient", "intelligent"},
        statuses={"poisoned"},  # must be cleared by AMLA
    )
    gs = GameState(
        game_id="test",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=set(), units={sharpshooter}),
        global_info=GlobalInfo(
            current_side=1, turn_number=10, time_of_day="dawn",
            village_gold=2, village_upkeep=1, base_income=2,
        ),
        sides=[SideInfo(player="x", recruits=[], current_gold=0,
                        base_income=2, nb_villages_controlled=0)],
    )

    advanced = _maybe_advance_unit(gs, sharpshooter)
    assert advanced.max_hp == 60, f"expected +3 max_hp, got {advanced.max_hp}"
    assert advanced.current_hp == 60, "AMLA should heal_full"
    # 36 + div100rounded(36*20) = 36 + (720+50)//100 = 36 + 7 = 43
    assert advanced.max_exp == 43, (
        f"expected max_exp 43 after first AMLA from 36, got {advanced.max_exp}"
    )
    assert advanced.current_exp == 0, "XP should reset (carry over excess)"
    assert "poisoned" not in advanced.statuses, (
        "AMLA must remove poisoned status (amla.cfg:22-24)"
    )


def test_amla_clears_slowed():
    """Same as above but with `slowed`. Both statuses are removed by
    distinct [effect][status][remove=...] entries in AMLA_DEFAULT."""
    from classes import Position, Unit
    from tools.replay_dataset import _maybe_advance_unit
    from classes import GameState, GlobalInfo, Map, SideInfo

    u = Unit(
        id="u1", name="Elvish Sharpshooter", name_id=0, side=1,
        is_leader=False, position=Position(0, 0),
        max_hp=57, max_moves=6, max_exp=36, cost=62,
        alignment=None, levelup_names=[],
        current_hp=20, current_moves=6, current_exp=36,
        has_attacked=False, attacks=[],
        resistances=[1.0]*6, defenses=[50]*14, movement_costs=[1]*14,
        abilities=set(), traits=set(),
        statuses={"slowed"},
    )
    gs = GameState(
        game_id="test",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=set(), units={u}),
        global_info=GlobalInfo(
            current_side=1, turn_number=10, time_of_day="dawn",
            village_gold=2, village_upkeep=1, base_income=2,
        ),
        sides=[SideInfo(player="x", recruits=[], current_gold=0,
                        base_income=2, nb_villages_controlled=0)],
    )
    advanced = _maybe_advance_unit(gs, u)
    assert "slowed" not in advanced.statuses


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
        (Path(__file__).parent / "unit_stats.json").read_text(encoding="utf-8")
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
        (Path(__file__).parent / "unit_stats.json").read_text(encoding="utf-8")
    )
    scorpion = db["units"]["Walking Corpse:scorpion"]
    # Scorpion's own [resistance] block sets blade=90 pierce=80
    # impact=110 fire=90 cold=110 arcane=80. None of these should
    # leak from WC's arcane=140 override.
    assert scorpion["resistance"]["blade"] == 90
    assert scorpion["resistance"]["pierce"] == 80
    assert scorpion["resistance"]["impact"] == 110
    assert scorpion["resistance"]["arcane"] == 80
