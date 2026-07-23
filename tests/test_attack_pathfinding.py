#!/usr/bin/env python3
"""Regression tests for WesnothSim._find_attack_hex (the implicit
move-to-attack pathfinder used in self-play rollouts).

The 2026-06-29 review found the BFS treated FRIENDLY units as
impassable walls, silently downgrading legal attacks routed past an
ally to end_turn. The fix makes the pathfinder mirror Wesnoth move
rules: enemies block entry, allies are PASS-THROUGH (but not a
landing hex), and a non-skirmisher stops on entering an enemy ZoC hex
(petrified enemies emit no ZoC).

These call the REAL production method on a hand-built state -- no
mirroring. Hexes lie on the y=0 row, which in odd-q layout makes
consecutive x mutually adjacent (a clean corridor); a second row
(y=1) parks an off-corridor ZoC emitter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.classes import (   # noqa: E402
    Alignment, Attack, DamageType, GameState, GlobalInfo, Hex, Map,
    Position, SideInfo, Terrain, Unit,
)
from tools.wesnoth_sim import WesnothSim   # noqa: E402


def _u(uid, side, x, y, *, name="Spearman", moves=5, abilities=None):
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=False, position=Position(x, y),
        max_hp=40, max_moves=moves, max_exp=32, cost=14,
        alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=40, current_moves=moves, current_exp=0,
        has_attacked=False,
        attacks=[Attack(type_id=DamageType.PIERCE, number_strikes=3,
                        damage_per_strike=7, is_ranged=False,
                        weapon_specials=set())],
        resistances=[1.0] * 6, defenses=[0.5] * 14,
        movement_costs=[1] * 14, abilities=set(abilities or set()),
        traits=set(), statuses=set(),
    )


def _sim(units, *, rows=(0,), width=8) -> WesnothSim:
    """Build a flat-terrain sim of `width` columns over the given rows,
    bypassing __init__'s turn setup (we only exercise _find_attack_hex,
    which reads self.gs)."""
    hexes = {Hex(position=Position(x, y), terrain_types={Terrain.FLAT},
                 modifiers=set())
             for x in range(width) for y in rows}
    sides = [SideInfo(player=f"S{i+1}", recruits=[], current_gold=100,
                      base_income=2, nb_villages_controlled=0)
             for i in range(2)]
    gs = GameState(
        game_id="t",
        map=Map(size_x=width, size_y=max(rows) + 1, mask=set(), fog=set(),
                hexes=hexes, units=set(units)),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=sides,
    )
    sim = WesnothSim.__new__(WesnothSim)
    sim.gs = gs
    return sim


def test_routes_through_friendly_unit():
    """Attacker can reach the only attack hex by passing THROUGH an
    ally (the bug returned None here, downgrading to end_turn)."""
    attacker = _u("atk", 1, 0, 0)
    ally = _u("ally", 1, 1, 0)          # blocks the corridor at (1,0)
    target = _u("tgt", 2, 3, 0)
    sim = _sim([attacker, ally, target])
    hex_ = sim._find_attack_hex(attacker, Position(3, 0))
    assert hex_ is not None, "attack through a friendly must be reachable"
    assert (hex_.x, hex_.y) == (2, 0), "lands on the target's near neighbour"


def test_enemy_unit_blocks_passthrough():
    """Contrast: an ENEMY on the same corridor hex is a wall -- the
    attack hex becomes unreachable (no friendly passthrough for foes)."""
    attacker = _u("atk", 1, 0, 0)
    blocker = _u("enemy_block", 2, 1, 0)   # enemy at (1,0) blocks entry
    target = _u("tgt", 2, 3, 0)
    sim = _sim([attacker, blocker, target])
    hex_ = sim._find_attack_hex(attacker, Position(3, 0))
    assert hex_ is None, "cannot pass THROUGH an enemy unit"


def test_zoc_stops_non_skirmisher():
    """A non-skirmisher entering an enemy ZoC hex stops there, so a
    target beyond the ZoC is unreachable this turn."""
    attacker = _u("atk", 1, 0, 0, moves=6)
    emitter = _u("zoc", 2, 3, 1)        # level-1 enemy off-corridor;
                                        # its ZoC covers (3,0)
    target = _u("tgt", 2, 5, 0)
    sim = _sim([attacker, emitter, target], rows=(0, 1))
    hex_ = sim._find_attack_hex(attacker, Position(5, 0))
    assert hex_ is None, "ZoC at (3,0) must stop the attacker short"


def test_skirmisher_ignores_zoc():
    """Same layout, but a skirmisher ignores ZoC and reaches the
    attack hex."""
    attacker = _u("atk", 1, 0, 0, moves=6, abilities={"skirmisher"})
    emitter = _u("zoc", 2, 3, 1)
    target = _u("tgt", 2, 5, 0)
    sim = _sim([attacker, emitter, target], rows=(0, 1))
    hex_ = sim._find_attack_hex(attacker, Position(5, 0))
    assert hex_ is not None, "skirmisher ignores ZoC"
    assert (hex_.x, hex_.y) == (4, 0)


def test_adjacent_attacker_uses_own_hex():
    """If already adjacent, the attacker's own hex is a zero-cost
    landing (no move needed)."""
    attacker = _u("atk", 1, 2, 0)
    target = _u("tgt", 2, 3, 0)
    sim = _sim([attacker, target])
    hex_ = sim._find_attack_hex(attacker, Position(3, 0))
    assert hex_ is not None and (hex_.x, hex_.y) == (2, 0)
