"""A minimal hand-built game state: two sides on a 10x10 flat board, a
Spearman and a leader each, built without the simulator or scenario
data."""
from wesnoth_ai.sim.classes import (
    Alignment, Attack, DamageType, GameState, GlobalInfo, Hex, Map,
    Position, SideInfo, Terrain, Unit,
)


def _u(uid, side, x, y, *, hp=40, name="Spearman", is_leader=False):
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=40, max_moves=5, max_exp=32, cost=14,
        alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=hp, current_moves=5, current_exp=0,
        has_attacked=False,
        attacks=[Attack(type_id=DamageType.PIERCE, number_strikes=3,
                        damage_per_strike=7, is_ranged=False,
                        weapon_specials=set())],
        resistances=[1.0]*6, defenses=[0.5]*14,
        movement_costs=[1]*14, abilities=set(), traits=set(),
        statuses=set(),
    )


def _gs():
    """Minimal 2-side game state with units and a few hexes."""
    units = {
        _u("u1", 1, 3, 3),
        _u("u2", 2, 4, 3),
        _u("ldr1", 1, 0, 0, is_leader=True),
        _u("ldr2", 2, 8, 8, is_leader=True),
    }
    hexes = {Hex(position=Position(x, y),
                 terrain_types={Terrain.FLAT}, modifiers=set())
             for x in range(10) for y in range(10)}
    sides = [SideInfo(player=f"S{i+1}", recruits=[], current_gold=100,
                      base_income=2, nb_villages_controlled=0)
             for i in range(2)]
    return GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=hexes, units=units),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=sides,
    )
