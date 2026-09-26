"""From-scratch sim builders for the test suite.

Replaces the retired corpus bootstrap (tests used to seed sims from
`replays_dataset/*.json.gz`; the corpus is permanently retired, user
decision 2026-06-12). Sims are built the same way production
self-play builds them: `wesnoth_ai.rules.scenario_pool` setup + gamestate from
the scenario .cfg / .map under `wesnoth_src/data/`.

Not a test module (name deliberately not `test_*`); imported by
test_sim_determinism / test_sim_advance / test_parallel_rollouts /
test_recruit_rejection / test_sim_self_play_smoke.
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.rules.scenario_pool import (   # noqa: E402
    ScenarioSetup, build_scenario_gamestate, load_factions,
    random_setup,
)
from tools.wesnoth_sim import WesnothSim   # noqa: E402

_FACTIONS_DIR = (Path(__file__).parent.parent / "wesnoth_src" / "data"
                 / "multiplayer" / "factions")


def require_scenario_data() -> None:
    """Skip the calling test when the vendored Wesnoth data tree is
    absent (wesnoth_src/ is environment, not repo content — see
    CLAUDE.md "Wesnoth data provenance" for the refresh command)."""
    if not _FACTIONS_DIR.is_dir():
        pytest.skip("wesnoth_src/data missing — refresh from the "
                    "Steam install (see CLAUDE.md)")


def scenario_setup(seed: int = 0, *, mini: bool = False,
                   scenario_id: Optional[str] = None) -> ScenarioSetup:
    """Deterministic-for-a-seed scenario + faction/leader draw, with
    an optional scenario override."""
    require_scenario_data()
    load_factions()
    setup = random_setup(random.Random(seed), forced_faction=None,
                         mini_maps=mini)
    if scenario_id is not None:
        setup = ScenarioSetup(
            scenario_id=scenario_id,
            faction1=setup.faction1, leader1=setup.leader1,
            faction2=setup.faction2, leader2=setup.leader2,
        )
    return setup


def fresh_scenario_sim(seed: int = 0, *, max_turns: int = 6,
                       mini: bool = False,
                       scenario_id: Optional[str] = None,
                       use_core: Optional[bool] = None) -> WesnothSim:
    """One from-scratch sim, the production way. `use_core` pins the
    state of record (None = the environment's default) for the tests
    that are about one of the two."""
    setup = scenario_setup(seed, mini=mini, scenario_id=scenario_id)
    gs = build_scenario_gamestate(setup)
    return WesnothSim(gs, scenario_id=setup.scenario_id,
                      max_turns=max_turns, use_core=use_core)


def twin_scenario_sims(seed: int = 0, *, max_turns: int = 6,
                       mini: bool = False,
                       scenario_id: Optional[str] = None,
                       ) -> Tuple[WesnothSim, WesnothSim]:
    """Two sims guaranteed to share an IDENTICAL starting state (one
    gamestate build, deep-copied into each). Determinism tests
    compare runs from these twins so they measure the SIM's
    determinism, not the builder's."""
    setup = scenario_setup(seed, mini=mini, scenario_id=scenario_id)
    gs = build_scenario_gamestate(setup)
    return (
        WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                   max_turns=max_turns),
        WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                   max_turns=max_turns),
    )


# A replay record, as tools/replay_extract writes one, of a 2p game whose
# scenario declares a third side (compare Caves of the Basilisk: a
# "Custom" statue side, controller=null). Coordinates are 0-indexed.
THREE_SIDE_FACTIONS = {1: "Rebels", 2: "Loyalists", 3: "Custom"}
THREE_SIDE_VILLAGES = {1: [(0, 0), (2, 0)], 2: [(9, 5), (7, 5), (9, 3)]}


def three_side_record(*, third_side_acts: bool = False, fog: bool = False,
                      turns: int = 3) -> dict:
    """The record of `turns` turns in which every side ends its turn at
    once. Side 1 (Rebels) owns two villages, side 2 (Loyalists) three,
    and side 3 holds one petrified statue and no village. With
    `third_side_acts` side 3 takes a turn after side 2 every round, as a
    mini map's tentacle side (controller=ai) does; otherwise it never
    does, as the engine plays a controller=null side."""
    width, height = 10, 6
    villages = {xy for owned in THREE_SIDE_VILLAGES.values() for xy in owned} | {(5, 5)}
    rows = [", ".join("Gg^Vh" if (x, y) in villages else "Gg" for x in range(width))
            for y in range(height)]
    border = ", ".join(["Xv"] * (width + 2))
    map_data = "\n".join([border] + [f"Xv, {r}, Xv" for r in rows] + [border])
    units = [("Elvish Captain", 1, 1, 1, {"is_leader": True}),
             ("Lieutenant", 2, 8, 4, {"is_leader": True}),
             ("Dwarvish Fighter", 3, 5, 2, {"petrified": True})]
    recruits = {1: ["Elvish Fighter"], 2: ["Spearman"], 3: []}
    one_turn = [["init_side", 1], ["end_turn"], ["init_side", 2], ["end_turn"]]
    if third_side_acts:
        one_turn += [["init_side", 3], ["end_turn"]]
    return {
        "game_id": "three_sides", "scenario_id": "", "map_data": map_data,
        "starting_units": [{"uid": i + 1, "type": t, "side": s, "x": x, "y": y,
                            "is_leader": False, **extra}
                           for i, (t, s, x, y, extra) in enumerate(units)],
        "starting_sides": [{"side": s, "faction": THREE_SIDE_FACTIONS[s], "gold": 100,
                            "recruit": recruits[s], "fog": fog, "shroud": False}
                           for s in (1, 2, 3)],
        "starting_villages": [{"x": x, "y": y, "side": s}
                              for s, owned in THREE_SIDE_VILLAGES.items() for x, y in owned],
        "commands": one_turn * turns,
    }


def replayed_state(record: dict, n_commands: int):
    """The record's state after its first `n_commands` commands, built
    as replay reconstruction builds it."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    gs = _build_initial_gamestate(record)
    _setup_scenario_events(gs, record.get("scenario_id", ""))
    for cmd in record["commands"][:n_commands]:
        _apply_command(gs, cmd)
    return gs


class Brawler:
    """A deterministic test driver: recruits like the dummy policy,
    then walks every unit toward the nearest enemy on the sim's own
    planner and attacks when adjacent."""

    def __init__(self):
        from wesnoth_ai.dummy_policy import DummyPolicy
        self.dummy = DummyPolicy()

    def select_action(self, gs, **kw):
        from tools.abilities import hex_neighbors
        from tools.pathfind_sim import ReachContext, unit_reach
        from wesnoth_ai.sim.classes import Position
        from wesnoth_ai.rewards import hex_distance
        from wesnoth_ai.visibility import is_scenery_unit
        side = gs.global_info.current_side
        units = sorted(gs.map.units, key=lambda u: u.id)
        mine = [u for u in units if u.side == side]
        leader = next((u for u in mine if u.is_leader), None)
        if leader is None:
            return {"type": "end_turn"}
        rec = self.dummy._try_recruit(leader, gs.sides[side - 1], gs, mine)
        if rec is not None:
            return rec
        enemies = [u for u in units if u.side in (1, 2) and u.side != side and not is_scenery_unit(u)]
        if not enemies:
            return self.dummy.select_action(gs, **kw)
        enemy_ids = {e.id for e in enemies}
        at = {(u.position.x, u.position.y): u for u in units}

        def dist(pos):
            return min(hex_distance(pos[0], pos[1], e.position.x, e.position.y) for e in enemies)

        for u in mine:
            if "petrified" in (u.statuses or ()):
                continue
            if not u.has_attacked and u.attacks:
                for nb in hex_neighbors(u.position.x, u.position.y):
                    e = at.get(nb)
                    if e is not None and e.id in enemy_ids:
                        return {"type": "attack", "start_hex": u.position,
                                "target_hex": Position(x=nb[0], y=nb[1]), "attack_index": 0}
            if u.current_moves <= 0:
                continue
            reach = unit_reach(u, gs, ReachContext.for_side(gs, side, exclude_unit=u))
            best = min(reach.landable, key=dist, default=None)
            if best is not None and dist(best) < dist((u.position.x, u.position.y)):
                return {"type": "move", "start_hex": u.position, "target_hex": Position(x=best[0], y=best[1])}
        return {"type": "end_turn"}
