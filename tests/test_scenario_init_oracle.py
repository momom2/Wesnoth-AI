"""The scenario-init oracle's halves that need no Wesnoth
(tools/scenario_init_oracle.py): the lobby writes the harness adds to a
command-line start, and the comparison itself, which must report a
difference in every field it claims to compare."""
import copy
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.scenario_init_oracle import (  # noqa: E402
    Declared, compare, engine_statuses, lobby_experience_modifier, lobby_parms, our_record,
    our_state,
)
from tools.scenario_pool import ScenarioSetup, classify_scenario  # noqa: E402


def test_lobby_writes_fill_only_what_a_side_does_not_declare():
    decl = Declared(scenario={"mp_village_gold": "4"},
                    sides={1: {"side": "1", "village_gold": "3", "fog": "no"}, 2: {"side": "2"}})
    parms = set(lobby_parms(decl))
    assert parms == {(1, "shroud", "no"), (1, "village_support", "1"),
                     (2, "fog", "yes"), (2, "shroud", "no"),
                     (2, "village_gold", "4"), (2, "village_support", "1")}


def test_lobby_experience_modifier_reads_the_scenario_like_the_engine():
    assert lobby_experience_modifier(Declared({"experience_modifier": "100"}, {})) == 100
    assert lobby_experience_modifier(Declared({"experience_modifier": "70%"}, {})) == 70  # not an int
    assert lobby_experience_modifier(Declared({}, {})) == 70


def test_not_living_is_dropped_only_where_it_names_its_three_parts():
    parts = ["undrainable", "unplagueable", "unpoisonable"]
    assert engine_statuses(["not_living"] + parts) == parts
    assert engine_statuses(["not_living", "undrainable"]) == ["not_living", "undrainable"]


def _engine_shaped(ours: dict) -> dict:
    """Our record in the shape lua/init_oracle.lua reports."""
    width = max(x for x, _ in ours["terrain"])
    height = max(y for _, y in ours["terrain"])
    global_bonus = ours["lawful_bonus"][(1, 1)]
    return {
        "time_of_day": ours["time_of_day"], "lawful_bonus": global_bonus,
        "lawful_bonus_exceptions": [{"x": x, "y": y, "lawful_bonus": b}
                                    for (x, y), b in ours["lawful_bonus"].items() if b != global_bonus],
        "terrain_rows": [[ours["terrain"][(x, y)] for x in range(1, width + 1)]
                         for y in range(1, height + 1)],
        "sides": [dict(s, controller="ai") for s in ours["sides"]]
                 + [{"side": s, "controller": "ai"} for s in sorted(ours["acting_sides"] - {1, 2})]
                 + [{"side": s, "controller": "null"} for s in sorted(ours["empty_sides"])],
        "units": copy.deepcopy(ours["units"]),
        "village_owners": copy.deepcopy(ours["village_owners"]),
    }


@pytest.fixture(scope="module")
def basilisk():
    logging.disable(logging.WARNING)
    try:
        sid = "multiplayer_Basilisk"
        setup = ScenarioSetup(sid, "Drakes", "Fire Drake", "Loyalists", "Red Mage",
                              category=classify_scenario(sid))
        return our_record(our_state(setup, 100))
    finally:
        logging.disable(logging.NOTSET)


def _disagreeing(engine: dict, ours: dict) -> set:
    return {name for name, res in compare(engine, ours).items() if res["agree"] != res["total"]}


def test_compare_agrees_with_itself_and_reports_each_perturbed_field(basilisk):
    engine = _engine_shaped(basilisk)
    assert _disagreeing(engine, basilisk) == set()

    def perturbed(edit) -> set:
        e = copy.deepcopy(engine)
        edit(e)
        return _disagreeing(e, basilisk)

    leader = next(u for u in engine["units"] if u["canrecruit"] and u["side"] == 1)
    statue = next(u for u in engine["units"] if u["side"] == 3)
    assert perturbed(lambda e: e["sides"][0].update(gold=99)) == {"side.gold"}
    assert perturbed(lambda e: e["sides"][1].update(village_gold=1)) == {"side.village_gold"}
    assert perturbed(lambda e: next(s for s in e["sides"] if s["side"] == 3).update(controller="ai")) \
        == {"side.turn"}
    assert perturbed(lambda e: next(u for u in e["units"] if u == leader).update(type="Sky Drake")) \
        == {"unit.type"}
    assert perturbed(lambda e: next(u for u in e["units"] if u == statue).update(
        max_hitpoints=statue["max_hitpoints"] + 1)) == {"unit.max_hitpoints"}
    assert perturbed(lambda e: next(u for u in e["units"] if u == statue).update(traits=["quick"])) \
        == {"unit.traits"}
    assert perturbed(lambda e: next(u for u in e["units"] if u == statue).update(traits=["remove_hp"])) \
        == set()                                       # a custom trait compares through its effects
    assert perturbed(lambda e: e["units"].remove(statue)) == {"unit.present"}
    assert perturbed(lambda e: e["village_owners"].append({"x": 1, "y": 1, "side": 1})) == {"village.owner"}
    assert perturbed(lambda e: e["terrain_rows"][0].__setitem__(0, "Zz^Zz")) == {"terrain"}
    assert perturbed(lambda e: e.update(lawful_bonus=e["lawful_bonus"] + 25)) == {"lawful_bonus"}
    assert perturbed(lambda e: e.update(time_of_day="dusk")) == {"time_of_day"}
