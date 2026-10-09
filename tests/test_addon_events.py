#!/usr/bin/env python3
"""Vendored add-on scenario events: [capture_village] + the Marshy
Fill turn-1 leader-MP chain ([store_unit]/[if]/[set_variable
sub=]/[modify_unit]) -- the 2026-08-06 whitelist-audit DISCUSS maps,
included on user order after sim support landed -- and the engine rules
of the turn start, the end of turn and advancement that the add-on
scenarios brought up.

Production path throughout: WesnothSim(build_scenario_gamestate(...))
fires prestart+start on the Rust core, exactly as replay reconstruction
does (`replay_dataset.record_core`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402
from tools.scenario_events import load_events_for_scenario  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402


def _fresh_sim(sid, leader1="Elvish Captain", leader2="Elvish Captain"):
    setup = ScenarioSetup(
        scenario_id=sid,
        faction1="Rebels", leader1=leader1,
        faction2="Rebels", leader2=leader2)
    return WesnothSim(build_scenario_gamestate(setup),
                      scenario_id=sid, max_turns=10)


def _core_of(units, sid="multiplayer_Weldyn_Channel", factions=("Rebels", "Rebels"), **extra):
    """The core of a two-side game on `sid`'s map from replay-record
    `units`, its scenario not set up."""
    from tools.replay_dataset import _build_initial_gamestate
    return gc.CoreState.from_state(_build_initial_gamestate({
        "game_id": "t", "scenario_id": sid, "factions": list(factions),
        "starting_sides": [{"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": units, "starting_villages": [], "commands": [], **extra}))


def _unit(cs, uid):
    return cs.core.unit_export(uid)


def _load_events(cs, events) -> None:
    """`events` (ScenarioEvent) as the core's event list, keeping its WML
    and location variables."""
    _fired, wml, stored = cs.core.events_export()
    cs.core.load_events([(ev.name, bool(ev.first_time_only), gc._event_actions(ev), ev.scenario_id, False)
                         for ev in events], wml, stored)


def _run_actions(cs, wml: str) -> None:
    """WML actions run on the core as an event of their own."""
    from tools.replay_extract import parse_wml
    from tools.scenario_events import collect_events
    _load_events(cs, collect_events(parse_wml(f"[multiplayer]\n[event]\nname=test\n{wml}[/event]\n"
                                              f"[/multiplayer]\n"), "test"))
    cs.core.fire_events(["test"])


def test_cold_war_prestart_capture_village_ownership():
    """WL Cold War: prestart [capture_village] gives side 1 the
    village at WML (33,18) and side 2 those at (48,19) and (3,4).
    Without the handler these were silently dropped -> 1g/turn income
    drift (audit 2026-08-06)."""
    sim = _fresh_sim("WL_Cold_War")
    owner = getattr(sim.gs.global_info, "_village_owner", None) or {}
    assert owner.get((32, 17)) == 1, owner
    assert owner.get((47, 18)) == 2, owner
    assert owner.get((2, 3)) == 2, owner


def test_summer_frosts_prestart_capture_village():
    """WL Summer Frosts: the single [capture_village] side=2 (40,7)."""
    sim = _fresh_sim("WL_Summer_Frosts")
    owner = getattr(sim.gs.global_info, "_village_owner", None) or {}
    assert owner.get((39, 6)) == 2, owner


def test_a_captured_village_is_counted_for_its_owner():
    """Income reads each side's village count, and the engine's count
    is the side's village set, so a [capture_village] moves both. The
    handler wrote the owner only: Cold War started at counts 1 and 1
    with 2 and 3 villages owned, Summer Frosts at 1 and 1 with 1 and 2,
    and every turn paid side 2 short."""
    for sid, counts in (("WL_Cold_War", [2, 3]), ("WL_Summer_Frosts", [1, 2])):
        sim = _fresh_sim(sid)
        assert [s.nb_villages_controlled for s in sim.gs.sides] == counts, sid
        assert sim.core.core.invariant_violation() is None, sid


def test_capture_village_moves_releases_and_skips_what_is_not_a_village():
    """wesnoth.map.set_owner (game_lua_kernel.cpp:1142-1193, 1.18.4):
    the old owner loses the village and the new one gains it, no side
    leaves it to nobody, and a location that is not a village is
    skipped."""
    from wesnoth_ai.classes import Terrain
    cs = _fresh_sim("WL_Cold_War").core

    def owners():
        return {(x, y): s for x, y, s in cs.core.village_owner_export()}

    def counts():
        return [s[4] for s in cs.core.sides_export()]

    _run_actions(cs, "[capture_village]\nside=1\nx=48\ny=19\n[/capture_village]\n")
    assert owners()[(47, 18)] == 1
    assert counts() == [3, 2]
    _run_actions(cs, "[capture_village]\nx=48\ny=19\n[/capture_village]\n")
    assert (47, 18) not in owners()
    assert counts() == [2, 2]
    field = next(h.position for h in cs.to_state().map.hexes if Terrain.VILLAGE not in h.terrain_types)
    _run_actions(cs, f"[capture_village]\nside=2\nx={field.x + 1}\ny={field.y + 1}\n[/capture_village]\n")
    assert (field.x, field.y) not in owners()
    assert cs.core.invariant_violation() is None


def test_the_sim_invariant_catches_a_village_count_off_its_owners():
    import pytest
    from dataclasses import replace
    sim = _fresh_sim("WL_Summer_Frosts")
    sim._assert_invariants(after_cmd="setup")
    sim.gs.sides[1] = replace(sim.gs.sides[1], nb_villages_controlled=1)
    from sim_test_helpers import commit_view
    commit_view(sim)
    with pytest.raises(AssertionError, match="village counts"):
        sim._assert_invariants(after_cmd="setup")


def _marshy_leader(cs) -> dict:
    """Side 1's leader at WML (18,1)."""
    return _unit(cs, cs.core.unit_id_at(17, 0))


def test_marshy_fill_leader_mp_shave_6mp():
    """WL Marshy Fill start event: side-1 leader at WML (18,1) with
    5 <= max_moves <= 8 gets current moves = 9 - moves on turn 1.
    Elvish Captain (5 MP) -> 4. max_moves untouched (the event writes
    the CURRENT-moves attribute only; modify_unit.lua:14-17,41)."""
    sim = _fresh_sim("WL_Marshy_Fill")
    leader = next(u for u in sim.gs.map.units
                  if u.side == 1 and u.position.x == 17
                  and u.position.y == 0)
    assert leader.max_moves == 5
    assert leader.current_moves == 4, leader.current_moves
    # side 2's leader is NOT filtered by the event: untouched.
    l2 = next(u for u in sim.gs.map.units if u.side == 2)
    assert l2.current_moves == l2.max_moves


def test_marshy_fill_leader_mp_branches():
    """The [if] branches on a re-fired fresh event list: moves >= 9
    -> 0; moves <= 4 -> untouched (condition greater_than=4 fails)."""
    cs = _fresh_sim("WL_Marshy_Fill").core
    # >= 9 branch: else-arm of the inner [if] sets 0.
    cs.core.update_unit(_marshy_leader(cs)["id"], {"current_moves": 9})
    _load_events(cs, load_events_for_scenario("WL_Marshy_Fill"))
    cs.core.fire_events(["start"])
    assert _marshy_leader(cs)["current_moves"] == 0
    # <= 4 branch: outer [if] condition false, no [else] -> untouched.
    cs.core.update_unit(_marshy_leader(cs)["id"], {"current_moves": 4})
    _load_events(cs, load_events_for_scenario("WL_Marshy_Fill"))
    cs.core.fire_events(["start"])
    assert _marshy_leader(cs)["current_moves"] == 4


def test_vendored_seamless_variant_shares_event_logic():
    """The Seamless Marshy Fill-(R) vendored copy carries the same id
    (WL_Marshy_Fill) and identical event logic; by-id event loading
    resolves to ONE cfg -- the sim behavior must be the same shave."""
    events = load_events_for_scenario("WL_Marshy_Fill")
    assert any(ev.name == "start" for ev in events)
    names = {a.tag for ev in events for a in ev.actions}
    assert "store_unit" in names and "if" in names


def test_quick_4mp_leader_current_moves_refreshed():
    """quick_4mp_leaders engine parity: after the auto-quick trait,
    the engine refreshes moves AND hitpoints to max (eras.lua:18-19).
    A 4-MP leader (Elder Wose) must therefore START with 5/5 MP --
    the missing refresh made every such leader one MP short on turn
    1 (2026-08-06, user-diagnosed from an Aethermaw replay)."""
    from tools import replay_dataset as rd
    gs = rd._build_initial_gamestate({
        "game_id": "t", "scenario_id": "multiplayer_Aethermaw",
        "factions": ["Undead", "Rebels"],
        "experience_modifier": 70,
        "starting_sides": [
            {"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": [
            {"uid": 1, "type": "Dark Sorcerer", "side": 1, "x": 28,
             "y": 17, "hp": 48, "max_hp": 48, "max_moves": 5,
             "is_leader": True},
            {"uid": 2, "type": "Elder Wose", "side": 2, "x": 20,
             "y": 23, "hp": 64, "max_hp": 64, "max_moves": 4,
             "is_leader": True}],
        "starting_villages": [], "commands": [],
    })
    wose = next(u for u in gs.map.units if u.id == "u2")
    assert "quick" in wose.traits
    assert wose.max_moves == 5
    assert wose.current_moves == 5, wose.current_moves
    assert wose.current_hp == wose.max_hp


def test_pickadvance_narrows_advancement_resolution():
    """Plan Unit Advance (mainline mod): a recorded pick REPLACES the
    unit's advances_to, so later [choose] indices index the NARROWED
    list. Fail-before: value=0 advanced a Fighter to Captain
    (vanilla index 0) where the engine made the picked Hero (CotB
    74713, root-caused 2026-08-06 with the user's viewer ledger)."""
    cs = _core_of([
        {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5, "y": 5, "is_leader": True},
        {"uid": 2, "type": "Elvish Fighter", "side": 1, "x": 7, "y": 7},
        {"uid": 3, "type": "Elvish Fighter", "side": 1, "x": 9, "y": 9},
        {"uid": 4, "type": "Elvish Captain", "side": 2, "x": 20, "y": 5, "is_leader": True}],
        experience_modifier=70)

    def advance(uid, choice):
        cs.core.update_unit(uid, {"current_exp": _unit(cs, uid)["max_exp"]})
        _choices, pick, _events = cs.core.advance_state_export()
        cs.core.set_advance_state([choice], list(pick), [])
        assert cs.core.advance_unit_id(uid)
        return _unit(cs, uid)

    # unit-scoped pick: only u2 narrowed
    cs.apply_command(["pickadvance", 7, 7, "Elvish Hero", "", 1, 0])
    assert _unit(cs, "u2")["pickadvance"] == ["Elvish Hero"]
    assert _unit(cs, "u3")["pickadvance"] is None

    # advancement: recorded choose value=0 must resolve on the
    # narrowed list -> Hero (vanilla list is [Captain, Hero]).
    adv = advance("u2", 0)
    assert adv["name"] == "Elvish Hero", adv["name"]
    # the advanced unit re-initializes: old narrowing cleared
    assert adv["pickadvance"] is None

    # game-scoped pick: all current same-side same-type units narrow
    # via the unit list; future map recorded for new inits.
    cs.apply_command(["pickadvance", 9, 9, "Elvish Hero", "Elvish Hero", 1, 1])
    assert _unit(cs, "u3")["pickadvance"] == ["Elvish Hero"]
    _choices, pick, _events = cs.core.advance_state_export()
    assert (1, "Elvish Fighter", ["Elvish Hero"]) in [(s, t, list(v)) for s, t, v in pick]

    # sanity: a pick naming an illegal type is ignored at resolution
    cs.core.update_unit("u3", {"pickadvance": ["Dwarvish Lord"]})
    adv3 = advance("u3", 0)
    assert adv3["name"] == "Elvish Captain", adv3["name"]

    # RECRUITS initialized after a game-override inherit it too: the
    # mod's initialize_unit runs on the "recruit" event (main.lua:231)
    # and reads game_override for the type. Fail-before: a Wolf Rider
    # recruited after game_override=Goblin Pillager advanced as
    # Goblin Knight (vanilla index 0) where the engine made a
    # Pillager — Hellhole 21368, weapon_oob on the Pillager's net at
    # turn 22; engine playback clean end-to-end (2026-08-07).
    cs.core.set_global_int("current_side", 1)
    cs.apply_command(["recruit", "Elvish Fighter", 6, 5, "1a2b3c4d"])
    fresh = cs.core.unit_id_at(6, 5)
    assert _unit(cs, fresh)["pickadvance"] == ["Elvish Hero"], (
        "post-override recruit must inherit the game pick"
    )
    adv4 = advance(fresh, 0)
    assert adv4["name"] == "Elvish Hero", adv4["name"]


def test_turn1_healing_gate_split():
    """Engine parity, play_controller.cpp:484-507 (1.18.4): healing
    is gated by do_healing() -- false ONLY for the game's very first
    side-init -- while MP refresh + income sit behind turn() > 1. A
    regenerating unit damaged on turn 1 heals at its own turn-1 init
    (Micro Isar tentacles, user-observed); nothing heals at the very
    first init; turn-1 inits never refresh MP."""
    cs = _core_of([
        {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5, "y": 5, "is_leader": True},
        {"uid": 2, "type": "Wose", "side": 2, "x": 20, "y": 5, "is_leader": True}],
        experience_modifier=70)
    for uid in ("u1", "u2"):
        cs.core.update_unit(uid, {"current_hp": _unit(cs, uid)["max_hp"] - 10})
    cs.core.update_unit("u2", {"current_moves": 1})          # must NOT refresh on turn 1

    cs.apply_command(["init_side", 1])   # the game's FIRST init
    u1 = _unit(cs, "u1")
    assert u1["current_hp"] == u1["max_hp"] - 10, "no healing at first init"

    cs.apply_command(["init_side", 2])   # turn-1, non-first init
    u2 = _unit(cs, "u2")
    assert u2["current_hp"] == u2["max_hp"] - 10 + 8, \
        f"regen must heal at turn-1 non-first init (got {u2['current_hp']})"
    assert u2["current_moves"] == 1, "no MP refresh on turn 1"

    cs.apply_command(["end_turn"])
    cs.apply_command(["init_side", 1])   # turn 2 begins
    u1 = _unit(cs, "u1")
    assert u1["current_moves"] == u1["max_moves"], "turn-2 init refreshes MP"
    cs.apply_command(["end_turn"])
    cs.apply_command(["init_side", 2])   # turn 2, side 2
    u2 = _unit(cs, "u2")
    # -10 +8 (t1 regen) = max-2; +8+2 at t2 clamps at max_hp.
    assert u2["current_hp"] == u2["max_hp"], f"turn-2 regen+rest should clamp to full (got {u2['current_hp']})"
    assert u2["current_moves"] == u2["max_moves"]


def test_map_header_start_positions():
    """Add-on maps embed their .map header (border_size=/usage=) in
    map_data; counting header lines as terrain rows shifted every
    start hex by +2 in y, silently src-missing every leader command
    of mini-map server replays (29/34 sampled 2p_mini_edited forked
    from turn 1, 2026-08-06 sweep)."""
    from tools.replay_extract import _parse_map_starting_positions
    md = ("border_size=1\nusage=map\n\n"
          "Wo, Wo, Wo, Wo\n"
          "Wo, 1 Ke, Gg, Wo\n"
          "Wo, Gg, 2 Ke, Wo\n"
          "Wo, Wo, Wo, Wo\n")
    pos = _parse_map_starting_positions(md)
    assert pos[1] == (0, 0), pos
    assert pos[2] == (1, 1), pos


def test_object_effects_survive_advancement():
    """Scenario [object] effects persist through advancement (Wesnoth
    stores them in the unit's [modifications] and re-applies on
    advance, like traits). Fail-before: Hornshark's MODIFY_BOWMAN
    firststrike vanished when the preplaced (28,24) Bowman leveled to
    Longbowman — every later defensive fight ran attacker-first and
    the HP ledger forked (16349, engine playback clean, user viewer
    frames 2026-08-07)."""
    cs = _core_of([
        {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5, "y": 5, "is_leader": True},
        {"uid": 2, "type": "Bowman", "side": 2, "x": 27, "y": 23},
        {"uid": 3, "type": "Dwarvish Lord", "side": 2, "x": 20, "y": 5, "is_leader": True}],
        sid="multiplayer_Hornshark_Island", factions=("Rebels", "Loyalists"))
    cs.setup_scenario("multiplayer_Hornshark_Island")

    def ranged_specials(u):
        return next(set(sp) for (_t, _n, _d, ranged, sp) in u["attacks"] if ranged)

    assert "firststrike" in ranged_specials(_unit(cs, "u2")), (
        "MODIFY_BOWMAN prestart object must grant ranged firststrike"
    )
    cs.core.update_unit("u2", {"current_exp": _unit(cs, "u2")["max_exp"]})
    cs.core.set_advance_state([0], [], [])
    assert cs.core.advance_unit_id("u2")
    adv = _unit(cs, "u2")
    assert adv["name"] == "Longbowman", adv["name"]
    assert "firststrike" in ranged_specials(adv), (
        "object-granted specials must survive advancement"
    )


def test_end_turn_mp_deficit_clears_resting():
    """unit::end_turn (unit.cpp:1280-1292, 1.18.4): a unit ending its
    side's turn with remaining MP != max MP loses `resting`, even if
    it never moved or fought -- MP-draining events count as activity.
    A unit at full MP keeps resting."""
    cs = _core_of([
        {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5, "y": 5, "is_leader": True},
        {"uid": 2, "type": "Elvish Fighter", "side": 1, "x": 7, "y": 5, "is_leader": False}])
    cs.core.set_global_int("current_side", 1)
    for uid in ("u1", "u2"):
        cs.core.update_unit(uid, {"statuses": sorted(set(_unit(cs, uid)["statuses"]) | {"resting"})})
    cs.core.update_unit("u1", {"current_moves": _unit(cs, "u1")["max_moves"] - 1})   # drained
    cs.apply_command(["end_turn"])
    assert "resting" not in _unit(cs, "u1")["statuses"], \
        "MP deficit at end_turn must clear resting"
    assert "resting" in _unit(cs, "u2")["statuses"], "full-MP unit keeps resting"


def test_micro_isar_tentacle_never_rest_heals():
    """Full chain for the Micro Isar 38859 fix (2026-08-07): the
    repeating `turn refresh` event {MODIFY_UNIT (role=monster) moves 0}
    (enclave_micro_isar.cfg:86-91) zeroes tentacle MP every side turn,
    so unit::end_turn's MP check clears `resting` and the tentacle
    heals regen-only +8, never +10. User-verified viewer frames:
    turn 3 heal 7->15, turn 4 heal 15->23 (not 25); our former +2 rest
    left a 1-HP survivor whose ZoC forked the whole game."""
    cs = _core_of([
        {"uid": 1, "type": "Drake Flare", "side": 1, "x": 0, "y": 5, "is_leader": True},
        {"uid": 2, "type": "Revenant", "side": 2, "x": 7, "y": 5, "is_leader": True}],
        sid="enclave_micro_isar", factions=("Drakes", "Undead"))
    cs.setup_scenario("enclave_micro_isar")

    def tent():
        return _unit(cs, cs.core.unit_id_at(3, 2))

    cs.apply_command(["init_side", 1])    # turn-1 spawn + refresh
    t = tent()
    assert t["side"] == 3 and t["wml_role"] == "monster"
    assert t["max_moves"] > 0 and t["current_moves"] == 0, \
        "turn refresh MODIFY_UNIT must zero monster MP"
    cs.core.update_unit(t["id"], {"current_hp": 5})        # wounded
    for cmd in (["end_turn"], ["init_side", 2], ["end_turn"], ["init_side", 3]):
        cs.apply_command(cmd)             # own init: +8 regen
    assert tent()["current_hp"] == 13, tent()["current_hp"]
    for cmd in (["end_turn"], ["init_side", 1], ["end_turn"], ["init_side", 2],
                ["end_turn"], ["init_side", 3]):
        cs.apply_command(cmd)             # MP 0 != max at its end of turn: no rest
    t = tent()
    assert t["current_hp"] == 21, \
        f"regen-only heal expected (13+8=21), got {t['current_hp']}"
    assert t["current_moves"] == 0, "turn-2 refresh must be re-zeroed"


def test_a_scenario_without_wml_is_said_once_and_an_import_failure_is_not_swallowed(
        monkeypatch, caplog):
    """A scenario whose WML is not found runs without its events, time
    areas and side modifications: that is warned about, once per id (the
    WL_Troll_Toll lookup bug ran silent this way). And if the event
    parser cannot be imported the setup raises instead of returning
    with no events, which played every game without its scenario."""
    import logging

    import pytest

    from tools import replay_dataset as rd
    cs = gc.CoreState.from_state(rd._build_initial_gamestate({"map_data": "Gg, Gg\nGg, Gg"}))
    monkeypatch.setattr(rd, "_SCENARIOS_WITHOUT_WML", set())
    with caplog.at_level(logging.WARNING, logger="replay_dataset"):
        cs.setup_scenario("no_such_scenario_id")
        cs.setup_scenario("no_such_scenario_id")
    assert cs.statics["_scenario_events"] == []
    assert sum("no_such_scenario_id" in r.getMessage() for r in caplog.records) == 1
    monkeypatch.setitem(sys.modules, "tools.scenario_events", None)
    with pytest.raises(ImportError):
        cs.setup_scenario("multiplayer_Hamlets")
