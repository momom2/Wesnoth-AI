#!/usr/bin/env python3
"""Vendored add-on scenario events: [capture_village] + the Marshy
Fill turn-1 leader-MP chain ([store_unit]/[if]/[set_variable
sub=]/[modify_unit]) -- the 2026-08-06 whitelist-audit DISCUSS maps,
included on user order after sim support landed.

Production path throughout: WesnothSim(build_scenario_gamestate(...))
fires prestart+start through tools/scenario_events, exactly as replay
reconstruction does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402
from tools.scenario_events import (  # noqa: E402
    fire_event, load_events_for_scenario,
)


def _fresh_sim(sid, leader1="Elvish Captain", leader2="Elvish Captain"):
    setup = ScenarioSetup(
        scenario_id=sid,
        faction1="Rebels", leader1=leader1,
        faction2="Rebels", leader2=leader2)
    return WesnothSim(build_scenario_gamestate(setup),
                      scenario_id=sid, max_turns=10)


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
    sim = _fresh_sim("WL_Marshy_Fill")
    leader = next(u for u in sim.gs.map.units
                  if u.side == 1 and u.position.x == 17
                  and u.position.y == 0)
    # >= 9 branch: else-arm of the inner [if] sets 0.
    leader.current_moves = 9
    fire_event(sim.gs, load_events_for_scenario("WL_Marshy_Fill"),
               "start")
    assert leader.current_moves == 0, leader.current_moves
    # <= 4 branch: outer [if] condition false, no [else] -> untouched.
    leader.current_moves = 4
    fire_event(sim.gs, load_events_for_scenario("WL_Marshy_Fill"),
               "start")
    assert leader.current_moves == 4, leader.current_moves


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
    from tools import replay_dataset as rd
    gs = rd._build_initial_gamestate({
        "game_id": "t", "scenario_id": "multiplayer_Weldyn_Channel",
        "factions": ["Rebels", "Rebels"],
        "experience_modifier": 70,
        "starting_sides": [
            {"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": [
            {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5,
             "y": 5, "is_leader": True},
            {"uid": 2, "type": "Elvish Fighter", "side": 1, "x": 7,
             "y": 7},
            {"uid": 3, "type": "Elvish Fighter", "side": 1, "x": 9,
             "y": 9},
            {"uid": 4, "type": "Elvish Captain", "side": 2, "x": 20,
             "y": 5, "is_leader": True}],
        "starting_villages": [], "commands": [],
    })
    u2 = next(u for u in gs.map.units if u.id == "u2")

    # unit-scoped pick: only u2 narrowed
    rd._apply_command(gs, ["pickadvance", 7, 7, "Elvish Hero", "", 1, 0])
    assert getattr(u2, "_pickadvance", None) == ["Elvish Hero"]
    u3 = next(u for u in gs.map.units if u.id == "u3")
    assert getattr(u3, "_pickadvance", None) is None

    # advancement: recorded choose value=0 must resolve on the
    # narrowed list -> Hero (vanilla list is [Captain, Hero]).
    u2.current_exp = u2.max_exp
    setattr(gs.global_info, "_advance_choices", [0])
    adv = rd._maybe_advance_unit(gs, u2)
    assert adv.name == "Elvish Hero", adv.name
    # the advanced unit re-initializes: old narrowing cleared
    assert getattr(adv, "_pickadvance", None) is None

    # game-scoped pick: all current same-side same-type units narrow
    # via the unit list; future map recorded for new inits.
    rd._apply_command(gs, ["pickadvance", 9, 9, "Elvish Hero",
                           "Elvish Hero", 1, 1])
    u3 = next(u for u in gs.map.units if u.id == "u3")
    assert getattr(u3, "_pickadvance", None) == ["Elvish Hero"]
    gmap = getattr(gs.global_info, "_pickadvance_game", {})
    assert gmap.get((1, "Elvish Fighter")) == ["Elvish Hero"]

    # sanity: a pick naming an illegal type is ignored at resolution
    u3.current_exp = u3.max_exp
    setattr(u3, "_pickadvance", ["Dwarvish Lord"])
    setattr(gs.global_info, "_advance_choices", [0])
    adv3 = rd._maybe_advance_unit(gs, u3)
    assert adv3.name == "Elvish Captain", adv3.name

    # RECRUITS initialized after a game-override inherit it too: the
    # mod's initialize_unit runs on the "recruit" event (main.lua:231)
    # and reads game_override for the type. Fail-before: a Wolf Rider
    # recruited after game_override=Goblin Pillager advanced as
    # Goblin Knight (vanilla index 0) where the engine made a
    # Pillager — Hellhole 21368, weapon_oob on the Pillager's net at
    # turn 22; engine playback clean end-to-end (2026-08-07).
    gs.global_info.current_side = 1
    rd._apply_command(gs, ["recruit", "Elvish Fighter", 6, 5,
                           "1a2b3c4d"])
    fresh = next(u for u in gs.map.units
                 if (u.position.x, u.position.y) == (6, 5))
    assert getattr(fresh, "_pickadvance", None) == ["Elvish Hero"], (
        "post-override recruit must inherit the game pick"
    )
    fresh.current_exp = fresh.max_exp
    setattr(gs.global_info, "_advance_choices", [0])
    adv4 = rd._maybe_advance_unit(gs, fresh)
    assert adv4.name == "Elvish Hero", adv4.name


def test_pickadvance_extractor_plumbing():
    """The extractor pairs [fire_event] raise="menu item pickadvance"
    with its dependent [input] and emits the compact pickadvance
    command (0-indexed hex, override strings, flags); ignore=yes
    picks are dropped."""
    import pathlib
    src = pathlib.Path("tools/replay_extract.py").read_text(
        encoding="utf-8")
    assert 'menu item pickadvance' in src
    assert '"pickadvance",' in src
    assert 'pending_pick_hex' in src
    # forced-choice mode: the [input] follows a RECRUIT with no
    # fire_event, so recruits must arm the pick target too.
    assert src.count("pending_pick_hex = (") >= 2


def test_turn1_healing_gate_split():
    """Engine parity, play_controller.cpp:484-507 (1.18.4): healing
    is gated by do_healing() -- false ONLY for the game's very first
    side-init -- while MP refresh + income sit behind turn() > 1. A
    regenerating unit damaged on turn 1 heals at its own turn-1 init
    (Micro Isar tentacles, user-observed); nothing heals at the very
    first init; turn-1 inits never refresh MP."""
    from tools import replay_dataset as rd
    gs = rd._build_initial_gamestate({
        "game_id": "t", "scenario_id": "multiplayer_Weldyn_Channel",
        "factions": ["Rebels", "Rebels"],
        "experience_modifier": 70,
        "starting_sides": [
            {"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": [
            {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5,
             "y": 5, "is_leader": True},
            {"uid": 2, "type": "Wose", "side": 2, "x": 20, "y": 5,
             "is_leader": True}],
        "starting_villages": [], "commands": [],
    })
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u2 = next(u for u in gs.map.units if u.id == "u2")
    u1.current_hp -= 10
    u2.current_hp -= 10
    u2.current_moves = 1          # must NOT refresh on turn 1

    rd._apply_command(gs, ["init_side", 1])   # the game's FIRST init
    u1 = next(u for u in gs.map.units if u.id == "u1")
    assert u1.current_hp == u1.max_hp - 10, "no healing at first init"

    rd._apply_command(gs, ["init_side", 2])   # turn-1, non-first init
    u2 = next(u for u in gs.map.units if u.id == "u2")
    assert u2.current_hp == u2.max_hp - 10 + 8, \
        f"regen must heal at turn-1 non-first init (got {u2.current_hp})"
    assert u2.current_moves == 1, "no MP refresh on turn 1"

    rd._apply_command(gs, ["end_turn"])
    rd._apply_command(gs, ["init_side", 1])   # turn 2 begins
    u1 = next(u for u in gs.map.units if u.id == "u1")
    assert u1.current_moves == u1.max_moves, "turn-2 init refreshes MP"
    rd._apply_command(gs, ["end_turn"])
    rd._apply_command(gs, ["init_side", 2])   # turn 2, side 2
    u2 = next(u for u in gs.map.units if u.id == "u2")
    # -10 +8 (t1 regen) = max-2; +8+2 at t2 clamps at max_hp.
    assert u2.current_hp == u2.max_hp,         f"turn-2 regen+rest should clamp to full (got {u2.current_hp})"
    assert u2.current_moves == u2.max_moves


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
    from tools import replay_dataset as rd
    from wesnoth_ai.classes import AttackSpecial  # noqa: F401
    gs = rd._build_initial_gamestate({
        "game_id": "t", "scenario_id": "multiplayer_Hornshark_Island",
        "factions": ["Rebels", "Loyalists"],
        "starting_sides": [
            {"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": [
            {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5,
             "y": 5, "is_leader": True},
            {"uid": 2, "type": "Bowman", "side": 2, "x": 27, "y": 23},
            {"uid": 3, "type": "Dwarvish Lord", "side": 2, "x": 20,
             "y": 5, "is_leader": True}],
        "starting_villages": [], "commands": [],
    })
    rd._setup_scenario_events(gs, "multiplayer_Hornshark_Island")
    u2 = next(u for u in gs.map.units if u.id == "u2")
    ranged = next(a for a in u2.attacks if a.is_ranged)
    assert "firststrike" in ranged.weapon_specials, (
        "MODIFY_BOWMAN prestart object must grant ranged firststrike"
    )
    u2.current_exp = u2.max_exp
    setattr(gs.global_info, "_advance_choices", [0])
    adv = rd._maybe_advance_unit(gs, u2)
    assert adv.name == "Longbowman", adv.name
    ranged2 = next(a for a in adv.attacks if a.is_ranged)
    assert "firststrike" in ranged2.weapon_specials, (
        "object-granted specials must survive advancement"
    )


def test_end_turn_mp_deficit_clears_resting():
    """unit::end_turn (unit.cpp:1078-1091, 1.18.4): a unit ending its
    side's turn with remaining MP != max MP loses `resting`, even if
    it never moved or fought -- MP-draining events count as activity.
    A unit at full MP keeps resting."""
    from tools import replay_dataset as rd
    gs = rd._build_initial_gamestate({
        "game_id": "t", "scenario_id": "multiplayer_Weldyn_Channel",
        "factions": ["Rebels", "Rebels"],
        "starting_sides": [
            {"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": [
            {"uid": 1, "type": "Elvish Captain", "side": 1, "x": 5,
             "y": 5, "is_leader": True},
            {"uid": 2, "type": "Elvish Fighter", "side": 1, "x": 7,
             "y": 5, "is_leader": False}],
        "starting_villages": [], "commands": [],
    })
    gs.global_info.current_side = 1
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u2 = next(u for u in gs.map.units if u.id == "u2")
    u1.statuses = set(u1.statuses) | {"resting"}
    u2.statuses = set(u2.statuses) | {"resting"}
    u1.current_moves = u1.max_moves - 1        # drained
    u2.current_moves = u2.max_moves            # untouched
    rd._apply_command(gs, ["end_turn"])
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u2 = next(u for u in gs.map.units if u.id == "u2")
    assert "resting" not in u1.statuses, \
        "MP deficit at end_turn must clear resting"
    assert "resting" in u2.statuses, "full-MP unit keeps resting"


def test_micro_isar_tentacle_never_rest_heals():
    """Full chain for the Micro Isar 38859 fix (2026-08-07): the
    repeating `turn refresh` event {MODIFY_UNIT (role=monster) moves 0}
    (enclave_micro_isar.cfg:86-91) zeroes tentacle MP every side turn,
    so unit::end_turn's MP check clears `resting` and the tentacle
    heals regen-only +8, never +10. User-verified viewer frames:
    turn 3 heal 7->15, turn 4 heal 15->23 (not 25); our former +2 rest
    left a 1-HP survivor whose ZoC forked the whole game."""
    from tools import replay_dataset as rd

    def tent(g):
        return next(u for u in g.map.units
                    if u.side == 3 and (u.position.x, u.position.y) == (3, 2))

    gs = rd._build_initial_gamestate({
        "game_id": "t", "scenario_id": "enclave_micro_isar",
        "factions": ["Drakes", "Undead"],
        "starting_sides": [
            {"side": 1, "gold": 100}, {"side": 2, "gold": 100}],
        "starting_units": [
            {"uid": 1, "type": "Drake Flare", "side": 1, "x": 0,
             "y": 5, "is_leader": True},
            {"uid": 2, "type": "Revenant", "side": 2, "x": 7,
             "y": 5, "is_leader": True}],
        "starting_villages": [], "commands": [],
    })
    rd._setup_scenario_events(gs, "enclave_micro_isar")
    rd._apply_command(gs, ["init_side", 1])    # turn-1 spawn + refresh
    t = tent(gs)
    assert getattr(t, "_wml_role", None) == "monster"
    assert t.max_moves > 0 and t.current_moves == 0, \
        "turn refresh MODIFY_UNIT must zero monster MP"
    t.current_hp = 5                            # wounded
    rd._apply_command(gs, ["end_turn"])
    rd._apply_command(gs, ["init_side", 2])
    rd._apply_command(gs, ["end_turn"])
    rd._apply_command(gs, ["init_side", 3])     # own init: +8 regen
    assert tent(gs).current_hp == 13, tent(gs).current_hp
    rd._apply_command(gs, ["end_turn"])         # MP 0 != max: no rest
    rd._apply_command(gs, ["init_side", 1])     # turn 2
    rd._apply_command(gs, ["end_turn"])
    rd._apply_command(gs, ["init_side", 2])
    rd._apply_command(gs, ["end_turn"])
    rd._apply_command(gs, ["init_side", 3])     # +8 only, NOT +10
    t = tent(gs)
    assert t.current_hp == 21, \
        f"regen-only heal expected (13+8=21), got {t.current_hp}"
    assert t.current_moves == 0, "turn-2 refresh must be re-zeroed"
