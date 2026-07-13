"""RNG accounting: every random resolution in a sim game must be
PINNED in the recorded command stream and the exported save, leaving
NOTHING for Wesnoth to resolve at playback.

Threat model (user, 2026-06-12): the dangerous failure class is not
the sim rolling unseeded RNG internally — twin-run determinism tests
catch that — but the sim implicitly recording a command in a form
where WESNOTH rolls the dice on playback. Sim-vs-sim agreement is
structurally blind to it: both runs share the wrong convention. The
concrete precedent is the retaliation bug: attacks were recorded
with defender_weapon=-1, the sim applied no counter, and Wesnoth's
battle_context auto-selected one at playback — silent divergence
with empty [checkup] blocks unable to flag it.

Known synced-RNG / user-choice sites (from the 1.18.4 engine
research, docs/wesnoth_rules.md):
  - attack: damage rolls (seed) + defender counter weapon choice
  - recruit: trait rolls (seed)
  - advancement: which advance_to the unit takes ([choose])

The end-state oracle: exported [attack] commands carry FULL
[checkup] blocks (per-strike chance/hits/damage/dies [result]
children) which Wesnoth's synced_checkup COMPARES at playback,
raising the SYNC error on divergence — so any manual replay
viewing verifies the sim's combat math (landed 2026-06-12; fully
non-interactive playback is impossible on the stock binary, see
BACKLOG). This file remains the automatable contract check: seeds
pinned, choices pinned, and checkup payloads present and
well-formed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from dummy_policy import DummyPolicy   # noqa: E402
from sim_test_helpers import fresh_scenario_sim   # noqa: E402
from tools.abilities import hex_neighbors   # noqa: E402

_HEX8 = re.compile(r"^[0-9a-f]{8}$")


def _teleport_adjacent(sim, attacker, defender):
    """State surgery: place `attacker` on a free hex next to
    `defender` (combat reads positions, not how they got there)."""
    occupied = {(u.position.x, u.position.y) for u in sim.gs.map.units}
    spot = next(
        ((nx, ny) for nx, ny in hex_neighbors(defender.position.x,
                                              defender.position.y)
         if (nx, ny) not in occupied
         and 0 <= nx < sim.gs.map.size_x
         and 0 <= ny < sim.gs.map.size_y),
        None)
    if spot is None:
        pytest.skip("no free hex adjacent to the defender")
    attacker.position.x, attacker.position.y = spot


def _leaders(sim):
    side = sim.gs.global_info.current_side
    att = next(u for u in sim.gs.map.units
               if u.side == side and u.attacks)
    dfd = next(u for u in sim.gs.map.units
               if u.side != side and u.attacks
               and u.side in (1, 2)
               and "petrified" not in (u.statuses or set()))
    return att, dfd


def test_attack_pins_seed_and_counter_choice():
    """Every recorded attack carries (a) a concrete 8-hex damage
    seed and (b) a CONCRETE defender weapon index whenever the
    defender has a matching-range weapon. (b) is the assertion that
    would have caught the retaliation bug on the day it was written:
    -1 is only a legal recording when no counter exists, because
    Wesnoth's battle_context AUTO-SELECTS on -1 at playback."""
    sim = fresh_scenario_sim(seed=12, max_turns=20, mini=True)
    att, dfd = _leaders(sim)
    _teleport_adjacent(sim, att, dfd)

    a_ranged = att.attacks[0].is_ranged
    has_counter = any(w.is_ranged == a_ranged for w in dfd.attacks)

    sim.step({"type": "attack", "start_hex": att.position,
              "target_hex": dfd.position, "attack_index": 0})
    rc = sim.command_history[-1]
    assert rc.kind == "attack", f"attack not recorded: {rc.kind}"

    seed = rc.cmd[7]
    assert _HEX8.match(str(seed)), (
        f"attack damage seed not pinned: {seed!r} -- Wesnoth would "
        f"roll its own dice on playback")
    d_weapon = rc.cmd[6]
    if has_counter:
        assert d_weapon >= 0, (
            f"defender has a matching-range counter but the recorded "
            f"defender_weapon is {d_weapon}; Wesnoth auto-selects on "
            f"-1 at playback while the sim applies NO counter -- "
            f"silent divergence (the 2026-06-12 retaliation bug)")
    else:
        assert d_weapon == -1, (
            f"no matching-range counter exists but defender_weapon="
            f"{d_weapon} was recorded")


def test_recruit_pins_trait_seed():
    """Every recruit command carries a concrete trait seed -- even
    for musthave-only races (the sim allocates the seed regardless;
    whether Wesnoth consumes it is the emitter's lazy-[random_seed]
    concern, tested below)."""
    sim = fresh_scenario_sim(seed=13, max_turns=8, mini=True)
    pol = DummyPolicy()
    for _ in range(80):
        if sim.done:
            break
        sim.step(pol.select_action(sim.gs, game_label="acct"))
    recruits = [rc for rc in sim.command_history
                if rc.kind == "recruit"]
    if not recruits:
        pytest.skip("DummyPolicy recruited nothing on this seed")
    for rc in recruits:
        assert _HEX8.match(str(rc.cmd[4])), (
            f"recruit trait seed not pinned: {rc.cmd!r}")


def test_advancement_pins_choice_in_history_and_export():
    """A level-up's advance-target choice must be recorded (extras
    advance_choices) and emitted as a [choose] block -- otherwise
    Wesnoth playback asks its OWN rng/choice machinery which
    advancement to take.

    Engineered deterministically: attacker at max_exp - 1 gains
    combat XP (>= opponent level >= 1) from ANY strike exchange,
    kill or not, so the advancement fires regardless of hit rolls."""
    from tools.openers import recruit_type
    from tools.sim_to_replay import build_save_wml
    from tools.replay_extract import parse_wml

    # Leaders can be max-level (AMLA-only, no advances_to); a fresh
    # level-1 recruit ALWAYS has an advancement target.
    sim = fresh_scenario_sim(seed=14, max_turns=20, mini=True)
    side = sim.gs.global_info.current_side
    unit_type = sim.gs.sides[side - 1].recruits[0]
    r_action = recruit_type(unit_type)(sim.gs, side)
    assert r_action is not None, "leader can't recruit at start"
    sim.step(r_action)
    sim.step({"type": "end_turn"})    # side 2's turn
    sim.step({"type": "end_turn"})    # back to us; recruit can act
    att = next(u for u in sim.gs.map.units
               if u.side == side and not u.is_leader)
    dfd = next(u for u in sim.gs.map.units
               if u.side != side and u.attacks
               and u.side in (1, 2)
               and "petrified" not in (u.statuses or set()))
    _teleport_adjacent(sim, att, dfd)
    att.current_exp = max(0, att.max_exp - 1)
    # Survival surgery: the counter-attack must not kill the
    # attacker before XP is awarded (dead units don't advance).
    att.current_hp = 80

    sim.step({"type": "attack", "start_hex": att.position,
              "target_hex": dfd.position, "attack_index": 0})
    rc = sim.command_history[-1]
    assert rc.kind == "attack"
    choices = rc.extras.get("advance_choices")
    assert choices, (
        "attacker crossed the XP threshold but no advance choice was "
        "recorded -- Wesnoth playback would pick the advancement "
        "itself")

    wml = build_save_wml(sim)
    root = parse_wml(wml)
    choose_blocks = []
    replays = root.all("replay")
    for rep in replays:
        for cmd in rep.all("command"):
            ch = cmd.first("choose")
            if ch is not None:
                choose_blocks.append(ch)
    assert len(choose_blocks) >= len(choices), (
        f"{len(choices)} advancement choice(s) in history but only "
        f"{len(choose_blocks)} [choose] block(s) in the exported "
        f"save")


def test_export_every_rng_consumer_has_followup():
    """WML-level audit of an exported game containing recruits and
    a combat: every [attack] must carry weapon= AND defender_weapon=
    and be followed by a dependent [random_seed]; every [recruit]
    must either be followed by [random_seed] (trait roll consumed)
    or carry a completed checkup ([result] child -- the seedless
    musthave-race form). Anything else leaves a roll to Wesnoth."""
    from tools.sim_to_replay import build_save_wml
    from tools.replay_extract import parse_wml

    sim = fresh_scenario_sim(seed=15, max_turns=8, mini=True)
    pol = DummyPolicy()
    for _ in range(40):
        if sim.done:
            break
        sim.step(pol.select_action(sim.gs, game_label="acct2"))
    if not sim.done:
        att, dfd = _leaders(sim)
        _teleport_adjacent(sim, att, dfd)
        sim.step({"type": "attack", "start_hex": att.position,
                  "target_hex": dfd.position, "attack_index": 0})

    kinds = {rc.kind for rc in sim.command_history}
    assert "attack" in kinds and "recruit" in kinds, (
        f"game must exercise both RNG sites; got {sorted(kinds)}")

    wml = build_save_wml(sim)
    root = parse_wml(wml)
    # The final [replay] block carries the command stream.
    commands = []
    for rep in root.all("replay"):
        cmds = rep.all("command")
        if cmds:
            commands = cmds
    assert commands, "no [command] stream in exported save"

    def _followed_by_seed(idx: int) -> bool:
        for nxt in commands[idx + 1:idx + 4]:
            if nxt.first("random_seed") is not None:
                return True
            # Another player action before any seed: nothing pinned.
            if any(nxt.first(t) is not None for t in
                   ("attack", "recruit", "move", "end_turn",
                    "init_side")):
                return False
        return False

    n_attacks = n_recruits = 0
    for i, cmd in enumerate(commands):
        atk = cmd.first("attack")
        if atk is not None:
            n_attacks += 1
            assert atk.attrs.get("weapon") is not None, "no weapon="
            assert atk.attrs.get("defender_weapon") is not None, (
                "attack without defender_weapon= -- Wesnoth "
                "auto-selects the counter at playback")
            assert _followed_by_seed(i), (
                "attack without a dependent [random_seed] -- Wesnoth "
                "rolls its own damage dice at playback")
            # VERIFIABLE checkup: per strike the engine compares two
            # [result] children ({chance,hits,damage} then {dies});
            # an empty [checkup] silently records instead of
            # verifying (synced_checkup.cpp). Require the payload
            # and its 2-results-per-strike shape.
            checkup = cmd.first("checkup")
            assert checkup is not None, "attack without [checkup]"
            results = checkup.all("result")
            assert len(results) >= 2 and len(results) % 2 == 0, (
                f"attack [checkup] carries {len(results)} [result] "
                f"children; expected an even count >= 2 (two per "
                f"strike) -- playback would verify nothing")
            for j in range(0, len(results), 2):
                first, second = results[j], results[j + 1]
                assert {"chance", "hits", "damage"} <= set(
                    first.attrs), f"result {j}: {first.attrs}"
                assert "dies" in second.attrs, (
                    f"result {j + 1}: {second.attrs}")
        rec = cmd.first("recruit")
        if rec is not None:
            n_recruits += 1
            checkup = cmd.first("checkup")
            completed = (checkup is not None
                         and checkup.first("result") is not None)
            assert _followed_by_seed(i) or completed, (
                "recruit with neither [random_seed] follow-up nor a "
                "completed checkup -- trait roll left to Wesnoth")
    assert n_attacks >= 1 and n_recruits >= 1
