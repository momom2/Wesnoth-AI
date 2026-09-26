#!/usr/bin/env python3
"""Tests for the recruit-rejection retry loop (per the legality-mask
contract in CLAUDE.md).

Behaviors verified:

  1. Encoder retains fog hexes -- they show up in `hex_positions` /
     `pos_to_hex` and are eligible for recruit-mask lookup.
  2. Recruit-affordability mask: an unaffordable unit-type slot
     gets `actor_valid=0`.
  3. `_recruit_rejected_hexes` state lives on `gs.global_info` and
     gates the recruit-hex mask (its encoder bit is asserted in
     tests/test_encode_raw_cache.py).
  4. Sim's `_action_to_command` for recruit detects god-view
     occupancy: returns the retry sentinel + adds to the rejection
     set. `step()` honors the sentinel as a no-op (no history
     append, no side advance, last_step_rejected=True).
  5. Harness `_would_recruit_bounce` correctly classifies recruit
     actions targeting occupied hexes.
  6. A bounce survives the rest of the turn and clears at the next
     init_side on both states of record: the Python state and the
     Rust-owned core, whose `sim.gs` is a view rebuilt after every
     command (a rejection written to the view alone would be lost).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pytest

from wesnoth_ai.classes import (
    Alignment, GameState, GlobalInfo, Hex, Map, Position, SideInfo,
    Terrain, TerrainModifiers, Unit,
)
from wesnoth_ai.dummy_policy import DummyPolicy
from wesnoth_ai.game_core import game_core_class
from sim_test_helpers import scenario_setup
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
from tools.wesnoth_sim import WesnothSim


def _u(uid, side, x, y, *, hp=40, cost=14, is_leader=False, name="Spearman"):
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=hp, max_moves=5, max_exp=32, cost=cost,
        alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=hp, current_moves=5, current_exp=0,
        has_attacked=False,
        attacks=[], resistances=[1.0]*6, defenses=[0.5]*14,
        movement_costs=[1]*14, abilities=set(), traits=set(),
        statuses=set(),
    )


def _hex(x, y, mods=None, terrain=Terrain.FLAT):
    return Hex(position=Position(x, y),
               terrain_types={terrain},
               modifiers=set(mods or []))


# ---------------------------------------------------------------------
# Encoder: fog hexes retained
# ---------------------------------------------------------------------

def test_encoder_retains_fog_hexes():
    """After the 2026-04-28 fix, hex_positions includes fog hexes
    (their TERRAIN is visible; only units there would be hidden)."""
    from wesnoth_ai.encoder import encode_raw

    hexes = {
        _hex(0, 0), _hex(0, 1), _hex(1, 0), _hex(1, 1, terrain=Terrain.CASTLE),
    }
    fog = {Position(1, 1)}     # one fog hex
    sides = [SideInfo(player=f"S{i+1}", recruits=[],
                      current_gold=100, base_income=2,
                      nb_villages_controlled=0)
             for i in range(2)]
    gs = GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=fog,
                hexes=hexes, units=set()),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=sides,
    )
    raw = encode_raw(gs, type_to_id={}, faction_to_id={"": 0})
    positions = {(p.x, p.y) for p in raw.hex_positions}
    assert (1, 1) in positions, "fog hex should be retained in encoded set"


# ---------------------------------------------------------------------
# Affordability mask
# ---------------------------------------------------------------------

def _full_gs(*, recruits=("Spearman",), gold=100, leader_pos=(2, 2),
             extra_units=()):
    """Build a GameState with one keep + adjacent castle + leader on
    keep + recruit list + free castle hex. The recruit BFS needs a
    'connected' castle network."""
    keep = _hex(leader_pos[0], leader_pos[1],
                mods=[TerrainModifiers.KEEP], terrain=Terrain.CASTLE)
    castle = _hex(leader_pos[0] + 1, leader_pos[1],
                  mods=[TerrainModifiers.CASTLE], terrain=Terrain.CASTLE)
    far_flat = _hex(5, 5)
    leader = _u("ldr", 1, leader_pos[0], leader_pos[1], is_leader=True)
    side1 = SideInfo(player="S1", recruits=list(recruits),
                     current_gold=gold, base_income=2,
                     nb_villages_controlled=0)
    side2 = SideInfo(player="S2", recruits=[], current_gold=100,
                     base_income=2, nb_villages_controlled=0)
    return GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes={keep, castle, far_flat},
                units={leader, *extra_units}),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=[side1, side2],
    )


def test_affordability_mask_zeros_unaffordable_recruit():
    """A recruit slot for a Wose (cost 17) when we have 10 gold
    must have actor_valid=0 -- the policy can't waste a decision
    on it."""
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.encoder import GameStateEncoder

    gs = _full_gs(recruits=("Wose",), gold=10)
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    encoded = encoder.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    # Recruit slots come after unit slots. The leader is the only
    # unit (1 slot); recruit slot for Wose is at index 1.
    actor_valid = masks.actor_valid.squeeze(0).cpu().numpy()
    R = encoded.recruit_tokens.size(1)
    if R == 0:
        pytest.skip("no recruit slots emitted (encoder filtered)")
    # Find the slot whose recruit_type is "Wose".
    slot = encoded.recruit_types.index("Wose")
    a = encoded.unit_tokens.size(1) + slot
    assert actor_valid[a] == 0.0, (
        "Unaffordable Wose slot should be masked off; got "
        f"actor_valid={actor_valid[a]}")


def test_affordability_mask_lets_affordable_recruit_through():
    """With enough gold, the same Wose slot is selectable."""
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.encoder import GameStateEncoder

    gs = _full_gs(recruits=("Wose",), gold=50)
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    encoded = encoder.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    actor_valid = masks.actor_valid.squeeze(0).cpu().numpy()
    R = encoded.recruit_tokens.size(1)
    if R == 0:
        pytest.skip("no recruit slots emitted")
    slot = encoded.recruit_types.index("Wose")
    a = encoded.unit_tokens.size(1) + slot
    assert actor_valid[a] == 1.0


# ---------------------------------------------------------------------
# Rejection state + dynamic-flag feature
# ---------------------------------------------------------------------

def test_rejected_hex_blocked_by_recruit_mask():
    """A hex in the rejection set is excluded from
    `_recruit_hex_mask`'s output."""
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.encoder import GameStateEncoder

    gs = _full_gs(recruits=("Spearman",), gold=100)
    rejected_pos = (3, 2)
    setattr(gs.global_info, "_recruit_rejected_hexes",
            {rejected_pos})
    encoder = GameStateEncoder()
    encoder.register_names(gs)
    encoded = encoder.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    target_valid = masks.target_valid.cpu().numpy()
    R = encoded.recruit_tokens.size(1)
    if R == 0:
        pytest.skip("no recruit slots")
    slot = encoded.recruit_types.index("Spearman")
    a = encoded.unit_tokens.size(1) + slot
    j = encoded.hex_positions.index(Position(*rejected_pos))
    assert target_valid[a, j] == 0.0, (
        "rejected hex should be excluded from recruit target mask")


# ---------------------------------------------------------------------
# Sim retry-recruit sentinel
# ---------------------------------------------------------------------

def test_sim_recruit_on_occupied_hex_signals_retry():
    """sim._action_to_command for recruit on an occupied hex adds
    to the rejection set and returns the retry sentinel; step()
    sets last_step_rejected and doesn't consume the turn."""
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(seed=8, max_turns=5, mini=True)
    # The state surgery below replaces the whole map; make sure no
    # game-over latch from the bootstrap state survives it.
    sim.done = False
    sim.winner = 0
    sim.ended_by = ""
    # Build a controlled scenario: leader on keep, fog-hidden enemy
    # on adjacent castle.
    gs = sim.gs
    gs.map.units.clear()
    keep_pos = (2, 2)
    enemy_pos = (3, 2)
    keep = _hex(*keep_pos, mods=[TerrainModifiers.KEEP],
                terrain=Terrain.CASTLE)
    castle = _hex(*enemy_pos, mods=[TerrainModifiers.CASTLE],
                  terrain=Terrain.CASTLE)
    other_castle = _hex(2, 3, mods=[TerrainModifiers.CASTLE],
                        terrain=Terrain.CASTLE)
    # Use a fresh hex set with our castle network.
    gs.map = Map(
        size_x=10, size_y=10, mask=set(), fog=set(),
        hexes={keep, castle, other_castle},
        units={
            _u("ldr1", 1, *keep_pos, is_leader=True),
            _u("hidden_enemy", 2, *enemy_pos),
            _u("ldr2", 2, 8, 8, is_leader=True),
        },
    )
    # Side 1 has gold + recruits.
    gs.sides[0] = SideInfo(player="S1", recruits=["Spearman"],
                           current_gold=100, base_income=2,
                           nb_villages_controlled=0)
    sim._begin_side_turn(1)

    pre_history_len = len(sim.command_history)
    pre_actions = dict(sim._actions_by_side)
    pre_side = sim.gs.global_info.current_side

    # Try to recruit on the occupied hex. Step should be a no-op.
    sim.step({
        "type":       "recruit",
        "unit_type":  "Spearman",
        "target_hex": Position(*enemy_pos),
    })
    assert sim.last_step_rejected is True
    assert sim.gs.global_info.current_side == pre_side  # turn not consumed
    assert sim._actions_by_side == pre_actions          # not bumped
    # History got the init_side from _begin_side_turn but no recruit cmd.
    new_kinds = [rc.kind for rc in sim.command_history[pre_history_len:]]
    assert "recruit" not in new_kinds
    # The hex is now in the rejection set.
    rej = getattr(sim.gs.global_info, "_recruit_rejected_hexes", set())
    assert (enemy_pos[0], enemy_pos[1]) in rej


# ---------------------------------------------------------------------
# Harness pre-check
# ---------------------------------------------------------------------

def test_harness_would_recruit_bounce_detects_occupied():
    from tools.selfplay_game import _would_recruit_bounce

    gs = _full_gs(extra_units=(_u("intruder", 2, 3, 2),))
    occupied_action = {
        "type":       "recruit",
        "unit_type":  "Spearman",
        "target_hex": Position(3, 2),
    }
    assert _would_recruit_bounce(occupied_action, gs) is True


def test_harness_would_recruit_bounce_passes_empty():
    from tools.selfplay_game import _would_recruit_bounce

    gs = _full_gs()
    free_action = {
        "type":       "recruit",
        "unit_type":  "Spearman",
        "target_hex": Position(3, 2),
    }
    assert _would_recruit_bounce(free_action, gs) is False


def test_harness_would_recruit_bounce_ignores_non_recruit():
    from tools.selfplay_game import _would_recruit_bounce

    gs = _full_gs()
    move_action = {
        "type":       "move",
        "start_hex":  Position(2, 2),
        "target_hex": Position(3, 2),
    }
    assert _would_recruit_bounce(move_action, gs) is False


def _sim(seed: int, use_core: bool) -> WesnothSim:
    setup = scenario_setup(seed, mini=True)
    return WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                      max_turns=6, use_core=use_core)


@pytest.mark.parametrize("use_core", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        game_core_class() is None, reason="wesnoth_core.GameCore not available")),
])
def test_a_recruit_bounce_survives_a_mid_turn_command(use_core):
    sim = _sim(3, use_core)
    action = DummyPolicy().select_action(sim.gs, game_label="bounce")
    if action.get("type") == "end_turn":
        pytest.skip("the driver opened with a turn-ending action")
    hexes = sorted(sim.gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
    spot = (hexes[4].position.x, hexes[4].position.y)

    sim.reject_recruit_hex(*spot)
    assert spot in (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None) or set())

    turn_before = sim.turn_number
    sim.step(action)
    assert sim.turn_number == turn_before, "the action ended the turn; pick another"
    assert spot in (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None) or set()), \
        "the bounce was lost when the state of record was re-read"


@pytest.mark.parametrize("use_core", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        game_core_class() is None, reason="wesnoth_core.GameCore not available")),
])
def test_the_bounce_clears_at_the_next_init_side(use_core):
    """Per TURN, not forever: the enemy may have moved away."""
    sim = _sim(3, use_core)
    hexes = sorted(sim.gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
    spot = (hexes[4].position.x, hexes[4].position.y)
    sim.reject_recruit_hex(*spot)
    sim.step({"type": "end_turn"})          # end_turn runs the next side's init_side
    assert spot not in (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None) or set())
