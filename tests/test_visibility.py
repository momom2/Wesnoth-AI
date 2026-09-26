"""Tests for the fog-of-war contract (`visibility.py`) and its
integration into `encoder.py` and `action_sampler.py`.

The contract:
  * Own-side units: always visible to own side.
  * Enemy units on hexes the side sees and NOT hiding: visible
    (what a side sees: tests/test_vision.py).
  * Enemy units on hexes it does not see: NOT visible.
  * Enemy units with active hide-cover ability AND not in
    `global_info._uncovered_units`: NOT visible.
  * Recruit phantoms emitted by the encoder: ONLY for the
    current side. Enemy recruit lists are fog-hidden.
  * Action-sampler legality (occupancy / enemy_mask): hidden
    enemies are treated as empty hexes -- the policy can
    attempt to move there (engine will reveal on contact) but
    cannot click-to-attack them.
"""
from __future__ import annotations


import pytest

from wesnoth_ai.classes import (GameState, Map, GlobalInfo, Unit, Hex, Position,
                     SideInfo, Terrain, Alignment)
from wesnoth_ai import visibility


# ---- helpers ------------------------------------------------------

def _hexes_grid(w: int, h: int = 1, terrain=Terrain.FLAT):
    """Build a w x h grid of plain hexes."""
    return {Hex(position=Position(x=x, y=y),
                terrain_types=frozenset({terrain}),
                modifiers=frozenset())
            for x in range(w) for y in range(h)}


def _unit(uid: str, x: int, side: int, max_moves: int = 2,
          abilities=frozenset(), is_leader=False, name='Test'):
    """A unit of type `name`: the placeholder 'Test' pays 1 MP per hex on
    boards without terrain codes; on coded terrain the vision tests use a
    real type, whose movement costs the resolver knows."""
    return Unit(
        id=uid, name=name, name_id='test', side=side,
        is_leader=is_leader, position=Position(x=x, y=0),
        max_hp=10, max_moves=max_moves, max_exp=20, cost=10,
        alignment=Alignment.NEUTRAL, levelup_names=tuple(),
        current_hp=10, current_moves=max_moves, current_exp=0,
        has_attacked=False, attacks=tuple(),
        resistances={}, defenses={}, movement_costs={},
        abilities=frozenset(abilities), traits=tuple(),
        statuses=frozenset(),
    )


def _state(units, hexes, current_side=1, turn=1, recruits=()):
    """Build a GameState with the given units + hexes."""
    return GameState(
        game_id='t',
        map=Map(size_x=20, size_y=4, mask=set(), fog=set(),
                hexes=hexes, units=set(units)),
        global_info=GlobalInfo(current_side=current_side, turn_number=turn,
                              time_of_day=None, village_gold=2,
                              village_upkeep=1, base_income=2),
        sides=[
            SideInfo(player=1, recruits=tuple(recruits),
                     current_gold=100, base_income=2,
                     nb_villages_controlled=0, faction='h'),
            SideInfo(player=2, recruits=tuple(recruits),
                     current_gold=100, base_income=2,
                     nb_villages_controlled=0, faction='h'),
        ],
    )


# ---- own visibility -----------------------------------------------

def test_own_units_always_visible():
    """Own-side units must always appear in `units_visible_to`,
    regardless of where they stand."""
    units = [
        _unit('mine_close', x=0, side=1),
        _unit('mine_far',   x=18, side=1),   # far from everyone
    ]
    s = _state(units, _hexes_grid(20))
    seen = visibility.units_visible_to(s, side=1)
    assert {u.id for u in seen} == {'mine_close', 'mine_far'}


# ---- enemy visibility ------------------------------------------------

def test_enemy_in_sight_is_visible():
    units = [
        _unit('mine', x=0, side=1, max_moves=3),    # sees x <= 4
        _unit('enemy_close', x=2, side=2),
    ]
    s = _state(units, _hexes_grid(20))
    seen = {u.id for u in visibility.units_visible_to(s, side=1)}
    assert 'enemy_close' in seen


def test_enemy_outside_sight_is_invisible():
    units = [
        _unit('mine', x=0, side=1, max_moves=3),    # sees x <= 4
        _unit('enemy_far', x=10, side=2),            # far outside
    ]
    s = _state(units, _hexes_grid(20))
    seen = {u.id for u in visibility.units_visible_to(s, side=1)}
    assert 'mine' in seen
    assert 'enemy_far' not in seen


def test_visibility_is_per_side():
    """Side 1 and side 2 may see different subsets of the same
    god-view unit list."""
    units = [
        _unit('a1', x=0, side=1, max_moves=2),
        _unit('a2', x=15, side=1, max_moves=2),
        _unit('b1', x=2, side=2, max_moves=2),    # near a1
        _unit('b2', x=10, side=2, max_moves=2),   # nobody near
    ]
    s = _state(units, _hexes_grid(20))
    seen1 = {u.id for u in visibility.units_visible_to(s, 1)}
    seen2 = {u.id for u in visibility.units_visible_to(s, 2)}
    # Side 1 sees its own + b1 (close to a1); not b2.
    assert seen1 == {'a1', 'a2', 'b1'}
    # Side 2 sees its own + a1 (close to b1); not a2.
    assert seen2 == {'b1', 'b2', 'a1'}


# ---- ambush handling ---------------------------------------------

def test_ambush_unit_in_forest_is_hidden_until_uncovered():
    """Ambush on forest: not in sight set until in `_uncovered_units`.
    The unit is within sight range, so the only reason it's hidden
    is the ambush ability.

    `_hide_cover_active` matches the engine's `*^F*` glob against
    the hex's WML terrain CODE from `global_info._terrain_codes`
    (NOT the `Hex.terrain_types` enum field), so this test
    pre-populates that dict at the lurker's hex. Real game states
    get these codes from the scenario parser; tests set them by hand.

    The code is `Gs^Fms`, mixed deciduous forest, deliberately: the
    hand-rolled defense-key table this predicate used until
    2026-09-13 did NOT list it, so the lurker stayed visible there.
    A code the old table happened to cover (`Gg^Fp`) would pass under
    both rules and prove nothing about the fix.

    The observer is a Spearman: forest costs it 2 of its 3 MP, so it
    reaches x=1 and sees x=2, where the lurker stands.
    """
    forest_hexes = {Hex(position=Position(x=x, y=0),
                       terrain_types=frozenset({Terrain.FOREST}),
                       modifiers=frozenset())
                   for x in range(20)}
    units = [
        _unit('mine', x=0, side=1, max_moves=3, name='Spearman'),
        _unit('lurker', x=2, side=2, max_moves=2,
              abilities={'ambush'}),                # inside sight, on forest
    ]
    s = _state(units, forest_hexes)
    # Populate WML terrain codes so the hex matches ambush's `*^F*`.
    s.global_info._terrain_codes = {(x, 0): 'Gs^Fms' for x in range(20)}
    seen = {u.id for u in visibility.units_visible_to(s, 1)}
    assert 'lurker' not in seen

    # Now mark it as uncovered (e.g., post-ambush trigger).
    s.global_info._uncovered_units = {'lurker'}
    seen = {u.id for u in visibility.units_visible_to(s, 1)}
    assert 'lurker' in seen


def test_ambush_off_cover_terrain_does_not_hide():
    """The same ambush unit on FLAT terrain is NOT hiding (cover
    condition not met). Should be visible if in sight range."""
    units = [
        _unit('mine', x=0, side=1, max_moves=3),
        _unit('lurker_outside_forest', x=2, side=2,
              abilities={'ambush'}),  # FLAT here, no cover
    ]
    s = _state(units, _hexes_grid(20))  # default FLAT
    seen = {u.id for u in visibility.units_visible_to(s, 1)}
    assert 'lurker_outside_forest' in seen


# ---- encoder integration -----------------------------------------

def _encode_raw(s):
    """Build the encoder's vocab dicts on the fly so encode_raw
    can run on a synthetic state. The dicts need (at minimum)
    the unit-type names that appear in `s`, plus a default
    faction id of 0 for the placeholder faction string 'h'.
    `encode_raw` is module-level (not a method) so we import it
    directly rather than going through a GameStateEncoder."""
    from wesnoth_ai.encoder import encode_raw
    type_to_id = {u.name: i for i, u in enumerate(s.map.units)}
    faction_to_id = {'h': 0}
    return encode_raw(s, type_to_id=type_to_id,
                      faction_to_id=faction_to_id)


def test_encoder_omits_fog_hidden_enemy_tokens():
    """Encoder's `encode_raw` must emit unit tokens only for
    visible units."""
    units = [
        _unit('mine_leader', x=0, side=1, max_moves=2, is_leader=True),
        _unit('enemy_close', x=2, side=2),       # within sight
        _unit('enemy_far',   x=15, side=2),      # outside sight
    ]
    s = _state(units, _hexes_grid(20))
    raw = _encode_raw(s)
    ids = list(raw.unit_ids)
    assert 'mine_leader' in ids
    assert 'enemy_close' in ids
    assert 'enemy_far' not in ids


def test_encoder_omits_cover_hidden_enemy_tokens():
    """A unit hidden by its ABILITY, not by distance, must also lose
    its token. The enemy here stands on a hex the side sees (a
    Spearman with 5 MP reaches x=2 through forest and sees x=3): cover
    is the only reason it can be absent, and `Gs^Fms` is a forest code
    the pre-2026-09-13 defense-key table did not list, so this fails
    against that rule.

    This is the half the corpus certification could not reach --
    `diff_replay` checks that recorded commands stay legal, never what
    the policy was shown.
    """
    units = [
        _unit('mine_leader', x=0, side=1, max_moves=5, is_leader=True, name='Spearman'),
        _unit('plain_enemy', x=2, side=2),
        _unit('lurker', x=3, side=2, abilities={'ambush'}),
    ]
    s = _state(units, _hexes_grid(20))
    s.global_info._terrain_codes = {(x, 0): 'Gs^Fms' for x in range(20)}
    ids = list(_encode_raw(s).unit_ids)
    assert 'mine_leader' in ids and 'plain_enemy' in ids, \
        "control: units in sight without cover keep their tokens"
    assert 'lurker' not in ids, \
        "an ambusher on ^Fms must not reach the policy's observation"

    # Uncovered by a previous ambush trigger: the token comes back.
    s.global_info._uncovered_units = {'lurker'}
    assert 'lurker' in list(_encode_raw(s).unit_ids)


def test_encoder_recruit_phantoms_only_for_current_side():
    """encode_raw should emit recruit phantoms only for the side
    currently acting. Enemy recruit lists are fog-hidden."""
    units = [
        _unit('mine_leader', x=0, side=1, is_leader=True),
        _unit('enemy_leader', x=2, side=2, is_leader=True),
    ]
    s = _state(units, _hexes_grid(20),
               current_side=1,
               recruits=('Soldier', 'Healer'))
    raw = _encode_raw(s)
    # recruit_is_ours should be all 1.0 (only own side emitted).
    assert (raw.recruit_is_ours == 1.0).all()
    # Number of phantoms == |our recruits|, not |our| + |their|.
    assert len(raw.recruit_is_ours) == 2


# ---- visible_hexes / visible_fraction -----------------------------

def test_visible_hexes_are_the_reach_and_the_ring_around_it():
    """A 2-MP unit on open ground reaches 2 hexes each way and sees the
    third (tests/test_vision.py covers terrain and the turn)."""
    units = [_unit('u', x=5, side=1, max_moves=2)]
    s = _state(units, _hexes_grid(20))
    assert visibility.visible_hexes_for(s, 1) == {(x, 0) for x in range(2, 9)}


def test_visible_fraction_in_unit_interval():
    units = [_unit('u', x=5, side=1, max_moves=2)]
    s = _state(units, _hexes_grid(20))
    f = visibility.visible_fraction_for(s, 1)
    assert 0.0 < f <= 1.0
    assert f == pytest.approx(7/20)


# ---- move onto hidden units: Wesnoth blocked/ambush semantics ----
# (2026-07-17: replaces the retired harness-side fog-bounce
# pre-check `_would_move_bounce_on_fog` -- moves onto hidden enemies
# now EXECUTE with the engine's partial-move resolution via
# tools/pathfind_sim.walk_move_path.)

def test_walk_blocked_by_hidden_enemy_keeps_mp():
    """A hidden unit ON a path hex stops the mover on the hex
    BEFORE it, KEEPS remaining MP (post_move zeroes MP only for
    ambush/ZoC-final, move.cpp:1041-1043), and reveals the
    blocker.

    Setup note: the mover gets max_moves=1, so it sees x <= 2 (its
    reach and the ring around it) and the lurker at distance 3 is
    fog-hidden and exerts no ZoC; the walk budget is an explicit 4."""
    from tools.pathfind_sim import walk_move_path
    units = [
        _unit('mover', x=0, side=1, max_moves=1),
        _unit('lurker', x=3, side=2),      # on the path, fog-hidden
    ]
    s = _state(units, _hexes_grid(6))
    mover = next(u for u in s.map.units if u.id == 'mover')
    out = walk_move_path(s, mover, [0, 1, 2, 3, 4], [0, 0, 0, 0, 0],
                         budget=4)
    assert out.stop_reason == "blocked"
    assert out.final_idx == 2               # stopped BEFORE (3,0)
    assert out.uncovered_ids == ['lurker']
    # 2 MP spent on flat terrain, 2 kept (NOT zeroed).
    assert out.mp_left == 2


def test_walk_ambush_by_hidden_hider_zeroes_mp():
    """Entering a hex adjacent to a hidden `hides` enemy stops the
    mover AT that hex, zeroes MP, and reveals the ambusher
    (check_for_ambushers, move.cpp:422-440)."""
    from tools.pathfind_sim import walk_move_path
    units = [
        _unit('mover', x=0, side=1, max_moves=4),
        # Ambusher OFF the path (y=1) but adjacent to path hex (2,0).
        # nightstalk covers via lawful_bonus<0; simpler: monkeypatch
        # is avoided by giving it `ambush` and forest terrain below.
        _unit('wose', x=2, side=2, abilities=frozenset({'ambush'})),
    ]
    hexes = _hexes_grid(6)
    # Put the ambusher's hex on forest so its cover is active.
    # Hide-cover resolution matches ambush's `*^F*` glob against the
    # raw terrain-code dict (`_terrain_codes`), so stamp the code
    # rather than the Hex enum. `Gs^Fms` is a forest the pre-2026-09-13
    # defense-key table missed, so this asserts the move truncation on
    # a hex where the old rule would NOT have stopped the mover -- the
    # one thing the corpus sweep cannot check.
    hexes = {h for h in hexes if not (h.position.x == 2 and h.position.y == 0)}
    hexes.add(Hex(position=Position(x=2, y=0),
                  terrain_types=frozenset({Terrain.FOREST}),
                  modifiers=frozenset()))
    s = _state(units, hexes)
    s.global_info._terrain_codes = {(2, 0): "Gs^Fms"}
    mover = next(u for u in s.map.units if u.id == 'mover')
    # Path passes adjacent to (2,0): entering (1,0) is adjacent.
    out = walk_move_path(s, mover, [0, 1], [0, 0])
    assert out.stop_reason == "ambush"
    assert out.final_idx == 1               # stopped AT the entered hex
    assert 'wose' in out.uncovered_ids
    assert out.mp_left == 0


def test_walk_passes_through_ally_and_backtracks_off_it():
    """Own-side units are pass-through (pathfind.cpp:777-786), but
    a move may not END on one: the walk backtracks off occupied end
    hexes (plot_turn, move.cpp:776-780) with MP refunded."""
    from tools.pathfind_sim import walk_move_path
    units = [
        _unit('mover', x=0, side=1, max_moves=4),
        _unit('buddy', x=2, side=1),
    ]
    s = _state(units, _hexes_grid(6))
    mover = next(u for u in s.map.units if u.id == 'mover')
    # Through the ally, landing beyond: fine.
    out = walk_move_path(s, mover, [0, 1, 2, 3], [0, 0, 0, 0])
    assert out.stop_reason == "end" and out.final_idx == 3
    assert out.mp_left == 1
    # Ordered to END on the ally's hex: backtrack to (1,0).
    out2 = walk_move_path(s, mover, [0, 1, 2], [0, 0, 0])
    assert out2.final_idx == 1
    assert out2.mp_left == 3                # only 1 MP charged


def test_legality_mask_offers_a_reachable_empty_hex():
    """Our unit on (0,0) with 2 MP on a two-hex line: the empty (1,0)
    is a legal target."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.action_sampler import _build_legality_masks

    s = _state([_unit('mine', x=0, side=1, max_moves=2)], _hexes_grid(2), current_side=1)
    enc = GameStateEncoder(d_model=8)
    enc.register_names(s)
    encoded = enc.encode(s)
    masks = _build_legality_masks(encoded, s)
    # `target_valid` is [A, H]: the unit's actor row, the hex's column.
    row = masks.target_valid[encoded.unit_ids.index('mine')]
    assert float(row[encoded.pos_to_hex[(1, 0)]].item()) > 0.0


def test_empty_state_zero_visibility():
    """No units OR no hexes -> 0.0 fraction, empty visible set."""
    assert visibility.visible_fraction_for(
        _state([], set()), 1) == 0.0
    assert visibility.visible_hexes_for(
        _state([], set()), 1) == set()
    # Side with no units on a populated map: also 0.
    enemy_only = [_unit('e', x=0, side=2)]
    s = _state(enemy_only, _hexes_grid(20))
    assert visibility.visible_fraction_for(s, 1) == 0.0
