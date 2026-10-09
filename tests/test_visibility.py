"""Tests for the fog-of-war contract (`visibility.py`, answered by the
Rust core) and its integration into `encoder.py` and `action_sampler.py`.

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

The boards are one row of grass (tests/helpers/parity_games.py), so a
unit with m movement points sees m + 1 hexes each way.
"""
from __future__ import annotations


import pytest

from wesnoth_ai import game_core as gc
from wesnoth_ai import visibility

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


# ---- helpers ------------------------------------------------------

def _board(units, *, width=20, code=None, recruits=None):
    """The core of a fogged one-row board and a view bound to it.
    `units`: (id, type, side, x, extras) with extras among max_moves,
    abilities, is_leader; `code` covers every hex."""
    from helpers.parity_games import record, state_of
    special = {(x, 0): code for x in range(width)} if code else None
    gs = state_of(record([(t, s, x, 0, bool(extra.get("is_leader"))) for _uid, t, s, x, extra in units],
                         width=width, height=1, special=special, fog=True, recruits=recruits))
    cs = gc.CoreState.from_state(gs)
    names = {}
    for (uid, _t, _s, x, extra), u in zip(units, sorted(gs.map.units, key=lambda u: u.position.x)):
        assert u.position.x == x
        names[u.id] = uid
        fields = {k: v for k, v in extra.items() if k in ("max_moves", "abilities")}
        if fields:
            cs.core.update_unit(u.id, fields)
    return cs, names


def _seen(cs, names, side):
    """The names of the units `side` sees."""
    return {names[u.id] for u in visibility.units_visible_to(gc.view_of(cs), side)}


# ---- own visibility -----------------------------------------------

def test_own_units_always_visible():
    """Own-side units must always appear in `units_visible_to`,
    regardless of where they stand."""
    cs, names = _board([("mine_close", "Spearman", 1, 0, {}), ("mine_far", "Spearman", 1, 18, {})])
    assert _seen(cs, names, 1) == {"mine_close", "mine_far"}


# ---- enemy visibility ------------------------------------------------

def test_enemy_in_sight_is_visible():
    cs, names = _board([("mine", "Spearman", 1, 0, {"max_moves": 3}),     # sees x <= 4
                        ("enemy_close", "Spearman", 2, 2, {})])
    assert "enemy_close" in _seen(cs, names, 1)


def test_enemy_outside_sight_is_invisible():
    cs, names = _board([("mine", "Spearman", 1, 0, {"max_moves": 3}),
                        ("enemy_far", "Spearman", 2, 10, {})])
    assert _seen(cs, names, 1) == {"mine"}


def test_visibility_is_per_side():
    """Side 1 and side 2 may see different subsets of the same
    god-view unit list."""
    two = {"max_moves": 2}
    cs, names = _board([("a1", "Spearman", 1, 0, two), ("b1", "Spearman", 2, 2, two),
                        ("b2", "Spearman", 2, 10, two), ("a2", "Spearman", 1, 15, two)])
    # Side 1 sees its own + b1 (close to a1); not b2.
    assert _seen(cs, names, 1) == {"a1", "a2", "b1"}
    # Side 2 sees its own + a1 (close to b1); not a2.
    assert _seen(cs, names, 2) == {"b1", "b2", "a1"}


# ---- ambush handling ---------------------------------------------

def test_ambush_unit_in_forest_is_hidden_until_uncovered():
    """Ambush on forest: not in sight set until in `_uncovered_units`.
    The unit is within sight range, so the only reason it's hidden is
    the ambush ability. The cover matches the engine's `*^F*` glob
    against the hex's terrain CODE.

    The code is `Gs^Fms`, mixed deciduous forest, deliberately: the
    hand-rolled defense-key table the predicate used until 2026-09-13
    did NOT list it, so the lurker stayed visible there. A code the old
    table happened to cover (`Gg^Fp`) would pass under both rules and
    prove nothing about the fix.

    The observer is a Spearman: forest costs it 2 of its 3 MP, so it
    reaches x=1 and sees x=2, where the lurker stands."""
    cs, names = _board([("mine", "Spearman", 1, 0, {"max_moves": 3}),
                        ("lurker", "Spearman", 2, 2, {"abilities": ["ambush"]})], code="Gs^Fms")
    assert "lurker" not in _seen(cs, names, 1)
    # Now mark it as uncovered (e.g., post-ambush trigger).
    lurker = next(uid for uid, name in names.items() if name == "lurker")
    cs.core.set_uncovered([lurker])
    assert "lurker" in _seen(cs, names, 1)


def test_ambush_off_cover_terrain_does_not_hide():
    """The same ambush unit on FLAT terrain is NOT hiding (cover
    condition not met). Should be visible if in sight range."""
    cs, names = _board([("mine", "Spearman", 1, 0, {"max_moves": 3}),
                        ("lurker_outside_forest", "Spearman", 2, 2, {"abilities": ["ambush"]})])
    assert "lurker_outside_forest" in _seen(cs, names, 1)


# ---- encoder integration -----------------------------------------

def _unit_names(cs, names):
    """The names of the units the side to move's encoding carries."""
    raw = cs.encode_raw(type_to_id={}, faction_to_id={})
    return [names[uid] for uid in raw.unit_ids], raw


def test_encoder_omits_fog_hidden_enemy_tokens():
    """Encoder's `encode_raw` must emit unit tokens only for
    visible units."""
    cs, names = _board([("mine_leader", "Spearman", 1, 0, {"max_moves": 2, "is_leader": True}),
                        ("enemy_close", "Spearman", 2, 2, {}),       # within sight
                        ("enemy_far", "Spearman", 2, 15, {})])       # outside sight
    ids, _raw = _unit_names(cs, names)
    assert "mine_leader" in ids and "enemy_close" in ids
    assert "enemy_far" not in ids


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
    cs, names = _board([("mine_leader", "Spearman", 1, 0, {"is_leader": True}),
                        ("plain_enemy", "Spearman", 2, 2, {}),
                        ("lurker", "Spearman", 2, 3, {"abilities": ["ambush"]})], code="Gs^Fms")
    ids, _raw = _unit_names(cs, names)
    assert "mine_leader" in ids and "plain_enemy" in ids, \
        "control: units in sight without cover keep their tokens"
    assert "lurker" not in ids, \
        "an ambusher on ^Fms must not reach the policy's observation"

    # Uncovered by a previous ambush trigger: the token comes back.
    lurker = next(uid for uid, name in names.items() if name == "lurker")
    cs.core.set_uncovered([lurker])
    assert "lurker" in _unit_names(cs, names)[0]


def test_encoder_recruit_phantoms_only_for_current_side():
    """encode_raw should emit recruit phantoms only for the side
    currently acting. Enemy recruit lists are fog-hidden."""
    cs, names = _board([("mine_leader", "Lieutenant", 1, 0, {"is_leader": True}),
                        ("enemy_leader", "Lieutenant", 2, 2, {"is_leader": True})],
                       recruits={1: ["Spearman", "Mage"], 2: ["Elvish Fighter"]})
    _ids, raw = _unit_names(cs, names)
    assert raw.recruit_types == ["Spearman", "Mage"]
    # recruit_is_ours should be all 1.0 (only own side emitted).
    assert (raw.recruit_is_ours == 1.0).all()


# ---- visible_hexes / visible_fraction -----------------------------

def test_visible_hexes_are_the_reach_and_the_ring_around_it():
    """A 2-MP unit on open ground reaches 2 hexes each way and sees the
    third (tests/test_vision.py covers terrain and the turn)."""
    cs, _names = _board([("u", "Spearman", 1, 5, {"max_moves": 2})])
    assert visibility.visible_hexes_for(gc.view_of(cs), 1) == {(x, 0) for x in range(2, 9)}


def test_visible_fraction_in_unit_interval():
    cs, _names = _board([("u", "Spearman", 1, 5, {"max_moves": 2})])
    assert visibility.visible_fraction_for(gc.view_of(cs), 1) == pytest.approx(7 / 20)


# ---- move onto hidden units: Wesnoth blocked/ambush semantics ----
# A move onto a hidden enemy EXECUTES with the engine's partial-move
# resolution, on the Rust core (rust/wesnoth_core/src/core_move.rs).

def _walk(units, path, *, special=None):
    """The side-1 unit on the path's first hex walks it on a fogged
    grass board (tests/helpers/parity_games.py): (landed hex, stop
    reason, the mover's movement left, the uncovered unit ids, the core).
    `units` as parity_games.record takes them."""
    from helpers.parity_games import record, state_of, unit_id_at
    from wesnoth_ai.game_core import CoreState, game_core_class
    if game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    cs = CoreState.from_state(state_of(record(units, special=special, fog=True)))
    mover = unit_id_at(cs, *path[0])
    cs.core.apply_move([p[0] for p in path], [p[1] for p in path], 1)
    _ox, _oy, lx, ly, reason = cs.core.last_move_walk_export()
    return (lx, ly), reason, cs.core.unit_export(mover)["current_moves"], set(cs.core.uncovered_export()), cs


ROW = [(x, 3) for x in range(1, 7)]


def test_walk_blocked_by_hidden_enemy_keeps_mp():
    """A hidden unit ON a path hex stops the mover on the hex
    BEFORE it, KEEPS remaining MP (post_move zeroes MP only for
    ambush/ZoC-final, move.cpp:1041-1043), and reveals the
    blocker.

    Setup note: the mover's max_moves is 1, so it sees x <= 3 (its
    reach and the ring around it) and the lurker at x=4 is fog-hidden
    and exerts no ZoC; it walks on its 5 current moves."""
    from helpers.parity_games import record, state_of, unit_id_at
    from wesnoth_ai.game_core import CoreState, game_core_class
    if game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    cs = CoreState.from_state(state_of(record(
        [("Spearman", 1, *ROW[0], False), ("Spearman", 2, *ROW[3], False)], fog=True)))
    mover, lurker = unit_id_at(cs, *ROW[0]), unit_id_at(cs, *ROW[3])
    cs.core.update_unit(mover, {"max_moves": 1})
    assert lurker not in cs.core.visible_ids(1)
    cs.core.apply_move([p[0] for p in ROW[:5]], [p[1] for p in ROW[:5]], 1)
    *_ordered, lx, ly, reason = cs.core.last_move_walk_export()
    assert reason == "blocked"
    assert (lx, ly) == ROW[2]               # stopped BEFORE the lurker
    assert set(cs.core.uncovered_export()) == {lurker}
    # 2 MP spent on flat terrain, 3 kept (NOT zeroed).
    assert cs.core.unit_export(mover)["current_moves"] == 3


def test_walk_ambush_by_hidden_hider_zeroes_mp():
    """Entering a hex adjacent to a hidden `hides` enemy stops the
    mover AT that hex, zeroes MP, and reveals the ambusher
    (check_for_ambushers, move.cpp:422-440). The Elvish Ranger stands
    off the path on `Gs^Fms`, a forest the pre-2026-09-13 defense-key
    table missed, so this asserts the move truncation on a hex where the
    old rule would NOT have stopped the mover -- the one thing the corpus
    sweep cannot check."""
    from tools.abilities import hex_neighbors
    ranger = next(p for p in hex_neighbors(*ROW[1]) if p not in hex_neighbors(*ROW[0]) and p not in ROW)
    landed, reason, mp_left, uncovered, cs = _walk(
        [("Spearman", 1, *ROW[0], False), ("Elvish Ranger", 2, *ranger, False)], ROW[:3],
        special={ranger: "Gs^Fms"})
    assert reason == "ambush"
    assert landed == ROW[1]                 # stopped AT the entered hex
    assert uncovered == {cs.core.unit_id_at(*ranger, 0)}
    assert mp_left == 0


def test_walk_passes_through_ally_and_backtracks_off_it():
    """Own-side units are pass-through (pathfind.cpp:777-786), but
    a move may not END on one: the walk backtracks off occupied end
    hexes (plot_turn, move.cpp:776-780) with MP refunded."""
    units = [("Spearman", 1, *ROW[0], False), ("Spearman", 1, *ROW[2], False)]
    # Through the ally, landing beyond: fine.
    landed, reason, mp_left, _unc, _cs = _walk(units, ROW[:4])
    assert (landed, reason, mp_left) == (ROW[3], "end", 2)
    # Ordered to END on the ally's hex: back to the hex before it.
    landed, _reason, mp_left, _unc, _cs = _walk(units, ROW[:3])
    assert (landed, mp_left) == (ROW[1], 4)  # only 1 MP charged


def test_legality_mask_offers_a_reachable_empty_hex():
    """Our unit on (0,0) with 2 MP on a two-hex line: the empty (1,0)
    is a legal target."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.action_sampler import _build_legality_masks

    cs, names = _board([("mine", "Spearman", 1, 0, {"max_moves": 2})], width=2)
    s = gc.view_of(cs)
    enc = GameStateEncoder(d_model=8)
    enc.register_names(s)
    encoded = enc.encode(s)
    masks = _build_legality_masks(encoded, s)
    # `target_valid` is [A, H]: the unit's actor row, the hex's column.
    mine = next(uid for uid, name in names.items() if name == "mine")
    row = masks.target_valid[encoded.unit_ids.index(mine)]
    assert float(row[encoded.pos_to_hex[(1, 0)]].item()) > 0.0


def test_empty_state_zero_visibility():
    """No hexes -> 0.0 fraction; a side with no units on a populated
    map sees nothing."""
    from wesnoth_ai.classes import GameState, GlobalInfo, Map
    empty = GameState(game_id="t", map=Map(size_x=0, size_y=0, mask=set(), fog=set(), hexes=set(), units=set()),
                      global_info=GlobalInfo(current_side=1, turn_number=1, time_of_day=None, village_gold=2,
                                             village_upkeep=1, base_income=2), sides=[])
    assert visibility.visible_fraction_for(empty, 1) == 0.0
    cs, _names = _board([("e", "Spearman", 2, 0, {})])
    view = gc.view_of(cs)
    assert visibility.visible_hexes_for(view, 1) == set()
    assert visibility.visible_fraction_for(view, 1) == 0.0
