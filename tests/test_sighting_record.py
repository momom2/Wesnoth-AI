"""What a side saw of the enemy's turn (the parity observation's sighting
record), as a player with move animations off sees it (user ruling
2026-10-01): an enemy unit is noted where it stands after each command, so
one that walks out of view is remembered where it stood before its move,
and one that crosses the view during a move is not seen at all. The record
reaches the side's next encoding as sighting tokens and is forgotten at the
side's own end of turn; the side's seen types keep the types for the whole
game."""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from wesnoth_ai import game_core as gc  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")

WIDTH, HEIGHT = 20, 7


def _path(start, goal, avoid):
    """A shortest route of adjacent hexes from `start` to `goal` on the
    open board, around `avoid`."""
    from tools.abilities import hex_neighbors
    prev = {start: None}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if p == goal:
            break
        for n in hex_neighbors(*p):
            if 0 <= n[0] < WIDTH and 0 <= n[1] < HEIGHT and n not in prev and n not in avoid:
                prev[n] = p
                queue.append(n)
    route, p = [], goal
    while p is not None:
        route.append(p)
        p = prev[p]
    return route[::-1]


def _two_hexes_into_fog(leg1, seen, keys, zoc):
    """The shortest continuation of `leg1` (which ends on a hex the watcher
    sees) whose last two hexes are fogged: the hex after the last seen one
    is then not the route's end, where a leak of the landing hex shows."""
    avoid = zoc | set(leg1[:-1])
    routes = []
    for end in (p for p in keys if p not in seen and p not in avoid):
        route = leg1 + _path(leg1[-1], end, avoid)[1:]
        if max(k for k, p in enumerate(route) if p in seen) + 2 < len(route):
            routes.append(route)
    return min(routes, key=lambda r: (len(r), r[-1]))


def _watch_game(start):
    """Side 2's Cavalryman on `start`, at side 2's first turn: (core, record,
    the board's hexes, the hexes side 1 sees, the hexes a route avoids
    around side 1's units)."""
    from helpers.parity_games import core_of, record
    from tools.abilities import hex_neighbors
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 7, 3, False),
                   ("Lieutenant", 2, 18, 3, True), ("Cavalryman", 2, *start, False)],
                  fog=True, width=WIDTH, height=HEIGHT)
    cs = core_of(data)
    for command in (["init_side", 1], ["end_turn"], ["init_side", 2]):
        cs.apply_command(command)
    keys = cs.geometry().keys
    seen = {keys[j] for j in np.flatnonzero(cs.core.seen_export(1))}
    zoc = {n for u in ((1, 3), (7, 3)) for n in hex_neighbors(*u)} | {(1, 3), (7, 3)}
    return cs, data, keys, seen, zoc


def _cavalryman_walked_into_fog():
    """Side 2's Cavalryman starts on a hex side 1 sees and walks two hexes
    into fog: (core after the move, the record with its commands, its
    start, its end)."""
    _cs, _data, keys, seen, zoc = _watch_game((17, 0))
    start = max((p for p in seen if p not in zoc), key=lambda p: (p[0], -p[1]))
    cs, data, keys, seen_now, zoc = _watch_game(start)
    assert seen_now == seen, "side 1's view does not depend on where the Cavalryman stands"
    route = _two_hexes_into_fog([start], seen, keys, zoc)
    move = ["move", [p[0] for p in route], [p[1] for p in route], 2]
    cs.apply_command(move)
    data["commands"] = [["init_side", 1], ["end_turn"], ["init_side", 2], move, ["end_turn"], ["init_side", 1]]
    return cs, data, start, route[-1]


def test_a_unit_crossing_the_sides_view_mid_move_is_not_seen():
    """Side 2's Cavalryman, out of side 1's view, crosses it and ends its
    move in fog: with move animations off, side 1 never saw it."""
    from helpers.parity_games import parity_raw, vocab_of
    cs, _data, keys, seen, zoc = _watch_game((17, 0))
    assert (17, 0) not in seen
    inside = max((p for p in seen if p not in zoc), key=lambda p: (p[0], -p[1]))
    route = _two_hexes_into_fog(_path((17, 0), inside, zoc), seen, keys, zoc)
    assert len(route) - 1 <= 8 and inside in route and route[-1] not in seen
    cs.apply_command(["move", [p[0] for p in route], [p[1] for p in route], 2])
    assert not [r for r in cs.core.sightings_export(1) if r[1] == "Cavalryman"]
    assert "Cavalryman" not in cs.core.seen_types(1, 2)
    for command in (["end_turn"], ["init_side", 1]):
        cs.apply_command(command)
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Cavalryman"]))
    assert raw.sight_xs.tolist() == []


def test_a_unit_that_walks_out_of_view_is_remembered_where_it_stood():
    """Side 1 sees the Cavalryman before its move, and it walks into fog:
    side 1 last saw it where it stood. The record keeps that hex until side
    1's end of turn, its next encoding carries it as a sighting token, and
    its type stays seen."""
    from helpers.parity_games import parity_raw, vocab_of
    cs, _data, start, _end = _cavalryman_walked_into_fog()
    rows = [r for r in cs.core.sightings_export(1) if r[1] == "Cavalryman"]
    assert [(r[4], r[5]) for r in rows] == [start]
    assert "Cavalryman" in cs.core.seen_types(1, 2)
    for command in (["end_turn"], ["init_side", 1]):
        cs.apply_command(command)
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Cavalryman"]))
    assert list(zip(raw.sight_xs.tolist(), raw.sight_ys.tolist())) == [start]
    cs.apply_command(["end_turn"])
    assert cs.core.sightings_export(1) == []
    assert "Cavalryman" in cs.core.seen_types(1, 2)


def test_the_belief_targets_are_the_hidden_enemies_tokens():
    """The belief head's targets: the token of each enemy unit the side
    cannot see, and a count of those with no token; own units' hexes are
    outside the loss's domain."""
    from helpers.parity_games import core_of, parity_raw, record, vocab_of
    from wesnoth_ai.belief_targets import belief_targets
    own = [("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 7, 3, False)]
    channel = {(x, y): "Wo" for x in (8, 9) for y in range(HEIGHT)}     # deep water blocks the view
    probe = core_of(record(own + [("Lieutenant", 2, 18, 3, True)], fog=True, width=WIDTH, height=HEIGHT,
                           special=channel))
    probe.apply_command(["init_side", 1])
    keys = probe.geometry().keys
    seen = {keys[j] for j in np.flatnonzero(probe.core.seen_export(1))}
    hidden_at = min((p for p in keys if p not in seen and p not in channel),
                    key=lambda p: (len(_path((7, 3), p, set())), p))
    assert len(_path((7, 3), hidden_at, set())) - 1 <= 6       # within the relevant set's fog radius
    cs = core_of(record(own + [("Lieutenant", 2, 18, 3, True), ("Cavalryman", 2, *hidden_at, False)],
                        fog=True, width=WIDTH, height=HEIGHT, special=channel))
    cs.apply_command(["init_side", 1])
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Cavalryman"]))
    slots = {(p.x, p.y): i for i, p in enumerate(raw.hex_positions)}
    t = belief_targets(raw, 1)
    assert slots[hidden_at] in t.hidden_tokens.tolist()
    hidden_hexes = [hidden_at, (18, 3)]
    assert t.n_untokened == sum(1 for p in hidden_hexes if p not in slots)
    assert not t.no_visible_unit[slots[(1, 3)]] and not t.no_visible_unit[slots[(7, 3)]]
    assert t.no_visible_unit[slots[hidden_at]]


def test_a_unit_that_leaves_the_board_unseen_stays_where_it_was_last_seen():
    """A neutral side can kill a unit in a side's fog: the player did not
    see it go, so the record keeps it, as a sighting token, until the
    side's end of turn."""
    from helpers.parity_games import parity_raw, unit_id_at, vocab_of
    cs, _data, last_seen, end = _cavalryman_walked_into_fog()
    cavalryman = unit_id_at(cs, *end)
    cs.core.remove_unit(cavalryman)
    cs.apply_command(["update_shroud"])                  # any command: the records follow it
    assert [(r[4], r[5]) for r in cs.core.sightings_export(1) if r[0] == cavalryman] == [last_seen]
    assert cs.core.sightings_gone_export(1) == [cavalryman]
    for command in (["end_turn"], ["init_side", 1]):
        cs.apply_command(command)
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Cavalryman"]))
    assert list(zip(raw.sight_xs.tolist(), raw.sight_ys.tolist())) == [last_seen]
    cs.apply_command(["end_turn"])
    assert cs.core.sightings_export(1) == [] and cs.core.sightings_gone_export(1) == []


def test_a_unit_that_leaves_the_board_in_view_leaves_the_record():
    from helpers.parity_games import unit_id_at
    cs, _data, _start, end = _cavalryman_walked_into_fog()
    lieutenant = unit_id_at(cs, 1, 3)
    cs.apply_command(["end_turn"])
    cs.apply_command(["init_side", 1])
    spearman = unit_id_at(cs, 7, 3)
    assert spearman and lieutenant
    # Side 2 sees side 1's Spearman; it leaves the board where side 2 sees it.
    assert spearman in {r[0] for r in cs.core.sightings_export(2)}
    cs.core.remove_unit(spearman)
    cs.apply_command(["update_shroud"])
    assert spearman not in {r[0] for r in cs.core.sightings_export(2)}
    assert cs.core.sightings_gone_export(2) == []


def test_the_certification_compares_the_sighting_records(tmp_path, monkeypatch):
    """diff_core --sightings holds the core's records against the oracle's
    after every command: clean on a game where a unit walks out of side 1's
    view, and a divergence once the oracle stops noting what a side sees."""
    import gzip
    import json
    from collections import Counter
    from tools import sighting_oracle
    from tools.diff_core import diff_core
    _cs, data, _start, _end = _cavalryman_walked_into_fog()
    path = tmp_path / "g.json.gz"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    counts = Counter()
    assert diff_core(path, sightings=True, encode_every=1, counts=counts) == []
    assert counts[("sightings", "rust")] == len(data["commands"])
    monkeypatch.setattr(sighting_oracle.SightingOracle, "_note_visible", lambda self, gs: None)
    out = diff_core(path, sightings=True)
    assert out and "sightings" in out[0]


def test_a_fight_the_side_defended_is_recorded_before_its_refog():
    """Side 2's Cavalryman kills side 1's Bowman, its only unit in view of
    it, and is hit back first. The fight was shown to side 1 before its fog
    closed over the Cavalryman, so side 1's record carries its hit points
    after the fight."""
    from helpers.parity_games import core_of, record, unit_id_at
    data = record([("Lieutenant", 1, 1, 3, True), ("Bowman", 1, 15, 3, False),
                   ("Cavalryman", 2, 16, 3, False), ("Lieutenant", 2, 18, 3, True)],
                  fog=True, width=WIDTH, height=HEIGHT)
    cs = core_of(data)
    for command in (["init_side", 1], ["end_turn"], ["init_side", 2]):
        cs.apply_command(command)
    cavalryman = unit_id_at(cs, 16, 3)
    full = [r for r in cs.core.sightings_export(1) if r[0] == cavalryman][0][2]
    cs.core.update_unit(unit_id_at(cs, 15, 3), {"current_hp": 1})
    # The Cavalryman misses, the Bowman's sword hits, the Cavalryman kills it.
    cs.core.apply_attack_scripted(16, 3, 15, 3, 0, 0, [False, True, True], [])
    assert cs.core.unit_id_at(15, 3) is None
    assert cavalryman not in set(cs.core.visible_ids(1)), "side 1 no longer sees it"
    hp = [r for r in cs.core.sightings_export(1) if r[0] == cavalryman][0][2]
    assert hp < full, "the record carries the hit points the fight left it"


def _round2_note_fight(self, gs):
    """Round 2's oracle: the fight read from the state after the command,
    the refog decided by whether a unit holds the defender's id."""
    from wesnoth_ai.visibility import is_scenery_unit, units_visible_to_python
    fight = gs.global_info._last_fight
    side = fight["defender_side"]
    if any(u.id == fight["defender"] for u in gs.map.units):
        return
    for u in units_visible_to_python(gs, side, vis_set=self._seen_before.get(side)):
        if u.side != side and not is_scenery_unit(u):
            self._record(side, u, u.position.x, u.position.y)


@pytest.mark.parametrize("units, attack, attacker_side, attacker_after", [
    ([("Lieutenant", 1, 1, 3, True), ("Bowman", 1, 15, 3, False, {"hp": 1}),
      ("Cavalryman", 2, 16, 3, False, {"max_exp": 1}), ("Lieutenant", 2, 18, 3, True)],
     [16, 3, 15, 3], 2, "Dragoon"),
    ([("Lieutenant", 1, 1, 3, True), ("Walking Corpse", 1, 15, 3, False), ("Lieutenant", 2, 5, 0, True),
      ("Spearman", 2, 16, 3, False, {"hp": 1})],
     [15, 3, 16, 3], 1, "Walking Corpse"),
], ids=["the attacker advances", "the corpse takes the defender's id"])
def test_a_defended_fight_is_certified_against_the_oracle(units, attack, attacker_side, attacker_after,
                                                          tmp_path, monkeypatch):
    """A fight kills the defending side's only unit in view of the attacker,
    and its fog closes over the attacker. The engine refogs the side when
    the fight ends and advances the attacker only after: the side's record
    holds the attacker as the fight left it. diff_core --sightings is clean,
    and diverges under round 2's oracle, which read the fight from the state
    after the command (the advanced attacker; the plague corpse under the
    dead defender's id)."""
    import gzip
    import json
    from helpers.parity_games import core_of, record
    from tools import sighting_oracle
    from tools.diff_core import diff_core
    data = record(units, fog=True, width=WIDTH, height=HEIGHT)
    turns = [["init_side", 1]] if attacker_side == 1 else [["init_side", 1], ["end_turn"], ["init_side", 2]]
    data["commands"] = [*turns, ["attack", *attack, 0, 0, "00000000"], ["end_turn"]]
    cs = core_of(data)
    for command in data["commands"][:-1]:
        cs.apply_command(command)
    attacker, side = cs.core.unit_id_at(attack[0], attack[1], 0), 3 - attacker_side
    assert cs.core.unit_export(attacker)["name"] == attacker_after
    assert attacker not in set(cs.core.visible_ids(side)), "the side's fog closed over the attacker"
    row = [r for r in cs.core.sightings_export(side) if r[0] == attacker][0]
    assert row[1] == units[1 if attacker_side == 1 else 2][0], "recorded before it advanced"
    path = tmp_path / "g.json.gz"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    assert diff_core(path, sightings=True, encode_every=1) == []
    monkeypatch.setattr(sighting_oracle.SightingOracle, "_note_fight", _round2_note_fight)
    out = diff_core(path, sightings=True)
    assert out and "sightings" in out[0]


def test_the_units_the_scenario_placed_are_not_seen_types():
    """Hornshark Island gives the Loyalists Woodsmen, whose Poacher line only
    the Knalgan Alliance recruits: neither a unit the scenario placed nor
    what it advances to is a seen type. A leader is."""
    from helpers.parity_games import record, state_of
    from tools.replay_dataset import _setup_scenario_events
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 14, 3, False, {"hp": 1}),
                   ("Mage", 1, 12, 3, False), ("Woodsman", 2, 15, 3, False, {"max_exp": 1}),
                   ("Lieutenant", 2, 18, 3, True)], fog=True, width=WIDTH, height=HEIGHT)
    gs = state_of(data)
    _setup_scenario_events(gs, "")
    cs = gc.CoreState.from_state(gs)
    woodsman = cs.core.unit_id_at(15, 3, 0)
    assert set(cs.core.scenario_unit_ids()) == {cs.core.unit_id_at(x, 3, 0) for x in (12, 14, 15)}
    for command in (["init_side", 1], ["end_turn"], ["init_side", 2], ["attack", 15, 3, 14, 3, 0, 0, "00000000"]):
        cs.apply_command(command)
    advanced = cs.core.unit_export(woodsman)["name"]
    assert advanced != "Woodsman" and woodsman in set(cs.core.visible_ids(1))
    assert cs.core.seen_types(1, 2) == ["Lieutenant"]
