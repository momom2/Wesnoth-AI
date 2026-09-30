"""What a side watched during the enemy's turn (the parity observation's
sighting record): an enemy unit that crosses the side's view and ends its
move in fog is remembered at the last hex the side saw it on, reaches the
side's next encoding as a sighting token, and is forgotten at the side's own
end of turn; the side's seen types keep it for the whole game."""
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


def test_a_unit_crossing_the_sides_view_is_remembered_where_last_seen():
    from helpers.parity_games import core_of, parity_raw, record, vocab_of
    from tools.abilities import hex_neighbors
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 7, 3, False),
                   ("Lieutenant", 2, 18, 3, True), ("Cavalryman", 2, 17, 0, False)],
                  fog=True, width=WIDTH, height=HEIGHT)
    cs = core_of(data)
    for command in (["init_side", 1], ["end_turn"], ["init_side", 2]):
        cs.apply_command(command)
    keys = cs.geometry().keys
    seen = {keys[j] for j in np.flatnonzero(cs.core.seen_export(1))}
    # Around side 1's units' zones of control, into the seen area and back
    # out into fog, within the Cavalryman's 8 moves.
    zoc = {n for u in ((1, 3), (7, 3)) for n in hex_neighbors(*u)} | {(1, 3), (7, 3)}
    start = (17, 0)
    inside = max((p for p in seen if p not in zoc), key=lambda p: (p[0], -p[1]))
    leg1 = _path(start, inside, zoc)
    fog = [p for p in keys if p not in seen and p not in zoc and p not in leg1]
    end = min(fog, key=lambda p: len(_path(inside, p, zoc | set(leg1[:-1]))))
    route = leg1 + _path(inside, end, zoc | set(leg1[:-1]))[1:]
    assert len(route) - 1 <= 8 and route[-1] not in seen
    last_seen = [p for p in route if p in seen][-1]

    cs.apply_command(["move", [p[0] for p in route], [p[1] for p in route], 2])
    # The step out of view is animated from the last seen hex toward the
    # next: side 1 last saw the Cavalryman entering that next hex.
    last_seen = route[route.index(last_seen) + 1]
    rows = [r for r in cs.core.sightings_export(1) if r[1] == "Cavalryman"]
    assert [(r[4], r[5]) for r in rows] == [last_seen]
    assert "Cavalryman" in cs.core.seen_types(1, 2)

    for command in (["end_turn"], ["init_side", 1]):
        cs.apply_command(command)
    raw = parity_raw(cs, vocab_of(["Lieutenant", "Spearman", "Cavalryman"]))
    assert list(zip(raw.sight_xs.tolist(), raw.sight_ys.tolist())) == [last_seen]

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


def _cavalryman_watched_into_fog():
    """The first test's game after side 2's move: the Cavalryman crossed
    side 1's view and ended in fog; (core, its last seen hex, a seen hex
    off its route)."""
    from helpers.parity_games import core_of, record
    from tools.abilities import hex_neighbors
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 7, 3, False),
                   ("Lieutenant", 2, 18, 3, True), ("Cavalryman", 2, 17, 0, False)],
                  fog=True, width=WIDTH, height=HEIGHT)
    cs = core_of(data)
    for command in (["init_side", 1], ["end_turn"], ["init_side", 2]):
        cs.apply_command(command)
    keys = cs.geometry().keys
    seen = {keys[j] for j in np.flatnonzero(cs.core.seen_export(1))}
    zoc = {n for u in ((1, 3), (7, 3)) for n in hex_neighbors(*u)} | {(1, 3), (7, 3)}
    inside = max((p for p in seen if p not in zoc), key=lambda p: (p[0], -p[1]))
    leg1 = _path((17, 0), inside, zoc)
    fog = [p for p in keys if p not in seen and p not in zoc and p not in leg1]
    end = min(fog, key=lambda p: len(_path(inside, p, zoc | set(leg1[:-1]))))
    route = leg1 + _path(inside, end, zoc | set(leg1[:-1]))[1:]
    cs.apply_command(["move", [p[0] for p in route], [p[1] for p in route], 2])
    last_seen = [p for p in route if p in seen][-1]
    return cs, route[route.index(last_seen) + 1], end


def test_a_unit_that_leaves_the_board_unseen_stays_where_it_was_last_seen():
    """A neutral side can kill a unit in a side's fog: the player did not
    see it go, so the record keeps it, as a sighting token, until the
    side's end of turn."""
    from helpers.parity_games import parity_raw, unit_id_at, vocab_of
    cs, last_seen, end = _cavalryman_watched_into_fog()
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
    cs, _, end = _cavalryman_watched_into_fog()
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
    after every command: clean on a game where a unit crosses side 1's
    view, and a divergence once the oracle forgets path sightings."""
    import gzip
    import json
    from collections import Counter
    from helpers.parity_games import core_of, record
    from tools import sighting_oracle
    from tools.diff_core import diff_core
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 7, 3, False),
                   ("Lieutenant", 2, 18, 3, True), ("Cavalryman", 2, 17, 0, False)],
                  fog=True, width=WIDTH, height=HEIGHT)
    cs = core_of(data)
    for command in (["init_side", 1], ["end_turn"], ["init_side", 2]):
        cs.apply_command(command)
    keys = cs.geometry().keys
    seen = {keys[j] for j in np.flatnonzero(cs.core.seen_export(1))}
    from tools.abilities import hex_neighbors
    zoc = {n for u in ((1, 3), (7, 3)) for n in hex_neighbors(*u)} | {(1, 3), (7, 3)}
    inside = max((p for p in seen if p not in zoc), key=lambda p: (p[0], -p[1]))
    leg1 = _path((17, 0), inside, zoc)
    fog = [p for p in keys if p not in seen and p not in zoc and p not in leg1]
    end = min(fog, key=lambda p: len(_path(inside, p, zoc | set(leg1[:-1]))))
    route = leg1 + _path(inside, end, zoc | set(leg1[:-1]))[1:]
    data["commands"] = [["init_side", 1], ["end_turn"], ["init_side", 2],
                        ["move", [p[0] for p in route], [p[1] for p in route], 2],
                        ["end_turn"], ["init_side", 1]]
    path = tmp_path / "g.json.gz"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    counts = Counter()
    assert diff_core(path, sightings=True, counts=counts) == []
    assert counts[("sightings", "rust")] == len(data["commands"])
    monkeypatch.setattr(sighting_oracle.SightingOracle, "_note_path", lambda self, gs, cmd: None)
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
