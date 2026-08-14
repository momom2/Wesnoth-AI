"""GBC label machinery tests (docs/gbc_spec.md par.2, amendment A1).

Drives the production scanner (tools/gbc_labels.py) on real sim
states and real replays -- no mirrored logic. The synthetic fogged
kill is the unit test amendment A1 explicitly requires: an event the
observer's fog did not admit must NOT label as achieved for that
observer, regardless of entity ownership.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.visibility import visible_hexes_for  # noqa: E402
from tools.gbc_labels import (  # noqa: E402
    Anchor, Event, _diff_events, _unit_fp, _village_fp,
    labels_for_anchor, scan_game,
)


def test_synthetic_fogged_kill_is_censored_per_side():
    """Kill an enemy unit standing outside side 1's fog: the death
    event must be observed by side 2 (its owner sees its own camp)
    and NOT by side 1 (amendment A1's censoring direction)."""
    import dataclasses
    sim = fresh_scenario_sim(seed=0, max_turns=6, mini=False)
    gs = sim.gs
    vis1 = visible_hexes_for(gs, 1)
    leader2 = next(u for u in gs.map.units if u.side == 2)
    # Plant a second side-2 unit next to its leader, outside side 1's
    # fog: the leader keeps side 2's vision alive after the kill.
    spot = next(
        (x, y)
        for x, y in visible_hexes_for(gs, 2)
        if (x, y) not in vis1
        and abs(x - leader2.position.x) + abs(y - leader2.position.y)
        in (1, 2)
        and not any((u.position.x, u.position.y) == (x, y)
                    for u in gs.map.units))
    victim = dataclasses.replace(
        leader2, id="gbc_test_victim",   # unit ids are strings ('u2')
        is_leader=False,
        position=dataclasses.replace(leader2.position,
                                     x=spot[0], y=spot[1]))
    gs.map.units.add(victim)
    prev_u, prev_v = _unit_fp(gs), _village_fp(gs)
    gs.map.units.discard(victim)          # the fogged death
    events = _diff_events(1, prev_u, prev_v, gs)
    death = [e for e in events if e.predicate == "dies"]
    assert len(death) == 1
    e = death[0]
    assert e.key == ("u", victim.id)
    assert 2 in e.observed_by, "owner side sees its own camp"
    assert 1 not in e.observed_by, "fogged observer must NOT see it"

    # Label side: an anchor whose roster contains the victim (e.g.
    # scouted earlier) still gets y=0 -- unobserved is unachieved.
    anchor = Anchor(seq=0, turn=e.turn, side=1,
                    goals={("u", victim.id): ("dies", 2, True)})
    rows = {(p, k): y for _key, p, k, y, _v in
            labels_for_anchor(anchor, events)}
    assert rows[("dies", 1)] == 0
    # The owner-side observer labels it achieved.
    anchor2 = Anchor(seq=0, turn=e.turn, side=2,
                     goals={("u", victim.id): ("dies", 2, True)})
    rows2 = {(p, k): y for _key, p, k, y, _v in
             labels_for_anchor(anchor2, events)}
    assert rows2[("dies", 1)] == 1


def test_label_windows_and_seq_ordering():
    ev = [Event(seq=5, turn=3, predicate="dies", key=("u", 9),
                entity_side=2, hex=(4, 4), observed_by=frozenset({1}))]
    goals = {("u", 9): ("dies", 2, True)}
    # Anchor before the event, same turn: k=1 window [3,3] hits.
    rows = {k: y for _key, p, k, y, _v in labels_for_anchor(
        Anchor(seq=1, turn=3, side=1, goals=goals), ev) if p == "dies"}
    assert rows == {1: 1, 2: 1, 3: 1}
    # Anchor AFTER the event in the same turn (seq ordering): no hit
    # at any k -- the past is not a prediction target.
    rows = {k: y for _key, p, k, y, _v in labels_for_anchor(
        Anchor(seq=9, turn=3, side=1, goals=goals), ev) if p == "dies"}
    assert rows == {1: 0, 2: 0, 3: 0}
    # Two turns earlier: k=1,2 miss, k=3 (window [1,3]) hits.
    rows = {k: y for _key, p, k, y, _v in labels_for_anchor(
        Anchor(seq=1, turn=1, side=1, goals=goals), ev) if p == "dies"}
    assert rows == {1: 0, 2: 0, 3: 1}


def test_scan_real_replay_yields_anchors_and_events():
    dataset = Path(__file__).parent.parent / "replays_dataset_imitation"
    files = sorted(dataset.glob("*.json.gz"))
    if not files:
        pytest.skip("imitation dataset not present")
    scan = scan_game(files[0])
    assert scan.anchors, "side-turn-start anchors recorded"
    assert scan.n_turns >= 1
    for a in scan.anchors[:5]:
        assert a.side in (1, 2)
        assert a.goals, "fog-honest goal roster non-empty"
        # Unit goals only for units visible to the mover; village
        # goals cover the whole map's villages (terrain truth).
        kinds = {k[0] for k in a.goals}
        assert kinds <= {"u", "v"}
    # Events are seq-ordered and carry observability sets.
    for e in scan.events:
        assert e.observed_by <= {1, 2}
        assert e.predicate in ("dies", "flips", "levels")
