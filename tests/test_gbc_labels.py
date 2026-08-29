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


def test_stream_labels_catch_fast_turn_events():
    """Project round-2 C4/C6: an event inside a TCS fast turn (no
    recorded state, but observed by the game loop) must land at its
    TRUE turn with fog read at action resolution -- the recorded-
    only diff stamped it with the next recorded state's turn,
    pushing it outside its label windows."""
    import copy
    import dataclasses
    from wesnoth_ai.gbc import (PRED_IDX, labels_for_game_states,
                                observe_state)

    from wesnoth_ai.gbc import diff_events_obs

    sim = fresh_scenario_sim(seed=0, max_turns=9, mini=False)
    gs = sim.gs
    leader2 = next(u for u in gs.map.units if u.side == 2)
    victim = dataclasses.replace(leader2, id="gbc_stream_victim",
                                 is_leader=False)
    gs.map.units.add(victim)

    # Incremental trace, exactly as note_observation builds it.
    events, anchor, prev, n = [], {}, None, 0

    def _note(state):
        nonlocal prev, n
        o = observe_state(state)
        anchor[id(state)] = n
        if prev is not None:
            events.extend(diff_events_obs(n, prev, o))
        prev = o
        n += 1

    anchor1 = copy.deepcopy(gs)              # recorded, turn 1
    _note(anchor1)
    gs.global_info.turn_number = 2           # fast turn: observed,
    _note(gs)                                # never recorded
    gs.map.units.discard(victim)             # ...and the death
    post = copy.deepcopy(gs)
    _note(post)
    gs.global_info.turn_number = 5
    anchor2 = copy.deepcopy(gs)              # recorded, turn 5
    _note(anchor2)

    rows = labels_for_game_states([anchor1, anchor2], [2, 2],
                                  trace=(events, anchor))
    key = ("u", "gbc_stream_victim", PRED_IDX["dies"])
    r1 = {r[:3]: r[3:] for r in (rows[0] or [])}
    assert key in r1, sorted(r1)
    assert any(r1[key]), \
        "turn-2 fast-turn death invisible from the turn-1 anchor"
    # From the turn-5 anchor the death is in the PAST: no label.
    r2 = {r[:3]: r[3:] for r in (rows[1] or [])}
    assert key not in r2 or not any(r2[key])


def test_fogless_game_observes_everything():
    """Project round-4: the --fogless-ratio slice must label with
    WHOLE-BOARD observability -- the sight-disc union censored
    labels by a fog the game does not have."""
    from wesnoth_ai.gbc import _observable_hexes
    sim = fresh_scenario_sim(seed=0, max_turns=6, mini=False)
    gs = sim.gs
    setattr(gs.global_info, "_fog", False)
    all_hexes = {(h.position.x, h.position.y) for h in gs.map.hexes}
    assert _observable_hexes(gs, 1) == all_hexes
    assert _observable_hexes(gs, 2) == all_hexes
    setattr(gs.global_info, "_fog", True)
    assert _observable_hexes(gs, 1) < all_hexes


def test_owner_always_observes_own_unit_death():
    """Round-4 adjacent: the owner's roster shrinks even when the
    death hex is fogged from EVERYONE (a lone unit deep in enemy
    territory takes its own disc with it)."""
    from wesnoth_ai.gbc import diff_events_obs
    prev = (1, {"v1": (2, "Spearman", (5, 5), 14, False)}, {},
            frozenset(), frozenset())
    cur = (2, {}, {}, frozenset(), frozenset())
    evs = diff_events_obs(1, prev, cur)
    assert len(evs) == 1 and evs[0].predicate == "dies"
    assert 2 in evs[0].observed_by
    assert 1 not in evs[0].observed_by
