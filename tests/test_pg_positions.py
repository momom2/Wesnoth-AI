"""The position of every logged decision (tools/pg_positions.py): one
position per decision of each side, refused ones as shadow rows, each seeing
exactly the recruit rejections that happened before it."""
from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pg_positions import PositionError, action_is_legal, decision_slots, positions_of, read_record  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.pg_records import Decision, decisions_of, side_counts  # noqa: E402
from wesnoth_ai.server_priors import KIND_END_TURN, KIND_MOVE, KIND_RECRUIT  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


def _decision(kind: int, cmd: int, side: int = 1) -> Decision:
    return Decision(side=side, cmd=cmd, refused=cmd < 0, action=(0, kind, 0, -1), n=1, mass=1.0, logp=0.0, H=0.0,
                    p_end=0.0, p_att=0.0, p_rec=0.0, p_att_actor=float("nan"), cdf=(0.0, 1.0), v=0.0)


def test_a_decision_sees_the_rejections_of_the_refused_recruits_before_it():
    """Two refused recruits, then the recruit the engine relocated, all at
    command 5: the three see 0, 1 and 2 of its 3 rejections (the last is the
    relocated recruit's own, which only later decisions see); a refusal
    after the last command sits at the end."""
    decisions = [_decision(KIND_MOVE, 4), _decision(KIND_RECRUIT, -1), _decision(KIND_RECRUIT, -1),
                 _decision(KIND_RECRUIT, 5), _decision(KIND_END_TURN, 6), _decision(KIND_RECRUIT, -1, side=2)]
    rejections = [[5, 1, 1], [5, 2, 2], [5, 3, 3], [8, 4, 4]]
    assert decision_slots(decisions, rejections, 8) == [(4, 0), (5, 0), (5, 1), (5, 2), (6, 0), (8, 0)]


@pytest.mark.parametrize("decisions, rejections, n", [
    ([_decision(KIND_MOVE, 5)], [[5, 1, 1]], 6),                                  # a rejection no recruit made
    ([_decision(KIND_RECRUIT, -1), _decision(KIND_RECRUIT, 5)], [[5, 1, 1]] * 3, 6),   # one too many
    ([_decision(KIND_MOVE, 5), _decision(KIND_MOVE, 5)], [], 6),                  # two decisions emit one command
    ([_decision(KIND_MOVE, 6)], [], 6),                                           # a command past the record
])
def test_a_log_that_does_not_fit_its_record_is_refused(decisions, rejections, n):
    with pytest.raises(PositionError):
        decision_slots(decisions, rejections, n)


def _games():
    from helpers.pg_games import played_games
    return played_games()


@needs_core
def test_every_logged_decision_has_its_position():
    """Each side's positions are its logged decisions one for one, the
    refused ones flagged as shadow rows, every logged action legal at its
    position (else the replay refuses) and of its command's kind, with the
    logged count of legal actions."""
    games = _games()
    shadows = 0
    for stem, path in games.records.items():
        rec = read_record(path)
        decisions = decisions_of(rec)
        pos = positions_of(rec, *games.vocab, game=stem)
        assert {s: len(v) for s, v in pos.sides.items()} == side_counts(decisions)
        for side, rows in pos.sides.items():
            logged = [i for i, d in enumerate(decisions) if d.side == side]
            assert [p.index for p in rows] == logged
            assert [p.shadow for p in rows] == [decisions[i].refused for i in logged]
            assert [p.action for p in rows] == [decisions[i].action for i in logged]
        assert pos.counts.get("command_mismatch", 0) == 0 and pos.counts.get("legal_count_mismatch", 0) == 0
        shadows += pos.counts["shadow"]
    assert shadows == 2


def _hex_token(raw, hex_) -> int:
    return [(h.x, h.y) for h in raw.hex_positions].index(hex_)


@needs_core
def test_a_refused_recruit_is_legal_where_it_was_decided_and_rejected_after():
    """The refused recruit's ordered hex is a legal target at its own
    position and is no longer one at the decision that follows it."""
    games = _games()
    rec = read_record(games.records["g1"])
    pos = positions_of(rec, *games.vocab)
    rows = pos.sides[1]
    checked = 0
    for shadow, after in zip(rows, rows[1:]):
        if not shadow.shadow:
            continue
        actor, kind, target, weapon = shadow.action
        assert kind == KIND_RECRUIT and action_is_legal(shadow.masks, shadow.action)
        hex_ = (shadow.raw.hex_positions[target].x, shadow.raw.hex_positions[target].y)
        moved = (actor, kind, _hex_token(after.raw, hex_), weapon)
        assert not action_is_legal(after.masks, moved)
        checked += 1
    assert checked == 2


def _differs(a, b) -> bool:
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray) and (x.shape != y.shape or not np.array_equal(x, y)):
            return True
    return False


def _recruit_hexes(p):
    """The hexes some recruit option may be placed on at position `p`."""
    m = p.masks
    U, R = m.n_units, m.n_recruits
    bits = np.unpackbits(m.union_valid[U:U + R], axis=1)[:, :m.n_hexes].any(axis=0)
    return [(p.raw.hex_positions[t].x, p.raw.hex_positions[t].y) for t in np.flatnonzero(bits)]


def _ordered_later(decisions, rows, i):
    """The hexes the recruits after decision i in its side's turn are
    ordered onto: a rejection elsewhere leaves every logged action legal."""
    out = set()
    for j in range(i + 1, len(decisions)):
        d = decisions[j]
        if d.side != decisions[i].side:
            break
        if d.kind == KIND_RECRUIT:
            h = rows[j].raw.hex_positions[d.action[2]]
            out.add((h.x, h.y))
    return out


def _relocation_case(games):
    """A recruit followed in its turn by another decision that may still
    recruit, and a hex that one could recruit on and no later recruit of
    the turn is ordered onto."""
    for path in games.records.values():
        rec = read_record(path)
        decisions = decisions_of(rec)
        rows = {p.index: p for side in (1, 2) for p in positions_of(rec, *games.vocab, verify=False).sides[side]}
        for i, d in enumerate(decisions[:-1]):
            nxt = decisions[i + 1]
            if d.kind != KIND_RECRUIT or d.refused or nxt.cmd != d.cmd + 1 or nxt.side != d.side:
                continue
            for h in _recruit_hexes(rows[i + 1]):
                if h not in _ordered_later(decisions, rows, i):
                    return rec, decisions, rows, i, h
    raise AssertionError("no game holds a recruit followed by a decision that may still recruit elsewhere")


@needs_core
def test_a_relocated_recruit_sees_none_of_its_own_rejection():
    """A recruit the engine relocated leaves its ordered hex rejected at its
    own command: the recruit decided without it (its masks unchanged), and
    the next decision of its turn may no longer recruit there. The parity
    encoding writes no rejection column: the masks carry it."""
    games = _games()
    rec, decisions, rows, i, hex_ = _relocation_case(games)
    relocated = copy.deepcopy(rec)
    relocated["rejections"] = sorted(relocated["rejections"] + [[decisions[i].cmd, *hex_]])
    moved = {p.index: p for side in (1, 2) for p in positions_of(relocated, *games.vocab, verify=False).sides[side]}
    assert not _differs(rows[i].masks, moved[i].masks)
    assert _differs(rows[i + 1].masks, moved[i + 1].masks)
    assert hex_ not in _recruit_hexes(moved[i + 1])


@needs_core
@pytest.mark.slow
def test_the_worker_processes_build_the_same_positions():
    """The loader's worker processes (the box's path) build each game as the
    calling thread does, and count the same decisions."""
    from tools.pg_positions import PositionLoader, side_lengths
    games = _games()
    here = PositionLoader(games.records, *games.vocab)
    pool = PositionLoader(games.records, *games.vocab, workers=1)
    assert side_lengths(games.records, workers=1) == side_lengths(games.records)
    pool.prefetch(games.records)
    for game in games.records:
        a, b = here.game(game), pool.game(game)
        assert a.counts == b.counts
        for side in (1, 2):
            assert [(p.action, p.shadow, p.logp) for p in a.sides[side]] == \
                   [(p.action, p.shadow, p.logp) for p in b.sides[side]]
            assert not any(_differs(p.raw, q.raw) or _differs(p.masks, q.masks)
                           for p, q in zip(a.sides[side], b.sides[side]))
    pool.close()
