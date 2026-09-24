"""The rewrite certificates of docs/xod_dominance_design_20260924.md
(tools/combat_dominance.py, tools/dominance_rewrites.py): the almost
dominance ratio on the design's worked example, relaxations that only
ever admit more, the two-attack table against the simulator's own
enumeration, and the setup and surround classes on small boards."""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.combat_dominance import (COMBOS, EQ, GT, INCOMP, LT, Combo, Dim,  # noqa: E402
                                    admits, compare_marginals, minimal_combos, side_turns)
from tools.dominance_rewrites import side_turn_rewrites, two_fight_joint  # noqa: E402
from tools.replay_dataset import _build_initial_gamestate  # noqa: E402

# The worked example of the design's section 4: a 6-4 and an 8-2 attack
# at 30% to hit; the target's hp after each, dead = -1.
SIX_FOUR_AT_20 = {20: 0.2401, 14: 0.4116, 8: 0.2646, 2: 0.0756, -1: 0.0081}
EIGHT_TWO_AT_20 = {20: 0.49, 12: 0.42, 4: 0.09}
SIX_FOUR_AT_16 = {16: 0.2401, 10: 0.4116, 4: 0.2646, -1: 0.0837}
EIGHT_TWO_AT_16 = {16: 0.49, 8: 0.42, -1: 0.09}


def _alive(m):
    dead = m.get(-1, 0.0)
    return {0: dead, 1: 1 - dead}


def _weapon_dims(cand, base):
    sym_a, _ = compare_marginals(_alive(cand), _alive(base), False)
    sym_h, eps = compare_marginals(cand, base, False)
    return [Dim("enemy_alive", "binary", sym_a), Dim("enemy_hp", "hp", sym_h, eps)]


def test_the_worked_example_is_almost_dominant_at_20_hp_and_rejected_at_16():
    dims = _weapon_dims(SIX_FOUR_AT_20, EIGHT_TWO_AT_20)
    assert dims[0].sym == GT                     # 6-4 can kill a 20-hp target, 8-2 cannot
    assert dims[1].sym == INCOMP and abs(dims[1].eps - 0.113) < 0.001
    assert admits(dims, "D", Combo(r2=0.15))
    assert not admits(dims, "D", Combo(r2=0.05)) and not admits(dims, "D", Combo())
    dims16 = _weapon_dims(SIX_FOUR_AT_16, EIGHT_TWO_AT_16)
    assert dims16[0].sym == LT                   # 8-2 kills a 16-hp target more often
    assert not any(admits(dims16, "D", c) for c in COMBOS)


def test_a_looser_combination_admits_everything_a_tighter_one_does():
    rng = random.Random(7)
    kinds = ("binary", "hp", "xp", "pos", "vis", "count")
    for _ in range(300):
        dims = []
        for k in range(rng.randint(1, 6)):
            kind = rng.choice(kinds)
            sym = EQ if kind == "pos" else rng.choice((GT, EQ, LT, INCOMP))
            dims.append(Dim(f"d{k}", kind, sym, eps=rng.random(),
                            distance=rng.choice((0, 1, 2, 3)) if kind == "pos" else 0,
                            guard=rng.random() < 0.5))
        tier = rng.choice(("D", "O"))
        for a, b in itertools.product(COMBOS, COMBOS):
            if a.within(b) and admits(dims, tier, a):
                assert admits(dims, tier, b), (dims, a, b)


def test_minimal_combinations_are_the_smallest_that_admit():
    dims = [Dim("enemy_alive", "binary", GT),
            Dim("pos:u1", "pos", EQ, distance=2, guard=False),
            Dim("seen_hexes", "vis", LT)]
    assert [c.name() for c in minimal_combos(dims, "D")] == ["r3l+r4"]
    dims[1] = Dim("pos:u1", "pos", EQ, distance=2, guard=True)
    assert [c.name() for c in minimal_combos(dims, "D")] == ["r3g+r4"]


def _board(width, units, height=3):
    border = ", ".join(["Xv"] * (width + 2))
    rows = [border] + ["Xv, " + ", ".join(["Gg"] * width) + ", Xv" for _ in range(height)] + [border]
    return _build_initial_gamestate({
        "game_id": "dom", "map_data": "\n".join(rows),
        "starting_units": [{"uid": i + 1, "type": t, "side": s, "x": x, "y": y, "is_leader": False, **extra}
                           for i, (t, s, x, y, extra) in enumerate(units)],
        "starting_sides": [{"side": k, "gold": 0, "recruit": [], "fog": True} for k in (1, 2)]})


def _marginal(parts, fn):
    out = {}
    for x, p in parts:
        v = fn(x)
        out[v] = out.get(v, 0.0) + p
    return out


def _simulated_window(gs, first, second):
    """The simulator's own children of `first`, then of `second` wherever
    the target survived (tools.swap_detector.enumerate_children_via_sim)."""
    from tools.swap_detector import enumerate_children_via_sim
    out = []
    for child, p in enumerate_children_via_sim(gs, first, advancement_choice="uniform"):
        if not any((u.position.x, u.position.y) == (first[3], first[4]) for u in child.map.units):
            out.append((child, p))
            continue
        for grand, q in enumerate_children_via_sim(child, second, advancement_choice="uniform"):
            out.append((grand, p * q))
    return out


def test_the_two_attack_table_matches_the_simulators_enumeration():
    """`two_fight_joint` composes the first fight's table with the second
    fight's from each state the first leaves; the simulator walks every
    strike pattern of both. Same marginals, the kill branch included."""
    # Bows against a 12-hp Spearman, who answers with one javelin: two
    # bow hits of the first attacker kill it.
    gs = _board(5, [("Bowman", 1, 2, 1, {}), ("Elvish Fighter", 1, 3, 0, {}),
                    ("Spearman", 2, 3, 1, {"hp": 12})])
    first = ["attack", 2, 1, 3, 1, 1, 1, "00000001"]
    second = ["attack", 3, 0, 3, 1, 1, 1, "00000002"]
    for order in ((first, second), (second, first)):
        joint = two_fight_joint(gs, *order)
        parts = _simulated_window(gs, *order)
        assert joint is not None and any(o["u3"][0] == 0 for o, _p in joint)
        for uid, (field, attr) in itertools.product(("u1", "u2", "u3"), ((1, "current_hp"), (5, "current_exp"))):
            mine = _marginal(joint, lambda o, u=uid, f=field: o[u][f])
            sims = _marginal(parts, lambda s, u=uid, a=attr: next(
                (getattr(x, a) for x in s.map.units if x.id == u), -1))
            assert set(mine) == set(sims), (uid, attr)
            assert all(abs(mine[v] - sims[v]) < 1e-9 for v in mine), (uid, attr)


def _rewrites(gs, commands):
    found = []
    for st in side_turns("dom", gs, [["init_side", 1]] + commands + [["end_turn"]]):
        rws, _counts = side_turn_rewrites(st)
        found += rws
    return found


def test_a_backstab_flanker_moved_after_the_attack_is_a_setup_rewrite():
    """The Thief strikes first, then a Spearman steps behind the target:
    moving it first doubles the dagger, which dominates at R0."""
    gs = _board(5, [("Thief", 1, 1, 1, {}), ("Spearman", 1, 4, 0, {}), ("Spearman", 2, 2, 1, {})])
    rws = _rewrites(gs, [["attack", 1, 1, 2, 1, 0, 0, "00000003"], ["move", [4, 3], [0, 0], 1]])
    setups = [r for r in rws if r.klass == "A"]
    assert setups and admits(setups[0].dims, "D", Combo())


def test_a_surround_move_is_k_unless_it_enables_the_attack():
    """A Spearman lands next to an 8-hp target, then another attacks it:
    attacking first keeps the mover's movement on the kill branch (K).
    A Lieutenant landing on the same hex gives the attacker leadership
    (7 damage becomes 9, one hit kills instead of two): the move enables
    the attack, and it is not K."""
    def run(mover):
        gs = _board(6, [("Spearman", 1, 2, 1, {}), (mover, 1, 4, 0, {}),
                        ("Spearman", 2, 3, 1, {"hp": 8})])
        return [r for r in _rewrites(gs, [["move", [4, 3], [0, 0], 1],
                                          ["attack", 2, 1, 3, 1, 0, 0, "00000004"]])
                if r.klass == "K"]
    ks = run("Spearman")
    assert len(ks) == 1 and ks[0].tier == "O" and ks[0].gains["banked_mp"] > 0
    assert admits(ks[0].dims, "O", Combo(r4=True))
    assert run("Lieutenant") == []
