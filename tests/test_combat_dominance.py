"""The rewrite certificates of docs/xod_dominance_design_20260924.md
(tools/combat_dominance.py, tools/dominance_rewrites.py) and the count
over games (tools/analysis/dominance_count.py): the almost-dominance
ratio on the design's worked example, what a combination drops, the
two-attack table against the simulator's own enumeration, and each
class on a small board."""
from __future__ import annotations

import gzip
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.analysis.dominance_count import count_game, merge  # noqa: E402
from tools.combat_dominance import (COMBOS, EQ, GT, INCOMP, LT, Combo, Dim,  # noqa: E402
                                    admits, compare_marginals, minimal_combos, side_turns)
from tools.dominance_rewrites import side_turn_rewrites, two_fight_joint  # noqa: E402
from tools.replay_dataset import _apply_command, _build_initial_gamestate  # noqa: E402

G, FOREST, VILLAGE, HILL = "Gg", "Gs^Fp", "Gg^Vh", "Hh"

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


def test_a_dropped_dimension_neither_blocks_nor_justifies():
    xp_only = [Dim("own_xp", "xp", GT), Dim("own_hp", "hp", EQ)]
    assert admits(xp_only, "D", Combo()) and not admits(xp_only, "D", Combo(r1=True))
    xp_for_hp = [Dim("own_xp", "xp", LT, eps=1.0), Dim("own_hp", "hp", GT)]
    assert not admits(xp_for_hp, "D", Combo()) and admits(xp_for_hp, "D", Combo(r1=True))
    view_only = [Dim("seen_hexes", "vis", GT), Dim("villages", "count", GT),
                 Dim("pos:u1", "pos", EQ, distance=1, guard=True), Dim("own_hp", "hp", EQ)]
    assert not any(admits(view_only, "D", c) for c in COMBOS)


def test_a_looser_combination_blocks_nothing_a_tighter_one_lets_through():
    """Tier O asks only that no kept dimension be worse, so it shows the
    blocking half of the filter alone: loosening an axis never blocks."""
    rng = random.Random(7)
    kinds = ("binary", "option", "hp", "xp", "pos", "vis", "count")
    for _ in range(300):
        dims = []
        for k in range(rng.randint(1, 6)):
            kind = rng.choice(kinds)
            sym = EQ if kind == "pos" else rng.choice((GT, EQ, LT, INCOMP))
            dims.append(Dim(f"d{k}", kind, sym, eps=rng.random(),
                            distance=rng.choice((0, 1, 2, 3)) if kind == "pos" else 0,
                            guard=rng.random() < 0.5))
        for a, b in itertools.product(COMBOS, COMBOS):
            if a.within(b) and admits(dims, "O", a):
                assert admits(dims, "O", b), (dims, a, b)


def test_minimal_combinations_are_the_smallest_that_admit():
    dims = [Dim("enemy_alive", "binary", GT),
            Dim("pos:u1", "pos", EQ, distance=2, guard=False),
            Dim("seen_hexes", "vis", LT)]
    assert [c.name() for c in minimal_combos(dims, "D")] == ["r3l+r4"]
    dims[1] = Dim("pos:u1", "pos", EQ, distance=2, guard=True)
    assert [c.name() for c in minimal_combos(dims, "D")] == ["r3g+r4"]


# ---------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------
def _game(rows, units, villages=(), fog=True):
    width = len(rows[0])
    border = ", ".join(["Xv"] * (width + 2))
    lines = [border] + ["Xv, " + ", ".join(r) + ", Xv" for r in rows] + [border]
    return {"game_id": "dom", "map_data": "\n".join(lines),
            "starting_units": [{"uid": i + 1, "type": t, "side": s, "x": x, "y": y, "is_leader": False, **extra}
                               for i, (t, s, x, y, extra) in enumerate(units)],
            "starting_villages": [{"x": x, "y": y, "side": s} for x, y, s in villages],
            "starting_sides": [{"side": k, "gold": 0, "recruit": [], "fog": fog} for k in (1, 2)]}


def _board(width, units, height=3, fog=True):
    return _build_initial_gamestate(_game([[G] * width] * height, units, fog=fog))


def _rewrites(gs, commands):
    found, counts = [], {}
    for st in side_turns("dom", gs, [["init_side", 1]] + commands + [["end_turn"]]):
        rws, c = side_turn_rewrites(st)
        found += rws
        counts.update(c)
    return found, counts


def _names(rw):
    return [c.name() for c in minimal_combos(rw.dims, rw.tier)]


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


# ---------------------------------------------------------------------
# W and H
# ---------------------------------------------------------------------
def test_a_dominated_weapon_is_a_weapon_rewrite_and_the_dominant_one_is_not():
    """A Mage's staff (5-1, answered by the Spearman's spear) against its
    missile (7-3 magical, answered by one javelin)."""
    gs = _board(5, [("Mage", 1, 1, 1, {}), ("Spearman", 2, 2, 1, {})])
    staff, _ = _rewrites(gs, [["attack", 1, 1, 2, 1, 0, 0, "00000001"]])
    assert [(r.klass, r.detail["weapon"], _names(r)) for r in staff] == [("W", 1, ["R0"])]
    assert staff[0].anchor == 1                  # the attack's index in the game
    missile, _ = _rewrites(gs, [["attack", 1, 1, 2, 1, 1, 1, "00000001"]])
    assert missile == []


def _hex_game(rows, extra_units=(), later=(), villages=()):
    """An Elvish Archer steps from (0,1) onto (1,1) and swords the
    Spearman on (2,1); (2,2) is another hex next to the target."""
    data = _game(rows, [("Elvish Archer", 1, 0, 1, {}), ("Spearman", 2, 2, 1, {}), *extra_units],
                 villages=villages)
    found, counts = _rewrites(_build_initial_gamestate(data),
                              [["move", [0, 1], [1, 1], 1], ["attack", 1, 1, 2, 1, 0, 0, "00000002"], *later])
    return [r for r in found if r.klass == "H"], counts


def test_a_better_attack_hex_is_admitted_by_r3_unless_a_later_move_needs_it():
    rows = [[G] * 5, [G] * 5, [G, G, FOREST, G, G]]
    found, _ = _hex_game(rows)
    hexes = [(r.klass, r.detail["hex"], _names(r)) for r in found]
    assert hexes == [("H", [2, 2], ["r3g"])]     # 70% defense in the forest, 40% on grass
    found, counts = _hex_game(rows, extra_units=[("Spearman", 1, 4, 2, {})],
                              later=[["move", [4, 3, 2], [2, 2, 2], 1]])
    assert found == [] and counts["h_not_isolated"] == 1


def test_the_guard_keeps_a_unit_on_a_village_it_holds():
    """From its own village (60%) to the forest (70%): a better fight,
    admitted by literal R3 and refused by the guard."""
    rows = [[G] * 5, [G, VILLAGE, G, G, G], [G, G, FOREST, G, G]]
    found, _ = _hex_game(rows, villages=[(1, 1, 1)])
    assert [(r.detail["hex"], _names(r)) for r in found] == [([2, 2], ["r3l"])]


# ---------------------------------------------------------------------
# A and K
# ---------------------------------------------------------------------
def test_a_backstab_flanker_moved_after_the_attack_is_a_setup_rewrite():
    """The Thief strikes first, then a Spearman steps behind the target:
    moving it first doubles the dagger, which dominates at R0."""
    gs = _board(5, [("Thief", 1, 1, 1, {}), ("Spearman", 1, 4, 0, {}), ("Spearman", 2, 2, 1, {})])
    rws, _ = _rewrites(gs, [["attack", 1, 1, 2, 1, 0, 0, "00000003"], ["move", [4, 3], [0, 0], 1]])
    assert [(r.klass, _names(r)) for r in rws] == [("A", ["R0"])]


def test_a_setup_move_that_is_legal_only_after_the_attack_is_not_a_rewrite():
    """A Lieutenant walks (0,3) -> (1,2) -> (0,2) and lands next to the
    Thief, whose attack its leadership raises. With a Grunt on (2,3),
    whose zone of control stops the walk on (1,2), the move is played
    after a Spearman kills the Grunt; played before the Thief's attack it
    would stop on (1,2), where its leadership still changes the fight but
    the move is not the one played. No setup rewrite then."""
    base_units = [("Thief", 1, 1, 1, {}), ("Lieutenant", 1, 0, 3, {}), ("Spearman", 2, 2, 1, {})]
    attack = ["attack", 1, 1, 2, 1, 0, 0, "00000003"]
    move = ["move", [0, 1, 0], [3, 2, 2], 1]
    free, _ = _rewrites(_board(7, base_units, height=5), [attack, move])
    assert [r.klass for r in free] == ["A"]
    units = base_units + [("Orcish Grunt", 2, 2, 3, {"hp": 1}), ("Spearman", 1, 2, 4, {})]
    kill = ["attack", 2, 4, 2, 3, 0, 0, "00000005"]
    played = _board(7, units, height=5)
    for cmd in (["init_side", 1], attack, kill, move):
        _apply_command(played, cmd)
    assert ("Lieutenant", 0, 2) in {(u.name, u.position.x, u.position.y) for u in played.map.units}
    assert not any(u.name == "Orcish Grunt" for u in played.map.units)
    blocked, _ = _rewrites(_board(7, units, height=5), [attack, kill, move])
    assert [r.klass for r in blocked if r.klass == "A"] == []


def test_an_attack_that_can_kill_has_no_setup_rewrite():
    """On the kill branch the played turn could drop the flanker's move;
    the rewrite would commit it."""
    gs = _board(5, [("Thief", 1, 1, 1, {}), ("Spearman", 1, 4, 0, {}), ("Spearman", 2, 2, 1, {"hp": 10})])
    rws, counts = _rewrites(gs, [["attack", 1, 1, 2, 1, 0, 0, "00000003"], ["move", [4, 3], [0, 0], 1]])
    assert [r.klass for r in rws if r.klass == "A"] == [] and counts["a_kill_in_reach"] == 1


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
                                          ["attack", 2, 1, 3, 1, 0, 0, "00000004"]])[0]
                if r.klass == "K"]
    ks = run("Spearman")
    assert len(ks) == 1 and ks[0].tier == "O"
    # Two hits of three at 60% kill: p = 0.648; the mover ends in the
    # target's zone of control, so it banks all 5 of its movement.
    assert ks[0].gains["banked_mp"] == pytest.approx(5 * 0.648)
    assert run("Lieutenant") == []


def test_a_k_move_that_clears_fog_needs_r4():
    """On a wide board the mover's walk clears fog no other unit sees:
    without the move (the kill branch) the side sees less, so R0 refuses
    and R4 admits. Without fog R0 admits."""
    def mins(fog):
        gs = _board(16, [("Spearman", 1, 6, 1, {}), ("Spearman", 1, 2, 0, {}),
                         ("Spearman", 2, 7, 1, {"hp": 8})], fog=fog)
        rws, _ = _rewrites(gs, [["move", [2, 3, 4, 5, 6, 7], [0] * 6, 1],
                                ["attack", 6, 1, 7, 1, 0, 0, "00000004"]])
        return [_names(r) for r in rws if r.klass == "K"]
    assert mins(True) == [["r4"]]
    assert mins(False) == [["R0"]]


# ---------------------------------------------------------------------
# Q and F
# ---------------------------------------------------------------------
def _order(units):
    """A Spearman then an Elvish Shaman on a 24-hp Grunt: neither kills
    alone, so both always attack and the second one takes the kill. The
    Shaman first slows the Grunt, which halves its answer to the Spearman."""
    gs = _board(5, units)
    rws, _ = _rewrites(gs, [["attack", 1, 1, 2, 1, 0, 0, "00000001"],
                            ["attack", 2, 2, 2, 1, 1, 0, "00000002"]])
    return [(r.klass, _names(r)) for r in rws if r.klass in ("Q", "F")]


def test_an_amla_counts_as_a_level_up():
    """A Royal Guard has no advancement: crossing its threshold is an
    AMLA, which keeps the type and heals it fully (the exact enumerator
    resolves it so), and counts as levelling."""
    from tools.combat_dominance import LEVELLED, end_xp
    gs = _board(5, [("Royal Guard", 1, 1, 1, {"max_exp": 8, "hp": 20}), ("Spearman", 2, 2, 1, {"hp": 1})])
    guard = next(u for u in gs.map.units if u.name == "Royal Guard")
    assert end_xp(guard, True, guard.name, 1, True) == LEVELLED       # a kill is worth 8
    assert end_xp(guard, True, guard.name, 1, False) == 1
    from tools.combat_dominance import attack_action, fight
    _apply_command(gs, ["init_side", 1])
    dist = fight(gs, attack_action(["attack", 1, 1, 2, 1, 0, 0]))
    kills = {k[0] for k in dist.probs if k[1] <= 0 and k[0] > 0}
    assert kills and all(hp > 20 for hp in kills) and {k[8] for k in dist.probs if k[0] > 0} == {"Royal Guard"}


def test_r1_admits_an_order_that_moves_kill_experience():
    grunt = ("Orcish Grunt", 2, 2, 1, {"hp": 24})
    assert _order([("Spearman", 1, 1, 1, {}), ("Elvish Shaman", 1, 2, 2, {}), grunt]) == [("Q", ["r1"])]
    # The same order takes a level-up from the Shaman: refused everywhere.
    assert _order([("Spearman", 1, 1, 1, {}), ("Elvish Shaman", 1, 2, 2, {"max_exp": 8}), grunt]) == []
    # It gives one to the Spearman: F.
    assert _order([("Spearman", 1, 1, 1, {"max_exp": 8}), ("Elvish Shaman", 1, 2, 2, {}), grunt]) == [("F", ["r1"])]


# ---------------------------------------------------------------------
# The count over games
# ---------------------------------------------------------------------
def test_the_count_reads_a_game_and_keeps_a_failed_one_out_of_the_denominators(tmp_path):
    """An Elvish Archer swords a Spearman from grass with a forest on
    each side of it: two attack-hex rewrites, one opportunity."""
    data = _game([[G, FOREST, G, G, G], [G] * 5, [G, G, FOREST, G, G]],
                 [("Elvish Archer", 1, 0, 1, {}), ("Spearman", 2, 2, 1, {})])
    data["commands"] = [["init_side", 1], ["move", [0, 1], [1, 1], 1], ["attack", 1, 1, 2, 1, 0, 0, "00000002"],
                        ["end_turn"], ["init_side", 2], ["end_turn"]]
    good = tmp_path / "good.json.gz"
    with gzip.open(good, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(b"not a game")
    outcomes = tmp_path / "outcomes.jsonl.gz"
    totals = merge(map(count_game, [("human", "good", str(good)), ("human", "bad", str(bad))]),
                   outcomes_out=outcomes, counts_out=tmp_path / "counts.json")
    t = totals["human"]
    assert t["tally"]["games"] == 1 and t["tally"]["games_failed"] == 1
    assert t["tally"]["decisions"] == 4 and t["tally"]["attacks"] == 1      # move, attack, two end_turns
    assert t["admitted"]["H|r3g"] == 2 and t["opportunities"]["H|r3g"] == 1
    assert t["opportunities"]["H*|r3g"] == 1 and "H|R0" not in t["opportunities"]
    lines = [json.loads(line) for line in gzip.open(outcomes, "rt", encoding="utf-8")]
    assert [x["game"] for x in lines] == ["good"] and "played" in lines[0]["outcomes"]["2"]
    assert json.loads((tmp_path / "counts.json").read_text())["games_done"] == 2
