"""The rewrite classes of docs/xod_dominance_design_20260924.md section 5,
read on a realized side-turn (tools/combat_dominance.py holds the
comparison and the combinations).

Every generator walks the side-turn's commands from its start state and
returns the rewrites whose vector some combination admits, with the
number of candidate rewrites it could not resolve (a fight the exact
enumerator refuses, a window past its particle cap).

Windows are the commands a rewrite touches and nothing else, so a
generator only proposes a rewrite whose commands are adjacent in the
played turn (the setup move of A may come any time after its attack):

  W  an attack with another weapon of the attacker;
  H  a move followed at once by the mover's attack, the move ending on
     another hex next to the target that the unit could reach;
  A  a move played after an attack, that changes the attack's
     distribution when played before it (a backstab flanker, a
     leadership unit, an illuminator), the move landing where it did;
  K  a move ending next to a target followed at once by another unit's
     attack on it, the move not changing that attack: played attack
     first, the move kept on the survive branch and dropped on the kill
     branch, where the mover keeps its movement (tier O);
  Q  two attacks in a row on one target by two units already next to
     it, swapped; F is a Q rewrite only R1 admits that raises an own
     unit's chance to level.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Set, Tuple

from tools.combat_dominance import (EQ, GT, INCOMP, LT, Dim, Rewrite, SideTurn, _unit_at,
                                    attack_action, fight, fight_dims, fight_gains,
                                    minimal_combos, numeric_dim)
from tools.replay_dataset import _apply_command

Hex = Tuple[int, int]


def _set_sym(cand: Set, base: Set, more_is_better: bool) -> str:
    if cand == base:
        return EQ
    bigger = cand > base
    smaller = cand < base
    if not bigger and not smaller:
        return INCOMP
    return GT if bigger == more_is_better else LT


def visibility_dims(cand_gs, base_gs, side: int, own_ids: Set[str]) -> List[Dim]:
    """What the side sees, which enemies it sees, and which of `own_ids`
    the enemy sees, candidate against played (wesnoth_ai.visibility)."""
    from wesnoth_ai.visibility import units_visible_to, visible_hexes_for
    enemy = 3 - side
    seen_c, seen_b = set(visible_hexes_for(cand_gs, side)), set(visible_hexes_for(base_gs, side))
    en_c = {u.id for u in units_visible_to(cand_gs, side) if u.side == enemy}
    en_b = {u.id for u in units_visible_to(base_gs, side) if u.side == enemy}
    shown_c = {u.id for u in units_visible_to(cand_gs, enemy) if u.id in own_ids}
    shown_b = {u.id for u in units_visible_to(base_gs, enemy) if u.id in own_ids}
    return [Dim("seen_hexes", "vis", _set_sym(seen_c, seen_b, True)),
            Dim("enemies_seen", "vis", _set_sym(en_c, en_b, True)),
            Dim("own_shown", "vis", _set_sym(shown_c, shown_b, False))]


class Outcomes:
    """The fight distributions a pass computes, kept per game as optional
    data (tools/game_record.py's `outcomes` shape): keyed by the game
    index of the attack command, then by what was fought ("played",
    "weapon:1", "hex:4,7", "setup:12")."""

    def __init__(self):
        self.by_command: Dict[str, Dict[str, object]] = {}

    def keep(self, st: SideTurn, i: int, label: str, dist) -> None:
        from tools.game_record import distribution_data
        if dist is not None:
            self.by_command.setdefault(str(st.first_index + i), {})[label] = distribution_data(dist)


def _rewrite(klass, st, window, detail, dims, tier="D", gains=None) -> Optional[Rewrite]:
    rw = Rewrite(klass, tier, st.turn, st.side, tuple(window), detail, dims, gains or {})
    return rw if minimal_combos(dims, tier) else None


# ---------------------------------------------------------------------
# W: another weapon
# ---------------------------------------------------------------------
def weapon_rewrites(st: SideTurn, gs, i: int, cmd: list, base, kept: Outcomes
                    ) -> Tuple[List[Rewrite], int]:
    att, dfd = _unit_at(gs, (cmd[1], cmd[2])), _unit_at(gs, (cmd[3], cmd[4]))
    out, inconclusive = [], 0
    for w in range(len(att.attacks)):
        if w == int(cmd[5]):
            continue
        cand = fight(gs, attack_action(cmd, w))
        kept.keep(st, i, f"weapon:{w}", cand)
        if cand is None:
            inconclusive += 1
            continue
        rw = _rewrite("W", st, (i,), {"weapon": w, "played": int(cmd[5])},
                      fight_dims(cand, base, att, dfd), gains=fight_gains(cand, base))
        if rw:
            out.append(rw)
    return out, inconclusive


# ---------------------------------------------------------------------
# H: another attack hex
# ---------------------------------------------------------------------
def _path_to(reach, start: Hex, dest: Hex) -> Optional[List[Hex]]:
    path, cur = [dest], dest
    while cur != start:
        cur = reach.prev.get(cur)
        if cur is None:
            return None
        path.append(cur)
    return path[::-1]


def _enemy_reach(gs, side: int) -> List[Tuple[Hex, Set[Hex]]]:
    """Each enemy unit's hex and the hexes it could move to next turn
    with its full movement, on the board `gs` shows."""
    from tools.pathfind_sim import ReachContext, unit_reach
    out = []
    for e in [u for u in gs.map.units if u.side == 3 - side]:
        ctx = ReachContext.for_side(gs, e.side, god_view=True, exclude_unit=e)
        r = unit_reach(e, gs, ctx, budget=int(e.max_moves))
        out.append(((e.position.x, e.position.y), set(r.landable)))
    return out


def _threats(enemy_reach: List[Tuple[Hex, Set[Hex]]], pos: Hex) -> int:
    """Enemy units standing next to `pos` or able to reach a hex next to
    it next turn."""
    from tools.abilities import hex_neighbors
    around = set(hex_neighbors(*pos))
    return sum(1 for at, reach in enemy_reach if at in around or around & reach)


def _support(gs, unit, pos: Hex) -> int:
    """Own healers and leadership units next to `pos`."""
    from tools.abilities import hex_neighbors
    around = set(hex_neighbors(*pos))
    keys = ("heals_4", "heals_8", "heals+4", "heals+8", "cures", "leadership")
    return sum(1 for u in gs.map.units
               if u.side == unit.side and u.id != unit.id
               and (u.position.x, u.position.y) in around
               and any(k in (u.abilities or ()) for k in keys))


def hex_rewrites(st: SideTurn, gs_move, j: int, move: list, attack: list, kept: Outcomes
                 ) -> Tuple[List[Rewrite], int]:
    """`move` (index j) lands the attacker where `attack` (j + 1) starts."""
    from tools.abilities import hex_neighbors
    from tools.pathfind_sim import ReachContext, defense_pct_at, unit_reach
    from wesnoth_ai.rewards import hex_distance
    start = (move[1][0], move[2][0])
    mover = _unit_at(gs_move, start)
    played_gs = copy.deepcopy(gs_move)
    _apply_command(played_gs, move)
    played_hex = (attack[1], attack[2])
    att, dfd = _unit_at(played_gs, played_hex), _unit_at(played_gs, (attack[3], attack[4]))
    if mover is None or att is None or dfd is None or att.id != mover.id:
        return [], 0
    base = fight(played_gs, attack_action(attack))
    if base is None:
        return [], 1
    ctx = ReachContext.for_side(gs_move, st.side, god_view=True, exclude_unit=mover)
    reach = unit_reach(mover, gs_move, ctx)
    target = (attack[3], attack[4])
    villages = getattr(played_gs.global_info, "_village_owner", {}) or {}
    enemy_reach = None                # computed once, on the played board
    out, inconclusive = [], 0
    for h in sorted(set(hex_neighbors(*target)) & set(reach.landable)):
        if h == played_hex:
            continue
        path = _path_to(reach, start, h)
        if path is None:
            continue
        cand_gs = copy.deepcopy(gs_move)
        _apply_command(cand_gs, ["move", [p[0] for p in path], [p[1] for p in path], st.side])
        moved = _unit_at(cand_gs, h)
        if moved is None or moved.id != mover.id:
            continue                                   # stopped short: ambush, block
        cand = fight(cand_gs, attack_action([attack[0], h[0], h[1], *attack[3:]]))
        kept.keep(st, j + 1, f"hex:{h[0]},{h[1]}", cand)
        if cand is None:
            inconclusive += 1
            continue
        if enemy_reach is None:
            enemy_reach = _enemy_reach(played_gs, st.side)
        cand_villages = getattr(cand_gs.global_info, "_village_owner", {}) or {}
        guard = (defense_pct_at(mover, cand_gs, *h) <= defense_pct_at(mover, played_gs, *played_hex)
                 and not (villages.get(played_hex) == st.side and cand_villages.get(h) != st.side)
                 and _support(cand_gs, mover, h) >= _support(played_gs, mover, played_hex)
                 and _threats(enemy_reach, h) <= _threats(enemy_reach, played_hex))
        own_villages = [(sum(1 for o in cand_villages.values() if o == st.side), 1.0)]
        base_villages = [(sum(1 for o in villages.values() if o == st.side), 1.0)]
        dims = fight_dims(cand, base, att, dfd) + [
            Dim(f"pos:{mover.id}", "pos", EQ, distance=hex_distance(*h, *played_hex), guard=guard),
            numeric_dim("villages", "count", own_villages, base_villages, lambda v: v, True),
        ] + visibility_dims(cand_gs, played_gs, st.side, {mover.id})
        rw = _rewrite("H", st, (j, j + 1), {"hex": list(h), "played": list(played_hex)}, dims,
                      gains=fight_gains(cand, base))
        if rw:
            out.append(rw)
    return out, inconclusive


# ---------------------------------------------------------------------
# A: a later move played before the attack
# ---------------------------------------------------------------------
def setup_rewrites(st: SideTurn, gs_i, i: int, attack: list, base, landings: Dict[int, tuple],
                   kept: Outcomes) -> Tuple[List[Rewrite], int]:
    """`landings[j]` = (mover id, landing hex, movement left) of every
    later move j of the side-turn, as played."""
    from tools.abilities import hex_neighbors
    att, dfd = _unit_at(gs_i, (attack[1], attack[2])), _unit_at(gs_i, (attack[3], attack[4]))
    near = set(hex_neighbors(attack[1], attack[2])) | set(hex_neighbors(attack[3], attack[4]))
    out, inconclusive = [], 0
    for j, (uid, land, mp_left) in sorted(landings.items()):
        if j <= i or land not in near or uid in (att.id, dfd.id):
            continue
        move = st.commands[j]
        cand_gs = copy.deepcopy(gs_i)
        _apply_command(cand_gs, move)
        moved = next((u for u in cand_gs.map.units if u.id == uid), None)
        if moved is None or (moved.position.x, moved.position.y) != land or moved.current_moves != mp_left:
            continue                                   # not the same move before the attack
        cand = fight(cand_gs, attack_action(attack))
        kept.keep(st, i, f"setup:{st.first_index + j}", cand)
        if cand is None:
            inconclusive += 1
            continue
        if cand.probs == base.probs:
            continue                                   # the move changes nothing: not a setup
        rw = _rewrite("A", st, (i, j), {"move": j, "mover": uid},
                      fight_dims(cand, base, att, dfd), gains=fight_gains(cand, base))
        if rw:
            out.append(rw)
    return out, inconclusive


# ---------------------------------------------------------------------
# K: the attack before a surround move that does not enable it
# ---------------------------------------------------------------------
def surround_rewrite(st: SideTurn, gs_move, j: int, move: list, attack: list) -> Tuple[List[Rewrite], int]:
    from tools.abilities import hex_neighbors
    start = (move[1][0], move[2][0])
    mover = _unit_at(gs_move, start)
    after_move = copy.deepcopy(gs_move)
    _apply_command(after_move, move)
    target = (attack[3], attack[4])
    att = _unit_at(gs_move, (attack[1], attack[2]))
    moved = next((u for u in after_move.map.units if mover is not None and u.id == mover.id), None)
    if (mover is None or att is None or moved is None or att.id == mover.id
            or (moved.position.x, moved.position.y) not in set(hex_neighbors(*target))):
        return [], 0
    before = fight(gs_move, attack_action(attack))
    after = fight(after_move, attack_action(attack))
    if before is None or after is None:
        return [], 1
    if before.probs != after.probs:
        return [], 0                                   # the move enables the attack: A, not K
    p_kill = sum(p for k, p in before.probs.items() if k[1] <= 0)
    if p_kill <= 1e-9:
        return [], 0
    # On the kill branch the mover stays with its movement; it could
    # still make the move (option dominance). What the side sees differs:
    # the move clears fog along its path and moves the mover's own
    # exposure.
    banked = int(mover.current_moves) - int(moved.current_moves)
    dims = visibility_dims(gs_move, after_move, st.side, {mover.id})
    rw = _rewrite("K", st, (j, j + 1), {"mover": mover.id, "p_kill": p_kill},
                  dims, tier="O", gains={"banked_mp": p_kill * banked})
    return ([rw] if rw else []), 0


# ---------------------------------------------------------------------
# Q and F: two attacks on one target, swapped
# ---------------------------------------------------------------------
# A unit's state after a window: (alive, hp, slowed, poisoned, type,
# experience, attacked); a dead unit reads (0, -1, 0, 0, type, -1, attacked).
UnitEnd = Tuple[int, int, int, int, str, int, int]


def _after_fight(unit, opp, hp, slowed, poisoned, utype, opp_dead, attacked) -> UnitEnd:
    from tools.combat_dominance import _level
    if hp <= 0:
        return (0, -1, 0, 0, utype, -1, attacked)
    if utype != unit.name:
        return (1, hp, int(slowed), int(poisoned), utype, 1_000_000, attacked)
    gain = _kill_xp_of(opp) if opp_dead else _level(opp.name)
    return (1, hp, int(slowed), int(poisoned), utype, unit.current_exp + gain, attacked)


def _kill_xp_of(unit) -> int:
    from tools.combat_dominance import _level
    from tools.combat_outcomes import _kill_xp
    return _kill_xp(_level(unit.name))


def _with(gs, unit, end: UnitEnd):
    """`gs` with `unit` set to `end` (removed when dead)."""
    from tools.replay_dataset import _rebuild_unit
    gs.map.units.discard(unit)
    if not end[0]:
        return None
    statuses = set(unit.statuses or ()) - {"slowed", "poisoned"}
    statuses |= ({"slowed"} if end[2] else set()) | ({"poisoned"} if end[3] else set())
    new = _rebuild_unit(unit, current_hp=end[1], statuses=statuses,
                        current_exp=end[5], has_attacked=unit.has_attacked or bool(end[6]))
    gs.map.units.add(new)
    return new


def two_fight_joint(gs, first: list, second: list) -> Optional[List[Tuple[Dict[str, UnitEnd], float]]]:
    """The exact joint outcome of two attacks on one target, `first` then
    `second`, over the two attackers and the target: the first fight's
    table, then the second fight from each state the first leaves the
    target and the first attacker in. None when a fight cannot be
    enumerated or the target levels in the first fight."""
    a1 = _unit_at(gs, (first[1], first[2]))
    a2 = _unit_at(gs, (second[1], second[2]))
    t = _unit_at(gs, (first[3], first[4]))
    d1 = fight(gs, attack_action(first))
    if d1 is None:
        return None
    a2_idle: UnitEnd = (1, a2.current_hp, int("slowed" in (a2.statuses or ())),
                        int("poisoned" in (a2.statuses or ())), a2.name, a2.current_exp, 0)
    out: List[Tuple[Dict[str, UnitEnd], float]] = []
    second_cache: Dict[tuple, object] = {}
    for k, p in d1.probs.items():
        a_hp, t_hp, a_sl, t_sl, a_po, t_po, _a_pe, t_pe, a_ty, t_ty = k
        e1 = _after_fight(a1, t, a_hp, a_sl, a_po, a_ty, t_hp <= 0, 1)
        if t_hp <= 0:
            out.append(({a1.id: e1, a2.id: a2_idle, t.id: (0, -1, 0, 0, t_ty, -1, 0)}, p))
            continue
        if t_ty != t.name or t_pe:
            return None
        et = _after_fight(t, a1, t_hp, t_sl, t_po, t_ty, a_hp <= 0, 0)
        key = (e1[0], et)
        if key not in second_cache:
            s = copy.deepcopy(gs)
            _with(s, _unit_at(s, (first[1], first[2])), e1)
            _with(s, _unit_at(s, (first[3], first[4])), et)
            second_cache[key] = (fight(s, attack_action(second)), et)
        d2, et_before = second_cache[key]
        if d2 is None:
            return None
        t_after_first = _unit_at(gs, (first[3], first[4]))
        for k2, p2 in d2.probs.items():
            b_hp, u_hp, b_sl, u_sl, b_po, u_po, _b_pe, _u_pe, b_ty, u_ty = k2
            e2 = _after_fight(a2, t, b_hp, b_sl, b_po, b_ty, u_hp <= 0, 1)
            if u_hp <= 0:
                eu = (0, -1, 0, 0, u_ty, -1, 0)
            elif u_ty != t_after_first.name:
                eu = (1, u_hp, u_sl, u_po, u_ty, 1_000_000, 0)
            else:
                gain = _kill_xp_of(a2) if b_hp <= 0 else _level_of(a2)
                eu = (1, u_hp, int(u_sl), int(u_po), u_ty, et_before[5] + gain, 0)
            out.append(({a1.id: e1, a2.id: e2, t.id: eu}, p * p2))
    return out


def _level_of(unit) -> int:
    from tools.combat_dominance import _level
    return _level(unit.name)


def _joint_dims(cand, base, ids: Dict[str, int], side: int, types: Dict[str, str]) -> List[Dim]:
    dims: List[Dim] = []
    for uid, uside in sorted(ids.items()):
        own = uside == side
        dims += [
            numeric_dim(f"alive:{uid}", "binary", cand, base, lambda o, u=uid: o[u][0], own),
            numeric_dim(f"hp:{uid}", "hp", cand, base, lambda o, u=uid: o[u][1], own),
            numeric_dim(f"slowed:{uid}", "binary", cand, base, lambda o, u=uid: o[u][2], not own),
            numeric_dim(f"poisoned:{uid}", "binary", cand, base, lambda o, u=uid: o[u][3], not own),
            numeric_dim(f"levelup:{uid}", "binary", cand, base,
                        lambda o, u=uid: int(o[u][0] == 1 and o[u][4] != types[u]), own),
            numeric_dim(f"xp:{uid}", "xp", cand, base, lambda o, u=uid: o[u][5], own),
        ]
        if own:
            dims.append(numeric_dim(f"attack_left:{uid}", "binary", cand, base,
                                    lambda o, u=uid: int(o[u][0] == 1 and not o[u][6]), True))
    return dims


def order_rewrite(st: SideTurn, gs_i, i: int, first: list, second: list) -> Tuple[List[Rewrite], int]:
    from tools.abilities import hex_neighbors
    target = (first[3], first[4])
    if (second[3], second[4]) != target or (first[1], first[2]) == (second[1], second[2]):
        return [], 0
    a1, a2 = _unit_at(gs_i, (first[1], first[2])), _unit_at(gs_i, (second[1], second[2]))
    t = _unit_at(gs_i, target)
    if a1 is None or a2 is None or t is None or (second[1], second[2]) not in set(hex_neighbors(*target)):
        return [], 0
    base = two_fight_joint(gs_i, first, second)
    cand = two_fight_joint(gs_i, second, first)
    if base is None or cand is None:
        return [], 1
    ids = {a1.id: a1.side, a2.id: a2.side, t.id: t.side}
    types = {a1.id: a1.name, a2.id: a2.name, t.id: t.name}
    dims = _joint_dims(cand, base, ids, st.side, types)
    mins = minimal_combos(dims, "D")
    if not mins:
        return [], 0
    levels = any(d.name.startswith("levelup:") and d.sym == GT and ids[d.name.split(":", 1)[1]] == st.side
                 for d in dims)
    klass = "F" if levels and all(c.r1 for c in mins) else "Q"

    def p_dead(joint, uid):
        return sum(p for o, p in joint if not o[uid][0])
    rw = Rewrite(klass, "D", st.turn, st.side, (i, i + 1),
                 {"first": a2.id, "second": a1.id, "target": t.id}, dims,
                 {"d_p_kill": p_dead(cand, t.id) - p_dead(base, t.id)})
    return [rw], 0


# ---------------------------------------------------------------------
# One side-turn
# ---------------------------------------------------------------------
def side_turn_rewrites(st: SideTurn, kept: Optional[Outcomes] = None
                       ) -> Tuple[List[Rewrite], Dict[str, int]]:
    """Every class over one realized side-turn. The counts: decisions
    and attacks of the side, candidates that could not be resolved.
    `kept` collects the fight distributions computed on the way."""
    kept = kept if kept is not None else Outcomes()
    counts = {"decisions": 0, "attacks": 0, "inconclusive": 0}
    rewrites: List[Rewrite] = []
    gs = copy.deepcopy(st.pre_state)
    states = []
    for cmd in st.commands:
        states.append(copy.deepcopy(gs))
        _apply_command(gs, cmd)
    landings: Dict[int, tuple] = {}
    for j, cmd in enumerate(st.commands):
        if cmd[0] == "move":
            before = _unit_at(states[j], (cmd[1][0], cmd[2][0]))
            after_gs = states[j + 1] if j + 1 < len(states) else gs
            moved = before and next((u for u in after_gs.map.units if u.id == before.id), None)
            if moved is not None:
                landings[j] = (moved.id, (moved.position.x, moved.position.y), moved.current_moves)
    for i, cmd in enumerate(st.commands):
        kind = cmd[0]
        if kind in ("move", "attack", "recruit", "end_turn"):
            counts["decisions"] += 1
        if kind != "attack":
            continue
        counts["attacks"] += 1
        gs_i = states[i]
        att, dfd = _unit_at(gs_i, (cmd[1], cmd[2])), _unit_at(gs_i, (cmd[3], cmd[4]))
        if att is None or dfd is None:
            continue
        base = fight(gs_i, attack_action(cmd))
        kept.keep(st, i, "played", base)
        if base is None:
            counts["inconclusive"] += 1
            continue
        for found, inc in (weapon_rewrites(st, gs_i, i, cmd, base, kept),
                           setup_rewrites(st, gs_i, i, cmd, base, landings, kept)):
            rewrites += found
            counts["inconclusive"] += inc
        if i >= 1 and st.commands[i - 1][0] == "move":
            for found, inc in (hex_rewrites(st, states[i - 1], i - 1, st.commands[i - 1], cmd, kept),
                               surround_rewrite(st, states[i - 1], i - 1, st.commands[i - 1], cmd)):
                rewrites += found
                counts["inconclusive"] += inc
        if i + 1 < len(st.commands) and st.commands[i + 1][0] == "attack":
            found, inc = order_rewrite(st, gs_i, i, cmd, st.commands[i + 1])
            rewrites += found
            counts["inconclusive"] += inc
    return rewrites, counts
