"""Every candidate rewrite of every XOD class on corpus side-turns,
admitted or not, with its full vector, its outcome distributions and
context flags. One JSON line per game in the output (gzip members).

    python census.py --games 40 --offset 0 --jobs 3 --out rows_000.jsonl.gz

Imports the generators' pieces from the exp/xod-dominance worktree and
edits nothing there.
"""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = r"C:/Users/amaur/Desktop/Perso/projects/Wesnoth_AI_xod"
CORPUS = Path(r"C:/Users/amaur/Desktop/Perso/projects/Wesnoth_AI/replays_dataset_imitation")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/tools")

from tools.analysis.dominance_count import _load, corpus_games  # noqa: E402
from tools.combat_dominance import (EQ, LEVELLED, LT, Dim, _level, _unit_at,  # noqa: E402
                                    attack_action, fight, fight_dims, fight_gains,
                                    minimal_combos, numeric_dim, side_turns)
from tools.dominance_rewrites import (_enemy_reach, _isolated, _joint_dims, _p_kill,  # noqa: E402
                                      _path_to, _support, _threats, two_fight_joint,
                                      visibility_dims)
from tools.replay_dataset import _apply_command  # noqa: E402


# ---------------------------------------------------------------------
# Compact records
# ---------------------------------------------------------------------
def dims_data(dims):
    return [[d.name, d.kind, d.sym, round(d.eps, 5), d.distance, int(d.guard)] for d in dims]


def dist_data(dist, att, dfd):
    """[a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe, a_adv, d_adv, p]; a_adv
    is 1 when the attacker ends as another type (advanced)."""
    out = []
    for k, p in dist.probs.items():
        a_adv = int(k[0] > 0 and k[8] != att.name)
        d_adv = int(k[1] > 0 and k[9] != dfd.name)
        out.append([k[0], k[1], int(bool(k[2])), int(bool(k[3])), int(bool(k[4])), int(bool(k[5])),
                    int(bool(k[6])), int(bool(k[7])), a_adv, d_adv, round(p, 12)])
    return out


def joint_data(joint, order):
    agg = collections.defaultdict(float)
    for o, p in joint:
        key = tuple(v for u in order for v in (o[u][0], o[u][1], o[u][2], o[u][3], o[u][5], o[u][6]))
        agg[key] += p
    return [list(k) + [round(p, 12)] for k, p in agg.items()]


def unit_ctx(u, gs):
    from tools.abilities import adjacent_curer, healer_heal_amount
    from tools.terrain_resolver import terrain_heals, strip_start_position
    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    pos = (u.position.x, u.position.y)
    code = codes.get(pos, "")
    try:
        t_heal = terrain_heals(strip_start_position(code)) if code else 0
    except Exception:  # noqa: BLE001
        t_heal = 0
    villages = getattr(gs.global_info, "_village_owner", {}) or {}
    return {"id": u.id, "name": u.name, "side": u.side, "level": _level(u.name),
            "exp": u.current_exp, "max_exp": u.max_exp, "hp": u.current_hp, "max_hp": u.max_hp,
            "traits": sorted(u.traits or ()), "abilities": sorted(u.abilities or ()),
            "statuses": sorted(u.statuses or ()), "leader": bool(u.is_leader),
            "pos": list(pos), "terrain_heal": t_heal, "village": pos in villages,
            "healer": healer_heal_amount(u, gs.map.units), "curer": adjacent_curer(u, gs.map.units),
            "moves": u.current_moves, "max_moves": u.max_moves, "attacked": bool(u.has_attacked)}


# ---------------------------------------------------------------------
# Context of one attack
# ---------------------------------------------------------------------
def followups(st, states, i, target_hex, target_id, exclude):
    """Own units (not in `exclude`) that could still attack the target at
    the state before command i: next to it, or able to reach a free hex
    next to it; and the later attacks on it actually played."""
    from tools.abilities import hex_neighbors
    from tools.pathfind_sim import ReachContext, unit_reach
    gs = states[i]
    around = set(hex_neighbors(*target_hex))
    occupied = {(u.position.x, u.position.y) for u in gs.map.units}
    free = around - occupied
    potential = 0
    for u in gs.map.units:
        if u.side != st.side or u.id in exclude or u.has_attacked or not u.attacks:
            continue
        if "petrified" in (u.statuses or ()):
            continue
        pos = (u.position.x, u.position.y)
        if pos in around:
            potential += 1
            continue
        if u.current_moves <= 0 or not free:
            continue
        ctx = ReachContext.for_side(gs, st.side, god_view=True, exclude_unit=u)
        if free & set(unit_reach(u, gs, ctx).landable):
            potential += 1
    return potential


def realized_later(st, states, after, target_id):
    n = 0
    for k in range(after, len(st.commands)):
        c = st.commands[k]
        if c[0] != "attack":
            continue
        t = _unit_at(states[k], (c[3], c[4]))
        if t is not None and t.id == target_id:
            n += 1
    return n


def attack_ctx(st, states, i, att, dfd, exclude, window_end):
    from tools.abilities import hex_neighbors
    gs = states[i]
    th = (dfd.position.x, dfd.position.y)
    other_enemies = sum(1 for u in gs.map.units if u.side == dfd.side and u.id != dfd.id
                        and (u.position.x, u.position.y) in set(hex_neighbors(*th))
                        and "petrified" not in (u.statuses or ()))
    after = states[window_end + 1]
    t_after = next((u for u in after.map.units if u.id == dfd.id), None)
    return {"target": unit_ctx(dfd, gs), "attacker": unit_ctx(att, gs),
            "followups_potential": followups(st, states, i, th, dfd.id, exclude),
            "followups_played": realized_later(st, states, window_end + 1, dfd.id),
            "enemies_next_to_target": other_enemies,
            "target_survived_realized": t_after is not None,
            "turn": st.turn, "fog": bool(getattr(gs.global_info, "_fog", True))}


# ---------------------------------------------------------------------
# Classes, every candidate kept
# ---------------------------------------------------------------------
def row(klass, st, i, detail, dims, tier="D", gains=None, flags=None, ctx=None, dists=None):
    return {"cls": klass, "turn": st.turn, "side": st.side, "i": i, "anchor": st.first_index + i,
            "tier": tier, "detail": detail, "dims": dims_data(dims), "gains": gains or {},
            "flags": flags or {}, "ctx": ctx or {}, "dists": dists or {},
            "minimal": [c.name() for c in minimal_combos(dims, tier)] if not (flags or {}).get("class_block") else []}


def weapons(st, states, i, cmd, base, att, dfd, ctx):
    gs = states[i]
    rows, tally = [], collections.Counter()
    for w in range(len(att.attacks)):
        if w == int(cmd[5]):
            continue
        cand = fight(gs, attack_action(cmd, w))
        if cand is None:
            tally["W_inconclusive"] += 1
            continue
        rows.append(row("W", st, i, {"weapon": w, "played": int(cmd[5])}, fight_dims(cand, base, att, dfd),
                        gains=fight_gains(cand, base), ctx=ctx,
                        dists={"cand": dist_data(cand, att, dfd), "base": dist_data(base, att, dfd)}))
    return rows, tally


def hexes(st, states, landings, j, ctx_fn):
    from tools.abilities import hex_neighbors
    from tools.pathfind_sim import ReachContext, defense_pct_at, unit_reach
    from wesnoth_ai.rewards import hex_distance
    move, attack = st.commands[j], st.commands[j + 1]
    gs_move, played_gs = states[j], states[j + 1]
    start = (move[1][0], move[2][0])
    mover = _unit_at(gs_move, start)
    played_hex = (attack[1], attack[2])
    att, dfd = _unit_at(played_gs, played_hex), _unit_at(played_gs, (attack[3], attack[4]))
    rows, tally = [], collections.Counter()
    if mover is None or att is None or dfd is None or att.id != mover.id:
        return rows, tally
    base = fight(played_gs, attack_action(attack))
    if base is None:
        tally["H_inconclusive"] += 1
        return rows, tally
    ctx = ctx_fn()
    reach = unit_reach(mover, gs_move, ReachContext.for_side(gs_move, st.side, god_view=True, exclude_unit=mover))
    target = (attack[3], attack[4])
    villages = getattr(played_gs.global_info, "_village_owner", {}) or {}
    enemy_reach = None
    later = range(j + 2, len(st.commands))
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
            tally["H_stopped_short"] += 1
            continue
        cand = fight(cand_gs, attack_action([attack[0], h[0], h[1], *attack[3:]]))
        if cand is None:
            tally["H_inconclusive"] += 1
            continue
        dims = fight_dims(cand, base, att, dfd)
        dists = {"cand": dist_data(cand, att, dfd), "base": dist_data(base, att, dfd)}
        strictly_worse = all(d.sym in (EQ, LT) for d in dims) and any(d.sym == LT for d in dims)
        d_cand, d_played = defense_pct_at(mover, cand_gs, *h), defense_pct_at(mover, played_gs, *played_hex)
        flags = {"cth_cand": d_cand, "cth_played": d_played}
        if strictly_worse:
            flags["fight_worse"] = True
            flags["class_block"] = "h_fight_worse"
            rows.append(row("H", st, j + 1, {"hex": list(h), "played": list(played_hex)}, dims,
                            gains=fight_gains(cand, base), flags=flags, ctx=ctx, dists=dists))
            continue
        isolated = _isolated(st, states, landings, later, mover.id, played_hex, h)
        if isolated is None:
            tally["H_inconclusive"] += 1
            continue
        if not isolated:
            flags["class_block"] = "h_not_isolated"
        if enemy_reach is None:
            enemy_reach = _enemy_reach(played_gs, st.side)
        cand_villages = getattr(cand_gs.global_info, "_village_owner", {}) or {}
        g_def = d_cand <= d_played
        g_vil = not (villages.get(played_hex) == st.side and cand_villages.get(h) != st.side)
        g_sup = _support(cand_gs, mover, h) >= _support(played_gs, mover, played_hex)
        thr_c, thr_p = _threats(enemy_reach, h), _threats(enemy_reach, played_hex)
        g_thr = thr_c <= thr_p
        flags.update({"g_def": g_def, "g_vil": g_vil, "g_sup": g_sup, "g_thr": g_thr,
                      "threats_cand": thr_c, "threats_played": thr_p})
        guard = g_def and g_vil and g_sup and g_thr
        own_villages = [(sum(1 for o in cand_villages.values() if o == st.side), 1.0)]
        base_villages = [(sum(1 for o in villages.values() if o == st.side), 1.0)]
        dims += [Dim(f"pos:{mover.id}", "pos", EQ, distance=hex_distance(*h, *played_hex), guard=guard),
                 numeric_dim("villages", "count", own_villages, base_villages, lambda v: v, True),
                 ] + visibility_dims(cand_gs, played_gs, st.side, {mover.id})
        rows.append(row("H", st, j + 1, {"hex": list(h), "played": list(played_hex)}, dims,
                        gains=fight_gains(cand, base), flags=flags, ctx=ctx, dists=dists))
    return rows, tally


def setups(st, states, i, base, landings, att, dfd, ctx):
    from tools.abilities import hex_neighbors
    attack, gs_i = st.commands[i], states[i]
    near = set(hex_neighbors(attack[1], attack[2])) | set(hex_neighbors(attack[3], attack[4]))
    rows, tally = [], collections.Counter()
    moves = [(j, landing) for j, landing in sorted(landings.items())
             if j > i and landing[1] in near and landing[0] not in (att.id, dfd.id)]
    kill_in_reach = _p_kill(base) > 1e-12
    for j, (uid, land, mp_left) in moves:
        move = st.commands[j]
        start = (move[1][0], move[2][0])
        before = _unit_at(gs_i, start)
        if before is None or before.id != uid:
            tally["A_mover_elsewhere"] += 1
            continue
        cand_gs = copy.deepcopy(gs_i)
        _apply_command(cand_gs, move)
        moved = next((u for u in cand_gs.map.units if u.id == uid), None)
        if moved is None or (moved.position.x, moved.position.y) != land or moved.current_moves != mp_left:
            tally["A_not_same_move"] += 1
            continue
        cand = fight(cand_gs, attack_action(attack))
        if cand is None:
            tally["A_inconclusive"] += 1
            continue
        if cand.probs == base.probs:
            tally["A_no_effect"] += 1
            continue
        flags = {"kill_in_reach": kill_in_reach, "mover": uid}
        isolated = _isolated(st, states, landings, range(i + 1, j), uid, start, land)
        if isolated is None:
            tally["A_inconclusive"] += 1
            continue
        if not isolated:
            flags["class_block"] = "a_not_isolated"
        elif kill_in_reach:
            flags["class_block"] = "a_kill_in_reach"
        rows.append(row("A", st, i, {"move": st.first_index + j, "mover": uid}, fight_dims(cand, base, att, dfd),
                        gains=fight_gains(cand, base), flags=flags, ctx=ctx,
                        dists={"cand": dist_data(cand, att, dfd), "base": dist_data(base, att, dfd)}))
    return rows, tally


def surround(st, states, j, ctx_fn):
    from tools.abilities import hex_neighbors
    move, attack = st.commands[j], st.commands[j + 1]
    gs_move, after_move = states[j], states[j + 1]
    start = (move[1][0], move[2][0])
    mover = _unit_at(gs_move, start)
    target = (attack[3], attack[4])
    att = _unit_at(gs_move, (attack[1], attack[2]))
    moved = next((u for u in after_move.map.units if mover is not None and u.id == mover.id), None)
    tally = collections.Counter()
    if (mover is None or att is None or moved is None or att.id == mover.id
            or (moved.position.x, moved.position.y) not in set(hex_neighbors(*target))):
        return [], tally
    before = fight(gs_move, attack_action(attack))
    after = fight(after_move, attack_action(attack))
    if before is None or after is None:
        tally["K_inconclusive"] += 1
        return [], tally
    if before.probs != after.probs:
        tally["K_enabling"] += 1
        return [], tally
    p_kill = _p_kill(before)
    if p_kill <= 1e-9:
        tally["K_no_kill"] += 1
        return [], tally
    banked = int(mover.current_moves) - int(moved.current_moves)
    dims = visibility_dims(gs_move, after_move, st.side, {mover.id})
    return [row("K", st, j + 1, {"mover": mover.id, "p_kill": p_kill}, dims, tier="O",
                gains={"banked_mp": p_kill * banked, "p_kill": p_kill}, ctx=ctx_fn())], tally


def order(st, states, i, first, second):
    tally = collections.Counter()
    gs_i = states[i]
    target = (first[3], first[4])
    if (second[3], second[4]) != target or (first[1], first[2]) == (second[1], second[2]):
        return [], tally
    a1, a2 = _unit_at(gs_i, (first[1], first[2])), _unit_at(gs_i, (second[1], second[2]))
    t = _unit_at(gs_i, target)
    if a1 is None or a2 is None or t is None:
        return [], tally
    base = two_fight_joint(gs_i, first, second)
    cand = two_fight_joint(gs_i, second, first)
    if base is None or cand is None:
        tally["Q_inconclusive"] += 1
        return [], tally
    ids = {a1.id: a1.side, a2.id: a2.side, t.id: t.side}
    dims = _joint_dims(cand, base, ids, st.side)
    roles = {a1.id: "a1", a2.id: "a2", t.id: "t"}
    for d in dims:
        kind, uid = d.name.split(":", 1)
        d.name = f"{kind}:{roles[uid]}"

    def p_of(joint, test):
        return sum(p for o, p in joint if test(o))
    gains = {"d_p_kill": p_of(cand, lambda o: not o[t.id][0]) - p_of(base, lambda o: not o[t.id][0])}
    for u, r in ((a1.id, "a1"), (a2.id, "a2")):
        gains[f"d_p_levelup_{r}"] = (p_of(cand, lambda o, u=u: o[u][5] == LEVELLED)
                                    - p_of(base, lambda o, u=u: o[u][5] == LEVELLED))
        gains[f"d_hp_{r}"] = (sum(p * max(0, o[u][1]) for o, p in cand) - sum(p * max(0, o[u][1]) for o, p in base))
    gains["d_target_hp"] = (sum(p * max(0, o[t.id][1]) for o, p in cand) - sum(p * max(0, o[t.id][1]) for o, p in base))
    ctx = attack_ctx(st, states, i, a1, t, {a1.id, a2.id}, i + 1)
    ctx["attacker2"] = unit_ctx(a2, gs_i)
    order_ids = [a1.id, a2.id, t.id]
    return [row("Q", st, i, {"first": a2.id, "second": a1.id, "target": t.id}, dims, gains=gains, ctx=ctx,
                dists={"cand": joint_data(cand, order_ids), "base": joint_data(base, order_ids)})], tally


def order_with_move(st, states, landings, i):
    """Q2: attack a1 (i), move a2 (i + 1), attack a2 (i + 2) on one target,
    played as move a2, attack a2, attack a1. The played turn moves a2 only
    where a1 did not kill; the rewrite always moves it, so a2 keeps its
    movement on fewer branches (`move_kept:a2`, an option dim)."""
    tally = collections.Counter()
    first, move, second = st.commands[i], st.commands[i + 1], st.commands[i + 2]
    target = (first[3], first[4])
    if (second[3], second[4]) != target or (i + 1) not in landings:
        return [], tally
    gs_i = states[i]
    a1 = _unit_at(gs_i, (first[1], first[2]))
    t = _unit_at(gs_i, target)
    uid, land, mp_left = landings[i + 1]
    a2_now = _unit_at(states[i + 2], (second[1], second[2]))
    if a1 is None or t is None or a2_now is None or a2_now.id != uid or land != (second[1], second[2]):
        return [], tally
    a2_start = _unit_at(gs_i, (move[1][0], move[2][0]))
    if a2_start is None or a2_start.id != uid or uid == a1.id:
        return [], tally
    moved = copy.deepcopy(gs_i)
    _apply_command(moved, move)
    a2 = next((u for u in moved.map.units if u.id == uid), None)
    if a2 is None or (a2.position.x, a2.position.y) != land or a2.current_moves != mp_left:
        tally["Q2_move_differs"] += 1
        return [], tally
    d_played = fight(gs_i, attack_action(first))
    d_moved = fight(moved, attack_action(first))
    if d_played is None or d_moved is None:
        tally["Q2_inconclusive"] += 1
        return [], tally
    if d_played.probs != d_moved.probs:
        tally["Q2_move_enables_first"] += 1
        return [], tally
    base = two_fight_joint(moved, first, second)
    cand = two_fight_joint(moved, second, first)
    if base is None or cand is None:
        tally["Q2_inconclusive"] += 1
        return [], tally
    ids = {a1.id: a1.side, a2.id: a2.side, t.id: t.side}
    dims = _joint_dims(cand, base, ids, st.side)
    roles = {a1.id: "a1", a2.id: "a2", t.id: "t"}
    for d in dims:
        kind, u = d.name.split(":", 1)
        d.name = f"{kind}:{roles[u]}"
    p_first_kills = _p_kill(d_played)
    dims.append(Dim("move_kept:a2", "option", LT if p_first_kills > 1e-12 else EQ))

    def p_of(joint, test):
        return sum(p for o, p in joint if test(o))
    gains = {"d_p_kill": p_of(cand, lambda o: not o[t.id][0]) - p_of(base, lambda o: not o[t.id][0]),
             "p_first_kills": p_first_kills}
    for u, r in ((a1.id, "a1"), (a2.id, "a2")):
        gains[f"d_p_levelup_{r}"] = (p_of(cand, lambda o, u=u: o[u][5] == LEVELLED)
                                    - p_of(base, lambda o, u=u: o[u][5] == LEVELLED))
        gains[f"d_hp_{r}"] = (sum(p * max(0, o[u][1]) for o, p in cand) - sum(p * max(0, o[u][1]) for o, p in base))
    gains["d_target_hp"] = (sum(p * max(0, o[t.id][1]) for o, p in cand) - sum(p * max(0, o[t.id][1]) for o, p in base))
    ctx = attack_ctx(st, states, i, a1, t, {a1.id, a2.id}, i + 2)
    ctx["attacker2"] = unit_ctx(a2_start, gs_i)
    order_ids = [a1.id, a2.id, t.id]
    return [row("Q2", st, i, {"first": a2.id, "second": a1.id, "target": t.id}, dims, gains=gains, ctx=ctx,
                dists={"cand": joint_data(cand, order_ids), "base": joint_data(base, order_ids)})], tally


def pair_structure(st, states):
    """Pairs of attacks on one target within the side-turn, by how they
    sit in the command list."""
    tally = collections.Counter()
    attacks = [k for k, c in enumerate(st.commands) if c[0] == "attack"]
    tid = {}
    for k in attacks:
        c = st.commands[k]
        t = _unit_at(states[k], (c[3], c[4]))
        tid[k] = t.id if t is not None else None
    for x, k in enumerate(attacks):
        nxt = next((m for m in attacks[x + 1:] if tid[m] is not None and tid[m] == tid[k]), None)
        if nxt is None:
            continue
        between = st.commands[k + 1:nxt]
        if not between:
            tally["pair_consecutive"] += 1
        elif all(c[0] == "move" for c in between) and len(between) == 1:
            tally["pair_one_move_between"] += 1
        else:
            tally["pair_other"] += 1
    return tally


def side_turn(st):
    rows, tally = [], collections.Counter()
    gs = copy.deepcopy(st.pre_state)
    states = []
    for cmd in st.commands:
        states.append(copy.deepcopy(gs))
        _apply_command(gs, cmd)
    states.append(gs)
    landings = {}
    for j, cmd in enumerate(st.commands):
        if cmd[0] == "move":
            before = _unit_at(states[j], (cmd[1][0], cmd[2][0]))
            moved = before and next((u for u in states[j + 1].map.units if u.id == before.id), None)
            if moved is not None:
                landings[j] = (moved.id, (moved.position.x, moved.position.y), moved.current_moves)
    tally.update(pair_structure(st, states))
    for i, cmd in enumerate(st.commands):
        if cmd[0] in ("move", "attack", "recruit", "end_turn"):
            tally["decisions"] += 1
        if cmd[0] != "attack":
            continue
        tally["attacks"] += 1
        gs_i = states[i]
        att, dfd = _unit_at(gs_i, (cmd[1], cmd[2])), _unit_at(gs_i, (cmd[3], cmd[4]))
        if att is None or dfd is None:
            tally["attack_units_missing"] += 1
            continue
        base = fight(gs_i, attack_action(cmd))
        if base is None:
            tally["inconclusive_played"] += 1
            continue
        cache = {}

        def ctx_fn(i=i, att=att, dfd=dfd):
            if "c" not in cache:
                cache["c"] = attack_ctx(st, states, i, att, dfd, {att.id}, i)
            return cache["c"]
        ctx = ctx_fn()
        ctx["base_p_kill"] = _p_kill(base)
        for part in (weapons(st, states, i, cmd, base, att, dfd, ctx),
                     setups(st, states, i, base, landings, att, dfd, ctx)):
            rows += part[0]
            tally.update(part[1])
        if i >= 1 and st.commands[i - 1][0] == "move":
            for part in (hexes(st, states, landings, i - 1, ctx_fn), surround(st, states, i - 1, ctx_fn)):
                rows += part[0]
                tally.update(part[1])
        if i + 1 < len(st.commands) and st.commands[i + 1][0] == "attack":
            part = order(st, states, i, cmd, st.commands[i + 1])
            rows += part[0]
            tally.update(part[1])
        if (i + 2 < len(st.commands) and st.commands[i + 1][0] == "move"
                and st.commands[i + 2][0] == "attack"):
            part = order_with_move(st, states, landings, i)
            rows += part[0]
            tally.update(part[1])
    return rows, tally


def census_game(item):
    _source, game_id, where = item
    logging.disable(logging.WARNING)
    t0 = time.time()
    rows, tally = [], collections.Counter()
    try:
        start, commands = _load(where)
        for st in side_turns(game_id, start, commands):
            r, c = side_turn(st)
            tally["side_turns"] += 1
            tally.update(c)
            rows += r
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"game": game_id, "error": repr(e), "trace": traceback.format_exc()[-2000:]}
    return {"game": game_id, "tally": dict(tally), "rows": rows, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    items = corpus_games(CORPUS, args.games, args.offset)
    out = args.out if args.out.is_absolute() else Path(__file__).parent / args.out
    t0 = time.time()
    with Pool(min(3, args.jobs)) as pool, open(out, "wb") as f:
        for n, r in enumerate(pool.imap_unordered(census_game, items, chunksize=1), 1):
            f.write(gzip.compress((json.dumps(r, separators=(",", ":")) + "\n").encode()))
            f.flush()
            print(f"{n}/{len(items)} {r['game'][:60]} {r.get('secs')} s rows {len(r.get('rows', []))} "
                  f"{'ERROR ' + r['error'] if 'error' in r else ''}", flush=True)
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
