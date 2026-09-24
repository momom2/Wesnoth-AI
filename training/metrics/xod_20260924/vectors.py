"""Candidate vectors with the dead-unit status reading corrected, and the
grouping of dimensions used by the pattern tables.

The tools read a dead unit's slowed / poisoned / petrified as 0
(combat_outcomes._canonical zeroes them; dominance_rewrites._after_fight
writes 0). For an enemy, where more status is better, that ranks a kill
below a slowed survivor. The corrected reading ranks a dead unit as the
extreme status value: for an enemy dead > alive with status > alive
without; for an own unit alive without > alive with > dead.
"""
from __future__ import annotations

from load import Dim, dims_of
from tools.combat_dominance import compare_marginals

FIGHT_STATUS = {  # name -> (status index, hp index, more is better) in a fight row
    "enemy_slowed": (3, 1, True), "enemy_poisoned": (5, 1, True), "enemy_petrified": (7, 1, True),
    "own_slowed": (2, 0, False), "own_poisoned": (4, 0, False), "own_petrified": (6, 0, False),
}
ROLE_OFFSET = {"a1": 0, "a2": 6, "t": 12}     # joint row: per unit (alive, hp, sl, po, xp, attacked)


def _marg(rows, fn):
    m = {}
    for r in rows:
        v = fn(r)
        m[v] = m.get(v, 0.0) + r[-1]
    return m


def corrected_dims(r):
    """The row's dims with every status dim recomputed, dead read as the
    extreme status value."""
    dims = dims_of(r)
    cand, base = r["dists"].get("cand"), r["dists"].get("base")
    if not cand or not base:
        return dims
    out = []
    for d in dims:
        new = None
        if r["cls"] in ("W", "H", "A") and d.name in FIGHT_STATUS:
            si, hi, more = FIGHT_STATUS[d.name]
            fn = (lambda x, si=si, hi=hi: 2 if x[hi] <= 0 else x[si])
            sym, eps = compare_marginals(_marg(cand, fn), _marg(base, fn), more)
            new = Dim(d.name, d.kind, sym, eps)
        elif r["cls"] in ("Q", "Q2") and d.name.split(":")[0] in ("slowed", "poisoned"):
            kind, role = d.name.split(":")
            off = ROLE_OFFSET[role]
            idx = off + (2 if kind == "slowed" else 3)
            fn = (lambda x, off=off, idx=idx: 2 if not x[off] else x[idx])
            more = role == "t"
            sym, eps = compare_marginals(_marg(cand, fn), _marg(base, fn), more)
            new = Dim(d.name, d.kind, sym, eps)
        out.append(new or d)
    return out


def is_dead_status(d):
    return d.name in FIGHT_STATUS or d.name.split(":")[0] in ("slowed", "poisoned")


def group(name):
    base, _, role = name.partition(":")
    if base.startswith("pos"):
        return "pos"
    if name in ("seen_hexes", "enemies_seen", "own_shown"):
        return "vis"
    if name == "villages":
        return "villages"
    if role:                                     # Q / Q2 per unit
        enemy = role == "t"
        return {"alive": "kill" if enemy else "death", "hp": "enemy_hp" if enemy else "own_hp",
                "slowed": "enemy_status" if enemy else "own_status",
                "poisoned": "enemy_status" if enemy else "own_status",
                "levelup": "enemy_levelup" if enemy else "own_levelup",
                "xp": "enemy_xp" if enemy else "own_xp",
                "attack_left": "attack_left", "move_kept": "move_kept"}[base]
    return {"enemy_alive": "kill", "own_alive": "death", "enemy_hp": "enemy_hp", "own_hp": "own_hp",
            "own_levelup": "own_levelup", "enemy_levelup": "enemy_levelup",
            "own_xp": "own_xp", "enemy_xp": "enemy_xp"}.get(name, "enemy_status" if name.startswith("enemy_")
                                                             else "own_status")
