"""Candidate relaxations over the census rows: for each, the rows it admits
that none of the 36 existing combinations admits (dead-unit statuses
corrected everywhere), per 1,000 decisions, with their gains.

    python analyze.py > relaxations.txt
"""
from __future__ import annotations

import collections
import statistics
import sys

sys.path.insert(0, ".")
import load  # noqa: E402
import vectors  # noqa: E402
from tools.combat_dominance import (COMBOS, EQ, GT, INCOMP, LT, Combo, Dim, admits,  # noqa: E402
                                    compare_marginals)

LEVELLED = 1_000_000
LOOSE_NO_R1 = Combo(False, 0.15, "literal", True)


def kill_xp(level):
    return 8 * level if level else 4


# ---------------------------------------------------------------------
# Reading quantities out of the stored distributions
# ---------------------------------------------------------------------
ROLE_CTX = {"a1": "attacker", "a2": "attacker2", "t": "target"}
OFF = {"a1": 0, "a2": 6, "t": 12}


def fight_value(r, x, what):
    """One quantity of one outcome row of a W/H/A fight."""
    att, tgt = r["ctx"]["attacker"], r["ctx"]["target"]
    if what == "own_hp":
        return x[0] if x[0] > 0 else -1
    if what == "enemy_hp":
        return x[1] if x[1] > 0 else -1
    if what == "own_xp":
        if x[0] <= 0:
            return -1
        xp = att["exp"] + (kill_xp(tgt["level"]) if x[1] <= 0 else tgt["level"])
        return LEVELLED if (x[8] or xp >= att["max_exp"]) else xp
    if what == "enemy_xp":
        if x[1] <= 0:
            return -1
        xp = tgt["exp"] + (kill_xp(att["level"]) if x[0] <= 0 else att["level"])
        return LEVELLED if (x[9] or xp >= tgt["max_exp"]) else xp
    if what == "enemy_poisoned":
        return x[5]
    if what == "kill":
        return int(x[1] <= 0)
    raise KeyError(what)


def joint_value(x, role, what):
    o = OFF[role]
    return {"alive": x[o], "hp": x[o + 1] if x[o] else -1, "slowed": x[o + 2], "poisoned": x[o + 3],
            "xp": x[o + 4], "attacked": x[o + 5]}[what]


def marg(rows, fn):
    m = collections.defaultdict(float)
    for x in rows:
        m[fn(x)] += x[-1]
    return dict(m)


def mean(m):
    return sum(v * p for v, p in m.items())


def is_joint(r):
    return r["cls"] in ("Q", "Q2")


def unit_of(r, dim_name):
    """The ctx record of the unit a dim is about."""
    if is_joint(r):
        return r["ctx"][ROLE_CTX[dim_name.split(":")[1]]]
    return r["ctx"]["target" if dim_name.startswith("enemy_") else "attacker"]


def opponent_level(r, dim_name):
    if is_joint(r):
        role = dim_name.split(":")[1]
        return r["ctx"]["target"]["level"] if role != "t" else r["ctx"]["attacker"]["level"]
    return r["ctx"]["attacker" if dim_name.startswith("enemy_") else "target"]["level"]


def hp_fn(r, name):
    if is_joint(r):
        role = name.split(":")[1]
        return lambda x: joint_value(x, role, "hp")
    return lambda x: fight_value(r, x, name)


def is_xp(d):
    return d.kind == "xp"


def is_hp(d):
    return d.kind == "hp"


# ---------------------------------------------------------------------
# Healing of the target at the start of its side's turn
# ---------------------------------------------------------------------
def healed(t, hp, poisoned):
    """The target's hp after its next healing (docs/wesnoth_rules.md
    "Healing"): the best of village/oasis, regenerate and adjacent healers,
    +2 rest only for a healthy unit (it was attacked, so it is not
    resting); poison cured instead of healed by a village, regenerate or
    an adjacent curer, held by an adjacent healer, else -8 floor 1."""
    if hp <= 0:
        return -1
    regen = "regenerate" in t["abilities"]
    rest = 2 if "healthy" in t["traits"] else 0
    if poisoned:
        if regen or t["terrain_heal"] > 0 or t["curer"] or t["healer"] > 0:
            return min(t["max_hp"], hp + rest)
        return max(1, hp - 8 + rest)
    heal = max(8 if regen else 0, t["terrain_heal"], t["healer"]) + rest
    return min(t["max_hp"], hp + heal)


def target_heals(t):
    return max(8 if "regenerate" in t["abilities"] else 0, t["terrain_heal"], t["healer"])


# ---------------------------------------------------------------------
# Transformations of a vector
# ---------------------------------------------------------------------
def drop(dims, test):
    return [d for d in dims if not test(d)]


def replace(dims, test, new):
    out = [d for d in dims if not test(d)]
    return out + ([new] if new is not None else [])


def xp_distance(r, d):
    u = unit_of(r, d.name)
    return u["max_exp"] - u["exp"]


def t_hopeless(horizon):
    """Drop the xp dim of a unit that stays more than `horizon` short of
    its threshold even if it takes the kill in this window."""
    def f(r, dims):
        return drop(dims, lambda d: is_xp(d)
                    and xp_distance(r, d) - kill_xp(opponent_level(r, d.name)) > horizon)
    return f


def own_xp_traded(dims):
    xs = [d for d in dims if d.kind == "xp" and d.name in ("xp:a1", "xp:a2")]
    return len(xs) == 2 and {xs[0].sym, xs[1].sym} & {GT} and {xs[0].sym, xs[1].sym} & {LT, INCOMP}


def t_prefer_xp(rule):
    """Q/Q2: keep only the preferred own unit's xp dim when xp is traded
    between the two attackers. rule 'intelligent': exactly one of them is
    intelligent; 'closest': the one nearer its threshold (absolute xp)."""
    def f(r, dims):
        if not is_joint(r) or not own_xp_traded(dims):
            return None
        a1, a2 = r["ctx"]["attacker"], r["ctx"]["attacker2"]
        if rule == "intelligent":
            i1, i2 = "intelligent" in a1["traits"], "intelligent" in a2["traits"]
            if i1 == i2:
                return None
            keep = "xp:a1" if i1 else "xp:a2"
        else:
            d1, d2 = a1["max_exp"] - a1["exp"], a2["max_exp"] - a2["exp"]
            if d1 == d2:
                return None
            keep = "xp:a1" if d1 < d2 else "xp:a2"
        return drop(dims, lambda d: d.name in ("xp:a1", "xp:a2") and d.name != keep)
    return f


def pooled(r, dims, kind_names, value, name, kind, more=True):
    if not is_joint(r):
        return dims
    sym, eps = compare_marginals(marg(r["dists"]["cand"], value), marg(r["dists"]["base"], value), more)
    return replace(dims, lambda d: d.name in kind_names, Dim(name, kind, sym, eps))


def t_pool_hp(r, dims):
    return pooled(r, dims, ("hp:a1", "hp:a2"),
                  lambda x: max(0, joint_value(x, "a1", "hp")) + max(0, joint_value(x, "a2", "hp")),
                  "hp:own_sum", "hp")


def t_pool_attack(r, dims):
    return pooled(r, dims, ("attack_left:a1", "attack_left:a2"),
                  lambda x: sum(int(joint_value(x, u, "alive") == 1 and not joint_value(x, u, "attacked"))
                                for u in ("a1", "a2")), "attack_left:count", "option")


def t_pool_xp(r, dims):
    def v(x):
        tot = 0
        for u in ("a1", "a2"):
            xp = joint_value(x, u, "xp")
            tot += r["ctx"][ROLE_CTX[u]]["max_exp"] if xp == LEVELLED else max(0, xp)
        return tot
    return pooled(r, dims, ("xp:a1", "xp:a2"), v, "xp:own_sum", "xp")


def t_heal(gate):
    """The target's hp compared after its next healing."""
    def f(r, dims):
        t = r["ctx"]["target"]
        if gate == "no_followup" and r["ctx"]["followups_potential"] > 0:
            return None
        if target_heals(t) == 0 and not any("poisoned" in s for s in t["statuses"]):
            # still applies: poison dealt this fight is handled below
            pass
        if is_joint(r):
            fn = (lambda x: healed(t, joint_value(x, "t", "hp"), joint_value(x, "t", "poisoned")))
            name = "hp:t"
        else:
            fn = (lambda x: healed(t, x[1] if x[1] > 0 else -1, x[5]))
            name = "enemy_hp"
        sym, eps = compare_marginals(marg(r["dists"]["cand"], fn), marg(r["dists"]["base"], fn), False)
        return replace(dims, lambda d: d.name == name, Dim(name, "hp", sym, eps))
    return f


def t_mean_hp(r, dims):
    out = []
    for d in dims:
        if d.kind == "hp" and d.sym == INCOMP:
            more = not (d.name == "enemy_hp" or d.name == "hp:t")
            fn = hp_fn(r, d.name)
            diff = mean(marg(r["dists"]["cand"], fn)) - mean(marg(r["dists"]["base"], fn))
            diff = diff if more else -diff
            sym = GT if diff > 1e-9 else (LT if diff < -1e-9 else EQ)
            out.append(Dim(d.name, "hp", sym, 0.0 if sym != LT else 1.0))
        else:
            out.append(d)
    return out


def t_r2_gain(level):
    def f(r, dims):
        return [Dim(d.name, d.kind, GT, 0.0) if (d.kind == "hp" and d.sym == INCOMP and d.eps <= level + 1e-9)
                else d for d in dims]
    return f


def t_eps(level):
    """R2 at a wider level: an hp dim within `level` passes (as R2 does)."""
    def f(r, dims):
        return [Dim(d.name, d.kind, EQ, 0.0) if (d.kind == "hp" and d.sym == INCOMP and d.eps <= level + 1e-9)
                else d for d in dims]
    return f


# ---------------------------------------------------------------------
# Override rules (admission by a condition, not by a transformed vector)
# ---------------------------------------------------------------------
def kill_dim(dims):
    return next(d for d in dims if d.name in ("enemy_alive", "alive:t"))


def own_death_ok(dims, leaders_only=None):
    return all(d.sym in (GT, EQ) for d in dims if d.name in ("own_alive", "alive:a1", "alive:a2"))


def own_leader_death_ok(r, dims):
    for d in dims:
        if d.name in ("own_alive", "alive:a1", "alive:a2"):
            u = unit_of(r, d.name) if d.name != "own_alive" else r["ctx"]["attacker"]
            if u["leader"] and d.sym not in (GT, EQ):
                return False
    return True


def o_kill_lex(gate):
    def f(r, dims):
        if r["tier"] != "D" or kill_dim(dims).sym != GT or not own_death_ok(dims):
            return False
        c = r["ctx"]
        t = c["target"]
        if gate == "all":
            return True
        if gate == "no_followup":
            return c["followups_potential"] == 0
        if gate == "target_heals8":
            return target_heals(t) >= 8
        if gate == "opens":
            return c["enemies_next_to_target"] >= 1 or t["terrain_heal"] > 0
        if gate == "leader":
            return t["leader"]
        raise KeyError(gate)
    return f


def o_leader(strict):
    def f(r, dims):
        if r["tier"] != "D" or not r["ctx"]["target"]["leader"] or kill_dim(dims).sym != GT:
            return False
        return own_death_ok(dims) if strict else own_leader_death_ok(r, dims)
    return f


# ---------------------------------------------------------------------
# Gains
# ---------------------------------------------------------------------
def gains_of(r):
    g = r["gains"]
    out = {"d_p_kill": g.get("d_p_kill", 0.0), "d_target_hp": g.get("d_target_hp", 0.0)}
    if is_joint(r):
        out["d_own_hp"] = g.get("d_hp_a1", 0.0) + g.get("d_hp_a2", 0.0)
        out["d_p_levelup"] = g.get("d_p_levelup_a1", 0.0) + g.get("d_p_levelup_a2", 0.0)
    else:
        out["d_own_hp"] = g.get("d_attacker_hp", 0.0)
        c = marg(r["dists"]["cand"], lambda x: fight_value(r, x, "own_xp") == LEVELLED).get(True, 0.0)
        b = marg(r["dists"]["base"], lambda x: fight_value(r, x, "own_xp") == LEVELLED).get(True, 0.0)
        out["d_p_levelup"] = c - b
    return out


def xp_gain_of_kept(r, dims_after):
    """For the preferred-xp rules: expected xp gained by the kept unit."""
    kept = [d.name for d in dims_after if d.name in ("xp:a1", "xp:a2")]
    if len(kept) != 1:
        return 0.0
    role = kept[0].split(":")[1]
    mx = r["ctx"][ROLE_CTX[role]]["max_exp"]

    def v(x):
        xp = joint_value(x, role, "xp")
        return mx if xp == LEVELLED else max(0, xp)
    return mean(marg(r["dists"]["cand"], v)) - mean(marg(r["dists"]["base"], v))


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def load_rows():
    games = [g for g in load.games() if "error" not in g]
    rows = []
    for g in games:
        for r in g["rows"]:
            if r["flags"].get("class_block"):
                continue
            r["game"] = g["game"]
            r["cdims"] = vectors.corrected_dims(r)
            r["any36"] = any(admits(r["cdims"], r["tier"], c) for c in COMBOS)
            r["r0"] = admits(r["cdims"], r["tier"], COMBOS[0])
            rows.append(r)
    dec = sum(g["tally"]["decisions"] for g in games)
    att = sum(g["tally"]["attacks"] for g in games)
    return rows, dec, att


def evaluate(rows, name, transform=None, override=None, combos=COMBOS, extra_stat=None):
    added = []
    for r in rows:
        if r["any36"]:
            continue
        if override is not None:
            ok = override(r, r["cdims"])
            dims_after = r["cdims"]
        else:
            dims_after = transform(r, r["cdims"])
            if dims_after is None:
                continue
            ok = any(admits(dims_after, r["tier"], c) for c in combos)
        if ok:
            added.append((r, dims_after))
    return added


def fmt(v):
    return f"{v:+.3f}"


def report(name, added, dec, extra_stat=None):
    n = len(added)
    opp = {(r["game"], r["anchor"]) for r, _ in added}
    by_cls = collections.Counter(r["cls"] for r, _ in added)
    line = (f"{name}: +{n} rewrites ({1000 * n / dec:.2f}/1k dec), +{len(opp)} attacks "
            f"({1000 * len(opp) / dec:.2f}/1k dec); by class {dict(by_cls)}")
    if n:
        gs = [gains_of(r) for r, _ in added]
        parts = []
        for k in ("d_p_kill", "d_target_hp", "d_own_hp", "d_p_levelup"):
            vals = [g[k] for g in gs]
            parts.append(f"{k} mean {fmt(statistics.fmean(vals))} max {fmt(max(vals))} min {fmt(min(vals))}")
        line += "\n    " + "; ".join(parts)
        if extra_stat:
            vals = [extra_stat(r, d) for r, d in added]
            line += f"\n    kept unit's E[xp] gain mean {fmt(statistics.fmean(vals))} max {fmt(max(vals))}"
    print(line)
    return added


def main():
    rows, dec, att = load_rows()
    print(f"rows (class conditions met) {len(rows)}, decisions {dec}, attacks {att}; "
          f"admitted by some of 36: {sum(r['any36'] for r in rows)}, R0: {sum(r['r0'] for r in rows)}")
    base36 = {(r["game"], r["anchor"]) for r in rows if r["any36"]}
    print(f"existing: attacks with an admitted rewrite under some combination {len(base36)} "
          f"({1000 * len(base36) / dec:.2f}/1k dec)\n")
    R = {}
    R["xp hopeless, horizon 0 (+ any combo)"] = evaluate(rows, "", transform=t_hopeless(0))
    R["xp hopeless, horizon 8"] = evaluate(rows, "", transform=t_hopeless(8))
    R["xp hopeless, horizon 16"] = evaluate(rows, "", transform=t_hopeless(16))
    R["prefer intelligent's xp (Q/Q2)"] = evaluate(rows, "", transform=t_prefer_xp("intelligent"))
    R["prefer closest-to-level xp (Q/Q2)"] = evaluate(rows, "", transform=t_prefer_xp("closest"))
    both = lambda rule, extra: (lambda r, d: (lambda x: None if x is None else extra(r, x))(t_prefer_xp(rule)(r, d)))  # noqa: E731
    R["prefer intelligent's xp + pooled own hp"] = evaluate(rows, "", transform=both("intelligent", t_pool_hp))
    R["prefer closest xp + pooled own hp"] = evaluate(rows, "", transform=both("closest", t_pool_hp))
    R["prefer intelligent's xp + pooled hp + pooled attacks"] = evaluate(
        rows, "", transform=both("intelligent", lambda r, d: t_pool_attack(r, t_pool_hp(r, d))))
    R["pooled own xp (sum)"] = evaluate(rows, "", transform=t_pool_xp)
    R["pooled own hp (sum)"] = evaluate(rows, "", transform=t_pool_hp)
    R["pooled attack_left (count)"] = evaluate(rows, "", transform=t_pool_attack)
    R["pooled hp + attack_left + xp"] = evaluate(rows, "", transform=lambda r, d: t_pool_xp(r, t_pool_attack(r, t_pool_hp(r, d))))
    R["Q2 move_kept dropped"] = evaluate(rows, "", transform=lambda r, d: drop(d, lambda x: x.name == "move_kept:a2"))
    R["Q2 move_kept dropped + pooled hp/attacks/xp"] = evaluate(
        rows, "", transform=lambda r, d: t_pool_xp(r, t_pool_attack(r, t_pool_hp(r, drop(d, lambda x: x.name == "move_kept:a2")))))
    R["leader kill first (own deaths not worse)"] = evaluate(rows, "", override=o_leader(True))
    R["leader kill first (own leader not worse)"] = evaluate(rows, "", override=o_leader(False))
    for gate in ("all", "no_followup", "target_heals8", "opens"):
        R[f"kill chance first, own deaths not worse, gate {gate}"] = evaluate(rows, "", override=o_kill_lex(gate))
    R["target hp after its healing, no follow-up"] = evaluate(rows, "", transform=t_heal("no_followup"))
    R["target hp after its healing, always"] = evaluate(rows, "", transform=t_heal("always"))
    R["hp almost-dominance counts as gain (eps <= 0.15)"] = evaluate(rows, "", transform=t_r2_gain(0.15))
    R["R2 at eps 0.25"] = evaluate(rows, "", transform=t_eps(0.25))
    R["R2 at eps 0.35"] = evaluate(rows, "", transform=t_eps(0.35))
    R["hp by mean (incomparable hp dims)"] = evaluate(rows, "", transform=t_mean_hp)
    for name, added in R.items():
        report(name, added, dec, extra_stat=xp_gain_of_kept if "prefer" in name else None)
    return rows, R, dec


if __name__ == "__main__":
    main()
