"""Context breakdowns for the user's examples, over the census rows
(dead-unit statuses corrected)."""
import collections
import statistics
import sys

sys.path.insert(0, ".")
import analyze as A  # noqa: E402
from tools.combat_dominance import GT, INCOMP, LT  # noqa: E402
from tools.replay_dataset import _stats_for  # noqa: E402

rows, dec, att = A.load_rows()
print(f"decisions {dec}, attacks {att}, rows {len(rows)}")


def bucket(v, edges):
    for lo, hi, lab in edges:
        if lo <= v < hi:
            return lab
    return edges[-1][2]


def p_kill(r, which):
    if A.is_joint(r):
        return sum(x[-1] for x in r["dists"][which] if not A.joint_value(x, "t", "alive"))
    return sum(x[-1] for x in r["dists"][which] if x[1] <= 0)


def key(r):
    return (r["game"], r["anchor"])


# ---------------------------------------------------------------- leader
print("\n== Target is the enemy leader ==")
lead = [r for r in rows if r["ctx"]["target"]["leader"]]
print(f"attacks on the leader with a candidate: {len({key(r) for r in lead})} "
      f"({1000 * len({key(r) for r in lead}) / dec:.2f}/1k dec); rewrites {len(lead)}")
kg = [r for r in lead if A.kill_dim(r["cdims"]).sym == GT]
print(f"rewrites raising the leader-kill chance: {len(kg)} on {len({key(r) for r in kg})} attacks; "
      f"admitted by some of 36: {sum(r['any36'] for r in kg)}")
for r in kg:
    worse = sorted({d.name for d in r["cdims"] if d.sym in (LT, INCOMP)})
    print(f"  {r['cls']} p_kill {p_kill(r, 'base'):.3f} -> {p_kill(r, 'cand'):.3f}; own hp {A.gains_of(r)['d_own_hp']:+.1f}; "
          f"attacker is leader {r['ctx']['attacker']['leader']}; admitted36 {r['any36']}; worse {worse}")
per = {key(r): r for r in lead}
bp = collections.Counter(bucket(p_kill(r, "base"), [(0, 1e-12, "0"), (1e-12, .1, "<0.1"), (.1, .3, "0.1-0.3"),
                                                    (.3, .6, "0.3-0.6"), (.6, 2, ">=0.6")]) for r in per.values())
print("played leader-kill chance per attack:", dict(bp))

# ---------------------------------------------------------------- intelligent
print("\n== Intelligent attackers in Q/Q2 ==")
qs = [r for r in rows if A.is_joint(r)]
one_int = [r for r in qs if ("intelligent" in r["ctx"]["attacker"]["traits"])
           != ("intelligent" in r["ctx"]["attacker2"]["traits"])]
traded = [r for r in one_int if A.own_xp_traded(r["cdims"])]


def int_role(r):
    return "a1" if "intelligent" in r["ctx"]["attacker"]["traits"] else "a2"


to_int = [r for r in traded if next(d for d in r["cdims"] if d.name == f"xp:{int_role(r)}").sym == GT]
print(f"Q/Q2 rewrites {len(qs)}; exactly one attacker intelligent {len(one_int)}; own xp traded {len(traded)}; "
      f"xp moved toward the intelligent one {len(to_int)}")
print("  their kill-chance symbol:", dict(collections.Counter(A.kill_dim(r["cdims"]).sym for r in to_int)))
print("  admitted by some of 36:", sum(r["any36"] for r in to_int))
xpg = [A.xp_gain_of_kept(r, [d for d in r["cdims"] if d.name != ("xp:a2" if int_role(r) == "a1" else "xp:a1")])
       for r in to_int]
print(f"  E[xp] gained by the intelligent unit: mean {statistics.fmean(xpg):+.2f}, max {max(xpg):+.2f}")
blk = collections.Counter()
for r in to_int:
    dims = [d for d in r["cdims"] if d.name not in ("xp:a1", "xp:a2")]
    w = sorted({A.vectors.group(d.name) for d in dims
                if d.sym in (LT, INCOMP) and not (d.kind == "hp" and d.eps <= 0.15)})
    blk[" ".join(w) or "(nothing)"] += 1
print("  other blocking groups once xp is settled (hp eps <= 0.15 passes):", dict(blk.most_common(8)))

# ---------------------------------------------------------------- xp distance
print("\n== XP distance to threshold vs XP at stake (Q/Q2 with own xp traded) ==")
tr = [r for r in qs if A.own_xp_traded(r["cdims"])]


def dist_bucket(u, stake):
    d = u["max_exp"] - u["exp"]
    if d <= stake:
        return "1 can level this window"
    if d <= 2 * stake:
        return "2 within one more kill"
    if d <= 4 * stake:
        return "3 within three more kills"
    return "4 further"


c = collections.Counter()
for r in tr:
    stake = A.kill_xp(r["ctx"]["target"]["level"])
    for role in ("a1", "a2"):
        d = next(x for x in r["cdims"] if x.name == f"xp:{role}")
        c[(dist_bucket(r["ctx"][A.ROLE_CTX[role]], stake), "gains xp" if d.sym == GT else "loses xp")] += 1
for k, v in sorted(c.items()):
    print(" ", k, v)
print(f"  ({len(tr)} rewrites)")

# ---------------------------------------------------------------- healing, follow-ups
print("\n== Target healing and follow-up attackers ==")


def heal_ctx(t):
    if "regenerate" in t["abilities"]:
        return "regenerates"
    if t["terrain_heal"] > 0:
        return "village/oasis"
    if t["healer"] > 0:
        return f"healer+{t['healer']}"
    return "none"


per_attack = {}
for r in rows:
    per_attack.setdefault(key(r), r)
print(f"attacks with any candidate: {len(per_attack)}")
print("  target heals next turn:", dict(collections.Counter(heal_ctx(r["ctx"]["target"]) for r in per_attack.values())))
print("  potential follow-up attackers (3 = 3+):",
      dict(sorted(collections.Counter(min(r["ctx"]["followups_potential"], 3) for r in per_attack.values()).items())))
print("  follow-up attacks played on it:",
      dict(sorted(collections.Counter(min(r["ctx"]["followups_played"], 3) for r in per_attack.values()).items())))

# ---------------------------------------------------------------- kill first
print("\n== Kill chance first extras (kill better, no own death worse, rejected by all 36) ==")
kl = [r for r in rows if not r["any36"] and A.o_kill_lex("all")(r, r["cdims"])]
print(f"{len(kl)} rewrites on {len({key(r) for r in kl})} attacks; by class {dict(collections.Counter(r['cls'] for r in kl))}")
lp = collections.Counter()
for r in kl:
    b, cnd = p_kill(r, "base"), p_kill(r, "cand")
    lp[("played 0" if b < 1e-12 else "played >0",
        bucket(cnd, [(0, .1, "cand <0.1"), (.1, .3, "cand 0.1-0.3"), (.3, .6, "cand 0.3-0.6"), (.6, 2, "cand >=0.6")]))] += 1
for k, v in sorted(lp.items()):
    print(" ", k, v)
print("  target heals:", dict(collections.Counter(heal_ctx(r["ctx"]["target"]) for r in kl)))
print("  potential follow-ups:",
      dict(sorted(collections.Counter(min(r["ctx"]["followups_potential"], 3) for r in kl).items())))
print("  enemies next to the target:",
      dict(sorted(collections.Counter(min(r["ctx"]["enemies_next_to_target"], 3) for r in kl).items())))
blk = collections.Counter()
for r in kl:
    w = sorted({A.vectors.group(d.name) for d in r["cdims"]
                if d.sym in (LT, INCOMP) and not (d.kind == "hp" and d.eps <= 0.15) and d.kind != "xp"})
    blk[" ".join(w)] += 1
print("  what blocks them (xp and eps<=0.15 hp ignored):", dict(blk.most_common(6)))
g = [A.gains_of(r) for r in kl]
for k in ("d_p_kill", "d_target_hp", "d_own_hp"):
    vals = sorted(x[k] for x in g)
    print(f"  {k}: mean {statistics.fmean(vals):+.3f} median {statistics.median(vals):+.3f} "
          f"max {vals[-1]:+.3f} min {vals[0]:+.3f}")
ratio = [x["d_own_hp"] / x["d_p_kill"] for x in g if x["d_p_kill"] > 1e-6]
print(f"  own hp given per unit of kill chance: median {statistics.median(ratio):+.1f}")

# ---------------------------------------------------------------- material reference
print("\n== Reference: expected material swing (cost x hp fraction, dead = 0), rejected by all 36 ==")
_COST = {}


def cost(name):
    if name not in _COST:
        try:
            _COST[name] = float(_stats_for(name).get("cost", 14))
        except Exception:  # noqa: BLE001
            _COST[name] = 14.0
    return _COST[name]


def value(r, which):
    if A.is_joint(r):
        tot = 0.0
        for role, sign in (("a1", 1), ("a2", 1), ("t", -1)):
            u = r["ctx"][A.ROLE_CTX[role]]
            cu = cost(u["name"])
            tot += sign * sum(x[-1] * cu * max(0, A.joint_value(x, role, "hp")) / u["max_hp"] for x in r["dists"][which])
        return tot
    a, t = r["ctx"]["attacker"], r["ctx"]["target"]
    ca, ct = cost(a["name"]), cost(t["name"])
    return sum(x[-1] * (ca * max(0, x[0]) / a["max_hp"] - ct * max(0, x[1]) / t["max_hp"]) for x in r["dists"][which])


sw = [(value(r, "cand") - value(r, "base"), r) for r in rows if not r["any36"] and r["tier"] == "D"]
pos = [(s, r) for s, r in sw if s > 0]
pa = {key(r) for _s, r in pos}
print(f"rewrites with a positive swing: {len(pos)} on {len(pa)} attacks ({1000 * len(pa) / dec:.1f}/1k dec); "
      f"mean {statistics.fmean(s for s, _ in pos):.2f} gold, median {statistics.median(s for s, _ in pos):.2f}, "
      f"above 1 gold {sum(s > 1 for s, _ in pos)}, above 3 gold {sum(s > 3 for s, _ in pos)}")
print("  by class:", dict(collections.Counter(r["cls"] for _s, r in pos)))
