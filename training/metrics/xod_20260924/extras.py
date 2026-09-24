"""Remaining counts: rewrites that need R4 alone, R1 against the hopeless-xp
rule, and worked examples of the Q2 and hp-trade families."""
import collections
import sys

sys.path.insert(0, ".")
import analyze as A  # noqa: E402
from tools.combat_dominance import COMBOS, Combo, admits  # noqa: E402

rows, dec, att = A.load_rows()


def key(r):
    return (r["game"], r["anchor"])


def adm(r, c, dims=None):
    return admits(dims if dims is not None else r["cdims"], r["tier"], c)


# Visibility: admitted with R4 added, not without, per base combination
for base in (Combo(), Combo(r3="guarded"), Combo(r3="literal"), Combo(True, 0.15, "literal", False)):
    with_r4 = Combo(base.r1, base.r2, base.r3, True)
    need = [r for r in rows if adm(r, with_r4) and not adm(r, base)]
    print(f"needs R4 on top of {base.name()}: {len(need)} rewrites, {len({key(r) for r in need})} attacks, "
          f"classes {dict(collections.Counter(r['cls'] for r in need))}")

# R1 against the hopeless-xp rule: R1 admissions that a unit near its threshold would block
for horizon in (8, 16):
    tr = A.t_hopeless(horizon)
    r1_only, blocked = 0, collections.Counter()
    for r in rows:
        with_r1 = [c for c in COMBOS if c.r1 and adm(r, c)]
        if not with_r1 or any(adm(r, c) for c in COMBOS if not c.r1):
            continue
        r1_only += 1
        dims = tr(r, r["cdims"])
        if not any(admits(dims, r["tier"], Combo(False, c.r2, c.r3, c.r4)) for c in with_r1):
            blocked[r["cls"]] += 1
    print(f"horizon {horizon}: rewrites admitted only with R1 {r1_only}; of them the hopeless rule in R1's place "
          f"rejects {sum(blocked.values())} {dict(blocked)}")

# Examples
q2 = [r for r in rows if r["cls"] == "Q2" and r["r0"]]
for r in q2[:3]:
    print("Q2 at R0:", r["game"][:45], r["turn"], r["detail"], {k: round(v, 3) for k, v in r["gains"].items()},
          [(d.name, d.sym) for d in r["cdims"] if d.sym != "="])
hp_tr = [r for r in rows if r["cls"] in ("Q", "Q2") and not r["any36"]
         and {d.name for d in r["cdims"] if d.sym not in ("=", ">")} <= {"hp:a1", "hp:a2", "xp:a1", "xp:a2"}
         and any(d.name.startswith("hp:a") and d.sym != "=" for d in r["cdims"])]
print(f"\nQ/Q2 blocked only by own hp/xp traded between the attackers: {len(hp_tr)}")
for r in hp_tr[:4]:
    a1, a2, t = r["ctx"]["attacker"], r["ctx"]["attacker2"], r["ctx"]["target"]
    print(f"  {r['cls']} {a1['name']}({a1['hp']}) then {a2['name']}({a2['hp']}) on {t['name']}({t['hp']}); "
          f"swap: {{'d_hp_a1': {r['gains']['d_hp_a1']:+.2f}, 'd_hp_a2': {r['gains']['d_hp_a2']:+.2f}, "
          f"'d_p_kill': {r['gains']['d_p_kill']:+.3f}}}")
