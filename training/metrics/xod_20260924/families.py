"""Blocking families of the candidates R0 rejects that have a gain,
dead-unit statuses corrected, with the existing combinations' coverage."""
import collections
import sys
sys.path.insert(0, ".")
import analyze
import vectors
from tools.combat_dominance import COMBOS, GT, INCOMP, dim_passes

rows, dec, att = analyze.load_rows()
R0 = COMBOS[0]
HURT = {"own_hp", "death", "own_status"}
DMG = {"enemy_hp", "kill", "enemy_status"}


def family(r, worse, better, traded):
    w = set(worse)
    if r["cls"] in ("Q", "Q2"):
        if "kill" in w:
            return "Q: kill chance worse"
        if "death" in w:
            return "Q: an attacker's survival worse"
        if "own_levelup" in w:
            return "Q: an own level-up worse"
        if "move_kept" in w:
            return "Q2: a2's move committed on the first attacker's kill branch (+ trades)"
        if "attack_left" in w:
            return "Q: attack_left traded between the attackers (+ hp/xp trades)"
        if w == {"own_xp"}:
            return "Q: only own xp traded between the attackers"
        if w <= {"own_hp", "own_xp"}:
            return "Q: own hp traded between the attackers (+ xp)"
        if w & {"enemy_hp", "enemy_status", "enemy_xp", "enemy_levelup"}:
            return "Q: target hp/status/xp worse"
        return "Q: other " + " ".join(sorted(w))
    if w <= {"pos", "vis", "villages"}:
        if "villages" in w:
            return "H: villages (with position/visibility)"
        return "H: position and/or visibility only"
    w -= {"pos", "vis", "villages"}
    if w == {"own_xp"} or w == {"enemy_xp"} or w == {"own_xp", "enemy_xp"}:
        return "W/H/A: only xp worse"
    if w == {"enemy_hp"} and worse["enemy_hp"] == {INCOMP}:
        return "W/H/A: only target hp incomparable"
    if w == {"own_hp"} and worse["own_hp"] == {INCOMP}:
        return "W/H/A: only own hp incomparable"
    if w & {"own_levelup"}:
        return "W/H/A: own level-up chance worse"
    if (w & DMG) and not (w & HURT) and (set(better) & HURT):
        return "W/H/A: attacker safer, target damage/kill worse"
    if (w & HURT) and not (w & DMG) and (set(better) & DMG):
        if "kill" in better:
            return "W/H/A: kill chance better, attacker hurt more"
        return "W/H/A: target damage better (no kill gain), attacker hurt more"
    if (w & DMG) and (set(better) & {"enemy_status"}) and not (set(better) & HURT):
        return "W/H/A: slow/poison gained, damage/kill worse"
    if (w & {"enemy_status"}) and not (w & HURT):
        return "W/H/A: slow/poison lost"
    return "W/H/A: other " + " ".join(sorted(w))


fam = collections.defaultdict(lambda: {"rows": 0, "att": set(), "some36": 0, "some36_att": set()})
eps_only = []
guard_fail = collections.Counter()
for r in rows:
    if r["r0"]:
        continue
    dims = r["cdims"]
    gains = [d for d in dims if d.sym == GT and d.justifies]
    if r["tier"] == "D" and not gains:
        continue
    worse = collections.defaultdict(set)
    for d in dims:
        if dim_passes(d, R0) is False:
            worse[vectors.group(d.name)].add(d.sym if d.kind != "pos" else "moved")
    better = {vectors.group(d.name) for d in gains}
    f = family(r, worse, better, None)
    e = fam[f]
    e["rows"] += 1
    e["att"].add((r["game"], r["anchor"]))
    if r["any36"]:
        e["some36"] += 1
        e["some36_att"].add((r["game"], r["anchor"]))
    hp_bad = [d for d in dims if d.kind == "hp" and d.sym == INCOMP]
    if set(worse) - {"pos", "vis", "villages"} <= {"enemy_hp", "own_hp", "own_xp", "enemy_xp"} and hp_bad \
            and all(s == {INCOMP} for g, s in worse.items() if g in ("enemy_hp", "own_hp")):
        eps_only.append(max(d.eps for d in hp_bad))
    if r["cls"] == "H":
        pos = next(d for d in dims if d.kind == "pos")
        if not pos.guard:
            fl = r["flags"]
            for k in ("g_def", "g_vil", "g_sup", "g_thr"):
                if not fl.get(k, True):
                    guard_fail[k] += 1
            guard_fail["any"] += 1
print(f"decisions {dec}, attacks {att}")
print("| family | rewrites | attacks | per 1k dec (attacks) | admitted by some of the 36 (rewrites / attacks) |")
for f, e in sorted(fam.items(), key=lambda kv: -kv[1]["rows"]):
    print(f"| {f} | {e['rows']} | {len(e['att'])} | {1000 * len(e['att']) / dec:.2f} | {e['some36']} / {len(e['some36_att'])} |")
print("\nhp-incomparable blocks (hp the only binding dims besides xp/pos/vis): eps of the worst hp dim")
b = collections.Counter()
for x in eps_only:
    for lo, hi in ((0, .05), (.05, .15), (.15, .25), (.25, .35), (.35, .5), (.5, 1.01)):
        if lo < x <= hi or (lo == 0 and x == 0):
            b[f"({lo},{hi}]"] += 1
print(dict(sorted(b.items())))
print("\nH candidates failing the guard, by condition (a candidate may fail several):", dict(guard_fail))
