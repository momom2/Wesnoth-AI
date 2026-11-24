"""One-off: the readings of a measure_games.py output."""
import json
import statistics as st
import sys
from collections import Counter, defaultdict

d = json.load(open(sys.argv[1], encoding="utf-8"))
G, A, R, D = d["games"], d["attacks"], d["recruits"], d["decisions"]


def pct(a, b):
    return f"{a} of {b} ({100.0 * a / b:.1f}%)" if b else f"{a} of 0"


print("== games ==")
print(len(G), "games;", Counter(g["ended_by"] for g in G), "turns median", st.median(g["turns"] for g in G))
print("matchups:", Counter(tuple(sorted((g["faction1"], g["faction2"]))) for g in G).most_common(40))
print("errors:", d["errors"][:5])

print("\n== zone of control ==")
print(d["zoc"])

print("\n== sampling ==")
n = len(D)
print("decisions:", n, "; the sampled action is the policy's most likely:", pct(sum(x["argmax"] for x in D), n))
for t in ("move", "attack", "recruit", "end_turn"):
    xs = [x for x in D if x["type"] == t]
    if xs:
        print(f"  {t}: {len(xs)}; most likely action taken {pct(sum(x['argmax'] for x in xs), len(xs))};"
              f" sampled with probability under 5%: {pct(sum(x['p'] < 0.05 for x in xs), len(xs))},"
              f" under 1%: {pct(sum(x['p'] < 0.01 for x in xs), len(xs))}")

print("\n== attacks ==")
print("attacks:", len(A), "; ordered from a distance (the simulator picks the hex):",
      pct(sum(bool(a.get("moved_first")) for a in A), len(A)))
arg_of = {}
for x in D:
    if x["type"] == "attack":
        arg_of[(x["game"], x["turn"], x["side"], round(x["p"], 12))] = x["argmax"]


def chosen_row(a):
    return next((w for w in a["weapons"] if w["w"] == a["chosen_w"]), None)


def dominated(row, rows):
    """Another weapon deals 25% more and takes no more, or deals as much
    (within 10%) and takes 3 hit points less. Expected values, before any
    cut-off at a death."""
    for o in rows:
        if o["w"] == row["w"]:
            continue
        more = (o["dealt"] >= 1.25 * row["dealt"] and o["dealt"] > row["dealt"] + 0.5
                and o["taken"] <= row["taken"] + 1e-9)
        safer = o["dealt"] >= 0.9 * row["dealt"] and o["taken"] <= row["taken"] - 3.0
        if more or safer:
            return True
    return False


SOFT = {"poison", "slow", "berserk", "plague", "drains"}
multi = [a for a in A if len(a["weapons"]) >= 2 and a.get("decision")]
print("attacks by a unit with two weapons or more:", len(multi))
plain = [a for a in multi if not any(set(w["specials"]) & SOFT for w in a["weapons"])]
for name, group in (("all", multi), ("no weapon with poison, slow, berserk, plague or drain", plain)):
    if not group:
        continue
    dom_chosen = dom_top = 0
    mass = []
    ratio = []
    for a in group:
        rows = a["weapons"]
        c = chosen_row(a)
        wp = {int(k): v for k, v in a["decision"]["weapon_p"].items()}
        top = max(wp, key=wp.get)
        top_row = next(w for w in rows if w["w"] == top)
        dom_chosen += dominated(c, rows)
        dom_top += dominated(top_row, rows)
        mass.append(sum(wp.get(w["w"], 0.0) for w in rows if dominated(w, rows)))
        best = max(w["dealt"] for w in rows)
        ratio.append(c["dealt"] / best if best > 0 else 1.0)
    print(f"  [{name}] {len(group)} attacks")
    print("    the weapon used is beaten by another (25% more dealt and no more taken, or as much dealt and 3 less taken):",
          pct(dom_chosen, len(group)))
    print("    the policy's MOST LIKELY weapon is so beaten:", pct(dom_top, len(group)))
    print("    mean probability the policy puts on a beaten weapon:", round(st.mean(mass), 3))
    print("    used weapon's expected damage over the best weapon's: median", round(st.median(ratio), 2),
          "; under 0.5 in", pct(sum(r < 0.5 for r in ratio), len(group)))

# Does the best weapon depend on the target, where the unit had several?
several = [a for a in multi if a["others"]]
differs = 0
dom_top_several = dom_top_single = 0
single = [a for a in multi if not a["others"]]
for a in several:
    best_here = max(a["weapons"], key=lambda w: w["dealt"] - w["taken"])["w"]
    others_best = {max(o["weapons"], key=lambda w: w["dealt"] - w["taken"])["w"] for o in a["others"] if o["weapons"]}
    differs += bool(others_best - {best_here})
for group, label in ((several, "several adjacent targets"), (single, "one adjacent target")):
    k = 0
    for a in group:
        wp = {int(k2): v for k2, v in a["decision"]["weapon_p"].items()}
        top = max(wp, key=wp.get)
        k += dominated(next(w for w in a["weapons"] if w["w"] == top), a["weapons"])
    print(f"  with {label}: the most likely weapon is beaten in {pct(k, len(group))}")
print("  with several adjacent targets, the best weapon (dealt minus taken) differs between them:",
      pct(differs, len(several)))

print("\n-- pairs where the most likely weapon is beaten (6 attacks or more) --")
pairs = defaultdict(list)
for a in multi:
    pairs[(a["att"], a["dfd"])].append(a)
table = []
for (att, dfd), group in pairs.items():
    if len(group) < 6:
        continue
    k = 0
    probs = defaultdict(list)
    vals = defaultdict(list)
    for a in group:
        wp = {int(k2): v for k2, v in a["decision"]["weapon_p"].items()}
        top = max(wp, key=wp.get)
        k += dominated(next(w for w in a["weapons"] if w["w"] == top), a["weapons"])
        for w in a["weapons"]:
            probs[w["name"]].append(wp.get(w["w"], 0.0))
            vals[w["name"]].append((w["dealt"], w["taken"]))
    table.append((k / len(group), len(group), att, dfd,
                  {n: (round(st.mean(p), 2), round(st.mean(v[0] for v in vals[n]), 1),
                       round(st.mean(v[1] for v in vals[n]), 1)) for n, p in probs.items()}))
for share, n_, att, dfd, info in sorted(table, reverse=True)[:14]:
    print(f"  {att} on {dfd}: beaten in {100 * share:.0f}% of {n_}; weapon: (mean probability, dealt, taken) {info}")

print("\n-- resistance of the defender to the weapon used (percent of damage taken) --")
rows = [(a, chosen_row(a)) for a in A if chosen_row(a) and chosen_row(a)["resist_pct"] is not None]
hist = Counter()
for a, c in rows:
    r = c["resist_pct"]
    hist["<=50" if r <= 50 else "51-70" if r <= 70 else "71-99" if r < 100 else "100" if r == 100 else ">100"] += 1
print("  ", dict(hist), "of", len(rows))
hard = [(a, c) for a, c in rows if c["resist_pct"] <= 60]
print("  attacks into a resistance of 40% or more:", pct(len(hard), len(rows)))
better_w = sum(any(o["dealt"] >= 1.5 * c["dealt"] and o["w"] != c["w"] for o in a["weapons"]) for a, c in hard)
better_t = sum(any(max((w["dealt"] for w in o["weapons"]), default=0) >= 1.5 * c["dealt"] for o in a["others"])
               for a, c in hard)
print("    of those, another weapon would deal 1.5 times as much:", pct(better_w, len(hard)))
print("    of those, another adjacent target would take 1.5 times as much:", pct(better_t, len(hard)))
argm = [arg_of.get((a["game"], a["turn"], a["side"], round(a["decision"]["p_joint"], 12)))
        for a, c in hard if a.get("decision")]
print("    of those, the attack was the policy's most likely action:", pct(sum(bool(x) for x in argm), len(argm)))
print("    examples:", Counter((a["att"], c["name"], c["type"], a["dfd"], c["resist_pct"]) for a, c in hard).most_common(14))

print("\n-- trades --")
tr = [(a, chosen_row(a)) for a in A if chosen_row(a)]
bad = [(a, c) for a, c in tr if c["taken"] >= 1.5 * c["dealt"] and c["taken"] - c["dealt"] >= 3 and a["dfd_hp"] > c["dealt"]]
print("  attacks expected to take 1.5 times what they deal (3 or more apart, no likely kill):", pct(len(bad), len(tr)))
argm = [arg_of.get((a["game"], a["turn"], a["side"], round(a["decision"]["p_joint"], 12)))
        for a, c in bad if a.get("decision")]
print("    of those, the policy's most likely action:", pct(sum(bool(x) for x in argm), len(argm)))
print("  expected dealt, median:", round(st.median(c["dealt"] for a, c in tr), 1), "; expected taken, median:",
      round(st.median(c["taken"] for a, c in tr), 1))
print("  chance to hit of the attacker, mean:", round(st.mean(c["a_cth"] for a, c in tr), 1),
      "; of the defender's answer, mean:", round(st.mean(c["d_cth"] for a, c in tr if c["d_cth"] is not None), 1))

print("\n== experience ==")
kills = [a for a in A if a["killed"]]
adv = sum(a["att_advanced"] + a["dfd_advanced"] for a in A)
print("kills:", len(kills), "; advancements:", adv, f"({adv / (2 * len(G)):.2f} a game-side)")
print("level-0 units among attackers:", pct(sum(a["att_level"] == 0 for a in A), len(A)),
      "; among defenders:", pct(sum(a["dfd_level"] == 0 for a in A), len(A)))
near = [a for a in A if a["att_max_exp"] and a["att_exp"] / a["att_max_exp"] >= 0.75]
print("attacks by a unit at 75% of its experience or more:", pct(len(near), len(A)))

print("\n== recruits ==")
picks = [r for r in R if r["chosen_type"] == "recruit"]
print("decisions with a recruit legal:", len(R), "; recruits made:", len(picks))
by_f = defaultdict(Counter)
cond_low = 0
for r in picks:
    by_f[r["faction"]][r["chosen_unit"]] += 1
    tot = sum(r["p_type"].values())
    cond_low += r["p_type"][r["chosen_unit"]] / tot < 0.10
print("recruits sampled at under 10% among the recruit options:", pct(cond_low, len(picks)))
pref = defaultdict(lambda: defaultdict(list))
for r in picks:
    tot = sum(r["p_type"].values())
    for t, p in r["p_type"].items():
        pref[r["faction"]][t].append(p / tot)
for f in sorted(by_f):
    tot = sum(by_f[f].values())
    line = ", ".join(f"{t} {c} ({100.0 * c / tot:.0f}%)" for t, c in by_f[f].most_common())
    print(f"  {f} ({tot}): {line}")
print("recruits from turn 4 on, by the enemy's faction:")
late = defaultdict(Counter)
for r in picks:
    if r["turn"] >= 4:
        late[(r["faction"], r["enemy_faction"])][r["chosen_unit"]] += 1
for key in sorted(late):
    tot = sum(late[key].values())
    line = ", ".join(f"{t} {c}" for t, c in late[key].most_common(5))
    print(f"  {key[0]} against {key[1]} ({tot}): {line}")
print("attacks into a resistance of 40% or more, by the attacker's faction and the defender's type:")
res = Counter()
tot_f = Counter()
for a, c in rows:
    tot_f[a["faction"]] += 1
    if c["resist_pct"] <= 60:
        res[(a["faction"], a["dfd"])] += 1
print("  ", res.most_common(12))
print("   attacks by faction:", dict(tot_f))
first = defaultdict(lambda: defaultdict(list))
seen = set()
for r in R:
    key = (r["game"], r["side"])
    if key in seen or r["turn"] != 1:
        continue
    seen.add(key)
    tot = sum(r["p_type"].values())
    for t, p in r["p_type"].items():
        first[r["faction"]][t].append(p / tot)
print("the first recruit of the game, the policy's probabilities (mean over game-sides):")
for f in sorted(first):
    line = ", ".join(f"{t} {100.0 * st.mean(v):.0f}%" for t, v in sorted(first[f].items(), key=lambda kv: -st.mean(kv[1])))
    print(f"  {f} ({len(next(iter(first[f].values())))} sides): {line}")
