"""One-off: among the unit types a default-era game can hold (recruits and
leaders of the six factions, and what they advance to), which facts have no
input column. Also the checkpoint's structure."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.replay_dataset import _stats_for  # noqa: E402
from wesnoth_ai import encoder as enc  # noqa: E402
from wesnoth_ai.rules.scenario_pool import load_factions  # noqa: E402

factions = load_factions()
print("factions:", sorted(factions))
start = set()
recruits = set()
for f in factions.values():
    start |= set(f.recruit) | set(f.leader_pool) | set(f.random_leader_pool)
    recruits |= set(f.recruit)
reach, todo = set(), list(start)
while todo:
    n = todo.pop()
    if n in reach:
        continue
    reach.add(n)
    try:
        todo += list(_stats_for(n).get("advances_to", []))
    except Exception as exc:
        print("no stats for", n, exc)
print("reachable types:", len(reach), "recruitable:", len(recruits))

specials, abilities, traits, many, level0, movetypes = {}, {}, {}, [], [], {}
levels = {}
for n in sorted(reach):
    s = _stats_for(n)
    levels[s["level"]] = levels.get(s["level"], 0) + 1
    if s["level"] == 0:
        level0.append((n, n in recruits, s["cost"]))
    if len(s["attacks"]) > enc.PARITY_WEAPON_SLOTS:
        many.append(n)
    for a in s["attacks"]:
        for sp in a.get("specials", []):
            specials.setdefault(sp, []).append(n)
    for ab in s.get("abilities", []):
        abilities.setdefault(ab, []).append(n)
    t = s.get("traits", {})
    for tr in list(t.get("musthave", [])) + list(t.get("pool", [])):
        traits.setdefault(tr, []).append(n)
    movetypes.setdefault(s["movement_type"], []).append(n)

print("levels:", dict(sorted(levels.items())))
print("level 0:", level0)
print("more than", enc.PARITY_WEAPON_SLOTS, "attacks:", many)
print("specials without a column:", {k: v[:6] for k, v in specials.items() if k not in enc.PARITY_SPECIALS})
print("specials with a column:", sorted(k for k in specials if k in enc.PARITY_SPECIALS))
print("abilities without a column:", {k: v[:8] for k, v in abilities.items() if k not in enc.PARITY_ABILITIES})
print("abilities with a column:", sorted(k for k in abilities if k in enc.PARITY_ABILITIES))
print("traits without a column:", {k: v[:6] for k, v in traits.items() if k not in enc.PARITY_TRAITS})
print("traits with a column:", sorted(k for k in traits if k in enc.PARITY_TRAITS))
print("movement types:", len(movetypes), sorted(movetypes))

# Recruitable types whose certain (musthave) traits the recruit row does not show.
must = {n: _stats_for(n)["traits"].get("musthave", []) for n in sorted(recruits)}
print("recruits with certain traits:", {n: m for n, m in must.items() if m})

# Does defence differ between two types of one movement type?
for mt, names in sorted(movetypes.items()):
    tables = {json.dumps(_stats_for(n)["defense"], sort_keys=True) for n in names}
    if len(tables) > 1:
        print("movement type", mt, "has", len(tables), "defence tables among", len(names), "types")

d = _stats_for("Vampire Bat")
print("Vampire Bat:", {k: d[k] for k in ("level", "cost", "traits", "movement_type")}, "defense", d["defense"])
d = _stats_for("Goblin Spearman")
print("Goblin Spearman:", {k: d[k] for k in ("level", "cost", "traits")})

from tools.eval_players import peek_checkpoint_arch  # noqa: E402
if len(sys.argv) > 1:
    print("checkpoint:", peek_checkpoint_arch(Path(sys.argv[1]), "audit"))
