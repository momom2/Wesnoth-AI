"""One-off: on a real mid-game position, decode the network's unit and
recruit rows back into words and compare them with the unit database and
with the unit's own record. Prints what a row holds and what it cannot hold.

    python -I -X utf8 spot_rows.py <checkpoint> <seed>
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

torch.set_num_threads(1)

from tools.eval_players import _load_policy  # noqa: E402
from tools.raw_player import RawPolicyPlayer  # noqa: E402
from tools.replay_dataset import _stats_for  # noqa: E402
from tools.wesnoth_sim import PvPDefaults, WesnothSim  # noqa: E402
from wesnoth_ai import encoder as enc  # noqa: E402
from wesnoth_ai.constants import COST_NORM, EXP_NORM, HP_NORM, MOVES_NORM  # noqa: E402
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup  # noqa: E402

ckpt, seed = Path(sys.argv[1]), int(sys.argv[2])
rng = random.Random(seed)
setup = random_setup(rng, category="ladder")
print("game:", setup.scenario_id, setup.faction1, "vs", setup.faction2)
gs = build_scenario_gamestate(setup, base_income=PvPDefaults().base_income)
sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=40)
policy = _load_policy(ckpt, torch.device("cpu"), label="audit")
player = RawPolicyPlayer(policy, 1.0, seed=seed, end_turn_offset=0.0, memory_slots=64)
n = 0
while not sim.done and (sim.gs.global_info.turn_number < 6 or n < 150):
    sim.step(player.select_action(sim.gs, game_label="spot", sim=sim))
    n += 1
gs = sim.gs
side = int(gs.global_info.current_side)
print("position: turn", gs.global_info.turn_number, "side to move", side, "after", n, "decisions")
raw = policy._inference_encoder.raw_of(gs)
print("unit row width:", raw.unit_feats.shape[1], "; hex row width:", raw.hex_dynamic_flags.shape[1],
      "; hexes in the stream:", len(raw.hex_positions), "of", len(gs.map.hexes))

DT = enc.PARITY_DAMAGE_TYPES


def decode(row):
    out = {
        "max_hp": round(row[0] * HP_NORM), "hp_frac": round(float(row[1]), 2),
        "max_moves": round(row[2] * MOVES_NORM), "max_exp": round(row[4] * EXP_NORM),
        "exp_frac": round(float(row[5]), 2), "cost": round(row[6] * COST_NORM),
        "leader": int(row[7]), "attacked": int(row[8]),
    }
    weapons = []
    for k in range(enc.PARITY_WEAPON_SLOTS):
        w = row[enc.PARITY_WEAPON_AT + k * enc.PARITY_WEAPON_COLS:][:enc.PARITY_WEAPON_COLS]
        if w[0] < 0.5:
            continue
        dtype = [DT[i] for i in range(len(DT)) if w[4 + i] > 0.5]
        specials = [enc.PARITY_SPECIALS[i] for i in range(len(enc.PARITY_SPECIALS)) if w[4 + len(DT) + i] > 0.5]
        weapons.append((round(w[1] * enc.WEAPON_DAMAGE_NORM), round(w[2] * enc.WEAPON_STRIKES_NORM),
                        "ranged" if w[3] > 0.5 else "melee", dtype, specials))
    out["weapons"] = weapons
    out["takes_pct"] = {DT[i]: round(100 - 100 * float(row[enc.PARITY_RESIST_AT + i])) for i in range(len(DT))}
    out["traits"] = [enc.PARITY_TRAITS[i] for i in range(len(enc.PARITY_TRAITS))
                     if row[enc.PARITY_TRAIT_AT + i] > 0.5]
    out["abilities"] = [enc.PARITY_ABILITIES[i] for i in range(len(enc.PARITY_ABILITIES))
                        if row[enc.PARITY_ABILITY_AT + i] > 0.5]
    out["tod"] = round(float(row[enc.PARITY_TOD_AT]) * enc.LAWFUL_BONUS_NORM)
    out["leadership"] = round(float(row[enc.PARITY_LEADERSHIP_AT]) * enc.LEADERSHIP_NORM)
    return out


by_id = {u.id: u for u in gs.map.units}
core = sim.core.core
shown = set()
mismatch = 0
checked = 0
for i, uid in enumerate(raw.unit_ids):
    u = by_id[uid]
    row = decode(raw.unit_feats[i])
    db = _stats_for(u.name)
    checked += 1
    ok = (row["takes_pct"] == {k: db["resistance"][k] for k in DT}
          and [w[3][0] for w in row["weapons"]] == [a["type"] for a in db["attacks"]][:3]
          and [w[2] for w in row["weapons"]] == [a["range"] for a in db["attacks"]][:3]
          and row["max_hp"] == u.max_hp and row["cost"] == db["cost"])
    mismatch += not ok
    if u.name in shown or len(shown) >= 5:
        continue
    shown.add(u.name)
    terrain = next((h for h in gs.map.hexes if h.position.x == u.position.x and h.position.y == u.position.y), None)
    print(f"\n{u.name} (side {u.side}) at {u.position.x + 1},{u.position.y + 1}: the row says", row)
    print("   database, NOT in the row: level", db["level"], "; advances to", db["advances_to"], "; movement type",
          db["movement_type"], "; chance to be hit by terrain", {k: v for k, v in db["defense"].items()
                                                                  if k in ("flat", "forest", "hills", "village", "castle")})
print(f"\nrows checked against the database (resistances, weapon types and ranges, max hp, cost): {checked},"
      f" mismatches: {mismatch}")
for r, name in enumerate(raw.recruit_types):
    row = decode(raw.recruit_feats[r])
    db = _stats_for(name)
    print(f"recruit option {name}: row traits {row['traits']}; database: level {db['level']}, certain traits"
          f" {db['traits'].get('musthave', [])}, possible {db['traits'].get('pool', [])}")
