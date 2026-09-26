#!/usr/bin/env python3
"""What a player sees that the network's tokens do not carry, counted over
corpus decisions on the Ladder maps.

Replays a seeded random sample of Ladder-map corpus games with the Python
applier and, at the state before each player command (move, attack,
recruit, end_turn), counts: units whose weapons differ from their type's
(traits), poisoned and slowed units, units on terrain classes the
multi-hot view merges (mushroom grove with cave, reef with shallow
water), a local time of day that differs from the board's, enemies hidden
by fog that the side saw this turn or last, and how much of the enemy's
threat the relevant set (obs8's hex basis) leaves without a token (every
third decision). Findings: docs/observation_parity_20260926.md.

The sample is `--sample` Ladder games in a seeded shuffle of the corpus
manifest; `--slices`/`--slice` split it for separate processes and
`--games` stops a slice after that many games. The 2026-09-26 record
(training/metrics/observation_parity_20260926/) is `--sample 100 --slices
4` with `--games` 18, 19, 13 and 20 for slices 0 to 3.

    python tools/analysis/observation_parity_census.py --sample 100 \
        --slices 4 --slice 0 --games 18 --out OUT_DIR
"""
import argparse
import gzip
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wesnoth_ai.paths import IMITATION_DATASET_DIR
from wesnoth_ai.rules.scenario_pool import LADDER_SCENARIO_IDS
from tools.replay_dataset import (_build_initial_gamestate, _setup_scenario_events, _apply_command,
                                  _lawful_bonus_for_turn, illuminated_lawful_bonus_at, _stats_for,
                                  _find_unit_at)
from tools.pathfind_sim import ReachContext, unit_reach, emits_zoc
from wesnoth_ai.sim.abilities import hex_neighbors
from wesnoth_ai.sim.classes import PLAYER_SIDES, opponent_of
from wesnoth_ai.visibility import units_visible_to, is_scenery_unit, relevant_hex_positions, visible_hexes_for
from wesnoth_ai.rules.terrain_resolver import strip_start_position


def conflated(code):
    """The terrain the multi-hot view files under another class."""
    code = strip_start_position(code or "")
    base, _, over = code.partition("^")
    if over.startswith("Tf") or base.startswith("Tb"):
        return "fungus"
    if base.startswith("Wwr"):
        return "reef"
    if base in ("Uu", "Uue") and not over:
        return "cave"
    return ""

PAIRED = {"move", "attack", "recruit", "end_turn"}
LADDER = set(LADDER_SCENARIO_IDS)
HIDDEN_TRAITS = ("dextrous", "strong", "fearless", "healthy", "feral")


def sample_files(n, corpus=IMITATION_DATASET_DIR, seed=20260926):
    rows = [json.loads(line) for line in open(corpus / "manifest.jsonl", encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked = []
    for r in rows:
        p = corpus / r["file"]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            head = f.read(300)
        sid = head.split('"scenario_id": "', 1)[-1].split('"', 1)[0]
        if sid in LADDER:
            picked.append((p, bool(r.get("fog", True))))
            if len(picked) >= n:
                break
    return picked


def base_attacks(name):
    return sorted((a["damage"], a["number"], a["range"]) for a in _stats_for(name).get("attacks", []))


def unit_attacks(u):
    return sorted((a.damage_per_strike, a.number_strikes, "ranged" if a.is_ranged else "melee")
                  for a in u.attacks)


def enemy_threat_ctx(gs, side, visible):
    """The enemy side's movement context as the mover can know it: the
    units the mover sees, the mover's units as the enemy's enemies."""
    enemy_side = opponent_of(side)
    ctx = ReachContext(side=enemy_side)
    for u in visible:
        pos = (u.position.x, u.position.y)
        ctx.occupied_visible.add(pos)
        if u.side == enemy_side:
            ctx.ally_hexes.add(pos)
            continue
        ctx.enemy_hexes.add(pos)
        if emits_zoc(u):
            ctx.zoc_hexes.update(hex_neighbors(*pos))
    return ctx


def census_game(path, fog_flag, c: Counter, rel_every=3):
    d = json.load(gzip.open(path, "rt", encoding="utf-8"))
    gs = _build_initial_gamestate(d)
    _setup_scenario_events(gs, d.get("scenario_id", ""))
    last_seen = {1: {}, 2: {}}
    first_seen = {}         # side -> enemy id -> turn last seen at a decision
    k = 0
    for cmd in d["commands"]:
        gi = gs.global_info
        side = gi.current_side
        if side in PLAYER_SIDES and cmd and cmd[0] in PAIRED:
            k += 1
            fog = bool(getattr(gi, "_fog", True))
            c["decisions"] += 1
            c["decisions_fog"] += fog
            turn = gi.turn_number
            vis = units_visible_to(gs, side)
            vis_ids = {u.id for u in vis}
            own = [u for u in vis if u.side == side and not is_scenery_unit(u)]
            opp = opponent_of(side)
            en_vis = [u for u in vis if u.side == opp and not is_scenery_unit(u)]
            en_all = [u for u in gs.map.units if u.side == opp and not is_scenery_unit(u)]
            units = own + en_vis
            c["units_seen"] += len(units)
            # statuses, own and enemy apart
            c["own_seen"] += len(own)
            c["own_poisoned"] += sum("poisoned" in (u.statuses or ()) for u in own)
            c["own_slowed"] += sum("slowed" in (u.statuses or ()) for u in own)
            c["dec_own_poisoned"] += any("poisoned" in (u.statuses or ()) for u in own)
            c["dec_enemy_poisoned"] += any("poisoned" in (u.statuses or ()) for u in en_vis)
            # terrain classes the encoder conflates, under the units
            codes = getattr(gi, "_terrain_codes", {}) or {}
            for u in units:
                cls = conflated(codes.get((u.position.x, u.position.y), ""))
                if cls:
                    c["units_on_" + cls] += 1
            # scenery the side does not see (statues under fog)
            if fog:
                seen = visible_hexes_for(gs, side)
                c["dec_fogged_scenery"] += any(is_scenery_unit(u) and (u.position.x, u.position.y) not in seen
                                               for u in gs.map.units)
                if en_vis and first_seen.get(side) is None:
                    first_seen[side] = turn
            st_units = [u for u in units if {"slowed", "poisoned"} & set(u.statuses or ())]
            c["units_slowed"] += sum("slowed" in (u.statuses or ()) for u in units)
            c["units_poisoned"] += sum("poisoned" in (u.statuses or ()) for u in units)
            c["dec_any_status"] += bool(st_units)
            # traits and per-unit weapons against the type's own
            for u in units:
                tr = set(u.traits or ())
                for t in HIDDEN_TRAITS:
                    c[f"units_trait_{t}"] += t in tr
                c["units_weapon_dev"] += unit_attacks(u) != base_attacks(u.name)
            c["dec_weapon_dev"] += any(unit_attacks(u) != base_attacks(u.name) for u in units)
            # local time of day against the global feature
            glob = _lawful_bonus_for_turn(turn, int(getattr(gi, "_tod_start_offset", 0) or 0))
            loc_dev = [u for u in units if illuminated_lawful_bonus_at(gs, u, turn) != glob]
            c["units_local_tod"] += len(loc_dev)
            if loc_dev:
                c["local_tod_" + d.get("scenario_id", "")] += len(loc_dev)
            c["dec_local_tod"] += bool(loc_dev)
            # fog memory
            if fog:
                hidden = [u for u in en_all if u.id not in vis_ids]
                seen_before = [u for u in hidden if last_seen[side].get(u.id, -99) >= turn - 1]
                c["fog_enemy_total"] += len(en_all)
                c["fog_enemy_hidden"] += len(hidden)
                c["fog_enemy_hidden_seen_recently"] += len(seen_before)
                c["fog_dec_hidden_seen_recently"] += bool(seen_before)
                leader = next((u for u in en_all if u.is_leader), None)
                c["fog_dec_enemy_leader_hidden"] += bool(leader and leader.id not in vis_ids)
            for u in en_vis:
                last_seen[side][u.id] = turn
            # attacks: the fighters' hidden per-unit state
            if cmd[0] == "attack":
                a = _find_unit_at(gs, cmd[1], cmd[2])
                b = _find_unit_at(gs, cmd[3], cmd[4])
                if a is not None and b is not None:
                    c["attacks"] += 1
                    c["att_status"] += bool({"slowed", "poisoned"} & (set(a.statuses or ()) | set(b.statuses or ())))
                    c["att_slowed"] += ("slowed" in (a.statuses or ())) or ("slowed" in (b.statuses or ()))
                    c["att_weapon_dev"] += (unit_attacks(a) != base_attacks(a.name)) or (unit_attacks(b) != base_attacks(b.name))
                    c["att_fearless"] += ("fearless" in (a.traits or ())) or ("fearless" in (b.traits or ()))
                    ca = conflated(codes.get((a.position.x, a.position.y), ""))
                    cb = conflated(codes.get((b.position.x, b.position.y), ""))
                    c["att_on_conflated"] += bool(ca or cb)
                    c["att_local_tod"] += (illuminated_lawful_bonus_at(gs, a, turn) != glob) or (illuminated_lawful_bonus_at(gs, b, turn) != glob)
            # the relevant set (obs8's hex basis) against what the enemy threatens
            if k % rel_every == 0:
                rel = relevant_hex_positions(gs, side)
                board = {(h.position.x, h.position.y) for h in gs.map.hexes}
                c["rel_dec"] += 1
                c["rel_hexes"] += len(rel)
                c["board_hexes"] += len(board)
                adj_en = {n for u in en_vis for n in hex_neighbors(u.position.x, u.position.y)} & board
                c["adj_enemy_hexes"] += len(adj_en)
                c["adj_enemy_missing"] += len(adj_en - rel)
                own_adj = {n for u in own for n in hex_neighbors(u.position.x, u.position.y)} & board
                ctx = enemy_threat_ctx(gs, side, vis)
                threat_from = set()
                for e in en_vis:
                    r = unit_reach(e, gs, ctx, budget=int(e.max_moves))
                    threat_from |= (set(r.landable) | {(e.position.x, e.position.y)}) & own_adj
                c["attack_from_hexes"] += len(threat_from)
                c["attack_from_missing"] += len(threat_from - rel)
                c["own_units"] += len(own)
                c["own_units_with_unseen_neighbour"] += sum(
                    bool((set(hex_neighbors(u.position.x, u.position.y)) & board) - rel) for u in own)
        _apply_command(gs, cmd)
    for sd in PLAYER_SIDES:
        if fog_flag:
            c[f"first_enemy_seen_turn_{first_seen.get(sd, 'never')}"] += 1
    c["games"] += 1
    c["games_fog"] += fog_flag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=100, help="Ladder games in the sample")
    ap.add_argument("--slices", type=int, default=1)
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--games", type=int, default=None,
                    help="stop the slice after this many games")
    ap.add_argument("--dataset", type=Path, default=IMITATION_DATASET_DIR,
                    help="the imitation corpus (its manifest.jsonl and game files)")
    ap.add_argument("--out", type=Path, required=True, help="directory for census_slice<N>.json")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    files = sample_files(args.sample, args.dataset)[args.slice::args.slices][:args.games]
    c = Counter()
    t0 = time.time()
    for p, fog in files:
        census_game(p, fog, c)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"census_slice{args.slice}.json"
    out.write_text(json.dumps(dict(c), indent=1), encoding="utf-8")
    print(f"slice {args.slice}: {c['games']} games, {c['decisions']} decisions, "
          f"{time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
