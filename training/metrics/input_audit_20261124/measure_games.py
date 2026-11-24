"""One-off audit (2026-11-24): fresh self-play games of one checkpoint at the
loop's decode (T=1, memory 64, no end-turn offset, fog), logging for every
attack what each weapon would do against the defender (the core's own fight
statistics) beside the policy's weapon probabilities, for every recruit
decision the policy's probabilities per unit type, and at every decision
whether the core's zone-of-control flags equal "the neighbours of the visible
enemy units of level 1 or more".

Reads the checkpoint, writes one JSON. Changes nothing in the repository.

    python -I -X utf8 measure_games.py <checkpoint> <out.json> <first seed> <games>
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

import tools.raw_player as rp  # noqa: E402
from tools.abilities import hex_neighbors  # noqa: E402
from tools.eval_players import _load_policy, _play_one_eval_game, _PolicyPair  # noqa: E402
from tools.replay_dataset import _stats_for  # noqa: E402
from tools.wesnoth_sim import PvPDefaults, WesnothSim  # noqa: E402
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup  # noqa: E402

STASH = {}
LAST = {}
OUT = {"games": [], "attacks": [], "recruits": [], "decisions": [], "zoc": {"positions": 0, "with_level0": 0,
                                                                         "mismatch": 0, "examples": []},
       "errors": []}
CTX = {}

_orig_enum = rp.enumerate_legal_actions_with_priors


def _enum(encoded, output, game_state):
    legal = _orig_enum(encoded, output, game_state)
    STASH["legal"] = legal
    return legal


rp.enumerate_legal_actions_with_priors = _enum


def level_of(name):
    try:
        return int(_stats_for(name).get("level", 1))
    except Exception:
        return 1


def pos(p):
    return [int(p.x), int(p.y)]


def check_zoc(sim, gs, side):
    """The core's zone flags for `side` against the rule read from the
    units: the neighbours of each visible enemy of level 1 or more that is
    not petrified."""
    obs = sim.core.observe(side)
    keys = obs.geometry.keys
    got = {tuple(keys[j]) for j in np.flatnonzero(np.asarray(obs.zoc)).tolist()}
    on_map = {tuple(k) for k in keys}
    visible_ids = {uid for uid, v in zip(obs.unit_ids, np.asarray(obs.visible).tolist()) if v}
    want = set()
    level0 = 0
    for u in gs.map.units:
        if u.side == side or u.id not in visible_ids:
            continue
        if "petrified" in (u.statuses or set()):
            continue
        if level_of(u.name) < 1:
            level0 += 1
            continue
        want |= set(hex_neighbors(u.position.x, u.position.y)) & on_map
    z = OUT["zoc"]
    z["positions"] += 1
    z["with_level0"] += 1 if level0 else 0
    if got != want:
        z["mismatch"] += 1
        if len(z["examples"]) < 5:
            z["examples"].append({"game": CTX["label"], "turn": int(gs.global_info.turn_number), "side": side,
                                  "extra": sorted(got - want)[:6], "missing": sorted(want - got)[:6]})


def note_decision(gs, sim, legal, action):
    side = int(gs.global_info.current_side)
    turn = int(gs.global_info.turn_number)
    LAST.clear()
    if not legal:
        return
    priors = np.array([la.prior for la in legal], dtype=np.float64)
    types = [la.action.get("type") for la in legal]
    chosen = next((k for k, la in enumerate(legal) if la.action is action), None)
    OUT["decisions"].append({
        "game": CTX["label"], "turn": turn, "side": side, "n": len(legal),
        "type": action.get("type"), "p": float(priors[chosen]) if chosen is not None else None,
        "p_max": float(priors.max()), "argmax": bool(chosen is not None and chosen == int(priors.argmax())),
        "p_end": float(sum(p for p, t in zip(priors, types) if t == "end_turn")),
    })
    try:
        check_zoc(sim, gs, side)
    except Exception as exc:  # the audit must not stop the game
        if len(OUT["errors"]) < 20:
            OUT["errors"].append("zoc: " + repr(exc))
    by_type = {}
    for la in legal:
        if la.action.get("type") == "recruit":
            by_type[la.action["unit_type"]] = by_type.get(la.action["unit_type"], 0.0) + float(la.prior)
    if by_type:
        OUT["recruits"].append({
            "game": CTX["label"], "turn": turn, "side": side, "faction": CTX["factions"][side],
            "enemy_faction": CTX["factions"][3 - side], "gold": int(gs.sides[side - 1].current_gold),
            "p_type": by_type, "chosen_type": action.get("type"), "chosen_unit": action.get("unit_type"),
            "p_end": float(sum(p for p, t in zip(priors, types) if t == "end_turn")),
            "p_move": float(sum(p for p, t in zip(priors, types) if t == "move")),
            "p_attack": float(sum(p for p, t in zip(priors, types) if t == "attack")),
        })
    if action.get("type") == "attack" and chosen is not None:
        me = legal[chosen]
        mine = [la for la in legal if la.actor_idx == me.actor_idx]
        p_actor = sum(la.prior for la in mine)
        attacks = [la for la in mine if la.action.get("type") == "attack"]
        same_target = [la for la in attacks if la.target_idx == me.target_idx]
        tot = sum(la.prior for la in same_target) or 1.0
        targets = {}
        for la in attacks:
            key = tuple(pos(la.action["target_hex"]))
            targets[key] = targets.get(key, 0.0) + float(la.prior)
        tsum = sum(targets.values()) or 1.0
        LAST.update({
            "turn": turn, "side": side, "p_joint": float(me.prior),
            "p_attack_given_actor": float(sum(la.prior for la in attacks) / p_actor) if p_actor else None,
            "weapon_p": {int(la.weapon_idx): float(la.prior / tot) for la in same_target},
            "target_p": {k: v / tsum for k, v in targets.items()},
            "start": pos(action["start_hex"]), "target": pos(action["target_hex"]),
        })


def fight_rows(core, att, dfd, ax, ay, dx, dy):
    """Per weapon of the attacker: what the fight's statistics give against
    this defender, the defender answering as the engine would choose."""
    rows = []
    a_stats = _stats_for(att.name)
    d_resist = _stats_for(dfd.name).get("resistance", {})
    for w in range(len(att.attacks)):
        base = a_stats["attacks"][w] if w < len(a_stats.get("attacks", [])) else {}
        cw = core.counter_weapon_choice(ax, ay, dx, dy, w)
        counter = int(cw[0]) if cw is not None else -1
        st = core.fight_stats(ax, ay, dx, dy, w, counter)
        if st is None:
            continue
        a_st, d_st = st
        dealt = a_st["cth"] / 100.0 * a_st["damage"] * a_st["n_attacks"]
        taken = d_st["cth"] / 100.0 * d_st["damage"] * d_st["n_attacks"] if d_st else 0.0
        rows.append({
            "w": w, "name": base.get("name"), "type": base.get("type"), "range": base.get("range"),
            "specials": base.get("specials", []), "resist_pct": d_resist.get(base.get("type")),
            "a_cth": a_st["cth"], "a_dmg": a_st["damage"], "a_n": a_st["n_attacks"], "rounds": a_st["rounds"],
            "d_cth": d_st["cth"] if d_st else None, "d_dmg": d_st["damage"] if d_st else None,
            "d_n": d_st["n_attacks"] if d_st else None, "dealt": dealt, "taken": taken,
        })
    return rows


def note_attack(sim, cmd, side):
    ax, ay, dx, dy, w = (int(v) for v in cmd[1:6])
    gs = sim.gs
    at = {(u.position.x, u.position.y): u for u in gs.map.units}
    att, dfd = at.get((ax, ay)), at.get((dx, dy))
    if att is None or dfd is None:
        return None
    core = sim.core.core
    rec = {
        "game": CTX["label"], "turn": int(gs.global_info.turn_number), "side": side,
        "faction": CTX["factions"].get(side), "att": att.name, "att_level": level_of(att.name),
        "att_hp": int(att.current_hp), "att_max_hp": int(att.max_hp), "att_exp": int(att.current_exp),
        "att_max_exp": int(att.max_exp), "att_id": att.id, "dfd": dfd.name, "dfd_level": level_of(dfd.name),
        "dfd_hp": int(dfd.current_hp), "dfd_max_hp": int(dfd.max_hp), "dfd_id": dfd.id, "chosen_w": w,
        "weapons": fight_rows(core, att, dfd, ax, ay, dx, dy), "decision": dict(LAST) if LAST else None,
        "others": [],
    }
    if rec["decision"]:
        rec["moved_first"] = rec["decision"]["start"] != [ax, ay]
        rec["decision"]["target_p"] = {f"{k[0]},{k[1]}": v for k, v in rec["decision"]["target_p"].items()}
        legal_targets = {tuple(int(c) for c in k.split(",")) for k in rec["decision"]["target_p"]}
        for nx, ny in hex_neighbors(ax, ay):
            other = at.get((nx, ny))
            if other is None or (nx, ny) == (dx, dy) or (nx, ny) not in legal_targets:
                continue
            rec["others"].append({"dfd": other.name, "dfd_hp": int(other.current_hp), "hex": [nx, ny],
                                  "weapons": fight_rows(core, att, other, ax, ay, nx, ny)})
    return rec


def play(policy, seed):
    rng = random.Random(seed)
    setup = random_setup(rng, category="ladder")
    pvp = PvPDefaults(starting_gold=100, village_gold=2, village_support=1, experience_modifier=70)
    gs = build_scenario_gamestate(setup, base_income=pvp.base_income)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=40)
    label = f"g{seed}"
    CTX.update({"label": label, "factions": {1: setup.faction1, 2: setup.faction2}})

    class Logging(rp.RawPolicyPlayer):
        def select_action(self, game_state, *, game_label="default", sim=None):
            STASH.clear()
            action = super().select_action(game_state, game_label=game_label, sim=sim)
            try:
                note_decision(game_state, sim_ref, STASH.get("legal") or [], action)
            except Exception as exc:
                if len(OUT["errors"]) < 20:
                    OUT["errors"].append("decision: " + repr(exc))
            return action

    sim_ref = sim
    player = Logging(policy, 1.0, seed=seed, end_turn_offset=0.0, memory_slots=64)
    orig = sim._apply_and_record

    def wrapped(cmd, side, extras=None):
        rec = None
        if cmd[0] == "attack":
            try:
                rec = note_attack(sim, cmd, side)
            except Exception as exc:
                if len(OUT["errors"]) < 20:
                    OUT["errors"].append("attack: " + repr(exc))
        out = orig(cmd, side, extras)
        if rec is not None:
            after = {u.id: u for u in sim.gs.map.units}
            a2, d2 = after.get(rec["att_id"]), after.get(rec["dfd_id"])
            rec["att_died"] = a2 is None
            rec["killed"] = d2 is None
            rec["att_advanced"] = a2 is not None and a2.name != rec["att"]
            rec["dfd_advanced"] = d2 is not None and d2.name != rec["dfd"]
            OUT["attacks"].append(rec)
        return out

    sim._apply_and_record = wrapped
    t0 = time.perf_counter()
    _play_one_eval_game(sim, _PolicyPair(player, label, 1), _PolicyPair(player, label, 2), game_label=label)
    OUT["games"].append({
        "label": label, "seed": seed, "scenario": setup.scenario_id, "faction1": setup.faction1,
        "faction2": setup.faction2, "leader1": setup.leader1, "leader2": setup.leader2,
        "winner": int(sim.winner), "turns": int(sim.gs.global_info.turn_number), "ended_by": sim.ended_by,
        "commands": len(sim.command_history), "seconds": round(time.perf_counter() - t0, 1),
    })
    print(json.dumps(OUT["games"][-1]), flush=True)


def main():
    ckpt, out, first, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    policy = _load_policy(ckpt, torch.device("cpu"), label="audit")
    for seed in range(first, first + n):
        play(policy, seed)
        out.write_text(json.dumps(OUT), encoding="utf-8")
    print("done", len(OUT["attacks"]), "attacks", len(OUT["recruits"]), "recruit decisions",
          OUT["zoc"], OUT["errors"][:5], flush=True)


if __name__ == "__main__":
    main()
