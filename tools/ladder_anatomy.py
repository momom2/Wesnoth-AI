"""Ladder-game anatomy: WHY don't ladder games decide?

Plays N ladder-only games with a checkpoint (real MCTSPolicy search)
and records, per turn: army sizes, total HP, gold, villages, minimum
army separation, combats initiated, leader HP and exposure. Also
captures, per DECISION, the value distribution MCTS surfaces at the
root (child Q-values -- the numbers PUCT actually compares; if their
spread is ~0 the search runs on priors and noise, not evaluation).

Distinguishes the four indecision shapes:
  never-meet   armies stay far apart (min separation never < ~6)
  no-commit    armies close but combats/turn stays ~0
  no-convert   combats happen, HP trades, nobody dies enough
  leader-hide  units die but leader exposure stays high

Writes: per-game JSONL (--out), a printed summary, and the first
--export-replays games as Wesnoth-loadable .bz2 replays.

Usage (on the training box):
    python tools/ladder_anatomy.py --checkpoint CKPT.pt
        [--games 20] [--sims 32] [--max-turns 100]
        [--out /workspace/anatomy.jsonl] [--export-replays 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics as st
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("ladder_anatomy")


def _hex_dist(a, b) -> int:
    """Hex distance on the offset grid (same metric the sim uses for
    adjacency: convert offset -> cube, max of abs deltas)."""
    def cube(p):
        x, y = p.x, p.y
        q = x
        r = y - (x - (x & 1)) // 2
        return (q, r, -q - r)
    ax, ay, az = cube(a)
    bx, by, bz = cube(b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def _turn_snapshot(gs, combats_this_turn: int) -> dict:
    sides = {1: [], 2: []}
    leaders = {}
    for u in gs.map.units:
        if u.side in sides:
            sides[u.side].append(u)
            if u.is_leader:
                leaders[u.side] = u
    min_sep = None
    for u1 in sides[1]:
        for u2 in sides[2]:
            d = _hex_dist(u1.position, u2.position)
            if min_sep is None or d < min_sep:
                min_sep = d
    leader_exposure = {}
    for side, ldr in leaders.items():
        enemies = sides[3 - side]
        if enemies:
            leader_exposure[side] = min(
                _hex_dist(ldr.position, e.position) for e in enemies)
    return {
        "turn": gs.global_info.turn_number,
        "units": {s: len(v) for s, v in sides.items()},
        "hp": {s: sum(u.current_hp for u in v)
               for s, v in sides.items()},
        "gold": {i + 1: sd.current_gold
                 for i, sd in enumerate(gs.sides[:2])},
        "villages": {i + 1: sd.nb_villages_controlled
                     for i, sd in enumerate(gs.sides[:2])},
        "min_separation": min_sep,
        "combats": combats_this_turn,
        "leader_hp": {s: leaders[s].current_hp for s in leaders},
        "leader_exposure": leader_exposure,
    }


def classify(turns: List[dict]) -> str:
    if not turns:
        return "empty"
    min_sep_ever = min((t["min_separation"] for t in turns
                        if t["min_separation"] is not None),
                       default=99)
    total_combats = sum(t["combats"] for t in turns)
    # units lost across the game (recruits complicate; use HP trend
    # of the back half)
    half = turns[len(turns) // 2:]
    hp_traded = (half[0]["hp"][1] + half[0]["hp"][2]
                 - half[-1]["hp"][1] - half[-1]["hp"][2]) if half else 0
    min_leader_exp = min((min(t["leader_exposure"].values())
                          for t in turns if t["leader_exposure"]),
                         default=99)
    if min_sep_ever > 5:
        return "never-meet"
    if total_combats < len(turns) * 0.3:
        return "no-commit"
    if min_leader_exp > 4:
        return "leader-hide"
    return "no-convert"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--aux-value-bonus", type=float, default=0.0,
                    help="Match the training run's value (reviewer "
                         "m2: dissecting an aux-trained checkpoint "
                         "with the bonus off measures the wrong "
                         "search regime).")
    ap.add_argument("--moves-left-utility", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=Path("logs/ladder_anatomy.jsonl"))
    ap.add_argument("--export-replays", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from tools.actor_pool import _zero_reward
    from tools.draw_tiebreak import DrawTiebreakConfig
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from tools.sim_to_replay import export_replay_from_scratch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from wesnoth_sim import PvPDefaults, WesnothSim

    raw = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    base = TransformerPolicy(
        device=device, d_model=a["d_model"],
        num_layers=a["num_layers"], num_heads=a["num_heads"],
        d_ff=a["d_ff"], aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")))
    base.load_checkpoint(args.checkpoint)
    policy = MCTSPolicy(base, MCTSConfig(
        n_simulations=args.sims,
        draw_tiebreak=DrawTiebreakConfig(cap=0.3),
        aux_value_bonus=args.aux_value_bonus,
        moves_left_utility=args.moves_left_utility))

    # Root value-distribution sink: per-search child-Q stats.
    q_stats: List[dict] = []

    def sink(root):
        qs = [e.q_value for e in getattr(root, "edges", [])
              if e.n_visits > 0]
        if len(qs) >= 2:
            q_stats.append({
                "n": len(qs), "mean": st.mean(qs),
                "std": st.pstdev(qs), "min": min(qs), "max": max(qs)})

    policy.search_stats_sink = sink

    pvp = PvPDefaults()
    cost = _recruit_cost_lookup()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shapes = []
    with args.out.open("w", encoding="utf-8") as fout:
        for g in range(args.games):
            setup = random_setup(rng)
            gs = build_scenario_gamestate(
                setup, base_income=pvp.base_income,
                village_gold=pvp.village_gold,
                village_upkeep=pvp.village_support,
                experience_modifier=pvp.experience_modifier)
            sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                             max_turns=args.max_turns)
            q_stats.clear()
            turns: List[dict] = []
            last_turn = -1
            combats = 0

            # Per-turn snapshots via a wrapper around select_action's
            # sim access: sample the state at each turn boundary.
            orig_select = policy.select_action

            def select_and_snap(gs_copy, *, game_label="g", sim=None):
                nonlocal last_turn, combats
                t = sim.gs.global_info.turn_number
                if t != last_turn:
                    if last_turn >= 0:
                        turns.append(_turn_snapshot(sim.gs, combats))
                    combats = 0
                    last_turn = t
                act = orig_select(gs_copy, game_label=game_label,
                                  sim=sim)
                if act.get("type") == "attack":
                    combats += 1
                return act

            policy.select_action = select_and_snap
            try:
                play_one_game(sim, policy, _zero_reward,
                              game_label=f"anat{g}", cost_lookup=cost)
            finally:
                policy.select_action = orig_select
            turns.append(_turn_snapshot(sim.gs, combats))
            shape = classify(turns)
            shapes.append(shape)
            qs_all = q_stats[:]
            row = {
                "game": g, "scenario": setup.scenario_id,
                "winner": sim.winner,
                "end_turn": sim.gs.global_info.turn_number,
                "shape": shape,
                "turns": turns,
                "rootq": {
                    "searches": len(qs_all),
                    "mean_std": (st.mean(x["std"] for x in qs_all)
                                 if qs_all else None),
                    "mean_range": (st.mean(x["max"] - x["min"]
                                           for x in qs_all)
                                   if qs_all else None),
                    "mean_q": (st.mean(x["mean"] for x in qs_all)
                               if qs_all else None),
                },
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            log.info(
                f"game {g}: {setup.scenario_id} winner={sim.winner} "
                f"turns={row['end_turn']} shape={shape} "
                f"rootQ std={row['rootq']['mean_std'] or 0:.4f} "
                f"range={row['rootq']['mean_range'] or 0:.4f} "
                f"mean={row['rootq']['mean_q'] or 0:+.3f}")
            if g < args.export_replays:
                try:
                    rp = args.out.parent / f"anatomy_game{g}.bz2"
                    export_replay_from_scratch(sim, rp)
                    log.info(f"  replay exported: {rp}")
                except Exception as e:             # noqa: BLE001
                    log.warning(f"  replay export failed: {e}")

    from collections import Counter
    log.info(f"== shapes: {dict(Counter(shapes))} ==")

    # Dump the games' experiences for diagnose_value_drift arms.
    with policy._lock:
        exps = list(policy._queue)
    if exps:
        import pickle
        dump = args.out.parent / "anatomy_experiences.pkl"
        with dump.open("wb") as f:
            pickle.dump(exps, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"dumped {len(exps)} experiences -> {dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
