"""Mini-map draw anatomy: WHAT does the policy do in mini games?

Investigates the 2026-07-30 mini-draw drift (docs/autonomous_run.md
cycles 41-42): a slow, weights-driven rise in turn-cap draws confined
to the MINI pool (0/17 at decision_step 2.40M -> 5/19 at 2.51M,
p=0.031). Plays N mini-pool games with a checkpoint under the
PRODUCTION search configuration (the box's launch flags, cycle 41
verified: sims=32, gumbel m=16, fpu 0.25, draw-tiebreak cap 0.3,
aux-score head, advice ON, turn cap jittered uniform [60, 100]) and
records per-turn per-SIDE trajectories, so drawn and decisive games
can be compared mechanically and stall symmetry measured:

  - per side per turn: units, total HP, gold, villages, min distance
    from any of the side's units to the ENEMY leader, attacks
    initiated, recruits, moves; at each side-turn end: fraction of MP
    left unused and units fully idle (no move, no attack).
  - per game: winner, ended_by (max_turns / max_actions /
    leader_killed / ...), the game's actual jittered cap, end state,
    engagement telemetry (tools/engagement_stats.py), noprogress
    quiet-streak summary.

Chunk-friendly: --append + --seed let successive short invocations
build one JSONL (the 9-min foreground guardrail); each game is
flushed as it finishes. Compare two checkpoints by running the same
seeds against each and diffing rows.

Usage:
    python tools/mini_anatomy.py --checkpoint training/checkpoints/campaign_live_20260730.pt \
        --games 6 --seed 100 --out logs/mini_anatomy_2515896.jsonl --append
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("mini_anatomy")


def _hex_dist(a, b) -> int:
    """Offset-grid hex distance (same cube conversion ladder_anatomy
    uses)."""
    def cube(p):
        x, y = p.x, p.y
        q = x
        r = y - (x - (x & 1)) // 2
        return (q, r, -q - r)
    ax, ay, az = cube(a)
    bx, by, bz = cube(b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def _side_snapshot(gs) -> dict:
    """Per-side board state at a turn boundary."""
    sides: Dict[int, list] = {1: [], 2: []}
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
    dist_to_enemy_leader = {}
    for s in (1, 2):
        enemy_ldr = leaders.get(3 - s)
        if enemy_ldr is not None and sides[s]:
            dist_to_enemy_leader[s] = min(
                _hex_dist(u.position, enemy_ldr.position)
                for u in sides[s])
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
        "dist_to_enemy_leader": dist_to_enemy_leader,
        "leader_hp": {s: leaders[s].current_hp for s in leaders},
    }


def _idle_stats(gs, side: int) -> dict:
    """MP usage for `side`'s units, read the moment its turn has just
    ended (before its next init_side resets current_moves)."""
    units = [u for u in gs.map.units if u.side == side]
    if not units:
        return {"n": 0, "idle": 0, "mp_unused_frac": None}
    idle = sum(1 for u in units
               if u.current_moves == u.max_moves and not u.has_attacked)
    tot_max = sum(u.max_moves for u in units)
    tot_left = sum(u.current_moves for u in units)
    return {"n": len(units), "idle": idle,
            "mp_unused_frac": (tot_left / tot_max) if tot_max else None}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32,
                    help="Box campaign value (launch script).")
    ap.add_argument("--seed", type=int, default=100,
                    help="Master seed; game g uses seed+g so chunked "
                         "invocations with disjoint seed ranges never "
                         "repeat a setup.")
    ap.add_argument("--category", default="mini",
                    choices=["mini", "ladder", "fogless", "drill"])
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-turns-min", type=int, default=60,
                    help="Production jitter floor (launch script "
                         "--max-turns-min default 60). Set == "
                         "--max-turns to disable jitter.")
    ap.add_argument("--out", type=Path,
                    default=Path("logs/mini_anatomy.jsonl"))
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from tools.actor_pool import _zero_reward
    from tools.draw_tiebreak import DrawTiebreakConfig
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.sim_self_play import (_recruit_cost_lookup, _roll_max_turns,
                                     play_one_game)
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.wesnoth_sim import PvPDefaults, WesnothSim

    raw = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    if args.device:
        device = torch.device(args.device)
    else:
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
    # Head/graft flags come from the CHECKPOINT (matching
    # sim_self_play's resume logic: aux/advice heads must exist at
    # construction or their weights are dropped on load).
    base = TransformerPolicy(
        device=device, d_model=a["d_model"],
        num_layers=a["num_layers"], num_heads=a["num_heads"],
        d_ff=a["d_ff"],
        aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")),
        advice=bool(raw.get("advice")),
        relevant_set_hexes=bool(raw.get("relevant_set_hexes")))
    base.load_checkpoint(args.checkpoint)
    step = raw.get("decision_step")
    # Production MCTS config = the box launch line's explicit flags
    # (--mcts-sims 32, --mcts-aux-score, --mcts-advice) + sim_self_play
    # argparse defaults for everything else (verified 2026-07-30:
    # argparse defaults == MCTSConfig dataclass defaults except
    # n_simulations). draw_tiebreak cap 0.3 is the argparse default
    # (search-only; training labels stay honest).
    cfg = MCTSConfig(
        n_simulations=args.sims,
        advice=bool(raw.get("advice")),
        draw_tiebreak=DrawTiebreakConfig(cap=0.3),
    )
    policy = MCTSPolicy(base, cfg)

    pvp = PvPDefaults()
    cost = _recruit_cost_lookup()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    endings = Counter()
    with args.out.open(mode, encoding="utf-8") as fout:
        for g in range(args.games):
            # Per-game RNG: scenario + cap jitter reproducible per
            # (seed+g) independent of how games are chunked across
            # invocations.
            grng = random.Random(args.seed + g)
            setup = random_setup(grng, category=args.category)
            cap = _roll_max_turns(grng, args.max_turns,
                                  args.max_turns_min)
            gs = build_scenario_gamestate(
                setup, starting_gold=None,
                base_income=pvp.base_income,
                village_gold=pvp.village_gold,
                village_upkeep=pvp.village_support,
                experience_modifier=pvp.experience_modifier)
            sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                             max_turns=cap)
            sim.enable_uniform_advancement()

            turns: List[dict] = []
            per_side_turn: Dict[tuple, Counter] = {}
            idle_log: List[dict] = []
            last_key: Optional[tuple] = None
            t0 = time.perf_counter()

            orig_select = policy.select_action

            def select_and_snap(gs_copy, *, game_label="g", sim=None):
                nonlocal last_key
                t = sim.gs.global_info.turn_number
                s = sim.gs.global_info.current_side
                key = (t, s)
                if key != last_key:
                    if last_key is not None:
                        # Previous side's turn just ended; its units
                        # still hold end-of-turn MP state.
                        prev_t, prev_s = last_key
                        st = _idle_stats(sim.gs, prev_s)
                        st.update({"turn": prev_t, "side": prev_s})
                        idle_log.append(st)
                    if last_key is None or t != last_key[0]:
                        turns.append(_side_snapshot(sim.gs))
                    last_key = key
                act = orig_select(gs_copy, game_label=game_label,
                                  sim=sim)
                tally = per_side_turn.setdefault(key, Counter())
                tally[act.get("type", "?")] += 1
                return act

            policy.select_action = select_and_snap
            try:
                outcome = play_one_game(
                    sim, policy, _zero_reward,
                    game_label=f"mini{args.seed + g}", cost_lookup=cost)
            finally:
                policy.select_action = orig_select
            turns.append(_side_snapshot(sim.gs))
            if last_key is not None:
                st = _idle_stats(sim.gs, last_key[1])
                st.update({"turn": last_key[0], "side": last_key[1]})
                idle_log.append(st)
            # Keep the probe memory-flat: experiences are not trained.
            with policy._lock:
                policy._queue.clear()

            actions_by_side: Dict[int, Counter] = {1: Counter(),
                                                   2: Counter()}
            for (t, s), c in per_side_turn.items():
                if s in actions_by_side:
                    actions_by_side[s].update(c)
            nop = None
            summ = getattr(sim, "noprogress_summary", None)
            if callable(summ):
                try:
                    nop = summ()
                except Exception:               # noqa: BLE001
                    nop = None
            row = {
                "seed": args.seed + g,
                "checkpoint": str(args.checkpoint),
                "decision_step": step,
                "scenario": setup.scenario_id,
                "factions": [setup.faction1, setup.faction2],
                "leaders": [setup.leader1, setup.leader2],
                "category": args.category,
                "max_turns_cap": cap,
                "winner": sim.winner,
                "ended_by": sim.ended_by,
                "end_turn": sim.gs.global_info.turn_number,
                "wall_seconds": round(time.perf_counter() - t0, 1),
                "actions_by_side": {s: dict(c) for s, c
                                    in actions_by_side.items()},
                "recruits": {1: outcome.n_recruits_s1,
                             2: outcome.n_recruits_s2},
                "end_gold": {1: outcome.side1_end_gold,
                             2: outcome.side2_end_gold},
                "closest_approach": {1: outcome.side1_closest_approach,
                                     2: outcome.side2_closest_approach},
                "turns": turns,
                "idle": idle_log,
                "noprogress": nop,
                "engagement": outcome.engagement,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            endings[sim.ended_by] += 1
            log.info(
                f"seed {args.seed + g}: {setup.scenario_id} cap={cap} "
                f"winner={sim.winner} ended_by={sim.ended_by} "
                f"turns={row['end_turn']} "
                f"atk(s1/s2)="
                f"{actions_by_side[1].get('attack', 0)}/"
                f"{actions_by_side[2].get('attack', 0)} "
                f"rec={outcome.n_recruits_s1}/{outcome.n_recruits_s2} "
                f"({row['wall_seconds']}s)")
    log.info(f"== endings: {dict(endings)} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
