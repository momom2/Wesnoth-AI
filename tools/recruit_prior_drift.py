"""Recruit-prior drift tripwire (cycle 30 decision, 2026-07-29).

WHY THIS EXISTS. Cycle 29 identified a "tried-and-cut tax" in
`extract_gumbel_policy_target`: at the box budget (32 sims, gumbel_m=16)
only ~16 of 100-1300 legal actions are visited and ~8 are cut after 1-2
sims, and those cut edges grade BELOW the v_mix shelter where the ~98%
never-sampled mass sits. So being sampled and cut costs probability mass
while never being sampled is free -- and CONCENTRATED-prior classes pay
most, midgame recruit being the most concentrated.

Cycle 30 decided NOT to change target extraction (its measured
consequences sit below every detection floor we have, and both candidate
levers diverge from the mctx reference the cycle-2 fix restored), and to
**arm a cheap tripwire instead**. This is that tripwire.

WHAT IT MEASURES. Per-class RAW prior mass on the SAME recruit-offered
states, for two or more checkpoints. Same states across checkpoints is
the whole point: it is a paired comparison, so map/faction/turn
composition cannot move the number.

The signal to watch is the **turn>=3 (midgame) recruit prior**. Over the
113k-step post-fix leg it fell 0.264 -> 0.122 (median 0.123 -> 0.037,
down in 49/53) while turn<=2 ROSE 0.700 -> 0.731 -- i.e. the drift is
midgame-specific, which is what makes it diagnostic rather than a
uniform policy shift.

ESCALATION RULE (cycle 30): the tax graduates from "recorded" to
"actionable" only if midgame recruit prior keeps bleeding below ~0.05
while strength is flat/negative, or if recruits/turn starts falling
(it was flat, 2.12 -> 2.19, when this was written).

WHICH LEVER (AMENDED 2026-07-29, cycle 34 -- read this before acting).
This file originally said to prefer `gumbel_m` 16->8. **That was wrong
and is now measured wrong.** A target-quality probe (~296 searches at
N in {8..512} on 20 preserved states) found m=16->8 at 32 sims moves
midgame recruit mass -0.0135 relative to m=16 -- the WRONG DIRECTION,
CI [-0.052..+0.015] -- leaves the cut band unchanged, and raises shelter
mass 0.20 -> 0.32. The reasoning behind the original guidance ("halve
the tried-and-cut population") was sound and the measurement still
refuted it, which is exactly why the lever is named here rather than
left as folklore.

Levers now in EVIDENCE order:
  1. Playout-cap randomization (already implemented,
     `--mcts-playout-cap`): full-move N=128 at matched average cost.
     128-sim targets roughly HALVE the class-level bias. Caveats, both
     unmeasured end-to-end: fewer targets per game, hotter label
     temperature.
  2. Extraction-semantics changes (a `max(q_hat, v_mix)` floor, or
     voiding 1-visit edges). Cheaply testable by re-running the
     target-quality probe on a new checkpoint -- NOT by Elo, which has
     +-68 resolution at n=100.
  NOT `gumbel_m` -> 8.

Do NOT raise the sim count as the remedy: quality-per-sim strictly
DECLINES above 32, and 64 is not better than 32 (0.223 vs 0.210, within
noise) at twice the cost. The information bottleneck is the m=16 draw
structure, not N.

USAGE
    # once: snapshot recruit-offered states (uses the production
    # select_action path, so states are on-distribution)
    python tools/recruit_prior_drift.py collect \
        --ckpt training/checkpoints/campaign_live_20260729.pt \
        --seeds 301,302,303,304 --outdir training/logs/recruit_snaps

    # then, at each new checkpoint:
    python tools/recruit_prior_drift.py compare \
        --snapdir training/logs/recruit_snaps \
        --ckpts seed_20260718.pt campaign_live_20260729.pt

Snapshots are checkpoint-independent (they are game STATES), so collect
once and re-compare as the lineage advances.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import pickle
import random
import statistics as st
import sys
from typing import Dict, List, Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


# ---------------------------------------------------------------------
# Pure analysis core (no torch) -- this is what the tests exercise.
# ---------------------------------------------------------------------

def summarize(rows: Sequence[dict]) -> Dict[str, float]:
    """Per-bucket recruit/end_turn prior stats for ONE checkpoint.

    Buckets are turn<=2 ("opening", where recruiting is near-forced) and
    turn>=3 ("midgame", the diagnostic bucket). Returns NaN-free zeros
    for an empty bucket rather than raising, so a caller can run this on
    a thin snapshot set without special-casing.
    """
    def _m(vals: List[float]) -> float:
        return st.mean(vals) if vals else 0.0

    def _med(vals: List[float]) -> float:
        return st.median(vals) if vals else 0.0

    early = [r for r in rows if r["turn"] <= 2]
    mid = [r for r in rows if r["turn"] >= 3]
    return {
        "n": len(rows),
        "n_early": len(early),
        "n_mid": len(mid),
        "rec_mean": _m([r["rec"] for r in rows]),
        "rec_median": _med([r["rec"] for r in rows]),
        "end_mean": _m([r["end"] for r in rows]),
        "rec_mean_early": _m([r["rec"] for r in early]),
        "rec_median_early": _med([r["rec"] for r in early]),
        "rec_mean_mid": _m([r["rec"] for r in mid]),
        "rec_median_mid": _med([r["rec"] for r in mid]),
        "end_mean_mid": _m([r["end"] for r in mid]),
    }


def paired_delta(rows_a: Sequence[dict], rows_b: Sequence[dict]
                 ) -> Dict[str, float]:
    """Paired b-minus-a recruit-prior delta over the SAME states.

    Requires equal length and assumes index alignment -- both hold
    because every checkpoint is scored over one shared snapshot list in
    the same order. Raises rather than silently truncating, because a
    length mismatch means the two runs did not see the same states and
    the pairing (the entire point) would be a lie.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired comparison needs identical state lists, got "
            f"{len(rows_a)} vs {len(rows_b)} -- the two checkpoints did "
            f"not score the same snapshots")
    d = [b["rec"] - a["rec"] for a, b in zip(rows_a, rows_b)]
    mid = [(a, b) for a, b in zip(rows_a, rows_b) if a["turn"] >= 3]
    d_mid = [b["rec"] - a["rec"] for a, b in mid]
    out = {
        "n": len(d),
        "mean": st.mean(d) if d else 0.0,
        "median": st.median(d) if d else 0.0,
        "up": sum(1 for v in d if v > 0),
        "n_mid": len(d_mid),
        "mean_mid": st.mean(d_mid) if d_mid else 0.0,
        "up_mid": sum(1 for v in d_mid if v > 0),
    }
    return out


def escalates(summary_latest: Dict[str, float],
              midgame_floor: float = 0.05) -> bool:
    """Cycle-30 escalation rule, as a function so it cannot drift from
    the prose. True when the midgame recruit prior has bled below the
    floor. Deliberately does NOT consider strength: strength is measured
    separately and far more expensively.

    On firing, see "WHICH LEVER" in this module's docstring -- the first
    thing to try is playout-cap randomization at full-move N=128, NOT
    `gumbel_m` 16->8, which was this file's original advice and was
    measured to move recruit mass the wrong way (cycle 34).
    """
    return (summary_latest.get("n_mid", 0) > 0
            and summary_latest["rec_mean_mid"] < midgame_floor)


# ---------------------------------------------------------------------
# Collection / scoring (need torch + the sim)
# ---------------------------------------------------------------------

def _class_of(action: dict) -> str:
    return action.get("type", "?")


def prior_aggregate(policy, gs) -> Dict[str, float]:
    """Per-class RAW prior mass for one state, via the production
    action-sampler path (no advice, no search)."""
    import torch
    from wesnoth_ai.action_sampler import (
        enumerate_legal_actions_with_priors)
    enc, mdl = policy._inference_encoder, policy._inference_model
    with torch.no_grad():
        e = enc.encode(gs)
        o = mdl(e)
        laps = enumerate_legal_actions_with_priors(
            e, o, gs, decision_step=policy._decision_step)
    agg: Dict[str, float] = {}
    for lap in laps:
        c = _class_of(lap.action)
        agg[c] = agg.get(c, 0.0) + lap.prior
    return agg


def cmd_collect(args) -> int:
    """Snapshot recruit-offered states by PLAYING games through the
    production select_action path, so states are on-distribution."""
    import torch
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions)
    from tools.wesnoth_sim import WesnothSim
    from tools.eval_sim import _load_policy

    torch.set_num_threads(args.threads)
    load_factions()
    pol = _load_policy(pathlib.Path(args.ckpt), None, "collect")
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for sd in [int(s) for s in args.seeds.split(",")]:
        setup = random_setup(random.Random(sd), forced_faction=None)
        sim = WesnothSim(build_scenario_gamestate(setup),
                         scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        if hasattr(sim, "enable_uniform_advancement"):
            sim.enable_uniform_advancement()
        snaps: List[dict] = []
        seen: Dict[tuple, int] = {}
        steps = 0
        while not sim.done and steps < args.max_steps:
            gs = sim.gs
            side = gs.global_info.current_side
            if side in (1, 2) and len(snaps) < args.per_game_cap:
                snap_gs = copy.deepcopy(gs)
                agg = prior_aggregate(pol, snap_gs)
                if agg.get("recruit", 0.0) > 0.0:
                    key = (gs.global_info.turn_number, side)
                    k = seen.get(key, 0)
                    seen[key] = k + 1
                    # First and sixth recruit-offered decision of each
                    # (turn, side): the turn start is where the
                    # recruit-vs-hoard choice is cleanest, the mid-turn
                    # sample catches post-move recruit decisions.
                    if k in (0, 5):
                        snaps.append(dict(
                            gs=snap_gs,
                            scenario_id=setup.scenario_id,
                            seed=sd,
                            turn=int(gs.global_info.turn_number),
                            side=int(side),
                            decision_in_turn=k,
                            gold=int(gs.sides[side - 1].current_gold),
                            n_units=sum(1 for u in gs.map.units
                                        if u.side == side),
                            prior_raw=agg,
                            max_turns=args.max_turns,
                        ))
            act = pol.select_action(copy.deepcopy(gs),
                                    game_label=f"s{sd}", sim=sim)
            if act is None:
                break
            sim.step(act)
            steps += 1
        # select_action accumulates deepcopied pending transitions; drop
        # them per game so a long collection doesn't balloon memory.
        with pol._lock:
            pol._pending.clear()
        with open(outdir / f"snaps_seed{sd}.pkl", "wb") as f:
            pickle.dump(snaps, f)
        print(f"seed {sd}: {len(snaps)} snapshots, {steps} steps, "
              f"turn={sim.gs.global_info.turn_number}, "
              f"winner={sim.winner}", flush=True)
    return 0


def load_snapshots(snapdir: pathlib.Path) -> List[dict]:
    snaps: List[dict] = []
    for p in sorted(snapdir.glob("snaps_seed*.pkl")):
        with open(p, "rb") as f:
            snaps.extend(pickle.load(f))
    return snaps


def cmd_compare(args) -> int:
    import torch
    from tools.scenario_pool import load_factions
    from tools.eval_sim import _load_policy

    torch.set_num_threads(args.threads)
    load_factions()
    snaps = load_snapshots(pathlib.Path(args.snapdir))
    if not snaps:
        print(f"no snapshots in {args.snapdir} -- run `collect` first")
        return 2
    print(f"{len(snaps)} states")

    results: Dict[str, List[dict]] = {}
    for ck in args.ckpts:
        path = pathlib.Path(ck)
        if not path.exists():
            path = ROOT / "training" / "checkpoints" / ck
        pol = _load_policy(path, None, str(ck))
        rows = []
        for sn in snaps:
            agg = prior_aggregate(pol, sn["gs"])
            rows.append(dict(turn=sn["turn"], gold=sn["gold"],
                             rec=agg.get("recruit", 0.0),
                             end=agg.get("end_turn", 0.0)))
        results[str(ck)] = rows
        del pol

    for ck, rows in results.items():
        s = summarize(rows)
        print(f"\n{ck}:")
        print(f"  ALL     : rec mean {s['rec_mean']:.3f} "
              f"median {s['rec_median']:.3f}  "
              f"end mean {s['end_mean']:.4f}  (n={s['n']})")
        print(f"  turn<=2 : rec mean {s['rec_mean_early']:.3f} "
              f"median {s['rec_median_early']:.3f}  (n={s['n_early']})")
        print(f"  turn>=3 : rec mean {s['rec_mean_mid']:.3f} "
              f"median {s['rec_median_mid']:.3f}  "
              f"end mean {s['end_mean_mid']:.4f}  (n={s['n_mid']})")
        if escalates(s, args.midgame_floor):
            print(f"  *** ESCALATE: midgame recruit prior "
                  f"{s['rec_mean_mid']:.3f} < {args.midgame_floor} -- "
                  f"see this file's escalation rule (try gumbel_m 16->8)")

    if len(args.ckpts) == 2:
        a, b = results[str(args.ckpts[0])], results[str(args.ckpts[1])]
        d = paired_delta(a, b)
        print(f"\npaired rec-prior delta "
              f"{args.ckpts[1]} - {args.ckpts[0]}:")
        print(f"  ALL     : mean {d['mean']:+.4f} median "
              f"{d['median']:+.4f}  UP in {d['up']}/{d['n']}")
        print(f"  turn>=3 : mean {d['mean_mid']:+.4f}  "
              f"UP in {d['up_mid']}/{d['n_mid']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="snapshot recruit-offered states")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--seeds", required=True,
                   help="comma-separated setup seeds")
    c.add_argument("--outdir", required=True)
    c.add_argument("--max-turns", type=int, default=100)
    c.add_argument("--per-game-cap", type=int, default=24)
    c.add_argument("--max-steps", type=int, default=2500)
    c.set_defaults(func=cmd_collect)

    p = sub.add_parser("compare", help="score checkpoints on snapshots")
    p.add_argument("--snapdir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="checkpoint paths or names under "
                        "training/checkpoints/")
    p.add_argument("--midgame-floor", type=float, default=0.05,
                   help="escalation threshold on midgame recruit prior "
                        "(cycle-30 rule)")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
