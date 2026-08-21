"""E2: is the TCS TARGET channel's advantage signal real or salt noise?

(docs/leg4_erosion_rootcause_20260820.md par.3 E2; user go 2026-08-21.)

The target pass (turn_search.plan_turn final pass) grades each
alternative with ONE materialization at ONE salt; the linear link
consumes a[i] = q_i - LOO_mean(other evaluated q). The Q8 probe
showed a single coordinate edit changes WHICH fights happen
downstream, so a single-salt advantage may be mostly combat luck.
This tool re-grades the SAME candidate sets at k independent salts
and reports the one-way intraclass correlation

    ICC(1) = (MSB - MSW) / (MSB + (k-1) * MSW)

over candidates (between-candidate signal vs within-candidate
across-salt noise) of exactly the advantage quantity the target
consumes.

Pre-registered thresholds (workflow synthesis): ICC <= 0.15 ->
channel is null (leg-5 fix needs lambda=1.0 AND replicated grading);
ICC >= 0.40 -> channel carries signal (pure-lambda story).

Run on the SEED head (the claim is the channel was null from
iteration 0):

    python tools/target_channel_icc.py \
        --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --states 60 --salts 4 --out training/metrics/e2_icc.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.turn_counterfactual_probe import load_policy  # noqa: E402
from tools.turn_search import (  # noqa: E402
    gumbel_top_k_alternatives, materialize, record_spine,
)
from tools.wesnoth_sim import WesnothSim  # noqa: E402

log = logging.getLogger("target_icc")


def coordinate_matrix(policy, sim0, side, ds, steps, commands, j,
                      st, n_alt, salts, rng, state_id):
    """Value matrix v[cand, salt] for coordinate j's candidate set
    (incumbent action + gumbel alternatives, drawn ONCE). A candidate
    stays only if valid at EVERY salt (balanced design)."""
    priors = np.array([a.prior for a in st.legal])
    et_idx = next((i for i, a in enumerate(st.legal)
                   if a.action.get("type") == "end_turn"), None)
    alt_idx = list(gumbel_top_k_alternatives(
        priors, st.action_idx, et_idx, n_alt, rng))
    cand_actions = [st.legal[st.action_idx].action] + [
        st.legal[i].action for i in alt_idx]
    rows: List[List[float]] = []
    for ci, act in enumerate(cand_actions):
        cand = commands[:j] + [act] + commands[j + 1:]
        vals = []
        for s in range(len(salts)):
            m = materialize(policy, sim0, side, cand, salts[s], ds)
            if m.invalid or math.isnan(m.value):
                vals = None
                break
            vals.append(float(m.value))
        if vals is not None:
            rows.append(vals)
    return np.array(rows) if len(rows) >= 3 else None


def advantages(v: np.ndarray) -> np.ndarray:
    """a[i,s] = v[i,s] - LOO mean over the OTHER candidates at the
    same salt -- the exact quantity turn_search's linear link
    consumes (turn_search.py:520-527)."""
    n = v.shape[0]
    tot = v.sum(axis=0, keepdims=True)
    return v - (tot - v) / (n - 1)


def icc_components(a: np.ndarray):
    """One-way random-effects ANOVA components for a [n_cand, k]."""
    n, k = a.shape
    grand = a.mean()
    row_means = a.mean(axis=1)
    ssb = k * ((row_means - grand) ** 2).sum()
    ssw = ((a - row_means[:, None]) ** 2).sum()
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    return msb, msw, n, k


def main(argv) -> int:
    from tools.scenario_pool import build_scenario_gamestate, random_setup

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--judge-checkpoint", type=Path, default=None,
                    help="Grade the value matrices with THIS "
                         "checkpoint's head while --checkpoint plays "
                         "the states/spines. Same --seed => identical "
                         "state stream, so two judges are compared on "
                         "the same coordinates (head-degradation vs "
                         "state-distribution attribution).")
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--salts", type=int, default=4)
    ap.add_argument("--n-alt", type=int, default=4)
    ap.add_argument("--max-coords", type=int, default=6,
                    help="coordinates probed per state (cost control)")
    ap.add_argument("--max-spine", type=int, default=40)
    ap.add_argument("--category", default="ladder")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--turn-sample-prob", type=float, default=0.25)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=args.log_level)

    rng_py = random.Random(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = load_policy(args.checkpoint, device=args.device)
    ds = policy._decision_step
    log.info(f"checkpoint {args.checkpoint} ds={ds}")
    judge = policy
    if args.judge_checkpoint is not None:
        judge = load_policy(args.judge_checkpoint, device=args.device)
        log.info(f"judge {args.judge_checkpoint} "
                 f"ds={judge._decision_step}")

    per_coord = []          # one dict per probed coordinate
    ssb_pool = ssw_pool = dfb_pool = dfw_pool = 0.0
    states_done = 0
    t0 = time.time()
    for g in range(args.games):
        if states_done >= args.states:
            break
        setup = random_setup(rng_py, category=args.category)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        label = f"icc_g{g}"
        prev_turnside = None
        while not sim.done:
            gi = sim.gs.global_info
            turnside = (gi.turn_number, gi.current_side)
            if turnside != prev_turnside:
                prev_turnside = turnside
                if (states_done < args.states
                        and rng.random() < args.turn_sample_prob):
                    sid = f"g{g}t{gi.turn_number}s{gi.current_side}"
                    side = gi.current_side
                    fork = sim.fork()
                    spine, _ = record_spine(policy, fork, side, ds, rng,
                                            max_spine=args.max_spine)
                    if spine:
                        commands = [s.action for s in spine]
                        salts = [f"icc:{sid}:s{s}"
                                 for s in range(args.salts)]
                        for j, st in enumerate(
                                spine[:args.max_coords]):
                            # `judge` does only the boundary forwards
                            # inside materialize; the state stream,
                            # spine, and alternatives all come from
                            # `policy` + the shared rng.
                            v = coordinate_matrix(
                                judge, fork, side, ds, spine,
                                commands, j, st, args.n_alt, salts,
                                rng, sid)
                            if v is None:
                                continue
                            # --salts 1 = blindness-only mode: blind
                            # is a within-salt property (production
                            # grades at ONE salt), so k=1 suffices
                            # for it; ICC needs k>=2.
                            if args.salts >= 2:
                                a = advantages(v)
                                msb, msw, n, k = icc_components(a)
                                icc = ((msb - msw)
                                       / (msb + (k - 1) * msw)
                                       if (msb + (k - 1) * msw) > 0
                                       else 0.0)
                                ssb_pool += msb * (n - 1)
                                dfb_pool += (n - 1)
                                ssw_pool += msw * n * (k - 1)
                                dfw_pool += n * (k - 1)
                            else:
                                icc = None
                                msb = float(np.var(v[:, 0], ddof=1))
                                msw = 0.0
                            per_coord.append({
                                "state": sid, "coord": j,
                                "n_cand": v.shape[0], "icc": icc,
                                "between_sd": math.sqrt(max(msb, 0)),
                                "within_sd": math.sqrt(max(msw, 0)),
                                # Structural fog blindness (2026-08-21
                                # finding): the boundary is encoded
                                # from the OPPONENT's fogged view, so
                                # where the opponent sees none of the
                                # mover's turn, every candidate gets
                                # the IDENTICAL value -- v has zero
                                # spread, deterministically.
                                "blind": bool(float(np.ptp(v)) < 1e-9),
                                "turn": gi.turn_number,
                            })
                        states_done += 1
                        log.info(
                            f"{sid}: {states_done}/{args.states} "
                            f"states, {len(per_coord)} coords, "
                            f"{time.time()-t0:.0f}s")
            if states_done >= args.states:
                break
            pre_state = copy.deepcopy(sim.gs)
            action = policy.select_action(pre_state, game_label=label,
                                          sim=sim)
            sim.step(action)
        policy.drop_pending(label)

    iccs = np.array([c["icc"] for c in per_coord
                     if c["icc"] is not None])
    msb_p = ssb_pool / dfb_pool if dfb_pool else float("nan")
    msw_p = ssw_pool / dfw_pool if dfw_pool else float("nan")
    k = args.salts
    icc_pooled = ((msb_p - msw_p) / (msb_p + (k - 1) * msw_p)
                  if dfb_pool and (msb_p + (k - 1) * msw_p) > 0
                  else float("nan"))
    blind = [c for c in per_coord if c["blind"]]
    sighted = [c for c in per_coord if not c["blind"]]
    s_iccs = np.array([c["icc"] for c in sighted
                       if c["icc"] is not None])
    report = {
        "checkpoint": str(args.checkpoint),
        "judge_checkpoint": (str(args.judge_checkpoint)
                             if args.judge_checkpoint else None),
        "category": args.category,
        "states": states_done,
        "coords": len(per_coord),
        "salts": k, "n_alt": args.n_alt,
        "blind_frac": (len(blind) / len(per_coord)
                       if per_coord else None),
        "sighted_icc_median": (float(np.median(s_iccs))
                               if len(s_iccs) else None),
        "sighted_icc_mean": (float(s_iccs.mean())
                             if len(s_iccs) else None),
        "sighted_within_sd_median": (float(np.median(
            [c["within_sd"] for c in sighted])) if sighted else None),
        "sighted_between_sd_median": (float(np.median(
            [c["between_sd"] for c in sighted])) if sighted else None),
        "icc_pooled": icc_pooled,
        "icc_median": float(np.median(iccs)) if len(iccs) else None,
        "icc_mean": float(iccs.mean()) if len(iccs) else None,
        "icc_frac_below_0p15": (float((iccs <= 0.15).mean())
                                if len(iccs) else None),
        "icc_frac_above_0p40": (float((iccs >= 0.40).mean())
                                if len(iccs) else None),
        "within_sd_median": (float(np.median(
            [c["within_sd"] for c in per_coord]))
            if per_coord else None),
        "between_sd_median": (float(np.median(
            [c["between_sd"] for c in per_coord]))
            if per_coord else None),
        "c51_atom_width": 2.0 / 50,
        "wall_seconds": round(time.time() - t0, 1),
        "per_coord": per_coord,
    }
    print(json.dumps({kk: vv for kk, vv in report.items()
                      if kk != "per_coord"}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n",
                            encoding="utf-8")
        log.info(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
