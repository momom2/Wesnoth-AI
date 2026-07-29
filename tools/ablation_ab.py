"""Advice-ablation A/B harness (T3-B, stage 2 of the attribution plan).

WHAT IT MEASURES: the ACTING-channel effect of the detector advice
signal -- the same checkpoint weights play head-to-head, arm A with
`MCTSConfig.advice=True` (search leaves encoded via encode_with_advice,
root-conditioned priors) vs arm B with `advice=False` (plain encode).
Verified equivalence: with no advice tokens attached the model skips
the advice cross-attention entirely (model.py `has_advice and
encoded.advice_tokens is not None` gate), so arm B is exactly the
ungrafted computation on identical weights.

WHAT IT DOES NOT MEASURE: trunk shaping through training -- gradients
the advice path routed into shared weights during the campaign benefit
BOTH arms equally and are invisible here. A null result does not mean
the advice signal contributed nothing to training; it means acting
does not depend on the advice channel.

Scoring is PURE game outcome (W/D/L, draws = half a point) per the
eval contract; the draw tiebreak stays search-internal (part of the
agent, matching production acting) and never enters the score.

Budget (box, 128 cores): ~250 decisions/game x sims x ~0.05-0.15 s
per forward => at --sims 8, ~2-5 min/game/core; 400 games across ~120
workers ~= 10-20 min. --sims 16 roughly doubles that; --sims 32 will
NOT fit 30 min at 400 games.

Null validation: --arm-a-advice off --arm-b-advice off (same agent
both sides) must return p ~= 0.5 within CI. --force-advice-graft lets
a pre-advice checkpoint exercise the ON path with a zero-init gate
(also an expected null; validates the encode_with_advice machinery).

Usage (box):
    python tools/ablation_ab.py --checkpoint <ckpt.pt> \
        --games 400 --sims 8 --max-turns 30 --workers 120 \
        --out /workspace/ablation_K.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pathlib
import random
import statistics
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Allow running from a scratch dir too (repo root passed via env).
_ENVROOT = os.environ.get("WESNOTH_AI_ROOT")
if _ENVROOT and _ENVROOT not in sys.path:
    sys.path.insert(0, _ENVROOT)

_W = {}          # per-worker-process state


def _load_policy(ckpt: pathlib.Path, force_advice: bool):
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = raw.get("arch", {}) or {}
    kw = {k: int(arch[k]) for k in
          ("d_model", "num_layers", "num_heads", "d_ff") if k in arch}
    advice = bool(raw.get("advice", False)) or force_advice
    pol = TransformerPolicy(aux_score=bool(raw.get("aux_score")),
                            moves_left=bool(raw.get("moves_left")),
                            advice=advice, **kw)
    pol.load_checkpoint(ckpt)
    return pol, advice


def _worker_init(ckpt_str: str, sims: int, cap: float,
                 a_advice: bool, b_advice: bool, force_graft: bool):
    """Load the checkpoint ONCE per worker process; build the two
    arm wrappers around the SAME TransformerPolicy (identical
    weights in memory -- the arms differ only in MCTSConfig.advice)."""
    import torch
    torch.set_num_threads(1)          # 1 process = 1 core; no oversub
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.draw_tiebreak import DrawTiebreakConfig
    from tools.scenario_pool import load_factions
    load_factions()
    base, has_advice = _load_policy(pathlib.Path(ckpt_str), force_graft)

    def cfg(advice_on: bool) -> MCTSConfig:
        return MCTSConfig(
            n_simulations=sims, batch_size=1,
            draw_tiebreak=DrawTiebreakConfig(cap=cap),
            playout_cap_randomization=False,
            advice=advice_on,
        )
    _W["A"] = MCTSPolicy(base, mcts_config=cfg(a_advice))
    _W["B"] = MCTSPolicy(base, mcts_config=cfg(b_advice))
    _W["has_advice"] = has_advice


def _play_one(task):
    """Play one game; returns (score_for_A, turns, decisions, err)."""
    pair_idx, setup_seed, a_is_side1 = task
    from tools.scenario_pool import random_setup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    try:
        setup = random_setup(random.Random(setup_seed),
                             forced_faction=None)
        sim = WesnothSim(build_scenario_gamestate(setup),
                         scenario_id=setup.scenario_id,
                         max_turns=_W["max_turns"])
        arm_of_side = {1: "A" if a_is_side1 else "B",
                       2: "B" if a_is_side1 else "A"}
        label = f"p{pair_idx}s{int(a_is_side1)}"
        steps = 0
        while not sim.done and steps < 1600:
            side = sim.gs.global_info.current_side
            pol = _W[arm_of_side.get(side, "A")]
            act = pol.select_action(copy.deepcopy(sim.gs),
                                    game_label=label, sim=sim)
            if act is None:
                break
            sim.step(act)
            steps += 1
        for k in ("A", "B"):
            _W[k].drop_pending(label)
        w = int(sim.winner or 0)
        if w == 0:
            score = 0.5
        else:
            score = 1.0 if arm_of_side[w] == "A" else 0.0
        return (score, int(sim.gs.global_info.turn_number), steps, None)
    except Exception as e:                                # noqa: BLE001
        return (None, 0, 0, f"{type(e).__name__}: {e}")


def _init_wrapper(ckpt, sims, cap, a_adv, b_adv, graft, max_turns):
    _worker_init(ckpt, sims, cap, a_adv, b_adv, graft)
    _W["max_turns"] = max_turns


def _elo(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--games", type=int, default=400,
                    help="total games (played as side-swapped pairs)")
    ap.add_argument("--sims", type=int, default=8,
                    help="MCTS sims per decision (default 8; 32 will "
                         "not fit the 30-min box budget at 400 games)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = cpu_count - 4")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--draw-tiebreak-cap", type=float, default=0.3,
                    help="search-internal only; never enters scoring")
    ap.add_argument("--arm-a-advice", choices=("on", "off"),
                    default="on")
    ap.add_argument("--arm-b-advice", choices=("on", "off"),
                    default="off")
    ap.add_argument("--force-advice-graft", action="store_true",
                    help="build the advice path (zero-init) even if "
                         "the checkpoint predates it -- validation "
                         "of the ON machinery; expected null")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    n_pairs = max(1, args.games // 2)
    workers = args.workers or max(1, (os.cpu_count() or 4) - 4)
    rng = random.Random(args.seed)
    tasks = []
    for i in range(n_pairs):
        s = rng.randint(0, 2**31 - 1)
        tasks.append((i, s, True))
        tasks.append((i, s, False))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    init_args = (str(args.checkpoint), args.sims, args.draw_tiebreak_cap,
                 args.arm_a_advice == "on", args.arm_b_advice == "on",
                 args.force_advice_graft, args.max_turns)
    scores, turns, errs = [], [], []
    with ctx.Pool(workers, initializer=_init_wrapper,
                  initargs=init_args) as pool:
        for score, t, steps, err in pool.imap_unordered(_play_one, tasks):
            if err is not None:
                errs.append(err)
                continue
            scores.append(score)
            turns.append(t)
            done = len(scores) + len(errs)
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} games "
                      f"({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0

    n = len(scores)
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = n - wins - draws
    p = (sum(scores) / n) if n else float("nan")
    se = (statistics.pstdev(scores) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = p - 1.96 * se, p + 1.96 * se
    result = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "arm_a_advice": args.arm_a_advice,
        "arm_b_advice": args.arm_b_advice,
        "force_advice_graft": bool(args.force_advice_graft),
        "sims": args.sims, "max_turns": args.max_turns,
        "games": n, "errors": len(errs),
        "W_D_L_for_A": [wins, draws, losses],
        "score_A": p, "score_se": se,
        "score_ci95": [lo, hi],
        "elo_A": _elo(p) if n else None,
        "elo_ci95": [_elo(max(lo, 1e-6)), _elo(min(hi, 1 - 1e-6))]
                    if n else None,
        "mean_turns": (sum(turns) / n) if n else None,
        "runtime_s": dt,
        "measures": "ACTING-channel effect of MCTSConfig.advice on "
                    "identical weights (head-to-head, side-swapped "
                    "ladder pool; pure-outcome scoring).",
        "does_not_measure": "Trunk shaping through training: advice-"
                            "routed gradients already in the shared "
                            "weights benefit both arms equally.",
        "err_samples": errs[:5],
    }
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
