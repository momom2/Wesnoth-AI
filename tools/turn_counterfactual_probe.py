"""TCS rung 0-1 probe: turn-commitment counterfactual measurements.

The falsification instrument for Turn-Commitment Search
(docs/tcs_spec.md par.8). Offline, no training. The search core
(spine / materialization / acceptance / target transform) lives in
`tools/turn_search.py` and is SHARED with the production
`TurnCommitPolicy` -- this file adds only the measurement harness:
placebo arm, Gumbel-baseline comparison, JSONL emission, and the
pre-registered collate.

Result 2026-08-14 (300 ladder states, imit_tierb_start +
tier_b_handoff_f1): revalidated accept 0.640/0.460, median accepted
delta 0.070/0.106, placebo 0.130/0.180, KL(matched) 0.184/0.383 vs
gumbel 0.343/0.463, rho(delta,survival) 0.016/0.061. User ruling:
TCS approved for production integration despite the KL gate
(perturbation-magnitude proxy) failing as pre-registered.

Run (full probe belongs on a rented many-core CPU box per the
standing eval rule; shard with --seed/--out and collate together):

    python tools/turn_counterfactual_probe.py \
        --checkpoint training/checkpoints/imit_tierb_start.pt \
        --games 40 --states 200 --category ladder \
        --out training/logs/tcs_probe/probe_s0.jsonl --seed 0

    python tools/turn_counterfactual_probe.py --collate \
        training/logs/tcs_probe/*.jsonl
"""
from __future__ import annotations

import argparse
import copy
import glob as _glob
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

import torch  # noqa: E402

from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from tools.mcts import (  # noqa: E402
    MCTSConfig, extract_gumbel_policy_target, mcts_search,
)
from tools.turn_search import (  # noqa: E402
    Materialized, gumbel_top_k_alternatives, materialize, record_spine,
    tcs_target_distribution, two_stage_accept,
)
from tools.wesnoth_sim import WesnothSim  # noqa: E402

__all__ = ["ProbeConfig", "probe_state", "tcs_target_kl", "spearman",
           "load_policy", "Materialized"]

log = logging.getLogger("turn_probe")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class ProbeConfig:
    n_alt:            int = 4      # alternatives per coordinate/round
    rounds:           int = 3      # hill-climb rounds (perturb+accept)
    reval_salts:      int = 3      # fresh salts in acceptance stage 2
    min_delta:        float = 0.01  # accept floor (float-jitter guard)
    max_spine:        int = 40     # hard cap on spine length
    matched_visits:   float = 16.0  # sigma span for the comparable KL
    lam:              float = 1.0  # prior discount in the KL target
    baseline:         str = "all"  # all | first | none
    baseline_sims:    int = 32
    placebo:          bool = True  # run the placebo arm per state
    mcts_cfg: MCTSConfig = field(default_factory=MCTSConfig)


def load_policy(ckpt: Path, device: str = "auto") -> TransformerPolicy:
    if device == "auto":
        # 2026-08-17: the A4 bake-off ran a full judge on CPU because
        # the launch omitted --device -- 26 ms forwards vs 3 ms.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = raw.get("arch", {})
    policy = TransformerPolicy(
        device=device,
        d_model=int(arch.get("d_model", 512)),
        num_layers=int(arch.get("num_layers", 6)),
        num_heads=int(arch.get("num_heads", 8)),
        d_ff=int(arch.get("d_ff", 2048)),
    )
    policy.load_checkpoint(ckpt)
    return policy


# ---------------------------------------------------------------------
# Probe-level metrics
# ---------------------------------------------------------------------

def tcs_target_kl(priors: np.ndarray, values: np.ndarray,
                  evaluated: np.ndarray, v_root: float,
                  max_visits: float, lam: float,
                  config: MCTSConfig, link: str = "exp") -> float:
    """KL(pi_TCS || prior) for one coordinate (the shared transform
    lives in `tcs_target_distribution`). Defaults to link="exp":
    this instrument's 2026-08-14 rung-1 baselines were measured
    under the exp transform and must stay comparable; pass
    link="linear" to measure the production leg-4 target."""
    tgt = tcs_target_distribution(priors, values, evaluated, v_root,
                                  max_visits, config, lam=lam,
                                  link=link)
    p = np.maximum(np.asarray(priors, dtype=np.float64), 1e-12)
    p = p / p.sum()
    return float((tgt * (np.log(tgt + 1e-12) - np.log(p))).sum())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy dependency (average ranks)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")

    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a), dtype=np.float64)
        vals, inv, cnt = np.unique(a, return_inverse=True,
                                   return_counts=True)
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        return sums[inv] / cnt[inv]

    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


# ---------------------------------------------------------------------
# One probed state
# ---------------------------------------------------------------------

def probe_state(policy: TransformerPolicy, sim0: WesnothSim, side: int,
                cfg: ProbeConfig, rng: np.random.Generator,
                state_id: str, placebo: bool = False) -> Dict:
    """Spine -> hill-climb rounds (two-stage acceptance) -> final KL.
    With `placebo=True`, the variant->value assignment is shuffled
    before the argmax each round (revalidation uses real values)."""
    ds = policy._decision_step
    spine, _ = record_spine(policy, sim0, side, ds, rng,
                            max_spine=cfg.max_spine)
    if not spine:
        return {"state_id": state_id, "empty": True}
    incumbent = [s.action for s in spine]
    steps = spine
    rounds_out: List[Dict] = []
    accepted_deltas: List[float] = []
    variant_tuples: List[Tuple[float, float, int, int]] = []

    for rnd in range(cfg.rounds):
        salt = f"probe:{state_id}:r{rnd}"
        inc = materialize(policy, sim0, side, incumbent, salt, ds)
        if inc.invalid:
            log.warning(f"{state_id}: incumbent materialization invalid")
            break
        cands: List[Tuple[int, int, Materialized]] = []
        for j, st in enumerate(steps):
            priors = np.array([a.prior for a in st.legal])
            et_idx = next((i for i, a in enumerate(st.legal)
                           if a.action.get("type") == "end_turn"), None)
            for alt_i in gumbel_top_k_alternatives(
                    priors, st.action_idx, et_idx, cfg.n_alt, rng):
                cand_cmds = (incumbent[:j]
                             + [st.legal[alt_i].action]
                             + incumbent[j + 1:])
                m = materialize(policy, sim0, side, cand_cmds, salt, ds)
                if m.invalid or math.isnan(m.value):
                    continue
                cands.append((j, alt_i, m))
                variant_tuples.append((
                    float(m.value - inc.value), m.survival,
                    int(m.stochastic),
                    int(m.vis_ids != inc.vis_ids)))
        if not cands:
            break
        deltas = np.array([m.value - inc.value for _, _, m in cands])
        sel = deltas.copy()
        if placebo:
            sel = rng.permutation(sel)
        best = int(np.argmax(sel))
        j, alt_i, best_m = cands[best]
        naive_delta = float(deltas[best])   # real delta of the pick
        naive_accept = float(sel[best]) > cfg.min_delta

        # Stage 2: paired re-evaluation at fresh salts. Deterministic
        # pairs replicate exactly; skip the redundant forwards.
        if not best_m.stochastic and not inc.stochastic:
            reval = np.array([naive_delta])
        else:
            best_cmds = (incumbent[:j]
                         + [steps[j].legal[alt_i].action]
                         + incumbent[j + 1:])
            reval_l = []
            for v in range(cfg.reval_salts):
                s2 = f"{salt}:v{v}"
                inc2 = materialize(policy, sim0, side, incumbent, s2, ds)
                var2 = materialize(policy, sim0, side, best_cmds, s2, ds)
                if inc2.invalid or var2.invalid:
                    continue
                reval_l.append(var2.value - inc2.value)
            reval = np.array(reval_l) if reval_l else np.array(
                [float("-inf")])
        accept, dbar, thr = two_stage_accept(reval, cfg.min_delta)
        rounds_out.append({
            "n_variants": len(cands), "naive_delta": naive_delta,
            "naive_accept": bool(naive_accept),
            "reval_mean": float(dbar), "reval_n": int(len(reval)),
            "threshold": float(thr), "accepted": bool(accept),
            "best_survival": best_m.survival,
        })
        if not accept:
            break
        accepted_deltas.append(float(dbar))
        # Grade-what-you-commit: the new incumbent is the MATERIALIZED
        # winner -- the commands that actually landed at the selection
        # salt, drops excluded (docs/tcs_spec.md par.5.1).
        incumbent = list(best_m.executed)
        steps, _ = record_spine(policy, sim0, side, ds, rng,
                                max_spine=cfg.max_spine,
                                actions=incumbent)
        if not steps:
            break
        incumbent = [s.action for s in steps]

    # Final KL pass over the final incumbent's coordinates: evaluate
    # one alternative set at a dedicated salt, no acceptance.
    kl_own: List[float] = []
    kl_matched: List[float] = []
    salt = f"probe:{state_id}:kl"
    inc = materialize(policy, sim0, side, incumbent, salt, ds)
    if not inc.invalid:
        for j, st in enumerate(steps):
            priors = np.array([a.prior for a in st.legal])
            et_idx = next((i for i, a in enumerate(st.legal)
                           if a.action.get("type") == "end_turn"), None)
            picks = gumbel_top_k_alternatives(
                priors, st.action_idx, et_idx, cfg.n_alt, rng)
            values = np.zeros(len(st.legal))
            evaluated = np.zeros(len(st.legal), dtype=bool)
            values[st.action_idx] = inc.value
            evaluated[st.action_idx] = True
            for alt_i in picks:
                cand = (incumbent[:j] + [st.legal[alt_i].action]
                        + incumbent[j + 1:])
                m = materialize(policy, sim0, side, cand, salt, ds)
                if m.invalid or math.isnan(m.value):
                    continue
                values[alt_i] = m.value
                evaluated[alt_i] = True
            n_ev = float(evaluated.sum())
            kl_own.append(tcs_target_kl(
                priors, values, evaluated, st.pre_value,
                n_ev, cfg.lam, cfg.mcts_cfg))
            kl_matched.append(tcs_target_kl(
                priors, values, evaluated, st.pre_value,
                cfg.matched_visits, cfg.lam, cfg.mcts_cfg))

    # Baseline arm: existing Gumbel search on the ORIGINAL spine's
    # pre-states (comparability: same states for both arms).
    baseline_kl: List[float] = []
    if cfg.baseline != "none" and not placebo:
        bcfg = MCTSConfig(n_simulations=cfg.baseline_sims)
        coords = spine if cfg.baseline == "all" else spine[:1]
        for st in coords:
            try:
                root = mcts_search(st.pre_fork, policy._inference_model,
                                   policy._inference_encoder, bcfg,
                                   decision_step=st.decision_step)
                extract_gumbel_policy_target(root, bcfg)
                stats = getattr(root, "_distill_stats", None) or {}
                if "kl_prior" in stats:
                    baseline_kl.append(float(stats["kl_prior"]))
            except Exception as e:  # noqa: BLE001
                log.warning(f"{state_id}: baseline search failed: {e!r}")

    return {
        "state_id": state_id, "placebo": placebo,
        "side": side,
        "turn": sim0.gs.global_info.turn_number,
        "K_spine": len(spine),
        "K_final": len(steps),
        "rounds": rounds_out,
        "accepted_deltas": accepted_deltas,
        "n_accepts": len(accepted_deltas),
        "variants": variant_tuples,     # (delta, survival, stoch, vis)
        "kl_own": kl_own, "kl_matched": kl_matched,
        "baseline_kl": baseline_kl,
    }


# ---------------------------------------------------------------------
# Game generation + state sampling
# ---------------------------------------------------------------------

def run_probe(args) -> int:
    from tools.scenario_pool import random_setup, build_scenario_gamestate

    cfg = ProbeConfig(
        n_alt=args.n_alt, rounds=args.rounds,
        reval_salts=args.reval_salts, min_delta=args.min_delta,
        baseline=args.baseline, baseline_sims=args.baseline_sims,
        placebo=not args.no_placebo,
    )
    rng_py = random.Random(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = load_policy(args.checkpoint, device=args.device)
    log.info(f"checkpoint {args.checkpoint} decision_step="
             f"{policy._decision_step}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f_out = out.open("w", encoding="utf-8")

    states_done = 0
    turn_lengths: List[int] = []
    total_actions = 0
    total_end_turns = 0
    t0 = time.time()
    for g in range(args.games):
        if states_done >= args.states:
            break
        setup = random_setup(rng_py, category=args.category)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        label = f"probe_g{g}"
        prev_turnside = None
        k_current = 0
        while not sim.done:
            gi = sim.gs.global_info
            turnside = (gi.turn_number, gi.current_side)
            if turnside != prev_turnside:
                if prev_turnside is not None:
                    turn_lengths.append(k_current)
                k_current = 0
                prev_turnside = turnside
                # Side-turn start: probe candidate.
                if (states_done < args.states
                        and rng.random() < args.turn_sample_prob):
                    sid = f"g{g}t{gi.turn_number}s{gi.current_side}"
                    rec = probe_state(policy, sim.fork(),
                                      gi.current_side, cfg, rng, sid)
                    f_out.write(json.dumps(rec) + "\n")
                    if cfg.placebo and not rec.get("empty"):
                        rec_p = probe_state(policy, sim.fork(),
                                            gi.current_side, cfg, rng,
                                            sid + "p", placebo=True)
                        f_out.write(json.dumps(rec_p) + "\n")
                    f_out.flush()
                    states_done += 1
                    log.info(f"probed {sid} ({states_done}/"
                             f"{args.states}, {time.time()-t0:.0f}s)")
            pre_state = copy.deepcopy(sim.gs)
            action = policy.select_action(pre_state, game_label=label,
                                          sim=sim)
            total_actions += 1
            k_current += 1
            if action.get("type") == "end_turn":
                total_end_turns += 1
            sim.step(action)
        policy.drop_pending(label)
        if k_current:
            turn_lengths.append(k_current)
        log.info(f"game {g}: winner={sim.winner} "
                 f"turns={sim.gs.global_info.turn_number} "
                 f"ended_by={sim.ended_by}")

    summary = {
        "summary": True, "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "category": args.category,
        "states": states_done,
        "rung0_K_mean": (float(np.mean(turn_lengths))
                         if turn_lengths else None),
        "rung0_K_median": (float(np.median(turn_lengths))
                           if turn_lengths else None),
        "rung0_end_turn_share": (total_end_turns / total_actions
                                 if total_actions else None),
        "total_actions": total_actions,
        "wall_seconds": round(time.time() - t0, 1),
    }
    f_out.write(json.dumps(summary) + "\n")
    f_out.close()
    log.info(f"wrote {out}: {states_done} states, "
             f"K_mean={summary['rung0_K_mean']}, "
             f"end_turn_share={summary['rung0_end_turn_share']}")
    return 0


# ---------------------------------------------------------------------
# Collate: the pre-registered decision rules (docs/tcs_spec.md par.8)
# ---------------------------------------------------------------------

def collate(paths: List[str]) -> int:
    real, plac, summaries = [], [], []
    for pattern in paths:
        for p in sorted(_glob.glob(pattern)):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("summary"):
                        summaries.append(rec)
                    elif rec.get("empty"):
                        continue
                    elif rec.get("placebo"):
                        plac.append(rec)
                    else:
                        real.append(rec)
    if not real:
        print("no records")
        return 1

    def _accept_rate(recs):
        return sum(1 for r in recs if r["n_accepts"] > 0) / len(recs)

    def _naive_rate(recs):
        return sum(1 for r in recs
                   if any(rd["naive_accept"] for rd in r["rounds"])
                   ) / len(recs)

    acc_real = _accept_rate(real)
    naive_real = _naive_rate(real)
    acc_plac = _accept_rate(plac) if plac else float("nan")
    all_acc = [d for r in real for d in r["accepted_deltas"]]
    med_delta = float(np.median(all_acc)) if all_acc else float("nan")
    kl_m = [k for r in real for k in r["kl_matched"]]
    kl_o = [k for r in real for k in r["kl_own"]]
    base = [k for r in real for k in r["baseline_kl"]]
    var = np.array([t for r in real for t in r["variants"]])
    rho = spearman(var[:, 0], var[:, 1]) if len(var) else float("nan")
    vis_mask = var[:, 3] > 0 if len(var) else np.array([])

    print(f"states: real={len(real)} placebo={len(plac)}")
    print(f"accept_rate (revalidated): {acc_real:.3f}   "
          f"naive: {naive_real:.3f}   placebo: {acc_plac:.3f}")
    print(f"median accepted delta: {med_delta:.4f}  (n={len(all_acc)})")
    print(f"KL per coordinate: own median="
          f"{np.median(kl_o) if kl_o else float('nan'):.4f}  "
          f"matched median="
          f"{np.median(kl_m) if kl_m else float('nan'):.4f}  "
          f"gumbel baseline median="
          f"{np.median(base) if base else float('nan'):.4f} "
          f"(n={len(base)})")
    print(f"rho(delta, survival) pooled: {rho:.3f}  "
          f"(n_variants={len(var)})")
    if len(var) and vis_mask.any() and (~vis_mask).any():
        print(f"fog split: mean delta vis-changed="
              f"{var[vis_mask, 0].mean():.4f} (n={int(vis_mask.sum())})"
              f"  unchanged={var[~vis_mask, 0].mean():.4f} "
              f"(n={int((~vis_mask).sum())})")
    for s in summaries:
        print(f"rung0[{s.get('seed')}]: K_mean={s['rung0_K_mean']} "
              f"K_median={s['rung0_K_median']} "
              f"end_turn_share={s['rung0_end_turn_share']}")

    # Pre-registered verdict (advisory print; ruling is the user's).
    print("--- pre-registered gates (docs/tcs_spec.md par.8) ---")
    kl_gate = (bool(kl_m) and bool(base)
               and np.median(kl_m) >= np.median(base))
    print(f"PROCEED needs: accept>=0.50 [{acc_real >= 0.5}], "
          f"median delta>=0.05 [{med_delta >= 0.05}], "
          f"KL(matched)>=baseline [{kl_gate}], "
          f"placebo < half real [{acc_plac < acc_real / 2}]")
    if not math.isnan(acc_plac) and acc_plac >= acc_real / 2:
        print("STOP signal: noise-climbing (placebo >= half real)")
    if naive_real >= 2 * max(acc_real, 1e-9):
        print("STOP signal: naive accepts >= 2x revalidated")
    return 0


# ---------------------------------------------------------------------

def main(argv) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collate", nargs="*", default=None,
                    metavar="JSONL",
                    help="Collate mode: compute the pre-registered "
                         "decision rules from probe JSONL files "
                         "(globs ok).")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/checkpoints/"
                                 "imit_tierb_start.pt"))
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--category", default="ladder",
                    choices=["ladder", "fogless", "mini"])
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--turn-sample-prob", type=float, default=0.15,
                    help="Probability a side-turn start gets probed.")
    ap.add_argument("--n-alt", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reval-salts", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=0.01)
    ap.add_argument("--baseline", default="all",
                    choices=["all", "first", "none"])
    ap.add_argument("--baseline-sims", type=int, default=32)
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cpu", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("training/logs/tcs_probe/probe.jsonl"))
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    if args.collate is not None:
        return collate(args.collate)
    return run_probe(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
