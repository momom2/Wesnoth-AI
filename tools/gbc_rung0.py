"""GBC rung-0b/0c: the two measurements that can kill the design
before any model code is written (docs/gbc_spec.md par.6).

0b -- DECOMPOSITION R^2. Is the turn-scale value differential
approximately linear in goal-event indicators? Held-out regression
of dV (value head at consecutive same-side turn-boundary anchors)
onto fog-censored event counts {dies_own, dies_enemy, flips_gained,
flips_lost, levels_own, levels_enemy}. PROCEED >= 0.25; ABANDON the
w/completion component below 0.15. This fitted vector is also the
frozen config-table `w` of review amendment A3.

0c -- PREMISE TEST (corrected form, review #7). On sampled
decision points from raw-policy self-play, sim-step every legal
action and compare Var_a[dV] against the value head's own noise
scale (C51 spread / cliffness) -- NOT against the atom width -- and
against the exact one-turn kill-probability differential from the
combat DP. PROCEED if the exact-kill differential is >= 5x the value
differential in normalized terms; ABANDON if Var_a[dV] is
comfortably above the value head's own noise scale.

    python tools/gbc_rung0.py 0b --games 120
    python tools/gbc_rung0.py 0c --states 60
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.gbc_labels import scan_game  # noqa: E402

log = logging.getLogger("gbc_rung0")

# The spec's w is keyed predicate x unit-type; cost-weighted counts
# + leader flags are the tractable regression form of that basis
# (unit cost is the type's scalar summary; the leader indicator is
# the game-ending special case). The coarse count-only basis
# measured R2=0.112 (2026-08-14, 120 games) -- below the 0.15 kill
# line -- so this richer basis is the deciding run.
FEATURES = ("dies_own", "dies_enemy", "flips_gained", "flips_lost",
            "levels_own", "levels_enemy",
            "dies_own_cost", "dies_enemy_cost",
            "dies_own_leader", "dies_enemy_leader")


def _load_policy(ckpt: Path, device: str = "cpu"):
    from tools.turn_counterfactual_probe import load_policy
    return load_policy(ckpt, device=device)


def _value(policy, gs, side: int) -> float:
    from tools.turn_search import forward_state, _value_for
    _, output, _ = forward_state(policy, gs, 0)
    return _value_for(output, gs, side)


# ---------------------------------------------------------------------
# 0b
# ---------------------------------------------------------------------

def _event_features(events, seq_lo: int, seq_hi: int,
                    side: int) -> np.ndarray:
    """Fog-censored event counts in (seq_lo, seq_hi], from `side`'s
    view -- only events `side` observed count (the head can only
    learn what its observer could see)."""
    f = dict.fromkeys(FEATURES, 0.0)
    for e in events:
        if not (seq_lo < e.seq <= seq_hi) or side not in e.observed_by:
            continue
        if e.predicate == "dies":
            who = "own" if e.entity_side == side else "enemy"
            f[f"dies_{who}"] += 1
            f[f"dies_{who}_cost"] += e.cost / 20.0   # ~grunt-cost unit
            f[f"dies_{who}_leader"] += float(e.is_leader)
        elif e.predicate == "levels":
            f["levels_own" if e.entity_side == side
              else "levels_enemy"] += 1
        elif e.predicate == "flips":
            if e.entity_side == side:
                f["flips_gained"] += 1
            if e.prev_side == side:
                f["flips_lost"] += 1
    return np.array([f[k] for k in FEATURES])


def _winner_map() -> Dict[str, int]:
    """game key (date_stem) -> winner side, from the certified
    outcomes ledger."""
    import gzip
    out: Dict[str, int] = {}
    p = Path("training/logs/replay_outcomes.jsonl.gz")
    if not p.exists():
        return out
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = int(e.get("winner_side") or 0)
            if w not in (1, 2):
                continue
            raw = Path(str(e.get("path", "")).replace("\\", "/"))
            out[f"{raw.parent.name}_{raw.stem}"] = w
    return out


def _collect(args, policy, winner_map: Optional[Dict[str, int]] = None):
    """Shared 0b/0d row collection: per same-side consecutive-turn
    window -- event features, dV from the value head, V at the
    window start, and (when winners are known) z from that side's
    perspective. Cumulative event features accumulate from game
    start through the window end (the events-vs-outcome test needs
    position-to-date, not one window)."""
    dataset = Path(args.dataset)
    files = sorted(dataset.glob("*.json.gz"))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[:args.games]

    X, Xc, Y, V0, Z, game_of = [], [], [], [], [], []
    matched = 0
    for gi, gz in enumerate(files):
        values: List[Tuple[int, int, int, float]] = []  # side,turn,seq,V

        def on_anchor(gs, anchor):
            values.append((anchor.side, anchor.turn, anchor.seq,
                           _value(policy, gs, anchor.side)))

        try:
            scan = scan_game(gz, on_anchor=on_anchor)
        except Exception as e:  # noqa: BLE001
            log.warning(f"{gz.name}: {e!r}")
            continue
        win = None
        if winner_map is not None:
            win = winner_map.get(gz.name[:-len(".json.gz")])
            if win is not None:
                matched += 1
        by_side: Dict[int, List] = defaultdict(list)
        for side, turn, seq, v in values:
            by_side[side].append((turn, seq, v))
        for side, rows in by_side.items():
            for (t0, s0, v0), (t1, s1, v1) in zip(rows, rows[1:]):
                if t1 != t0 + 1:
                    continue      # skip gaps (skipped turns)
                X.append(_event_features(scan.events, s0, s1, side))
                Xc.append(_event_features(scan.events, -1, s1, side))
                Y.append(v1 - v0)
                V0.append(v0)
                Z.append(0.0 if win is None
                         else (1.0 if win == side else -1.0))
                game_of.append(gi)
        if (gi + 1) % 20 == 0:
            log.info(f"collect: {gi + 1}/{len(files)}, {len(Y)} rows")
    if winner_map is not None:
        log.info(f"winner ledger matched {matched}/{len(files)} games")
    return (np.array(X), np.array(Xc), np.array(Y), np.array(V0),
            np.array(Z), np.array(game_of))


def rung0b(args) -> Dict:
    policy = _load_policy(args.checkpoint, args.device)
    X, _Xc, Y, _V0, _Z, game_of = _collect(args, policy)
    n_games = len(set(game_of.tolist()))
    holdout = game_of % 5 == 0          # ~20% of games held out
    Xt, Yt, Xh, Yh = X[~holdout], Y[~holdout], X[holdout], Y[holdout]

    def _fit(Xa, Ya):
        A = np.hstack([Xa, np.ones((len(Xa), 1))])
        w, *_ = np.linalg.lstsq(A, Ya, rcond=None)
        return w

    w = _fit(Xt, Yt)
    Ah = np.hstack([Xh, np.ones((len(Xh), 1))])
    resid = Yh - Ah @ w
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((Yh - Yh.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    out = {
        "rows": int(len(Y)), "games": n_games,
        "holdout_rows": int(holdout.sum()),
        "r2_holdout": round(r2, 4),
        "w": {k: round(float(v), 5) for k, v in
              zip(FEATURES + ("bias",), w)},
        "dv_std": round(float(Y.std()), 4),
    }
    print(json.dumps(out, indent=2))
    verdict = ("PROCEED" if r2 >= 0.25 else
               "MARGINAL (w component weak)" if r2 >= 0.15 else
               "ABANDON w/completion component")
    print(f"0b gate (>=0.25 proceed, <0.15 abandon w): R2={r2:.3f} "
          f"-> {verdict}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2),
                                  encoding="utf-8")
    return out


# ---------------------------------------------------------------------
# 0d -- the Gate-2 attribution test (user challenge 2026-08-14):
# is the low 0b R^2 the fault of EVENTS (they under-determine value)
# or of the VALUE HEAD (its movement is miscalibrated noise)?
# Three outcome-AUCs on held-out games decide:
#   events(cumulative) -> z : do events predict the true outcome?
#   V(anchor)          -> z : does the head predict it?
#   dV residual        -> z : does the head's event-orthogonal
#                             movement carry ANY outcome info?
# User's hypothesis (head miscalibrated) predicts: events->z decent,
# residual->z ~ 0.5. "Events don't matter" predicts events->z weak.
# "Head sees real non-event signal" predicts residual->z > 0.5.
# ---------------------------------------------------------------------

def rung0d(args) -> Dict:
    from tools.gbc_heads import auc as _auc
    policy = _load_policy(args.checkpoint, args.device)
    wm = _winner_map()
    if not wm:
        raise SystemExit("outcomes ledger missing")
    X, Xc, Y, V0, Z, game_of = _collect(args, policy, winner_map=wm)
    known = Z != 0.0
    X, Xc, Y, V0, Z, game_of = (a[known] for a in
                                (X, Xc, Y, V0, Z, game_of))
    holdout = game_of % 5 == 0
    y01 = (Z > 0).astype(int)

    def _fit(Xa, Ya):
        A = np.hstack([Xa, np.ones((len(Xa), 1))])
        w, *_ = np.linalg.lstsq(A, Ya, rcond=None)
        return w

    # (1) cumulative events -> z (linear score is AUC-sufficient)
    w_ev = _fit(Xc[~holdout], Z[~holdout])
    score_ev = np.hstack([Xc, np.ones((len(Xc), 1))])[holdout] @ w_ev
    auc_events = _auc(list(score_ev), list(y01[holdout]))
    # (2) the head's own value -> z
    auc_v = _auc(list(V0[holdout]), list(y01[holdout]))
    # (3) the head's event-orthogonal MOVEMENT -> z
    w_dv = _fit(X[~holdout], Y[~holdout])
    resid = Y - np.hstack([X, np.ones((len(X), 1))]) @ w_dv
    auc_resid = _auc(list(resid[holdout]), list(y01[holdout]))
    # Reference: window events alone -> z (weak by construction).
    w_w = _fit(X[~holdout], Z[~holdout])
    score_w = np.hstack([X, np.ones((len(X), 1))])[holdout] @ w_w
    auc_window = _auc(list(score_w), list(y01[holdout]))

    out = {
        "rows": int(known.sum()), "holdout_rows": int(holdout.sum()),
        "auc_events_cumulative": round(float(auc_events), 4),
        "auc_value_head": round(float(auc_v), 4),
        "auc_dv_residual": round(float(auc_resid), 4),
        "auc_events_window": round(float(auc_window), 4),
    }
    print(json.dumps(out, indent=2))
    print("reading: events>>0.5 & residual~0.5 -> HEAD MISCALIBRATED "
          "(user hypothesis); events~0.5 -> events under-determine; "
          "residual>>0.5 -> head sees real non-event signal")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2),
                                  encoding="utf-8")
    return out


# ---------------------------------------------------------------------
# 0c
# ---------------------------------------------------------------------

def rung0c(args) -> Dict:
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate,
    )
    from tools.turn_search import forward_state, _value_for
    from tools.wesnoth_sim import WesnothSim
    import torch  # noqa: F401

    policy = _load_policy(args.checkpoint, args.device)
    rng_py = random.Random(args.seed)
    rng = np.random.default_rng(args.seed)

    var_dv: List[float] = []
    noise: List[float] = []
    var_kill: List[float] = []
    states_done = 0
    g = 0
    while states_done < args.states and g < args.states:  # 1 state/game
        g += 1
        setup = random_setup(rng_py, category=args.category)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        # Roll the raw policy to a random depth, then measure there.
        depth = int(rng.integers(5, args.max_depth))
        for _ in range(depth):
            if sim.done:
                break
            _, output, legal = forward_state(policy, sim.gs, 0)
            if not legal:
                break
            p = np.array([max(a.prior, 0.0) for a in legal])
            p = p / p.sum() if p.sum() > 0 else None
            idx = int(rng.choice(len(legal), p=p)) if p is not None \
                else int(rng.integers(len(legal)))
            try:
                sim.step(legal[idx].action)
            except Exception:  # noqa: BLE001
                break
        if sim.done:
            continue
        side = sim.gs.global_info.current_side
        _, output, legal = forward_state(policy, sim.gs, 0)
        if len(legal) < 4:
            continue
        v0 = _value_for(output, sim.gs, side)
        # The value head's own per-state noise scale: C51 spread.
        cliff = float(output.cliffness.squeeze().item()) \
            if getattr(output, "cliffness", None) is not None else 0.0
        acts = legal if len(legal) <= args.max_actions else \
            [legal[i] for i in rng.choice(len(legal),
                                          args.max_actions,
                                          replace=False)]
        dvs, kills = [], []
        for a in acts:
            f = sim.fork()
            f._seed_salt = f"gbc0c:{g}"
            try:
                f.step(a.action)
            except Exception:  # noqa: BLE001
                continue
            _, out2, _ = forward_state(policy, f.gs, 0)
            dvs.append(_value_for(out2, f.gs, side) - v0)
            pk = 0.0
            if a.action.get("type") == "attack":
                try:
                    from tools.combat_outcomes import (
                        enumerate_attack_outcomes,
                    )
                    enum = enumerate_attack_outcomes(
                        sim.gs, a.action, advancement_choice="uniform")
                    if enum is not None:
                        # OutcomeKey = (a_hp, d_hp, ...); defender
                        # dead <=> d_hp == 0 (canonicalized).
                        pk = float(sum(
                            m for key, m in enum.probs.items()
                            if key[1] <= 0))
                except Exception:  # noqa: BLE001
                    pk = 0.0
            kills.append(pk)
        if len(dvs) < 4:
            continue
        var_dv.append(float(np.std(dvs)))
        noise.append(cliff)
        var_kill.append(float(np.std(kills)))
        states_done += 1
        if states_done % 10 == 0:
            log.info(f"0c: {states_done}/{args.states}")
    out = {
        "states": states_done,
        "median_std_dv": round(float(np.median(var_dv)), 4),
        "median_cliffness": round(float(np.median(noise)), 4),
        "median_std_kill": round(float(np.median(var_kill)), 4),
        "ratio_kill_over_dv": round(
            float(np.median(var_kill))
            / max(float(np.median(var_dv)), 1e-9), 2),
    }
    print(json.dumps(out, indent=2))
    print(f"0c gate: exact-kill differential >= 5x value "
          f"differential: {out['ratio_kill_over_dv'] >= 5.0}; "
          f"dV above V's own noise scale (premise FALSE) : "
          f"{out['median_std_dv'] > 2 * out['median_cliffness']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2),
                                  encoding="utf-8")
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rung", choices=("0b", "0c", "0d"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/checkpoints/"
                                 "imit_tierb_start.pt"))
    ap.add_argument("--dataset", type=Path,
                    default=Path("replays_dataset_imitation"))
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=120)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--category", default="ladder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s "
                               "%(message)s", datefmt="%H:%M:%S")
    {"0b": rung0b, "0c": rung0c, "0d": rung0d}[args.rung](args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
