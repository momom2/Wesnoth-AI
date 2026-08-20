"""E1 frozen-state policy-shape probe (leg-4 root-cause plan,
docs/leg4_erosion_rootcause_20260820.md §3).

Scores N checkpoints on IDENTICAL states -- the measurement leg 4
never took: every in-leg statistic averaged over the leg's own
drifting self-play states, so weights-drift and state-drift were
confounded. Here the state set is frozen (the human-probe holdout
stream, same deterministic order as supervised_train._evaluate), so
any between-checkpoint difference is pure weights.

Per state x checkpoint, from the REAL enumeration path
(action_sampler.enumerate_legal_actions_with_priors):
  n_legal, policy entropy H, H/log(n_legal), top-1 mass,
  top80 (top-1 > 0.8, the mcts_policy confident-decision gauge),
  p(end_turn), policy mass by action type vs the legal set's own
  type composition, and Spearman rho(actor mass, actor legal count)
  -- the p^lambda mobility-ranking signature.

Usage:
    python tools/policy_shape_probe.py \
        seed=training/checkpoints/seed_imit_tierb_start.pt \
        pin=training/checkpoints/2516k-b-294k-l4-495k.pt \
        [--dataset-dir replays_dataset_imitation] [--max-states 1200] \
        [--out training/metrics/policy_shape_YYYYMMDD.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

log = logging.getLogger("policy_shape_probe")


def _rankdata(xs):
    """Average ranks with ties (1-based)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def state_stats(policy, gs) -> dict | None:
    from wesnoth_ai.action_sampler import (
        enumerate_legal_actions_with_priors)
    with torch.no_grad():
        encoded = policy._encoder.encode(gs)
        output = policy._model(encoded)
        laps = enumerate_legal_actions_with_priors(encoded, output, gs)
    if not laps:
        return None
    ps = [max(lap.prior, 0.0) for lap in laps]
    z = sum(ps)
    if z <= 0:
        return None
    ps = [p / z for p in ps]
    n = len(laps)
    ent = -sum(p * math.log(p) for p in ps if p > 0)
    mass = defaultdict(float)
    legal = defaultdict(int)
    actor_mass = defaultdict(float)
    actor_n = defaultdict(int)
    for lap, p in zip(laps, ps):
        t = str(lap.action.get("type", "?"))
        mass[t] += p
        legal[t] += 1
        actor_mass[lap.actor_idx] += p
        actor_n[lap.actor_idx] += 1
    actors = sorted(actor_mass)
    rho = _spearman([actor_mass[a] for a in actors],
                    [float(actor_n[a]) for a in actors])
    return {
        "n_legal": n,
        "H": ent,
        "H_norm": ent / math.log(n) if n > 1 else 1.0,
        "top1": max(ps),
        "top80": max(ps) > 0.8,
        "p_end_turn": mass.get("end_turn", 0.0),
        "mass_by_type": dict(mass),
        "legal_by_type": dict(legal),
        "rho_actor_mobility": rho,
    }


def holdout_stream(dataset_dir: Path, max_states: int):
    """Same deterministic holdout stream _evaluate consumes."""
    from tools.supervised_train import _pair_stream_serial
    man_path = dataset_dir / "manifest.jsonl"
    holdout_names = {r["file"] for r in
                     (json.loads(line) for line in
                      man_path.open(encoding="utf-8"))
                     if r["holdout"]}
    files = sorted(p for p in dataset_dir.iterdir()
                   if p.name in holdout_names)
    n = 0
    for item in _pair_stream_serial(files):
        if item[0] != "pair":
            continue
        if n >= max_states:
            return
        n += 1
        yield item[1]


def aggregate(rows: list) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    type_mass = defaultdict(float)
    type_legal = defaultdict(int)
    for r in rows:
        for t, m in r["mass_by_type"].items():
            type_mass[t] += m
        for t, c in r["legal_by_type"].items():
            type_legal[t] += c
    n_states = len(rows)
    total_legal = sum(type_legal.values())
    return {
        "n_states": n_states,
        "n_legal_mean": mean("n_legal"),
        "H_mean": mean("H"),
        "H_norm_mean": mean("H_norm"),
        "top1_mean": mean("top1"),
        "top80_share": (sum(1 for r in rows if r["top80"]) / n_states
                        if n_states else None),
        "p_end_turn_mean": mean("p_end_turn"),
        "rho_actor_mobility_mean": mean("rho_actor_mobility"),
        "policy_mass_by_type": {t: m / n_states
                                for t, m in sorted(type_mass.items())},
        "legal_composition_by_type": {t: c / total_legal
                                      for t, c in sorted(type_legal.items())},
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+",
                    help="LABEL=CKPT_PATH (repeatable)")
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset_imitation"))
    ap.add_argument("--max-states", type=int, default=1200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=args.log_level)

    from tools.eval_sim import _load_policy
    policies = {}
    for spec in args.checkpoints:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"want LABEL=PATH, got {spec!r}")
        policies[label] = _load_policy(Path(path), args.device,
                                       label=label)

    per_ckpt = {lab: [] for lab in policies}
    t0 = time.time()
    n_states = 0
    for gs in holdout_stream(args.dataset_dir, args.max_states):
        row_ok = True
        rows = {}
        for lab, pol in policies.items():
            try:
                s = state_stats(pol, gs)
            except Exception as e:                    # noqa: BLE001
                log.debug(f"state skipped for {lab}: {e!r}")
                s = None
            if s is None:
                row_ok = False
                break
            rows[lab] = s
        # Paired design: a state counts only if EVERY checkpoint
        # scored it, so the sets stay identical.
        if not row_ok:
            continue
        for lab, s in rows.items():
            per_ckpt[lab].append(s)
        n_states += 1
        if n_states % 100 == 0:
            log.info(f"{n_states} states in {time.time()-t0:.0f}s")

    report = {lab: aggregate(rows) for lab, rows in per_ckpt.items()}
    print(json.dumps(report, indent=1, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True)
                            + "\n", encoding="utf-8")
        log.info(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
