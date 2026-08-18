"""Q3 luck probe (A7a, credit-assignment design review 2026-08-17):
does combat-dice luck explain a material share of game outcomes?

Replays human games (bit-exact reconstructor) and, at every attack,
compares the EXACT expected outcome (`enumerate_attack_outcomes`,
the combat-oracle DP) against what the recorded dice actually did.
The per-game sum of (actual - expected), signed from side 1's
perspective, is that game's luck score L -- an exactly-zero-mean
martingale increment sum. Two covariates:

  L_hp   -- HP differential luck (all attrition counts)
  L_cost -- kill luck weighted by unit gold cost (the material-
            margin analog; blind to non-lethal attrition)

PRE-REGISTERED DECISION RULE (docs/credit_assignment_design_20260817.md
Q3; fixed BEFORE first full run):
  - out-of-sample rho^2 < 0.05 on decisive games for BOTH
    covariates  -> luck does not decide these games; the luck-ledger
    proposal dies for free.
  - rho^2 >= 0.15 AND induced label shrinkage < 10%
    -> ship the ledger in the restart leg (with widened C51
    support, restart-time only).
  - in between -> judgment call, back to the user.

Also reported: attack coverage (the DP bails on advancement-
possible fights and complexity caps -- the bail rate bounds how
much luck the ledger could even see), fitted alpha, variance
reduction Var(z - alpha*L)/Var(z).

Usage:
    python tools/luck_probe.py [--games 500] [--dataset
        replays_dataset_imitation] [--jobs 6] [--out PATH.csv]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("luck_probe")


def _unit_cost(u) -> float:
    c = getattr(u, "cost", None)
    if c:
        return float(c)
    try:
        from tools.replay_dataset import _stats_for
        return float(_stats_for(u.type_name).get("cost", 0) or 0)
    except Exception:                                   # noqa: BLE001
        return 0.0


def probe_game(task) -> Optional[Dict]:
    """Replay one game; return its luck record or None on failure.
    `task` = (gz_path, winner_side) -- winners live in the dataset
    MANIFEST (manifest.jsonl), not in the game records."""
    gz_path, winner = task
    from tools.combat_outcomes import enumerate_attack_outcomes
    from tools.replay_dataset import (
        _apply_command, _build_initial_gamestate, _find_unit_at,
        _setup_scenario_events,
    )
    try:
        if winner not in (1, 2):
            return None
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        gs = _build_initial_gamestate(data)
        _setup_scenario_events(gs, data.get("scenario_id", ""))
    except Exception as e:                              # noqa: BLE001
        log.warning(f"{gz_path.name}: setup failed: {e!r}")
        return None

    l_hp = l_cost = 0.0
    n_measured = n_bailed = 0
    for cmd in data.get("commands", []):
        kind = cmd[0] if cmd else "?"
        if kind == "attack":
            ax, ay, dx, dy, a_weapon = (cmd[1], cmd[2], cmd[3],
                                        cmd[4], cmd[5])
            seed_hex = cmd[7] if len(cmd) > 7 else ""
            att = _find_unit_at(gs, ax, ay)
            dfd = _find_unit_at(gs, dx, dy)
            dist = None
            if att is not None and dfd is not None and seed_hex:
                try:
                    dist = enumerate_attack_outcomes(
                        gs,
                        {"type": "attack",
                         "start_hex": SimpleNamespace(x=ax, y=ay),
                         "target_hex": SimpleNamespace(x=dx, y=dy),
                         "attack_index": a_weapon},
                        advancement_choice=None)
                except Exception:                       # noqa: BLE001
                    dist = None
            if dist is None:
                if att is not None and dfd is not None and seed_hex:
                    n_bailed += 1
                # Apply and move on: unmeasured luck.
                try:
                    _apply_command(gs, cmd)
                except Exception as e:                  # noqa: BLE001
                    log.warning(f"{gz_path.name}: apply failed mid-"
                                f"game: {e!r}")
                    return None
                continue
            a_hp0, d_hp0 = att.current_hp, dfd.current_hp
            a_cost, d_cost = _unit_cost(att), _unit_cost(dfd)
            e_hp = e_cost = 0.0
            for key, p in dist.probs.items():
                a_hp, d_hp = key[0], key[1]
                e_hp += p * ((d_hp0 - d_hp) - (a_hp0 - a_hp))
                e_cost += p * (d_cost * (d_hp <= 0)
                               - a_cost * (a_hp <= 0))
            try:
                _apply_command(gs, cmd)
            except Exception as e:                      # noqa: BLE001
                log.warning(f"{gz_path.name}: apply failed mid-game: "
                            f"{e!r}")
                return None
            post_a = next((u for u in gs.map.units
                           if u.id == dist.attacker_id), None)
            post_d = next((u for u in gs.map.units
                           if u.id == dist.defender_id), None)
            a_hp1 = post_a.current_hp if post_a is not None else 0
            d_hp1 = post_d.current_hp if post_d is not None else 0
            act_hp = (d_hp0 - d_hp1) - (a_hp0 - a_hp1)
            act_cost = (d_cost * (post_d is None)
                        - a_cost * (post_a is None))
            sign = 1.0 if att.side == 1 else -1.0
            l_hp += sign * (act_hp - e_hp)
            l_cost += sign * (act_cost - e_cost)
            n_measured += 1
        else:
            try:
                _apply_command(gs, cmd)
            except Exception as e:                      # noqa: BLE001
                log.warning(f"{gz_path.name}: apply failed mid-game: "
                            f"{e!r}")
                return None
    return {
        "file": gz_path.name,
        "z1": 1.0 if winner == 1 else -1.0,
        "l_hp": l_hp,
        "l_cost": l_cost,
        "n_measured": n_measured,
        "n_bailed": n_bailed,
    }


def _cv_rho2(z, lv, folds: int = 5, seed: int = 0
             ) -> Tuple[float, float]:
    """Out-of-sample rho^2 for z ~ alpha*L (5-fold CV) and the
    full-sample fitted alpha (per unit L, unstandardized)."""
    import numpy as np
    z = np.asarray(z, dtype=np.float64)
    lv = np.asarray(lv, dtype=np.float64)
    n = len(z)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    preds = np.zeros(n)
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        lt = lv[train]
        var = float(lt.var())
        a = (float(((lt - lt.mean()) * (z[train] - z[train].mean()))
                   .mean()) / var) if var > 0 else 0.0
        b = float(z[train].mean() - a * lt.mean())
        preds[test] = a * lv[test] + b
    ss_res = float(((z - preds) ** 2).sum())
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    var_l = float(lv.var())
    alpha = (float(((lv - lv.mean()) * (z - z.mean())).mean())
             / var_l) if var_l > 0 else 0.0
    return r2_oos, alpha


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--dataset", type=Path,
                    default=Path("replays_dataset_imitation"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path,
                    default=Path("training/metrics/luck_probe.csv"))
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    manifest = {}
    mpath = args.dataset / "manifest.jsonl"
    for line in mpath.open(encoding="utf-8"):
        m = json.loads(line)
        if m.get("winner_side") in (1, 2) and not m.get("holdout"):
            manifest[m["file"]] = int(m["winner_side"])
    files = [(args.dataset / f, w) for f, w in sorted(manifest.items())
             if (args.dataset / f).exists()]
    if not files:
        print(f"no manifest winners under {args.dataset}")
        return 2
    rng = random.Random(args.seed)
    sample = rng.sample(files, min(args.games, len(files)))
    print(f"probing {len(sample)}/{len(files)} games "
          f"({args.jobs} jobs)")

    rows: List[Dict] = []
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for r in pool.imap_unordered(probe_game, sample,
                                         chunksize=4):
                if r is not None:
                    rows.append(r)
                    if len(rows) % 50 == 0:
                        print(f"  {len(rows)} games done")
    else:
        for p in sample:
            r = probe_game(p)
            if r is not None:
                rows.append(r)

    if len(rows) < 30:
        print(f"only {len(rows)} usable games -- refusing to fit")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import numpy as np
    z = np.array([r["z1"] for r in rows])
    meas = np.array([r["n_measured"] for r in rows], dtype=float)
    bail = np.array([r["n_bailed"] for r in rows], dtype=float)
    tot = meas + bail
    print(f"\ngames used: {len(rows)} | attacks/game "
          f"mean {tot.mean():.1f} | DP coverage "
          f"{meas.sum():.0f}/{tot.sum():.0f} measured "
          f"(bail rate {bail.sum() / max(tot.sum(), 1):.3f})")
    print(f"outcome mix: s1 {int((z > 0).sum())} / "
          f"s2 {int((z < 0).sum())}")
    for name in ("l_hp", "l_cost"):
        lv = np.array([r[name] for r in rows])
        rho2, alpha = _cv_rho2(z, lv)
        corr = (float(np.corrcoef(z, lv)[0, 1])
                if lv.std() > 0 else 0.0)
        zc = z - alpha * lv
        print(f"\n{name}: OOS rho^2 = {rho2:+.4f} | in-sample "
              f"pearson {corr:+.4f} (rho^2 {corr * corr:.4f}) | "
              f"alpha {alpha:+.5f}")
        print(f"  var reduction Var(z-aL)/Var(z) = "
              f"{float(zc.var() / z.var()):.4f} | label shrinkage "
              f"mean|z-aL|/mean|z| = "
              f"{float(np.abs(zc).mean() / np.abs(z).mean()):.4f}")
    print(f"\nper-game rows: {args.out}")
    print("gates (pre-registered): rho^2 < 0.05 both -> ledger dies; "
          ">= 0.15 + shrinkage < 10% -> ship with widened C51 "
          "support; else -> user judgment call")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
