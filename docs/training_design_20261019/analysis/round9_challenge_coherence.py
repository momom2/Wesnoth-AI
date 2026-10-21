"""Round 9, challenge stance: what a game-level (coherent) experiment buys over
per-decision coins, read from batch 1's records (seed section 7).

Inputs (read-only):
  training/metrics/near_tie_coins_20261020/rows_b0*.jsonl   (60,000 games: z, the seven luck sums S)
  training/metrics/near_tie_coins_20261020/readout_b_offline.json  (beta, per-type label means and SEs)

Outputs: printed; saved beside this script as .out.

1. The result's residual variance sigma_R^2 = E[(z - beta.S)^2] over batch 1's
   decisive games (beta as the readout froze it), and with a setup-cell mean
   removed, cross-fitted over two folds (cells: scenario x ordered faction pair).
2. The coin labels' second moment V_T implied by the readout's pooled SE.
3. For a fixed rule that switches n decisions per side-game, the games a
   per-decision coin design needs for a given SE on the rule's total effect,
   against a game-level design that applies the rule to one seat (one-sided)
   or opposite perturbations to the two seats (antithetic).
4. The measured instance: the in-band attack-promoting switches (types whose
   runner-up is an attack) read by batch 1's coins, against S0(a)'s
   attack +1 seat at 6,400 games (seed section 7).

numpy and the standard library only.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "training" / "metrics" / "near_tie_coins_20261020"
ELO_PER_Z = 347.4  # 400 / ln 10 / 2: Elo per unit of mean result z in {-1, +1} near 0

COLUMNS = ["hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200"]


def load_rows():
    zs, Ss, cells = [], [], []
    for path in sorted(REC.glob("rows_b0*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("decisive"):
                    continue
                zs.append(float(r["z"]))
                Ss.append([float(r["S"][c]) for c in COLUMNS])
                cells.append((r["scenario_id"], r["faction_a"], r["faction_b"]))
    return np.array(zs), np.array(Ss), cells


def residual_variance(z, S, beta, cells):
    R = z - S @ beta
    out = {"games": int(len(z)), "var_z": float(np.mean(z ** 2)), "E_R2_beta": float(np.mean(R ** 2))}
    # Cross-fitted cell mean: fold by game index parity.
    idx = np.arange(len(z))
    resid = np.empty_like(R)
    for fold in (0, 1):
        train = idx % 2 != fold
        sums, counts = defaultdict(float), defaultdict(int)
        for i in idx[train]:
            sums[cells[i]] += R[i]
            counts[cells[i]] += 1
        for i in idx[~train]:
            c = cells[i]
            b = sums[c] / counts[c] if counts[c] else 0.0
            resid[i] = R[i] - b
    out["E_R2_beta_cell_xfit"] = float(np.mean(resid ** 2))
    out["cells"] = len(set(cells))
    return out


def rule_games(n_per_side_game, V_T, sigma_R2, se_elo):
    """Games for an SE of se_elo on a rule's total effect per side-game.
    Coins: the rule's total is the sum of its n labels per side-game, labels
    uncorrelated within a game (current_design.md 2), two side-games a game:
        SE_z^2 = n V_T / (2 N).
    One-sided game-level: one seat carries the rule, the result is the
    contrast:  SE_z^2 = sigma_R^2 / N.
    Antithetic game-level (+delta on one seat, -delta on the other, a rule
    linear in delta): the contrast doubles:  SE_z^2 = sigma_R^2 / (4 N)
    for the one-seat effect."""
    se_z = se_elo / ELO_PER_Z
    coins = n_per_side_game * V_T / (2 * se_z ** 2)
    one = sigma_R2 / se_z ** 2
    anti = sigma_R2 / (4 * se_z ** 2)
    return coins, one, anti


def main():
    readout = json.loads((REC / "readout_b_offline.json").read_text())
    beta = np.array(readout["beta"]["values"])
    z, S, cells = load_rows()
    rv = residual_variance(z, S, beta, cells)
    print("1. Residual variance of the result over batch 1's decisive games")
    print(f"   games {rv['games']}, E[z^2] {rv['var_z']:.4f}")
    print(f"   E[(z - beta.S)^2] {rv['E_R2_beta']:.4f} (beta as frozen in the readout)")
    print(f"   with a cross-fitted cell mean ({rv['cells']} cells: scenario x ordered factions) {rv['E_R2_beta_cell_xfit']:.4f}")

    pooled = readout["labels"]["pooled"]
    n_lab, games, se = pooled["n"], pooled["games"], pooled["se"]
    # Clustered SE of the pooled mean: Var = sum_g (sum_i T_i)^2 / n^2; with
    # labels uncorrelated within a game this is n E[T^2] / n^2.
    V_T = se ** 2 * n_lab
    print("\n2. Coin labels' second moment implied by the pooled SE")
    print(f"   labels {n_lab}, games {games}, pooled SE {se:.6f} -> V_T = SE^2 n = {V_T:.3f}")
    print(f"   check: V_T * P(1-P) at P 0.25 = {V_T * 0.1875:.3f} against E[R^2] {rv['E_R2_beta']:.3f}")

    sigma_R2 = rv["E_R2_beta"]
    print("\n3. Games for an SE of 4.4 Elo on a fixed rule's total effect (n switched decisions per side-game)")
    print("   n      coins      one-sided  antithetic  coins/one  coins/anti")
    for n in (1, 2, 4.1, 6.4, 10, 20, 40):
        c, o, a = rule_games(n, V_T, sigma_R2, 4.4)
        print(f"   {n:<5}  {c:9.0f}  {o:9.0f}  {a:10.0f}  {c / o:9.2f}  {c / a:10.2f}")
    print(f"   derived: coins/one-sided = n V_T / (2 sigma_R^2) = {V_T / (2 * sigma_R2):.3f} n;"
          f" coins/antithetic = {4 * V_T / (2 * sigma_R2):.3f} n")

    print("\n4. Measured instance: in-band attack promotion")
    by_type = readout["labels"]["by_pair"] if "by_pair" in readout["labels"] else readout["labels"].get("by_type")
    side_games = readout["band"]["per_side_game"]["side_games"]
    tot_mean, tot_var, n_tot = 0.0, 0.0, 0.0
    for name in ("move>attack", "end_turn>attack", "recruit>attack"):
        t = by_type[name]
        n_sg = t["n"] / side_games
        tot_mean += n_sg * t["mean"]
        # The type's mean SE is clustered by game; a rule summing its labels
        # per side-game has SE n_sg * se_mean (the mean times the count).
        tot_var += (n_sg * t["se"]) ** 2
        n_tot += n_sg
        print(f"   {name:16s} {n_sg:6.3f} per side-game, mean {t['mean']:+.4f} +- {t['se']:.4f}")
    print(f"   switch all of them: {tot_mean * ELO_PER_Z:+.1f} +- {math.sqrt(tot_var) * ELO_PER_Z:.1f} Elo"
          f" from {games} coin games ({n_tot:.2f} switches per side-game)")
    c, o, a = rule_games(n_tot, V_T, sigma_R2, math.sqrt(tot_var) * ELO_PER_Z)
    print(f"   derived games for that SE: coins {c:.0f} (read: {games}), one-sided {o:.0f}, antithetic {a:.0f}")
    print("   S0(a), attack +1 on one seat, raw z: -41.3 +- 4.4 Elo from 6,400 games (seed section 7)")
    se_raw_6400 = math.sqrt(1.0 / 6400) * ELO_PER_Z
    print(f"   derived one-sided SE at 6,400 games: raw z {se_raw_6400:.2f} Elo, luck-corrected {math.sqrt(sigma_R2 / 6400) * ELO_PER_Z:.2f} Elo")
    ratio = games / 6400 * (math.sqrt(tot_var) * ELO_PER_Z / 4.4) ** 2
    print(f"   measured games ratio for equal SE (coins / S0(a) one-sided raw): {ratio:.1f}")


if __name__ == "__main__":
    main()
