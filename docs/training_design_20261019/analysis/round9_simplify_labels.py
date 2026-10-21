"""Round 9, simplify stance: the coin labels' measured variance, the forecast's SE, the margin slope and the prices.

Inputs (all records of the 2026-10-20/21 box, training/metrics/near_tie_coins_20261020/):
  coins_b00.jsonl, coins_b06.jsonl  the coin rows of 10,000 of batch 1's 60,000 games (the others stay on HF);
  rows_b0*.jsonl                    the walk's per-game rows (z, setup, luck sums) of all 60,000;
  readout_b_offline.json            the readout of all 60,000 (margin-bin means);
  matches.txt, walls.txt            each match's wall; ../own2_base_20261020/match.walls.

Reports:
  1. Per-decision labels T_i = s_i w_i (z - pred) on the 10,000 games, with pred either the luck
     columns as batch 1 used them (configs/luck_beta_20261019.json, b = 0) or the frozen
     coefficients of this stance (beta refit plus the shrunk cell, fitted on b01-b05's 50,000 games,
     so out of sample for b00 and b06; round9_simplify_residual.py's predictor). Measured E[T^2]
     against 1/(P(1-P)) x residual share; the per-side-game sum's variance against the sum of its
     labels' variances (within-game correlation); the two movers' sums' correlation.
  2. The forecast's SE: a held-out welfare sum over a switched share q of band decisions, per
     side-game, in Elo, scaled to 4 batches of 100,000 games; and the adoption match's SE, so the
     information ratio of the two readings per game.
  3. The slope of the mean label on the margin G across the seven 0.05-nat bins (weighted least
     squares on readout_b_offline.json), with its SE.
  4. Prices: coin games an hour in the full 10,000-game matches; the walk, rows, plant and readout
     time per match; the two-network match's games an hour; dollars a game at the box's $0.577 an hour.
numpy and the standard library only.
"""
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COIN = ROOT / "training/metrics/near_tie_coins_20261020"
OWN2 = ROOT / "training/metrics/own2_base_20261020"
BETA = ROOT / "configs/luck_beta_20261019.json"
ELO = 800 / math.log(10)
PRICE_H = 0.577
RNG = np.random.default_rng(20261021)

spec = importlib.util.spec_from_file_location("res", HERE / "round9_simplify_residual.py")
res = importlib.util.module_from_spec(spec)
spec.loader.exec_module(res)


def coin_rows(names):
    out = {}
    for n in names:
        with open(COIN / f"coins_{n}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "manifest" in r or "error" in r or not r.get("decisive"):
                    continue
                out[(r["match"], r["game"])] = r
    return out


def labels(coins, walk, pred_fn, cols):
    keys = [k for k in coins if k in walk]
    wr = [walk[k] for k in keys]
    t = res.arrays(wr, cols)
    R = t[0] - pred_fn(t)
    per_game = []
    for k, Rg in zip(keys, R):
        b = coins[k]["band"]
        side = np.array(b["side"])
        p = np.array(b["p"], float)
        d = np.array(b["d"], float)
        s = np.where(side == int(coins[k]["side_a"]), 1.0, -1.0)
        w = (d - p) / (p * (1 - p))
        T = s * w * Rg
        per_game.append((T, s, 1.0 / (p * (1 - p)), Rg))
    return per_game


def section1_2(per_game, label, r_share):
    T_all = np.concatenate([g[0] for g in per_game])
    inv = np.concatenate([g[2] for g in per_game])
    n_games = len(per_game)
    m2 = float(np.mean(T_all ** 2))
    pred_m2 = float(np.mean(inv)) * r_share
    WA = np.array([g[0][g[1] > 0].sum() for g in per_game])
    WB = np.array([g[0][g[1] < 0].sum() for g in per_game])
    diag = np.array([(g[0] ** 2).sum() for g in per_game])
    var_side = (np.var(WA) + np.var(WB)) / 2
    diag_side = diag.mean() / 2
    corr_ab = float(np.corrcoef(WA, WB)[0, 1])
    per_side = len(T_all) / (2 * n_games)
    print(f"[{label}] {n_games} games, {len(T_all)} band decisions ({per_side:.1f} per side-game)")
    print(f"  E[T^2] {m2:.3f} (SE {np.std(T_all ** 2) / math.sqrt(len(T_all)):.3f}); "
          f"1/(P(1-P)) x out-of-sample residual share = {pred_m2:.3f}")
    print(f"  mean label {T_all.mean():+.5f} +- {math.sqrt(m2 / len(T_all)):.5f}")
    print(f"  per side-game: Var(sum of its labels) {var_side:.1f} against the sum of their squares {diag_side:.1f}: "
          f"ratio {var_side / diag_side:.3f}; corr(mover A's sum, mover B's sum) {corr_ab:+.3f}")
    # forecast: switched share q of decisions, chosen at random (independent of the labels, as a
    # student fitted on other batches is)
    for q in (0.25, 0.5):
        sums = []
        for g in per_game:
            pick = RNG.random(len(g[0])) < q
            for sg in (1, -1):
                sums.append(g[0][pick & (g[1] == sg)].sum())
        v = float(np.var(sums))
        n_side_4 = 4 * 100_000 * 2
        se = math.sqrt(v / n_side_4) * ELO
        print(f"  forecast at switched share {q}: variance per side-game {v:.1f}; SE on 4 batches of 100,000 games "
              f"{se:.2f} Elo")
    se_match = ELO / math.sqrt(37_800)
    v_half = float(np.var([g[0][g[1] == sg].sum() for g in per_game for sg in (1, -1)]))
    print(f"  adoption match, 37,800 games: SE {se_match:.2f} Elo. Information per game about a rule's value: "
          f"match 1/Var(z) = 1 against labels 2/(q x per-side-game variance) = {2 / (0.5 * v_half):.4f} at q 0.5, "
          f"a ratio of {0.5 * v_half / 2:.0f}")
    return var_side / diag_side


def section3():
    d = json.loads((COIN / "readout_b_offline.json").read_text())["labels"]["by_bin"]
    x, y, se = [], [], []
    for name, v in d.items():
        lo, hi = (float(a) for a in re.findall(r"[\d.]+", name))
        x.append((lo + hi) / 2)
        y.append(v["mean"])
        se.append(v["se"])
    x, y, w = np.array(x), np.array(y), 1 / np.array(se) ** 2
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    b = cov @ X.T @ (w * y)
    chi = float(np.sum(w * (y - X @ b) ** 2))
    print(f"\n[margin slope] mean label = {b[0]:+.4f} + {b[1]:+.4f} G (SE of slope {math.sqrt(cov[1, 1]):.4f}, "
          f"z {b[1] / math.sqrt(cov[1, 1]):+.2f}); residual chi2 {chi:.1f} on {len(x) - 2} df; "
          f"the label at G = 0: {b[0]:+.4f} +- {math.sqrt(cov[0, 0]):.4f}")


def section4():
    games, secs, post = 0, 0, []
    for line in (COIN / "matches.txt").read_text().splitlines():
        m = re.search(r"(b0\d): games (\d+) .* (\d+) s,", line)
        if m and m.group(1) in ("b01", "b02", "b03", "b04", "b05"):
            games += int(m.group(2))
            secs += int(m.group(3))
    for line in (COIN / "walls.txt").read_text().splitlines():
        m = re.search(r"(walk|coin rows|plant rows|readout readout_b) (b0[1-5])? ?rc=0 ok (\d+) s", line)
        if m and (m.group(2) or "readout" in line):
            post.append(int(m.group(3)))
    rate = games / secs * 3600
    post_per = sum(post) / 5
    allin = 10_000 / (10_000 / rate * 3600 + post_per) * 3600
    print(f"\n[prices] coin games: {games} in {secs} s, {rate:,.0f} games an hour, ${PRICE_H / rate:.3e} a game "
          f"at ${PRICE_H} an hour; walk + rows + plant + readout {post_per:.0f} s per 10,000-game match, all-in "
          f"{allin:,.0f} an hour, ${PRICE_H / allin:.3e} a game; at $0.42 an hour ${0.42 / allin:.3e}")
    m = re.search(r"(\d+) s, (\d+) games, (\d+) decisive", (OWN2 / "match.walls").read_text())
    s2, g2, d2 = (int(v) for v in m.groups())
    r2 = d2 / s2 * 3600
    print(f"[prices] two-network match own2 vs own1: {d2} decisive in {s2} s, {r2:,.0f} an hour, "
          f"${PRICE_H / r2:.3e} a decisive game; 12,800 games ${12_800 * PRICE_H / r2:.2f}")
    print(f"[prices] batch of 100,000 coin games all-in ${100_000 * PRICE_H / allin:.2f} "
          f"({100_000 / allin:.2f} h); adoption match of 37,800 pool games ${37_800 * PRICE_H / rate:.2f}")


def main():
    cfg = json.loads(BETA.read_text())
    cols = cfg["columns"]
    walk_rows = res.load([COIN / f"rows_b0{k}.jsonl" for k in range(7)])
    walk = {(r["match"], r["game"]): r for r in walk_rows}
    fit_rows = [r for r in walk_rows if r["match"] in ("b01", "b02", "b03", "b04", "b05")]
    preds, _ = res.predictors(res.arrays(fit_rows, cols), (cfg["intercept"], np.array(cfg["beta"])))
    coins = coin_rows(["b00", "b06"])
    held = [walk[k] for k in coins if k in walk]
    t = res.arrays(held, cols)
    vz = float(np.var(t[0]))
    share_cfg = float(np.var(t[0] - preds["beta_config"](t))) / vz
    share_frz = float(np.var(t[0] - preds["beta+shrunk_cell"](t))) / vz
    print(f"held-out games b00 + b06: residual share {share_cfg:.3f} (batch 1's columns), {share_frz:.3f} "
          f"(frozen beta + shrunk cell fitted on b01-b05)\n")
    section1_2(labels(coins, walk, preds["beta_config"], cols), "batch 1's label (columns, b = 0)", share_cfg)
    section1_2(labels(coins, walk, preds["beta+shrunk_cell"], cols), "frozen beta + shrunk cell", share_frz)
    section3()
    section4()


if __name__ == "__main__":
    main()
