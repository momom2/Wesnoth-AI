"""Round 5 challenge: measurements on existing records (read-only, numpy + stdlib).

A. How much of a game's result variance the setup explains (map, seat,
   factions), against the dice's linear luck columns, on the luck walk's
   rows (training/metrics/match_luck_20261019/rows.jsonl).
B. The decode-knob search (training/metrics/decode_knobs_20261012/search.jsonl):
   the between-vector variance of true win rates, and a response-surface fit.
C. The margin census (training/metrics/critic_step1_20261009/prior_gaps.games.jsonl):
   decisions switched per side-game by a Gaussian logit perturbation.

Run: python round5_challenge_records.py  (writes round5_challenge_records.txt beside it)
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "training/metrics/match_luck_20261019/rows.jsonl"
SEARCH = ROOT / "training/metrics/decode_knobs_20261012/search.jsonl"
GAPS = ROOT / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
OUT = Path(__file__).with_suffix(".txt")
LUCK_COLUMNS = ["hp", "material", "material_nonterminal", "terminal",
                "value_50", "value_100", "value_200"]
AA_MATCHES = {"gen_own1_selfplay", "own64_selfplay", "kv00"}

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def load_rows():
    rows = [json.loads(line) for line in ROWS.open()]
    return [r for r in rows if r["decisive"]]


def anova_between_variance(groups):
    """Method-of-moments variance of the group means' true values (one-way
    random effects, unbalanced): (MS_between - MS_within) / n0."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 1]
    k = len(groups)
    sizes = np.array([len(g) for g in groups], float)
    n = sizes.sum()
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    return (ms_between - ms_within) / n0, ms_within


def setup_share(rows, label):
    """Share of Var(z) carried by the setup cell, with the match's mean removed."""
    by_match = defaultdict(list)
    for r in rows:
        by_match[r["match"]].append(r)
    z_dm, cells = [], []
    for match_rows in by_match.values():
        mean = np.mean([r["z"] for r in match_rows])
        for r in match_rows:
            z_dm.append(r["z"] - mean)
            cells.append(r)
    z_dm = np.array(z_dm)
    total = z_dm.var()
    say(f"[{label}] games {len(z_dm)}, Var(z - match mean) = {total:.4f}")
    keys = {
        "seat x map": lambda r: (r["side_a"], r["scenario_id"]),
        "factions (A, B)": lambda r: (r["faction_a"], r["faction_b"]),
        "seat x map x factions": lambda r: (r["side_a"], r["scenario_id"],
                                            r["faction_a"], r["faction_b"]),
    }
    for name, key in keys.items():
        groups = defaultdict(list)
        for zi, r in zip(z_dm, cells):
            groups[key(r)].append(zi)
        between, within = anova_between_variance(list(groups.values()))
        boot = []
        group_list = list(groups.values())
        rng = random.Random(0)
        for _ in range(200):
            sample = [group_list[rng.randrange(len(group_list))]
                      for _ in group_list]
            b, _ = anova_between_variance(sample)
            boot.append(b / total)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        say(f"  {name:24s} cells {len(groups):5d}  between-cell share "
            f"{between / total:+.4f}  [{lo:+.4f}, {hi:+.4f}] (cluster bootstrap)")
    return z_dm, cells


def ridge_fit(x, y, alpha):
    p = x.shape[1]
    return np.linalg.solve(x.T @ x + alpha * np.eye(p), x.T @ y)


def design(rows, use_luck, use_setup, levels):
    columns = []
    if use_luck:
        luck = np.array([[r["S"][c] for c in LUCK_COLUMNS] for r in rows])
        columns.append(luck)
    if use_setup:
        for name, index in levels.items():
            idx = np.array([index.get(KEY_FNS[name](r), -1) for r in rows])
            onehot = np.zeros((len(rows), len(index)))
            ok = idx >= 0
            onehot[np.arange(len(rows))[ok], idx[ok]] = 1.0
            columns.append(onehot)
    return np.hstack(columns)


KEY_FNS = {
    "seat_map": lambda r: (r["side_a"], r["scenario_id"]),
    "factions": lambda r: (r["faction_a"], r["faction_b"]),
    "seat_faction_a": lambda r: (r["side_a"], r["faction_a"]),
    "map_faction_a": lambda r: (r["scenario_id"], r["faction_a"]),
    "map_faction_b": lambda r: (r["scenario_id"], r["faction_b"]),
}


def cross_fitted_ratios(z_dm, rows, seeds=10):
    levels = {name: {} for name in KEY_FNS}
    for r in rows:
        for name, fn in KEY_FNS.items():
            levels[name].setdefault(fn(r), len(levels[name]))
    results = defaultdict(list)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, 2, len(rows))
        for label, use_luck, use_setup, alpha in [
                ("luck columns (7)", True, False, 1e-6),
                ("setup cells (ridge)", False, True, 30.0),
                ("luck + setup", True, True, 30.0)]:
            x = design(rows, use_luck, use_setup, levels)
            if use_luck:
                scale = x[:, :7].std(axis=0)
                x = x.copy()
                x[:, :7] /= scale
            resid = np.empty_like(z_dm)
            for f in (0, 1):
                train, test = fold != f, fold == f
                xt = np.hstack([x[train], np.ones((train.sum(), 1))])
                beta = ridge_fit(xt, z_dm[train], alpha)
                xs = np.hstack([x[test], np.ones((test.sum(), 1))])
                resid[test] = z_dm[test] - xs @ beta
            results[label].append(resid.var() / z_dm.var())
    for label, values in results.items():
        say(f"  {label:22s} cross-fitted Var(resid)/Var(z) = {np.mean(values):.4f} "
            f"(sd over {seeds} splits {np.std(values):.4f})")


def decode_knobs():
    events = [json.loads(line) for line in SEARCH.open()]
    vectors = {v["index"]: v for v in events[0]["vectors"]}
    stage1 = [e for e in events if e.get("event") == "match" and e["stage"] == 1]
    wins = np.array([e["wins"] for e in stage1], float)
    n = np.array([e["decisive"] for e in stage1], float)
    p = wins / n
    pbar = wins.sum() / n.sum()
    noise = (pbar * (1 - pbar) / n).mean()
    true_var = p.var(ddof=1) - noise
    say(f"stage 1: {len(stage1)} vectors x {int(n.mean())} games; mean win rate {pbar:.4f}")
    say(f"  observed var of win rates {p.var(ddof=1):.5f}, binomial noise {noise:.5f}, "
        f"true between-vector var {true_var:.5f} (sd {math.sqrt(max(true_var, 0)):.4f} "
        f"win-rate points)")
    chi = (p.var(ddof=1) * (len(p) - 1)) / noise
    say(f"  dispersion statistic (var/noise)*(k-1) = {chi:.1f} on {len(p) - 1} df")
    # all unbiased per-vector data: every stage's games are fresh seeds
    allm = [e for e in events if e.get("event") in ("match", "confirmation")]
    x_rows, y, w = [], [], []
    for e in allm:
        v = vectors[e["vector"]]
        eo = v["raw_end_turn_offset"] + 1.5
        x_rows.append([1.0, eo, eo * eo, v["raw_attack_offset"], v["raw_recruit_offset"],
                       v["raw_attack_tod_slope"]])
        y.append(e["wins"] / e["decisive"])
        w.append(e["decisive"])
    x_mat, y, w = np.array(x_rows), np.array(y), np.array(w, float)
    wsqrt = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(x_mat * wsqrt[:, None], y * wsqrt, rcond=None)
    resid = y - x_mat @ beta
    sigma2 = (w * resid ** 2).sum() / (len(y) - x_mat.shape[1])
    cov = sigma2 * np.linalg.inv((x_mat * w[:, None]).T @ x_mat)
    se = np.sqrt(np.diag(cov))
    names = ["const", "eo+1.5", "(eo+1.5)^2", "attack", "recruit", "tod_slope"]
    say("  weighted fit of win rate on knobs (all stages, fresh seeds each; eo centred at -1.5):")
    for nm, b, s in zip(names, beta, se):
        say(f"    {nm:12s} {b:+.4f} +- {s:.4f}")
    binomial = float(np.mean(y * (1 - y)))
    say(f"    weighted residual variance {sigma2:.3f} against binomial {binomial:.3f}: "
        f"dispersion {sigma2 / binomial:.2f}")


def margin_census(deltas=(0.05, 0.1, 0.2, 0.4), draws=64):
    rng = np.random.default_rng(0)
    side_games = 0
    switched = {d: 0.0 for d in deltas}
    small = 0
    decisions = 0
    stop_top = 0
    for line in GAPS.open():
        game = json.loads(line)
        side_games += 1
        for reading in game["readings"]:
            gaps = reading.get("gaps")
            if not gaps:
                continue
            decisions += 1
            if reading.get("top_type") == "end_turn":
                stop_top += 1
            g = np.concatenate([[0.0], np.asarray(gaps, float)])
            if g[1] < 0.1:
                small += 1
            for d in deltas:
                noise = rng.normal(0, d, (draws, len(g)))
                switched[d] += np.mean(np.argmax(-g[None, :] + noise, axis=1) != 0)
    say(f"side-games {side_games}, multi-legal decisions {decisions} "
        f"({decisions / side_games:.0f} per side-game), end_turn top at {stop_top / side_games:.1f} per side-game")
    say(f"  decisions with top-two margin < 0.1 nats: {small / side_games:.1f} per side-game")
    for d in deltas:
        say(f"  iid N(0, {d}^2) logit noise on the top 8: {switched[d] / side_games:.1f} "
            f"switched decisions per side-game")


def material_value(rows):
    """Result units per gold of material that the dice move: the slope of z on the
    material luck column alone, two-fold cross-fitted (the walk's own estimator)."""
    s = np.array([r["S"]["material"] for r in rows])
    z = np.array([r["z"] for r in rows], float)
    slopes = []
    rng = np.random.default_rng(0)
    for _ in range(20):
        fold = rng.integers(0, 2, len(z))
        for f in (0, 1):
            train = fold != f
            x = s[train] - s[train].mean()
            slopes.append((x * (z[train] - z[train].mean())).sum() / (x * x).sum())
    x = s - s.mean()
    resid = z - z.mean() - np.mean(slopes) * x
    se = math.sqrt(resid.var() / (x * x).sum())
    say(f"material luck: slope of z per gold of material {np.mean(slopes):.5f} +- {se:.5f} "
        f"(sd of S_material per game {s.std():.1f} gold)")


def main():
    rows = load_rows()
    say("== A. setup and luck shares of the result's variance")
    z_dm, cells = setup_share(rows, "all 23 matches, decisive")
    cross_fitted_ratios(z_dm, cells)
    aa = [r for r in rows if r["match"] in AA_MATCHES]
    z_aa, cells_aa = setup_share(aa, "A/A matches gen_own1_selfplay, own64_selfplay, kv00")
    by = defaultdict(list)
    for r in rows:
        by[r["match"]].append(r["z"])
    o = by["own1_eo25_vs_own1"]
    material_value(rows)
    say(f"own1 at -2.5 against own1 at -1.5: mean z {np.mean(o):+.4f} +- {np.std(o) / math.sqrt(len(o)):.4f} "
        f"over {len(o)} decisive games")
    say()
    say("== B. decode-knob search")
    decode_knobs()
    say()
    say("== C. margin census (parity3, 100 side-games)")
    margin_census()
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
