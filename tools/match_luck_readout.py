"""The readout of tools/match_luck.py's rows: how much of the result's
variance the luck correction removes, and what it does to a match's Elo.

For each control variate (`CONTROL_VARIATES`: one or more of the rows'
luck columns S, tools/match_luck.py: hp; material; material_terminal, the
board's material counted when no leader dies beside the terminal column;
material_and_terminal, the fighters' material beside the terminal column;
the saturating values; joint, every column), the corrected result is
G = z - beta . S, with beta the
least-squares fit of z on S under two-fold cross-fitting: the games are
split in two halves at random (seeded), and each half is corrected with
the beta fitted on the other, so no game's own result enters the beta
that corrects it. Read on the decisive games (the PURE convention: a
capped game is no result):

    ratio            Var(G) / Var(z), with a bootstrap 95% interval over
                     games (the cross-fitted G held fixed);
    equivalent games REFERENCE_GAMES x ratio: the corrected games that
                     reach the standard error of REFERENCE_GAMES
                     uncorrected ones;
    bias             mean(z - G) = mean(beta . S) with its standard error:
                     zero in expectation over all games; on the decisive
                     games only, it is zero when the luck does not decide
                     which games end;
    Elo              400 log10(s / (1 - s)) of the mean score s = (1 +
                     mean) / 2, for z (the PURE fit of a two-player match)
                     and for G, the standard errors by the delta method.

Each match is read with its own fit, and all matches together with one
pooled fit, split by game length and by map. The mean of every luck
column over all games, capped ones included, is the check that the luck
terms have mean zero.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.match_luck import COLUMNS

CONTROL_VARIATES: Dict[str, Tuple[str, ...]] = {
    "hp": ("hp",),
    "material": ("material",),
    "material_terminal": ("material_nonterminal", "terminal"),
    "material_and_terminal": ("material", "terminal"),
    **{c: (c,) for c in COLUMNS if c.startswith("value_")},
    "joint": COLUMNS,
}
REFERENCE_GAMES = 800
# Game-length buckets in turns: (first, last) inclusive.
TURN_BUCKETS = ((1, 15), (16, 20), (21, 25), (26, 35), (36, 10_000))
# A split cell with fewer decisive games is reported without a ratio.
MIN_SPLIT_GAMES = 30
COUNT_KEYS = ("attacks", "enumerated", "outside_support", "not_enumerable", "counter_weapon_differs",
              "no_units", "recruits", "advancements")


def load_rows(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
def two_folds(n: int, seed: int) -> np.ndarray:
    """0 or 1 for each of n games, half of them each, at random."""
    folds = np.zeros(n, dtype=int)
    folds[np.random.default_rng(seed).permutation(n)[n // 2:]] = 1
    return folds


def fit_beta(z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The least-squares slope of z on the columns of X (an intercept
    fitted beside it)."""
    zc = z - z.mean()
    Xc = X - X.mean(axis=0)
    beta, *_ = np.linalg.lstsq(Xc, zc, rcond=None)
    return beta


def cross_fit(z: np.ndarray, X: np.ndarray, folds: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """(G = z - X beta with each fold's beta fitted on the other fold,
    [fold 0's beta, fold 1's beta])."""
    G = z.astype(float).copy()
    betas = []
    for f in (0, 1):
        train, test = folds != f, folds == f
        beta = fit_beta(z[train].astype(float), X[train])
        G[test] = z[test] - X[test] @ beta
        betas.append(beta)
    return G, betas


def variance_ratio(z: np.ndarray, G: np.ndarray) -> float:
    return float(np.var(G, ddof=1) / np.var(z, ddof=1))


def bootstrap_interval(z: np.ndarray, G: np.ndarray, n_boot: int, seed: int,
                       level: float = 0.95) -> Tuple[float, float]:
    """The percentile interval of Var(G) / Var(z) over games resampled
    with replacement."""
    rng = np.random.default_rng(seed)
    n = len(z)
    ratios = []
    for start in range(0, n_boot, 200):
        idx = rng.integers(0, n, size=(min(200, n_boot - start), n))
        vz = np.var(z[idx], axis=1, ddof=1)
        vg = np.var(G[idx], axis=1, ddof=1)
        ratios.append(vg[vz > 0] / vz[vz > 0])
    r = np.concatenate(ratios)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(r, tail)), float(np.quantile(r, 1.0 - tail))


def mean_se(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


def elo_from_mean(m: float, se: float) -> Tuple[float, float]:
    """The Elo difference of a mean result m in [-1, 1] (score (1 + m) / 2)
    and its standard error from the mean's."""
    s = min(max((1.0 + m) / 2.0, 1e-9), 1.0 - 1e-9)
    elo = 400.0 * math.log10(s / (1.0 - s))
    return elo, 400.0 / math.log(10.0) / (s * (1.0 - s)) * se / 2.0


def corrected(z: np.ndarray, X: np.ndarray, *, seed: int, n_boot: int) -> Tuple[dict, np.ndarray]:
    """(the readout of one control variate on games (z, X), the
    cross-fitted G)."""
    folds = two_folds(len(z), seed)
    G, betas = cross_fit(z, X, folds)
    ratio = variance_ratio(z, G)
    lo, hi = bootstrap_interval(z, G, n_boot, seed + 1)
    mz, sz = mean_se(z.astype(float))
    mg, sg = mean_se(G)
    bias, bias_se = mean_se(z - G)
    elo_z, elo_z_se = elo_from_mean(mz, sz)
    elo_g, elo_g_se = elo_from_mean(mg, sg)
    out = {"games": int(len(z)), "betas": [b.tolist() for b in betas],
           "ratio": ratio, "ratio_ci95": [lo, hi], "equivalent_games": REFERENCE_GAMES * ratio,
           "mean_z": mz, "se_z": sz, "mean_g": mg, "se_g": sg,
           "bias": bias, "bias_se": bias_se, "bias_t": bias / bias_se if bias_se > 0 else 0.0,
           "elo_z": elo_z, "elo_z_se": elo_z_se, "elo_g": elo_g, "elo_g_se": elo_g_se}
    return out, G


# ---------------------------------------------------------------------
# The readout
# ---------------------------------------------------------------------
def luck_matrix(rows: Sequence[dict], columns: Sequence[str]) -> np.ndarray:
    return np.array([[float(r["S"][c]) for c in columns] for r in rows], dtype=float).reshape(len(rows), len(columns))


def turn_bucket(turns: Optional[int]) -> str:
    for first, last in TURN_BUCKETS:
        if turns is not None and first <= int(turns) <= last:
            return f"{first}-{last}" if last < 10_000 else f"{first}+"
    return "unknown"


def column_means(rows: Sequence[dict]) -> Dict[str, dict]:
    """Every luck column's mean over `rows` with its standard error, and
    the root-mean-square L per computed attack."""
    out = {}
    enumerated = sum(r["counts"].get("enumerated", 0) + r["counts"].get("outside_support", 0) for r in rows)
    for c in COLUMNS:
        m, se = mean_se(np.array([r["S"][c] for r in rows], dtype=float))
        ss = sum(r["SS"][c] for r in rows)
        out[c] = {"mean": m, "se": se, "t": m / se if se > 0 else 0.0,
                  "sd_game": se * math.sqrt(len(rows)),
                  "rms_attack": math.sqrt(ss / enumerated) if enumerated else 0.0}
    return out


def match_facts(rows: Sequence[dict], errors: Sequence[dict]) -> dict:
    counts: Counter = Counter()
    for r in rows:
        counts.update({k: r["counts"].get(k, 0) for k in COUNT_KEYS})
    temperatures = sorted({(r.get("raw_temperature_a"), r.get("raw_temperature_b")) for r in rows}, key=str)
    procedures = sorted({(r.get("procedure_a"), r.get("procedure_b")) for r in rows}, key=str)
    labels = sorted({(r.get("label_a"), r.get("label_b")) for r in rows}, key=str)
    return {"games": len(rows) + len(errors), "walked": len(rows),
            "decisive": sum(1 for r in rows if r["decisive"]),
            "no_result": sum(1 for r in rows if not r["decisive"]),
            "errors": len(errors), "error_kinds": dict(Counter(e["error"].split(":")[0] for e in errors)),
            "labels": labels, "procedures": procedures, "temperatures": temperatures,
            "argmax": all(a == 0 and b == 0 for a, b in temperatures),
            "turns_mean": float(np.mean([r["turns"] for r in rows])) if rows else 0.0,
            "counts": dict(counts)}


def readout_of(rows: Sequence[dict], *, seed: int, n_boot: int) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Every control variate on the decisive games of `rows`, and each
    one's G (in the order of the decisive rows)."""
    decisive = [r for r in rows if r["decisive"]]
    z = np.array([r["z"] for r in decisive], dtype=float)
    out, gs = {}, {}
    if len(decisive) < 4 or np.var(z) == 0:
        return out, gs
    for name, columns in CONTROL_VARIATES.items():
        out[name], gs[name] = corrected(z, luck_matrix(decisive, columns), seed=seed, n_boot=n_boot)
    return out, gs


def splits(rows: Sequence[dict], gs: Dict[str, np.ndarray], key) -> Dict[str, dict]:
    """The ratio of each control variate within each cell of `key` (a
    function of a row), from the pooled fit's G."""
    decisive = [r for r in rows if r["decisive"]]
    cells: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(decisive):
        cells[key(r)].append(i)
    z = np.array([r["z"] for r in decisive], dtype=float)
    out = {}
    for cell in sorted(cells):
        idx = np.array(cells[cell])
        entry = {"decisive": int(len(idx))}
        if len(idx) >= MIN_SPLIT_GAMES and np.var(z[idx]) > 0:
            entry["mean_z"] = float(z[idx].mean())
            entry.update({name: variance_ratio(z[idx], g[idx]) for name, g in gs.items()})
        out[cell] = entry
    return out


def readout(rows: Sequence[dict], *, seed: int = 0, n_boot: int = 2000) -> dict:
    """The whole readout (module docstring) of match_luck rows."""
    rows = sorted(rows, key=lambda r: (r["match"], r["game"]))
    walked = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    matches = {}
    for name in sorted({r["match"] for r in rows}):
        mine = [r for r in walked if r["match"] == name]
        entry = match_facts(mine, [e for e in errors if e["match"] == name])
        if mine:
            entry["luck_means"] = column_means(mine)
            entry["corrected"], _ = readout_of(mine, seed=seed, n_boot=n_boot)
        matches[name] = entry
    pooled, gs = readout_of(walked, seed=seed, n_boot=n_boot)
    out = {"reference_games": REFERENCE_GAMES, "seed": seed, "bootstrap": n_boot,
           "control_variates": {k: list(v) for k, v in CONTROL_VARIATES.items()},
           "matches": matches, "pooled": pooled}
    if walked:
        out["pooled_facts"] = match_facts(walked, errors)
        out["pooled_luck_means"] = column_means(walked)
    if gs:
        out["pooled_by_match"] = splits(walked, gs, lambda r: r["match"])
        out["pooled_by_turns"] = splits(walked, gs, lambda r: turn_bucket(r.get("turns")))
        out["pooled_by_map"] = splits(walked, gs, lambda r: str(r.get("scenario_id")))
        # The capped games as draws (z = 0): an estimand over every game,
        # where the luck terms' zero mean needs no condition.
        z = np.array([r["z"] for r in walked], dtype=float)
        out["pooled_all_games_capped_as_draws"] = {
            name: corrected(z, luck_matrix(walked, cols), seed=seed, n_boot=n_boot)[0]
            for name, cols in CONTROL_VARIATES.items()}
    return out


# ---------------------------------------------------------------------
# The text report
# ---------------------------------------------------------------------
def _cv_lines(corr: Dict[str, dict], indent: str = "  ") -> List[str]:
    lines = [f"{indent}{'control variate':<18} {'Var(G)/Var(z)':>13} {'95% interval':>16} {'games=800':>9} "
             f"{'Elo(G)':>15} {'bias t':>7}"]
    for name, c in corr.items():
        lo, hi = c["ratio_ci95"]
        lines.append(f"{indent}{name:<18} {c['ratio']:>13.3f} {f'[{lo:.3f}, {hi:.3f}]':>16} "
                     f"{c['equivalent_games']:>9.0f} {c['elo_g']:>+8.1f} +- {c['elo_g_se']:<4.1f} {c['bias_t']:>+7.2f}")
    return lines


def _split_lines(title: str, table: Dict[str, dict]) -> List[str]:
    names = list(CONTROL_VARIATES)
    lines = [title, "  " + f"{'cell':<36} {'n':>5} " + " ".join(f"{n[:9]:>9}" for n in names)]
    for cell, e in table.items():
        vals = " ".join(f"{e[n]:>9.3f}" if n in e else f"{'-':>9}" for n in names)
        lines.append("  " + f"{cell[:36]:<36} {e['decisive']:>5} " + vals)
    return lines


def report_text(result: dict) -> str:
    lines: List[str] = []
    for name, m in result["matches"].items():
        lines.append(f"== {name}: {m['games']} games, {m['decisive']} decisive, {m['no_result']} no result, "
                     f"{m['errors']} errors {m['error_kinds'] or ''}; labels {m['labels']}; "
                     f"argmax {m['argmax']}; mean turns {m['turns_mean']:.1f}")
        lines.append(f"  counts {m['counts']}")
        corr = m.get("corrected") or {}
        if corr:
            any_cv = next(iter(corr.values()))
            lines.append(f"  PURE: mean z {any_cv['mean_z']:+.4f} +- {any_cv['se_z']:.4f}, "
                         f"Elo {any_cv['elo_z']:+.1f} +- {any_cv['elo_z_se']:.1f}")
            lines.extend(_cv_lines(corr))
        means = m.get("luck_means") or {}
        if means:
            lines.append("  luck column means over all games (t), sd per game, rms per attack: "
                         + ", ".join(f"{c} {v['t']:+.2f} {v['sd_game']:.3g} {v['rms_attack']:.3g}"
                                     for c, v in means.items()))
    if result.get("pooled"):
        f = result["pooled_facts"]
        lines.append(f"== pooled: {f['walked']} games walked, {f['decisive']} decisive, {f['errors']} errors")
        lines.extend(_cv_lines(result["pooled"]))
        mt = result["pooled"].get("material_terminal")
        if mt:
            scales = [b[1] / b[0] if b[0] else float("nan") for b in mt["betas"]]
            lines.append(f"  material_terminal: a won game is worth {scales[0]:.0f} and {scales[1]:.0f} gold "
                         f"(the two folds' fits)")
        lines.append("  all games, capped as draws:")
        lines.extend(_cv_lines(result["pooled_all_games_capped_as_draws"], indent="    "))
        lines.extend(_split_lines("== pooled fit, ratio by match", result["pooled_by_match"]))
        lines.extend(_split_lines("== pooled fit, ratio by game length (turns)", result["pooled_by_turns"]))
        lines.extend(_split_lines("== pooled fit, ratio by map", result["pooled_by_map"]))
    return "\n".join(lines)
