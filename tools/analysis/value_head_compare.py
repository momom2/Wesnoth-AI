"""Paired comparison of two per-phase value-head records
(tools/analysis/value_head_by_phase.py --out): is head B better than
head A at telling the winner, phase by phase?

Both records score the same holdout states, so the comparison is
paired by game: per turn bucket, each game's same-turn score (the
share of its turns in the bucket where the winner's value is above
the loser's; ties count half) is computed for A and for B, and the
per-game difference B - A is the sample. Reported per bucket: the
two means, the mean difference with its 95% confidence interval
(t over games and a game-level bootstrap), the paired t-test and the
Wilcoxon signed-rank p-values, how many games moved each way, and
the Brier difference with the same game-clustered interval.

Usage:
    python tools/analysis/value_head_compare.py A.json B.json \\
        --label-a seed --label-b plus_1 --out compare_seed_plus_1
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.analysis.value_head_by_phase import BUCKETS, bucket_of  # noqa: E402

BOOTSTRAP_RESAMPLES = 10_000


def load_rows(path: Path) -> Dict[Tuple[str, int, int], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(r["game"], int(r["turn"]), int(r["side"])): r for r in data["rows"]}


def same_turn_score(winner_value: float, loser_value: float) -> float:
    if winner_value > loser_value:
        return 1.0
    return 0.5 if winner_value == loser_value else 0.0


def _per_game_scores(rows_a, rows_b, lo: int, hi: int):
    """Per game: (mean same-turn score A, mean B, mean Brier A, mean
    Brier B) over the bucket's turns present in both records; plus
    the pooled (value, win) lists per game for the AUC bootstrap."""
    turns: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    brier: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    pooled: Dict[str, List[Tuple[float, float, bool]]] = defaultdict(list)
    for (game, turn, side), ra in rows_a.items():
        if not (lo <= turn <= hi):
            continue
        rb = rows_b.get((game, turn, side))
        if rb is None:
            continue
        win = ra["side"] == ra["winner_side"]
        pooled[game].append((ra["value"], rb["value"], win))
        brier[game].append((((ra["value"] + 1) / 2 - win) ** 2, ((rb["value"] + 1) / 2 - win) ** 2))
        if side != ra["winner_side"]:
            continue
        loser_key = (game, turn, 3 - side)
        la, lb = rows_a.get(loser_key), rows_b.get(loser_key)
        if la is None or lb is None:
            continue
        turns[game].append((same_turn_score(ra["value"], la["value"]),
                            same_turn_score(rb["value"], lb["value"])))
    games = sorted(turns)
    a = np.array([np.mean([t[0] for t in turns[g]]) for g in games])
    b = np.array([np.mean([t[1] for t in turns[g]]) for g in games])
    brier_games = sorted(brier)
    ba = np.array([np.mean([t[0] for t in brier[g]]) for g in brier_games])
    bb = np.array([np.mean([t[1] for t in brier[g]]) for g in brier_games])
    return games, a, b, ba, bb, pooled


def _ci_t(d: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(mean, low, high): the 95% t interval of the mean difference."""
    from scipy import stats
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else None), None, None
    se = float(d.std(ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    m = float(d.mean())
    return m, m - half, m + half


def _bootstrap_mean(d: np.ndarray, rng: np.random.Generator) -> Tuple[Optional[float], Optional[float]]:
    if len(d) < 2:
        return None, None
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_RESAMPLES, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _pvalues(d: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Paired t-test and Wilcoxon signed-rank on the per-game differences."""
    from scipy import stats
    if len(d) < 2 or float(d.std(ddof=1)) == 0.0:
        return None, None
    p_t = float(stats.ttest_1samp(d, 0.0).pvalue)
    nonzero = d[d != 0]
    p_w = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 else 1.0
    return p_t, p_w


def rank_auc(scores: np.ndarray, wins: np.ndarray) -> Optional[float]:
    """Pooled AUC by ranks (Mann-Whitney), ties counted half; the same
    number as value_head_by_phase.pooled_auc in n log n."""
    from scipy.stats import rankdata
    n_pos = int(wins.sum())
    n_neg = len(wins) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[wins].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pooled_auc_bootstrap(pooled: Dict[str, List[Tuple[float, float, bool]]],
                          rng: np.random.Generator, resamples: int = 2000):
    """Pooled AUC of A and B and the game-level bootstrap interval of
    their difference (games resampled with replacement)."""
    games = sorted(pooled)
    if len(games) < 2:
        return None, None, None, None
    per_game = {g: (np.array([t[0] for t in pooled[g]]), np.array([t[1] for t in pooled[g]]),
                    np.array([t[2] for t in pooled[g]], dtype=bool)) for g in games}

    def auc_pair(sel: Sequence[str]):
        va = np.concatenate([per_game[g][0] for g in sel])
        vb = np.concatenate([per_game[g][1] for g in sel])
        w = np.concatenate([per_game[g][2] for g in sel])
        return rank_auc(va, w), rank_auc(vb, w)

    auc_a, auc_b = auc_pair(games)
    diffs = []
    for _ in range(resamples):
        sel = [games[i] for i in rng.integers(0, len(games), size=len(games))]
        xa, xb = auc_pair(sel)
        if xa is not None and xb is not None:
            diffs.append(xb - xa)
    if not diffs:
        return auc_a, auc_b, None, None
    return auc_a, auc_b, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare(rows_a, rows_b, seed: int = 0) -> Dict[str, dict]:
    rng = np.random.default_rng(seed)
    out: Dict[str, dict] = {}
    for lo, hi in BUCKETS + [(1, 10 ** 6)]:
        name = bucket_of(lo) if (lo, hi) in BUCKETS else "all"
        games, a, b, ba, bb, pooled = _per_game_scores(rows_a, rows_b, lo, hi)
        d = b - a
        mean_d, lo_t, hi_t = _ci_t(d)
        lo_b, hi_b = _bootstrap_mean(d, rng)
        p_t, p_w = _pvalues(d)
        db = bb - ba
        brier_d, brier_lo, brier_hi = _ci_t(db)
        auc_a, auc_b, auc_lo, auc_hi = _pooled_auc_bootstrap(pooled, rng)
        out[name] = {
            "n_games": len(games),
            "same_turn_a": float(a.mean()) if len(a) else None,
            "same_turn_b": float(b.mean()) if len(b) else None,
            "diff": mean_d, "diff_ci_t": [lo_t, hi_t], "diff_ci_bootstrap": [lo_b, hi_b],
            "p_paired_t": p_t, "p_wilcoxon": p_w,
            "games_better": int((d > 0).sum()), "games_worse": int((d < 0).sum()),
            "games_tied": int((d == 0).sum()),
            "brier_a": float(ba.mean()) if len(ba) else None,
            "brier_b": float(bb.mean()) if len(bb) else None,
            "brier_diff": brier_d, "brier_diff_ci_t": [brier_lo, brier_hi],
            "pooled_auc_a": auc_a, "pooled_auc_b": auc_b,
            "pooled_auc_diff_ci_bootstrap": [auc_lo, auc_hi],
        }
    return out


def markdown(result: Dict[str, dict], label_a: str, label_b: str) -> str:
    def f(v, spec="+.3f"):
        return "-" if v is None else format(v, spec)

    def ci(pair, spec="+.3f"):
        return "-" if pair[0] is None else f"[{f(pair[0], spec)}, {f(pair[1], spec)}]"

    def p(v):
        return "-" if v is None else (f"{v:.3f}" if v >= 0.001 else f"{v:.1e}")

    lines = [f"| turns | games | same-turn AUC {label_a} | same-turn AUC {label_b} | diff | "
             "95% CI (t) | 95% CI (bootstrap) | p paired t | p Wilcoxon | better / worse / tied | "
             f"Brier {label_a} | Brier {label_b} | Brier diff 95% CI | pooled AUC diff 95% CI |",
             "|---" * 14 + "|"]
    for name, r in result.items():
        lines.append(
            f"| {name} | {r['n_games']} | {f(r['same_turn_a'], '.3f')} | {f(r['same_turn_b'], '.3f')} | "
            f"{f(r['diff'])} | {ci(r['diff_ci_t'])} | {ci(r['diff_ci_bootstrap'])} | "
            f"{p(r['p_paired_t'])} | {p(r['p_wilcoxon'])} | "
            f"{r['games_better']} / {r['games_worse']} / {r['games_tied']} | "
            f"{f(r['brier_a'], '.3f')} | {f(r['brier_b'], '.3f')} | {ci(r['brier_diff_ci_t'])} | "
            f"{ci(r['pooled_auc_diff_ci_bootstrap'])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path, help="baseline record (A)")
    ap.add_argument("b", type=Path, help="candidate record (B)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="path stem for .md and .json")
    args = ap.parse_args(argv)
    rows_a, rows_b = load_rows(args.a), load_rows(args.b)
    shared = len(set(rows_a) & set(rows_b))
    print(f"{len(rows_a)} states in A, {len(rows_b)} in B, {shared} shared")
    result = compare(rows_a, rows_b, args.seed)
    report = markdown(result, args.label_a, args.label_b)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps({
            "a": str(args.a), "b": str(args.b), "label_a": args.label_a,
            "label_b": args.label_b, "shared_states": shared, "buckets": result},
            indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
