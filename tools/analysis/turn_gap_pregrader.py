"""Forward-only pre-graders against the playout truth of a turn_gap run.

Pre-registered in docs/turn_gap_prereg_20260904.md ("Pre-grader
measurement"): for every candidate turn with playouts, the playout mean
is the truth; `value_post` (the value head on the post-turn state) and
`hp_margin_post` (mover HP minus opponent HP) are the pre-graders.
Reported per pre-grader:

- the global least-squares fit y = a x + b over all candidates and the
  residual SD around it, plain and within-position (the estimand:
  position-level offsets removed);
- the noise floor: the mean standard error of the playout means, so a
  residual SD near it means the pre-grader is as good as the playouts;
- whether every alternative whose gap to its base is >= threshold is
  ranked above the base by the pre-grader;
- how often the pre-grader's top candidate is the playout top, or
  within 0.1 of it.

Usage:
    python tools/analysis/turn_gap_pregrader.py RESULT.json [--threshold 0.25]

Works on partial files (`<out>.partial.json`).
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PREGRADERS = ("value_post", "hp_margin_post")


def candidates_of(record: Dict) -> List[Dict]:
    """Base first, then the alternatives; only candidates with playouts."""
    cands = [record["base"]] + list(record.get("alternatives", []))
    return [c for c in cands if c.get("outcomes")]


def playout_mean_se(candidate: Dict) -> Tuple[float, float]:
    xs = np.asarray(candidate["outcomes"], dtype=float)
    se = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
    return float(xs.mean()), se


def pregrader_value(candidate: Dict, key: str) -> Optional[float]:
    """The pre-grader's read of a candidate. A turn that ended the game
    has no value_post; the value of a terminal state is its outcome
    (the repeated terminal result), which is what a value head reads."""
    x = candidate.get(key)
    if x is None and key == "value_post" and candidate.get("terminal_in_turn"):
        return float(candidate["outcomes"][0])
    return None if x is None else float(x)


def collect(records: Sequence[Dict], key: str):
    """(pre-grader values, playout means, playout counts, position ids)
    over candidates where the pre-grader is defined; plus the count of
    skipped ones."""
    xs, ys, ns, pos, skipped = [], [], [], [], 0
    for rec in records:
        for cand in candidates_of(rec):
            x = pregrader_value(cand, key)
            if x is None:
                skipped += 1
                continue
            xs.append(x)
            ys.append(playout_mean_se(cand)[0])
            ns.append(len(cand["outcomes"]))
            pos.append(int(rec["index"]))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns, dtype=float), np.asarray(pos), skipped


def fit_line(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Least squares y = a x + b, weighted by `w` (the playout counts:
    a candidate graded on few playouts says less about the fit)."""
    if len(x) < 2 or float(np.var(x)) == 0.0:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    a, b = np.polyfit(x, y, 1, w=None if w is None else np.sqrt(w))
    return float(a), float(b)


def weighted_sd(values: np.ndarray, w: np.ndarray, ddof: int) -> Optional[float]:
    """SD of `values` around their weighted mean, weights normalized to
    the sample size (so ddof keeps its meaning)."""
    if len(values) <= ddof:
        return None
    w = w * len(w) / w.sum()
    mean = float((w * values).sum() / w.sum())
    return math.sqrt(float((w * (values - mean) ** 2).sum()) / (len(values) - ddof))


def within_position_sd(residuals: np.ndarray, pos: np.ndarray,
                       w: Optional[np.ndarray] = None) -> Optional[float]:
    """Pooled SD of the residuals around their per-position (weighted)
    mean; positions with one candidate contribute nothing."""
    w = np.ones(len(residuals)) if w is None else w
    ss, dof = 0.0, 0
    for p in np.unique(pos):
        m = pos == p
        r, wp = residuals[m], w[m]
        if len(r) < 2:
            continue
        wp = wp * len(wp) / wp.sum()
        mean = float((wp * r).sum() / wp.sum())
        ss += float((wp * (r - mean) ** 2).sum())
        dof += len(r) - 1
    return math.sqrt(ss / dof) if dof else None


def rank_checks(records: Sequence[Dict], key: str, threshold: float) -> Dict:
    """Confirmed alternatives (gap >= threshold on the playouts): does
    the pre-grader put each above its base? Plus top-pick agreement."""
    confirmed = []
    n_unranked = 0
    top_agree, top_close, n_positions = 0, 0, 0
    for rec in records:
        cands = candidates_of(rec)
        base = cands[0]
        base_x = pregrader_value(base, key)
        base_mean = playout_mean_se(base)[0]
        for alt in cands[1:]:
            gap = playout_mean_se(alt)[0] - base_mean
            if gap < threshold:
                continue
            alt_x = pregrader_value(alt, key)
            if base_x is None or alt_x is None:
                n_unranked += 1
                continue
            rank = "above" if alt_x > base_x else ("tie" if alt_x == base_x else "below")
            confirmed.append({"index": rec["index"], "gap": round(gap, 3),
                              "base": round(base_x, 4), "alt": round(alt_x, 4),
                              "rank": rank, "ranked_above": rank == "above"})
        graded = [(c, pregrader_value(c, key)) for c in cands]
        graded = [(c, x) for c, x in graded if x is not None]
        if len(graded) < 2:
            continue
        n_positions += 1
        means = [playout_mean_se(c)[0] for c, _ in graded]
        pick = int(np.argmax([x for _, x in graded]))
        best = int(np.argmax(means))
        top_agree += pick == best
        top_close += means[best] - means[pick] <= 0.1
    ranks = [c["rank"] for c in confirmed]
    return {
        "confirmed": confirmed,
        "n_confirmed": len(confirmed),
        "n_confirmed_unranked": n_unranked,
        "n_confirmed_ranked_above": ranks.count("above"),
        "n_confirmed_tied": ranks.count("tie"),
        "n_confirmed_ranked_below": ranks.count("below"),
        "all_confirmed_ranked_above": bool(ranks) and all(r == "above" for r in ranks),
        "n_positions_ranked": n_positions,
        "top_pick_agrees": top_agree,
        "top_pick_within_0.1": top_close,
    }


def analyze(records: Sequence[Dict], threshold: float) -> Dict:
    ses = [playout_mean_se(c)[1] for r in records for c in candidates_of(r)]
    ses = [s for s in ses if not math.isnan(s)]
    out = {"n_positions": len(records),
           "n_candidates": int(sum(len(candidates_of(r)) for r in records)),
           "playout_mean_se_mean": float(np.mean(ses)) if ses else None,
           "threshold": threshold, "pregraders": {}}
    noise_var = float(np.mean([s ** 2 for s in ses])) if ses else 0.0
    for key in PREGRADERS:
        x, y, n, pos, skipped = collect(records, key)
        a, b = fit_line(x, y, n)
        resid = y - (a * x + b)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.var(x) > 0 else None
        within = within_position_sd(resid, pos, n)
        out["pregraders"][key] = {
            "n": int(len(x)), "n_skipped": skipped,
            "slope": a, "intercept": b, "correlation": corr,
            "residual_sd": weighted_sd(resid, n, 2) if len(resid) > 2 else None,
            "residual_sd_within_position": within,
            # The playout means carry their own noise (mean SE^2 over
            # the candidates); what is left is the pre-grader's error.
            "residual_sd_within_position_noise_corrected": (
                None if within is None else math.sqrt(max(0.0, within ** 2 - noise_var))),
            "playout_mean_sd": float(y.std(ddof=1)) if len(y) > 1 else None,
            **rank_checks(records, key, threshold),
        }
    return out


def verdict(result: Dict) -> str:
    """The pre-registered rule for the value head: kill at residual SD
    >= 0.3 or a confirmed alternative ranked below its base; alive at
    <= 0.2 with every confirmed alternative above."""
    v = result["pregraders"]["value_post"]
    sd = v["residual_sd_within_position"]
    if sd is None:
        return "undetermined (no within-position residuals)"
    if sd >= 0.3:
        return "KILL (residual SD >= 0.3)"
    if v["n_confirmed_ranked_below"]:
        return "KILL (a confirmed alternative ranked below its base)"
    if sd <= 0.2:
        if v["n_confirmed"] == 0:
            return "ALIVE on the residual SD; the ranking check is vacuous (no confirmed alternative)"
        if v["n_confirmed_tied"]:
            return "inconclusive (residual SD passes, a confirmed alternative ties its base)"
        return "ALIVE"
    return "inconclusive (between the kill and the pass)"


def fmt(v, spec=".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def report(result: Dict) -> str:
    lines = [f"positions {result['n_positions']}, candidates with playouts "
             f"{result['n_candidates']}, playout-mean SE (noise floor) "
             f"{fmt(result['playout_mean_se_mean'])}, threshold {result['threshold']:g}"]
    for key, p in result["pregraders"].items():
        lines.append(
            f"{key}: n={p['n']} (skipped {p['n_skipped']}), fit slope {fmt(p['slope'])} "
            f"intercept {fmt(p['intercept'])}, corr {fmt(p['correlation'])}, "
            f"residual SD {fmt(p['residual_sd'])} plain / "
            f"{fmt(p['residual_sd_within_position'])} within-position / "
            f"{fmt(p['residual_sd_within_position_noise_corrected'])} noise-corrected "
            f"(playout-mean SD {fmt(p['playout_mean_sd'])}; fits weighted by playouts)")
        lines.append(
            f"  confirmed alternatives (gap >= threshold): {p['n_confirmed']} ranked "
            f"(above {p['n_confirmed_ranked_above']}, tie {p['n_confirmed_tied']}, below "
            f"{p['n_confirmed_ranked_below']}) + {p['n_confirmed_unranked']} without a read; "
            f"top pick agrees {p['top_pick_agrees']}/{p['n_positions_ranked']}, within 0.1: "
            f"{p['top_pick_within_0.1']}/{p['n_positions_ranked']}")
        for c in p["confirmed"]:
            lines.append(f"    position {c['index']}: gap {c['gap']:+.3f}, base {fmt(c['base'], 'g')}, "
                         f"alt {fmt(c['alt'], 'g')}, {c['rank'].upper() if c['rank'] != 'above' else 'above'}")
    lines.append(f"verdict (value head): {verdict(result)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result", type=Path)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--json", type=Path, help="also write the analysis here")
    args = ap.parse_args(argv)
    data = json.loads(args.result.read_text(encoding="utf-8"))
    records = [r for r in data["positions"] if r.get("base", {}).get("outcomes")]
    if not records:
        print("no positions with playouts yet")
        return 1
    result = analyze(records, args.threshold)
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
