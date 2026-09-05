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


def collect(records: Sequence[Dict], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(pre-grader values, playout means, position ids) over candidates
    where the pre-grader is defined; plus the count of skipped ones."""
    xs, ys, pos, skipped = [], [], [], 0
    for rec in records:
        for cand in candidates_of(rec):
            x = cand.get(key)
            if x is None:
                skipped += 1
                continue
            xs.append(float(x))
            ys.append(playout_mean_se(cand)[0])
            pos.append(int(rec["index"]))
    return np.asarray(xs), np.asarray(ys), np.asarray(pos), skipped


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or float(np.var(x)) == 0.0:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def within_position_sd(residuals: np.ndarray, pos: np.ndarray) -> Optional[float]:
    """Pooled SD of the residuals around their per-position mean
    (positions with one candidate contribute nothing)."""
    ss, dof = 0.0, 0
    for p in np.unique(pos):
        r = residuals[pos == p]
        if len(r) < 2:
            continue
        ss += float(((r - r.mean()) ** 2).sum())
        dof += len(r) - 1
    return math.sqrt(ss / dof) if dof else None


def rank_checks(records: Sequence[Dict], key: str, threshold: float) -> Dict:
    """Confirmed alternatives (gap >= threshold on the playouts): does
    the pre-grader put each above its base? Plus top-pick agreement."""
    confirmed, ranked_above = [], []
    top_agree, top_close, n_positions = 0, 0, 0
    for rec in records:
        cands = candidates_of(rec)
        base = cands[0]
        if base.get(key) is None:
            continue
        base_mean = playout_mean_se(base)[0]
        for alt in cands[1:]:
            gap = playout_mean_se(alt)[0] - base_mean
            if gap >= threshold and alt.get(key) is not None:
                above = float(alt[key]) > float(base[key])
                confirmed.append({"index": rec["index"], "gap": round(gap, 3),
                                  "base": round(float(base[key]), 4),
                                  "alt": round(float(alt[key]), 4),
                                  "ranked_above": bool(above)})
                ranked_above.append(above)
        graded = [c for c in cands if c.get(key) is not None]
        if len(graded) < 2:
            continue
        n_positions += 1
        means = [playout_mean_se(c)[0] for c in graded]
        pick = int(np.argmax([float(c[key]) for c in graded]))
        best = int(np.argmax(means))
        top_agree += pick == best
        top_close += means[best] - means[pick] <= 0.1
    return {
        "confirmed": confirmed,
        "n_confirmed": len(confirmed),
        "n_confirmed_ranked_above": int(sum(ranked_above)),
        "all_confirmed_ranked_above": bool(ranked_above) and all(ranked_above),
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
    for key in PREGRADERS:
        x, y, pos, skipped = collect(records, key)
        a, b = fit_line(x, y)
        resid = y - (a * x + b)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.var(x) > 0 else None
        out["pregraders"][key] = {
            "n": int(len(x)), "n_skipped": skipped,
            "slope": a, "intercept": b, "correlation": corr,
            "residual_sd": float(resid.std(ddof=2)) if len(resid) > 2 else None,
            "residual_sd_within_position": within_position_sd(resid, pos),
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
    if sd >= 0.3 or (v["n_confirmed"] and not v["all_confirmed_ranked_above"]):
        return "KILL (value-head pre-grading)"
    if sd <= 0.2 and v["all_confirmed_ranked_above"]:
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
            f"{fmt(p['residual_sd_within_position'])} within-position "
            f"(playout-mean SD {fmt(p['playout_mean_sd'])})")
        lines.append(
            f"  confirmed alternatives (gap >= threshold): {p['n_confirmed']}, ranked above "
            f"base by the pre-grader: {p['n_confirmed_ranked_above']}; top pick agrees "
            f"{p['top_pick_agrees']}/{p['n_positions_ranked']}, within 0.1: "
            f"{p['top_pick_within_0.1']}/{p['n_positions_ranked']}")
        for c in p["confirmed"]:
            lines.append(f"    position {c['index']}: gap {c['gap']:+.3f}, base {fmt(c['base'], 'g')}, "
                         f"alt {fmt(c['alt'], 'g')}, {'above' if c['ranked_above'] else 'BELOW'}")
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
