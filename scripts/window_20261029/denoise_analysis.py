"""Why do the heads-only rounds' gains decline? A CPU reading of the 8 split
gradients each round's pass saved (Next CPU work, "the target pointer's
query, denoised"; the 2026-11-05 09:43 review's M1). Read-only on
rounds_local/round_KK/pass.pt; no GPU, Idle priority, 2 threads.

Per round, in the step's own coordinates (reduced AdamW on the head maps,
pg_step.direction's heads_only arm) and per block (target_q_proj, the other
head maps, all):
- the signal's squared norm, the mean cross-split dot product of the 8
  splits' directions (unbiased: the splits' games are independent), the
  noise per split, and the signal share of the round's direction;
- between consecutive rounds, the raw cosine and the cosine of the signals
  (the raw dot product over the two signal norms, since the rounds' noises
  are independent): does the signal rotate, or does the noise hide it?
- target_q_proj denoised by a rank-r truncation, or a soft threshold of its
  singular values, chosen on half the splits and scored on the other half,
  over all 35 balanced partitions: the cosine with the other half's estimate
  of the signal, against the raw direction's. An indicator only: the rounds'
  gain is measured in the policy's KL metric by the evaluator on the GPU, and
  a deployment needs a paired half-A reading, pre-registered, as one factor.

Run from the run tree (its wesnoth_ai is the running code):

    python <stage1>/scripts/window_20261029/denoise_analysis.py --rounds 2 3 4 5 6 7 8 --out <json>
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

from wesnoth_ai.pg_step import head_of, pooled_gradient, reduced_adamw, second_moment

RUNS = Path("C:/Users/momom2/wesnoth_runs")
RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 384)
THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)   # multiples of the noise bulk edge
Q = "model.target_q_proj.weight"


def idle(threads: int) -> None:
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        pass
    torch.set_num_threads(threads)


def split_directions(p):
    """The 8 splits' heads-only directions, as {block: (8, n) float64}."""
    heads = [n for n in p["names"] if head_of(n)]
    v_all = second_moment(p)
    v = {n: v_all[n] for n in heads}
    rows = {Q: [], "other": []}
    for i in range(len(p["splits"])):
        g = pooled_gradient(p, [i])
        d = reduced_adamw({n: g[n] for n in heads}, v)
        rows[Q].append(d[Q].double().flatten())
        rows["other"].append(torch.cat([d[n].double().flatten() for n in heads if n != Q]))
    out = {b: torch.stack(r) for b, r in rows.items()}
    out["all"] = torch.cat([out[Q], out["other"]], dim=1)
    return out


def gram_stats(x):
    """Signal, noise and share from the splits' Gram matrix."""
    k = x.shape[0]
    gram = x @ x.T
    off = (gram.sum() - gram.diag().sum()) / (k * (k - 1))
    diag = gram.diag().mean()
    total = gram.sum() / k ** 2
    return {"signal_sq": float(off), "noise_sq_per_split": float(diag - off), "direction_sq": float(total),
            "signal_share": float(off / total) if total > 0 else None}


def denoise_scores(q, shape):
    """Split-half scores of the truncations and soft thresholds of target_q_proj."""
    k = q.shape[0]
    rank_scores = {r: [] for r in RANKS}
    soft_scores = {t: [] for t in THRESHOLDS}
    raw_scores = []
    for a in itertools.combinations(range(k), k // 2):
        if 0 not in a:
            continue                      # each partition once; both halves are used below
        b = tuple(i for i in range(k) if i not in a)
        for h1, h2 in ((a, b), (b, a)):
            xa, xb = q[list(h1)].mean(0), q[list(h2)].mean(0)
            # the signal's norm from the two halves' dot product; skip a partition where it reads <= 0
            s_sq = float(xa @ xb)
            if s_sq <= 0:
                continue
            s = math.sqrt(s_sq)

            def score(x):
                n = float(x.norm())
                return float(x @ xb) / (n * s) if n > 0 else 0.0

            raw_scores.append(score(xa))
            u, sv, vh = torch.linalg.svd(xa.view(shape), full_matrices=False)
            for r in RANKS:
                rank_scores[r].append(score(((u[:, :r] * sv[:r]) @ vh[:r]).flatten()))
            # the noise bulk edge of a half's mean: entries of variance noise / (len(h1) * numel)
            noise_sq = float(((q[list(h1)] - xa) ** 2).sum()) / (len(h1) - 1) / len(h1)
            edge = math.sqrt(noise_sq / xa.numel()) * (math.sqrt(shape[0]) + math.sqrt(shape[1]))
            for t in THRESHOLDS:
                shrunk = torch.clamp(sv - t * edge, min=0.0)
                soft_scores[t].append(score(((u * shrunk) @ vh).flatten()))

    def summary(xs):
        if not xs:
            return None
        m = sum(xs) / len(xs)
        return {"mean": m, "n": len(xs)}
    return {"raw": summary(raw_scores), "rank": {str(r): summary(v) for r, v in rank_scores.items()},
            "soft": {str(t): summary(v) for t, v in soft_scores.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    idle(a.threads)
    report, prev = {}, None
    for k in a.rounds:
        path = RUNS / "rounds_local" / f"round_{k:02d}" / "pass.pt"
        p = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        x = split_directions(p)
        shape = tuple(p["total"][Q].shape)
        r = {"games": p["games"], "blocks": {b: gram_stats(t) for b, t in x.items()}}
        r["q_share_of_signal"] = r["blocks"][Q]["signal_sq"] / r["blocks"]["all"]["signal_sq"]
        mean_all = x["all"].mean(0)
        if prev is not None:
            raw = float(mean_all @ prev[0]) / float(mean_all.norm() * prev[0].norm())
            s1, s0 = r["blocks"]["all"]["signal_sq"], prev[1]
            sig = float(mean_all @ prev[0]) / math.sqrt(s1 * s0) if s1 > 0 and s0 > 0 else None
            r["vs_previous"] = {"raw_cos": raw, "signal_cos": sig}
        r["denoise_q"] = denoise_scores(x[Q], shape)
        report[k] = r
        prev = (mean_all, r["blocks"]["all"]["signal_sq"])
        b = r["blocks"]["all"]
        print(f"round {k}: signal |s| {math.sqrt(max(b['signal_sq'], 0)):.4g}, noise/split "
              f"{math.sqrt(max(b['noise_sq_per_split'], 0)):.4g}, share {b['signal_share']:.3f}, "
              f"q's share of the signal {r['q_share_of_signal']:.3f}"
              + (f", vs previous raw cos {r['vs_previous']['raw_cos']:.3f} signal cos "
                 f"{r['vs_previous']['signal_cos'] if r['vs_previous']['signal_cos'] is None else round(r['vs_previous']['signal_cos'], 3)}"
                 if "vs_previous" in r else ""), flush=True)
        dq = r["denoise_q"]
        best_r = max(dq["rank"].items(), key=lambda kv: kv[1]["mean"] if kv[1] else -1)
        best_t = max(dq["soft"].items(), key=lambda kv: kv[1]["mean"] if kv[1] else -1)
        if dq["raw"]:
            print(f"  target_q_proj split-half cos with the signal: raw {dq['raw']['mean']:.3f}, best rank {best_r[0]} "
                  f"{best_r[1]['mean']:.3f}, best soft threshold {best_t[0]} x edge {best_t[1]['mean']:.3f} "
                  f"({dq['raw']['n']} half-pairs)", flush=True)
        del p, x
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")   # written on the run, round by round


if __name__ == "__main__":
    main()
