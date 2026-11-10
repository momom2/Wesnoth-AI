"""How many earlier rounds should a step's averaged direction take? A CPU
reading of the rounds' saved passes and directions (prereg departure 37's
average; the question departure 43 leaves open). Read-only on
rounds_local/round_KK/{pass.pt,direction.pt}; no GPU, Idle priority, 2
threads.

For each target round k, in the step's own coordinates (the heads_only arm:
reduced AdamW on the head maps, pg_step.direction), the step along the
average of d_k and the K rounds before it, u_K = averaged([d_k, d_k-1, ...,
d_k-K]) (pg_step.averaged: unit norms, equal weights), is scored by its
cosine with round k's signal s_k, the expected direction of round k's games:

    cos_K = u_K . s_k / (|u_K| |s_k|)

- |s_k|^2 is the mean cross-split dot product of round k's 8 split
  directions x_i (unbiased: the splits' games are independent), and d_k is
  their mean, so E[d_k] = s_k.
- u_K . s_k: the earlier rounds' part P_K is independent of round k's games,
  so P_K . d_k is unbiased for P_K . s_k. Round k's own part d_k / |d_k|
  reads |s_k|^2 / |d_k| in expectation.
- |u_K| is the step's own norm, its noise included.

So cos_K reads how much of each step points along round k's own signal: the
stale part (the earlier rounds' signal rotated away) and the noise both
lower it. K = 2 is departure 37's average (--average 2).

    python <lean worktree>/scripts/window_20261029/average_window.py --rounds 2-18 --max-k 12 --out <json>
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from wesnoth_ai.pg_step import head_of, pooled_gradient, reduced_adamw, second_moment  # noqa: E402

RUNS = Path("C:/Users/momom2/wesnoth_runs/rounds_local")


def idle(threads: int) -> None:
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as error:                          # priority is best effort, but never silent
        print(f"WARNING: not at Idle priority ({error!r})", flush=True)
    torch.set_num_threads(threads)


def flat(d, names):
    return torch.cat([d[n].double().flatten() for n in names])


def round_reading(k: int):
    """(d_k flat, |s_k|^2, the head names) from round k's pass."""
    p = torch.load(RUNS / f"round_{k:02d}" / "pass.pt", map_location="cpu", weights_only=True, mmap=True)
    heads = [n for n in p["names"] if head_of(n)]
    v_all = second_moment(p)
    v = {n: v_all[n] for n in heads}
    xs = []
    for i in range(len(p["splits"])):
        g = pooled_gradient(p, [i])
        xs.append(flat(reduced_adamw({n: g[n] for n in heads}, v), heads))
    x = torch.stack(xs)
    gram = x @ x.T
    m = x.shape[0]
    signal_sq = float((gram.sum() - gram.diag().sum()) / (m * (m - 1)))
    return x.mean(0), signal_sq, heads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True, help="first-last target rounds, e.g. 4-18")
    ap.add_argument("--max-k", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    idle(a.threads)
    first, last = (int(t) for t in a.rounds.split("-"))
    saved = {}                     # round j: its direction.pt over the heads, unit norm
    report = {}
    for k in range(first, last + 1):
        d_k, s_sq, heads = round_reading(k)
        saved_k = torch.load(RUNS / f"round_{k:02d}" / "direction.pt", map_location="cpu", weights_only=True)
        dk_saved = flat(saved_k, heads)
        check = float(dk_saved @ d_k) / float(dk_saved.norm() * d_k.norm())
        size = float(d_k.norm())
        own = d_k / size
        r = {"signal_sq": s_sq, "direction_sq": size ** 2, "signal_share": s_sq / size ** 2,
             "saved_vs_recomputed_cos": check, "cos": {}}
        past = torch.zeros_like(d_k)
        for K in range(0, a.max_k + 1):
            if K > 0:
                j = k - K
                if j < 2 or j not in saved:
                    break
                past = past + saved[j]
            u = (own + past) / (K + 1)
            dot = (s_sq / size + float(past @ d_k)) / (K + 1)
            r["cos"][K] = dot / (float(u.norm()) * math.sqrt(s_sq)) if s_sq > 0 else None
        report[k] = r
        saved[k] = dk_saved / dk_saved.norm()
        best = max((c, K) for K, c in r["cos"].items() if c is not None) if s_sq > 0 else None
        print(f"round {k}: share {r['signal_share']:.3f}, saved/recomputed cos {check:.4f}, "
              + " ".join(f"K{K}:{c:.3f}" for K, c in r["cos"].items() if c is not None)
              + (f", best K {best[1]} ({best[0]:.3f})" if best else ""), flush=True)
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    ks = sorted({K for r in report.values() for K in r["cos"]})
    for K in ks:
        vals = [r["cos"][K] for k, r in report.items() if K in r["cos"] and r["cos"][K] is not None and k >= 8]
        if vals:
            print(f"K {K:2d}: mean cos over rounds 8 on {sum(vals) / len(vals):.3f} ({len(vals)} rounds)", flush=True)


if __name__ == "__main__":
    main()
