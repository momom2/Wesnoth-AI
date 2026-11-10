"""How much of a step's KL is the pass's own noise (CPU, float32; Idle priority).

Stage C's direction is P g, g the mean gradient of one pass's games: the
policy's gradient plus the noise of a finite pass. The first-order gain
<g_true, P g> takes the noise at zero mean, while the KL, a quadratic form in
the step, pays for it in full. Two halves A and B of the pass's splits give
directions d_A and d_B with the same signal and independent noises, and
reduced AdamW is linear in g under the pass's one second moment, so with n_A
and n_B their games and N the pass's

    (n_A d_A + n_B d_B) / N              is the pass's own direction,
    sqrt(n_A n_B) (d_A - d_B) / N        holds no signal, and noise of the
                                         covariance the pass's direction has.

The KL of the second over the KL of the first, at one step, on the evaluator's
float32 path (tools/pg_eval.py EvalContext.kl, a few held-out games, every
game-side), is the noise's share n of the step's KL, read for the trunk at
stage C's full-network step and for the heads at its heads-only step. A
direction with share n gains sqrt(1 - n) of what its signal alone would gain
per root KL, and a pass M times as large has the share n / (n + M (1 - n)).

Before any forward, by module group and with the splits' gradients weighted by
v^0, v^-1/2 and v^-1 (the plain metric, reduced AdamW's gain <g', g / sqrt v>,
and its direction's own squared norm): rho_1, the mean off-diagonal of the
eight splits' Gram matrix over the mean of its diagonal (the cosine of two
splits' gradients), and the signal's share of the pooled direction,
K rho_1 / (1 + (K - 1) rho_1).

    python scripts/window_20261029/trunk_direction_noise.py --out training/metrics/trunk_direction_noise_20261110.json
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

METRICS = {"plain": 0.0, "gain (v^-1/2)": 0.5, "direction (v^-1)": 1.0}   # the power of 1/v that weights g_i g_j
PAIRINGS = {"first four against last four": ((0, 1, 2, 3), (4, 5, 6, 7)),
            "even against odd": ((0, 2, 4, 6), (1, 3, 5, 7))}


def split_grams(gs, v, groups) -> dict:
    """{group: {metric: the K x K matrix of sum g_i g_j v^-power over the
    group's weights}}."""
    import torch
    out = {g: {m: torch.zeros(len(gs), len(gs), dtype=torch.float64) for m in METRICS} for g in groups}
    for group, members in groups.items():
        for name in members:
            rows = torch.stack([g[name].reshape(-1) for g in gs]).double()
            var = v[name].reshape(-1).double()
            live = var > 0                              # reduced AdamW moves no weight whose v is 0
            safe = torch.where(live, var, torch.ones_like(var))
            for metric, power in METRICS.items():
                weight = safe.pow(-power) if power else torch.ones_like(safe)
                out[group][metric] += (rows * torch.where(live, weight, torch.zeros_like(weight))) @ rows.T
    return out


def signal_share(gram) -> dict:
    """rho_1 and the pooled direction's signal share from a splits' Gram
    matrix (module docstring)."""
    k = gram.shape[0]
    own = float(gram.diagonal().mean())
    cross = float((gram.sum() - gram.diagonal().sum()) / (k * (k - 1)))
    if not own > 0:                                     # a group the pass's loss does not reach
        return {"own": own, "cross": cross, "rho_1": None, "signal_share": None}
    rho = cross / own
    return {"own": own, "cross": cross, "rho_1": rho, "signal_share": k * rho / (1 + (k - 1) * rho)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=4, help="held-out games the KL reads")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--offline", action="store_true", help="the splits' cosines alone: no forward")
    ap.add_argument("--pooled", type=Path, default=ROOT / "training/metrics/trunk_kl_by_module_20261110.json",
                    help="a reading of the pass's own direction on the same games and steps, taken when it is there")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as error:                          # a reading, not a run: priority is best effort
        print(f"WARNING: not at Idle priority ({error!r})", flush=True)
    import torch
    torch.set_num_threads(a.threads)
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import STAGE_C, group_of, held_out_context
    from wesnoth_ai.pg_step import direction, head_of, pooled_gradient, second_moment

    t0 = time.time()
    p = load_pass(STAGE_C / "pass.pt")
    names = list(p["names"])
    arms = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]
    steps = {"trunk": arms["reduced_adamw"]["s"], "heads": arms["heads_only"]["s"]}
    v = second_moment(p)
    K = len(p["splits"])
    games = [int(n) for n in p["split_games"]]
    out = {"split_games": games, "steps": steps, "splits": {}, "kl": {}}

    def write():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---- offline: the splits' cosines, by module group -----------------------
    groups = collections.defaultdict(list)
    for n in names:
        groups[group_of(n, head_of)].append(n)
    groups["trunk"] = [n for n in names if not head_of(n)]
    gs = [pooled_gradient(p, [i]) for i in range(K)]
    for group, by_metric in split_grams(gs, v, groups).items():
        out["splits"][group] = {"weights": sum(v[n].numel() for n in groups[group]),
                                **{metric: signal_share(gram) for metric, gram in by_metric.items()}}
    del gs
    write()
    for group, r in sorted(out["splits"].items(), key=lambda kv: -kv[1]["weights"]):
        print(f"{group:18s} {r['weights']:>9d} weights  " + "  ".join(
            f"{metric}: " + ("no gradient" if r[metric]["rho_1"] is None else
                             f"rho {r[metric]['rho_1']:+.4f} share {r[metric]['signal_share']:+.3f}")
            for metric in METRICS), flush=True)
    if a.offline:
        print(f"{time.time() - t0:.0f} s", flush=True)
        return 0

    # ---- the KL of each half-difference, and of the pass's own direction -----
    d = direction(p, "reduced_adamw")
    members = {"trunk": set(groups["trunk"]), "heads": set(groups["heads"])}

    def part(full, group):
        return {n: (full[n] if n in members[group] else torch.zeros_like(full[n])) for n in names}

    context, held = held_out_context(a.games)
    out["games"], out["game_sides"] = held, len(context.sides)
    before = json.loads(a.pooled.read_text(encoding="utf-8")) if a.pooled.exists() else {}
    for group, s in steps.items():
        known = before.get("groups", {}).get(group, {}).get("kl", {}).get(repr(s)) if before.get("games") == held else None
        kl = known if known is not None else context.kl(part(d, group), [s])[s]
        out["kl"][group] = {"s": s, "pooled": kl, "pooled_from": a.pooled.name if known is not None else "read here",
                            "noise": {}}
        print(f"{time.time() - t0:7.0f} s  {group:6s} the pass's direction     KL {kl:.4g}", flush=True)
        write()
    for label, (first, second) in PAIRINGS.items():
        n_a, n_b = (sum(games[i] for i in half) for half in (first, second))
        d_a, d_b = (direction(p, "reduced_adamw", gradient=pooled_gradient(p, half)) for half in (first, second))
        total = n_a + n_b
        back = max(float(((n_a * d_a[n] + n_b * d_b[n]) / total - d[n]).abs().max()) for n in names)
        scale = max(float(d[n].abs().max()) for n in names)
        if back > 1e-4 * scale:
            raise ValueError(f"{label}: the halves' directions do not sum to the pass's ({back:.3g} of {scale:.3g})")
        c = math.sqrt(n_a * n_b) / total
        noise = {n: c * (d_a[n] - d_b[n]) for n in names}
        del d_a, d_b
        for group, s in steps.items():
            kl = context.kl(part(noise, group), [s])[s]
            r = out["kl"][group]
            r["noise"][label] = {"kl": kl, "share": kl / r["pooled"]}
            print(f"{time.time() - t0:7.0f} s  {group:6s} {label:28s} KL {kl:.4g}  over the pass's {kl / r['pooled']:.3f}",
                  flush=True)
            write()
    context.close()
    for group, r in out["kl"].items():
        share = sum(x["share"] for x in r["noise"].values()) / len(r["noise"])
        r["noise_share"] = share
        r["gain_kept"] = math.sqrt(max(0.0, 1.0 - share))
    write()
    print(json.dumps({g: {k: r[k] for k in ("pooled", "noise_share", "gain_kept")} for g, r in out["kl"].items()},
                     indent=1), f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
