"""Where a full-network step's KL goes, by module (CPU, float32; Idle priority).

Stage C's full-network reduced_adamw direction d (its pass.pt) is cut into
module groups d_m (heads, attention, MLPs, norms, slot memory, ...). For each
group:
  KL_m   the evaluator's mean per-decision KL(pi_k || pi at theta_k + s d_m)
         on a few held-out games (tools/pg_eval.py EvalContext.kl, every
         game-side in the subsample), at s = stage C's full step (the heads
         also at the heads-only step, which checks the subset against stage C's
         eps);
  G_m    the first-order gain s c X_m, X_m the cross-split <E g, P E g> over
         the group (unbiased: the eight splits' sums, less their own squares)
         and c the units constant that maps the heads' X onto half A's first
         order (stage C's heads_only arm);
and e_m = G_m / sqrt(KL_m), the group's gain per root KL, which does not depend
on s while KL is quadratic in it. A step that gives each group its own share of
the KL (c_m proportional to G_m / KL_m, the cross terms neglected) reads
sqrt(sum e_m^2) against the full direction's own G / sqrt(KL).

    python scripts/window_20261029/trunk_kl_by_module.py --games 4 --out training/metrics/trunk_kl_by_module_20261110.json
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
RUNS = Path("C:/Users/momom2/wesnoth_runs")
STAGE_C = RUNS / "stage_c_local"
CHECKPOINT = ROOT / "training" / "checkpoints" / "own2.pt"


def group_of(name: str, head_of) -> str:
    """The module group of parameter `name`."""
    if head_of(name):
        return "heads"
    bare = name[len("model."):] if name.startswith("model.") else name
    parts = bare.split(".")
    if parts[0] == "encoder" and len(parts) > 3 and parts[1] == "layers":
        kind = parts[3]
        return ("encoder.attention" if kind == "self_attn" else "encoder.mlp" if kind.startswith("linear")
                else "encoder.norms" if kind.startswith("norm") else "encoder.other")
    return parts[0]


def context_on(games):
    """The evaluator's context on the held-out `games` of stage 1: CPU,
    float32, every game-side in the KL subsample."""
    import torch
    from tools.pg_eval import EvalConfig, EvalContext
    from tools.pg_positions import RecordIndex
    from wesnoth_ai.pg_records import read_labels
    keep = set(games)
    rows = [r for r in read_labels(RUNS / "selfplay_t1_local" / "labels.jsonl") if r.heldout and r.game in keep]
    index = RecordIndex(sorted(q for q in (RUNS / "selfplay_t1_local").glob("games_s*") if q.is_dir()))
    return EvalContext(CHECKPOINT, {g: index.path(g) for g in games}, rows, torch.device("cpu"),
                       EvalConfig(workers=2, slots=64, kl_fraction=1.0))


def held_out_context(n_games: int):
    """(`context_on` `n_games` of stage 1's held-out games, spread evenly
    over their sorted names; the games)."""
    from wesnoth_ai.pg_records import read_labels
    held = sorted({r.game for r in read_labels(RUNS / "selfplay_t1_local" / "labels.jsonl") if r.heldout})
    games = held[:: max(1, len(held) // n_games)][:n_games]
    return context_on(games), games


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=4, help="held-out games the KL reads")
    ap.add_argument("--threads", type=int, default=4)
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
    from wesnoth_ai.pg_step import direction, head_of, pooled_gradient, second_moment

    t0 = time.time()
    p = load_pass(STAGE_C / "pass.pt")
    names = list(p["names"])
    arms = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]
    s_full, s_heads = arms["reduced_adamw"]["s"], arms["heads_only"]["s"]
    d = direction(p, "reduced_adamw")
    v = second_moment(p)
    w = {n: torch.where(v[n] > 0, v[n].float().clamp_min(1e-38).rsqrt(), torch.zeros_like(v[n].float())) for n in names}
    K = len(p["splits"])
    gs = [pooled_gradient(p, [i]) for i in range(K)]
    groups = collections.defaultdict(list)
    for n in names:
        groups[group_of(n, head_of)].append(n)
    X = {}
    for gname, members in groups.items():
        total = sum(float((sum(g[n] for g in gs) ** 2 * w[n]).sum()) for n in members)
        own = sum(float((g[n] ** 2 * w[n]).sum()) for g in gs for n in members)
        X[gname] = (total - own) / (K * (K - 1))
    c = arms["heads_only"]["A"]["first_order"]["estimate"] / (s_heads * X["heads"])
    del gs

    context, games = held_out_context(a.games)
    out = {"games": games, "game_sides": len(context.sides), "s_full": s_full, "s_heads": s_heads, "c": c,
           "X": X, "groups": {}}
    if a.out.exists():                                  # a stopped reading goes on from the groups it wrote
        before = json.loads(a.out.read_text(encoding="utf-8"))
        if (before.get("games"), before.get("s_full")) == (games, s_full):
            out["groups"] = before["groups"]

    def write():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    plan = [("heads", [s_heads, s_full]), ("trunk", [s_full, s_full / 2])]
    plan += [(g, [s_full]) for g in sorted(groups, key=lambda g: -X[g]) if g != "heads"]
    for gname, steps in plan:
        if gname in out["groups"]:
            continue
        members = [n for n in names if not head_of(n)] if gname == "trunk" else groups[gname]
        dm = {n: (d[n] if n in members else torch.zeros_like(d[n])) for n in names}
        kl = context.kl(dm, steps)
        x = sum(X[g] for g in groups if g != "heads") if gname == "trunk" else X[gname]
        out["groups"][gname] = {"kl": {repr(s): kl[s] for s in steps}, "X": x,
                                "gain_at_s_full": c * s_full * x,
                                "gain_per_root_kl": c * s_full * x / math.sqrt(kl[s_full]) if kl[s_full] > 0 else None}
        print(f"{time.time() - t0:7.0f} s  {gname:18s} KL {kl}  X {x:.4g}", flush=True)
        write()
    context.close()
    e = {g: r["gain_per_root_kl"] for g, r in out["groups"].items() if g not in ("trunk",) and r["gain_per_root_kl"]}
    out["summary"] = {
        "trunk_kl_sum_of_groups": sum(r["kl"][repr(s_full)] for g, r in out["groups"].items() if g not in ("trunk", "heads")),
        "trunk_kl": out["groups"]["trunk"]["kl"][repr(s_full)],
        "trunk_gain_per_root_kl": out["groups"]["trunk"]["gain_per_root_kl"],
        "heads_gain_per_root_kl": out["groups"]["heads"]["gain_per_root_kl"],
        "best_by_group_gain_per_root_kl": math.sqrt(sum(x * x for g, x in e.items() if g != "heads")),
        "best_with_heads_gain_per_root_kl": math.sqrt(sum(x * x for x in e.values())),
    }
    write()
    print(json.dumps(out["summary"], indent=1), f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
