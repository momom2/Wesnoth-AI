"""Why the trunk has not trained: two CPU readings of stage C's saved files
(read-only; Idle priority, 4 threads).

1. Where a full-network step's KL and gain go. Reduced AdamW is elementwise,
   so the heads_only arm's direction is the reduced_adamw direction on the
   head maps (pg_step.heads_only). The two arms' KL searches then say how far
   the full step moves the heads against the heads-only step (the ratio of
   their s), and so how much of the full step's first-order gain on half A
   the heads carry. Also the trunk's norm by module.
2. Whether the plant reached the gradient. The planted labels add delta Y to
   each game's result (tools/pg_plant.py), so the planted pass's gradient less
   the unplanted one is delta u in expectation, u = plant_grads.pt's u. The
   planted games are the main pass's first 5,000 training games, and pass.pt
   holds only the 20,000 games' sums, so the main pass's mean stands in for
   the same games' (both unbiased for the same expected gradient). The
   projection on u is read for the heads and the trunk apart, its SE from the
   planted pass's eight splits.

    python scripts/window_20261029/trunk_diagnosis.py --out training/metrics/trunk_diagnosis_20261110.json
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STAGE_C = Path("C:/Users/momom2/wesnoth_runs/stage_c_local")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-c", type=Path, default=STAGE_C)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:                                   # a reading, not a run: priority is best effort
        pass
    import torch
    torch.set_num_threads(4)
    from tools.pg_stage_c import load_pass
    from wesnoth_ai.pg_step import direction, head_of, norm, pooled_gradient

    sc = args.stage_c
    out = {}
    main_pass = load_pass(sc / "pass.pt")
    arms = json.loads((sc / "stage_c.json").read_text(encoding="utf-8"))["arms"]
    d = direction(main_pass, "reduced_adamw")
    heads = {n: t for n, t in d.items() if head_of(n)}
    trunk = {n: t for n, t in d.items() if not head_of(n)}
    by_module = collections.defaultdict(float)
    for n, t in trunk.items():
        bare = n[len("model."):] if n.startswith("model.") else n
        by_module[bare.split(".")[0]] += float(t.pow(2).sum())
    total = sum(by_module.values())
    s_full, s_heads = arms["reduced_adamw"]["s"], arms["heads_only"]["s"]
    a_full, a_heads = arms["reduced_adamw"]["A"], arms["heads_only"]["A"]
    out["kl_and_gain"] = {
        "norm": {"full": norm(d), "heads": norm(heads), "trunk": norm(trunk)},
        "trunk_norm2_share": {k: v / total for k, v in sorted(by_module.items(), key=lambda x: -x[1]) if v / total > 1e-4},
        "s": {"reduced_adamw": s_full, "heads_only": s_heads, "ratio": s_heads / s_full},
        "heads_part_first_order_in_full_step": a_heads["first_order"]["estimate"] * s_full / s_heads,
        "full_step_first_order": a_full["first_order"]["estimate"],
        "half_A": {arm: {k: a["A"][k]["estimate"] for k in ("first_order", "exact")} for arm, a in arms.items()},
        "best_mix_exact": math.hypot(a_heads["exact"]["estimate"], a_full["exact"]["estimate"]),
    }

    planted = load_pass(sc / "pass_planted.pt")
    grads = torch.load(sc / "plant_grads.pt", map_location="cpu", weights_only=True)
    names = list(planted["names"])
    u = {n: sum(part[n].float() for part in grads["u"]) for n in names}
    g_main = {n: main_pass["total"][n].float() for n in names}

    def dot(a, b, keep):
        return sum(float((a[n] * b[n]).sum()) for n in names if keep(n))

    out["plant_in_gradient"] = {"delta": float(grads["delta"]), "planted_games": planted["games"],
                                "main_games": main_pass["games"]}
    split_g = [pooled_gradient(planted, [i]) for i in range(len(planted["splits"]))]
    for part, keep in (("all", lambda n: True), ("heads", lambda n: bool(head_of(n))),
                       ("trunk", lambda n: not head_of(n))):
        uu = dot(u, u, keep)
        diff = {n: planted["total"][n].float() - g_main[n] for n in names}
        per = [dot({n: g[n] - g_main[n] for n in names}, u, keep) / uu for g in split_g]
        mean = sum(per) / len(per)
        se = math.sqrt(sum((x - mean) ** 2 for x in per) / (len(per) - 1) / len(per))
        out["plant_in_gradient"][part] = {"u_norm2": uu, "projection": dot(diff, u, keep) / uu, "se": se,
                                          "cosine": dot(diff, u, keep) / math.sqrt(dot(diff, diff, keep) * uu)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
