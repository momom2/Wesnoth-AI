"""How much of stage C's raw gradient lies on the direction that pushes the
memory's pattern? A reader of the pass and of trunk_memory_project.py's
push.pt (the step's part in the span of the push gradients).

It gives the gradient's cosine with the step, with the push and with the rest
of the step; the step's first-order gain g . d and the push's and the rest's
parts of it; between the pass's halves, the gradient's signal (g_a . g_b holds
no noise) and its share on the push's line; and the same by module group.

    python scripts/window_20261029/trunk_memory_gradient.py --push <dir>/push.pt --out training/metrics/trunk_memory_gradient_20261110.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--push", type=Path, required=True, help="trunk_memory_project.py's push.pt")
    ap.add_argument("--threads", type=int, default=2)
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
    from trunk_kl_by_module import STAGE_C
    from trunk_memory_groups import memory_group
    from wesnoth_ai.pg_step import direction, head_of, pooled_gradient

    push = torch.load(a.push, weights_only=True)
    names = list(push)
    passed = load_pass(STAGE_C / "pass.pt")
    s = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]

    def pick(d):
        return {n: (d[n].detach().float().cpu() if n in d else torch.zeros_like(push[n])) for n in names}

    def one(x, y, n):
        return float((x[n].double() * y[n].double()).sum())

    def dot(x, y):
        return sum(one(x, y, n) for n in names)

    g, d = pick(direction(passed, "sgd")), pick(direction(passed, "reduced_adamw"))
    half = len(passed["splits"]) // 2
    g_half = [pick(pooled_gradient(passed, h)) for h in (tuple(range(half)), tuple(range(half, len(passed["splits"]))))]
    unread = [n for n in passed["names"] if n not in push]
    del passed
    rest = {n: d[n] - push[n] for n in names}
    norm = {key: dot(v, v) ** 0.5 for key, v in (("gradient", g), ("step", d), ("push", push), ("rest", rest))}
    gain = {"step": dot(g, d), "push": dot(g, push), "rest": dot(g, rest)}
    on = [dot(h, push) / norm["push"] for h in g_half]          # each half's gradient on the push's unit vector
    signal = dot(g_half[0], g_half[1])
    groups = {}
    for n in names:
        row = groups.setdefault(memory_group(n, head_of), {"numel": 0, "gradient_sq": 0.0, "step_sq": 0.0,
                                                            "push_sq": 0.0, "gain": 0.0, "gain_on_push": 0.0})
        row["numel"] += push[n].numel()
        row["gradient_sq"] += one(g, g, n)
        row["step_sq"] += one(d, d, n)
        row["push_sq"] += one(push, push, n)
        row["gain"] += one(g, d, n)
        row["gain_on_push"] += one(g, push, n)
    out = {"s": s, "push": a.push.name, "names": len(names), "pass_names_unread": len(unread), "norm": norm,
           "gain": gain, "gain_x_s": {key: s * value for key, value in gain.items()},
           "cosine_with_gradient": {key: gain[key] / (norm["gradient"] * norm[key]) for key in gain},
           "halves": {"gradient_signal_share": signal / norm["gradient"] ** 2,
                      "signal_share_on_push": on[0] * on[1] / signal,
                      "gradient_share_on_push": (gain["push"] / norm["push"]) ** 2 / norm["gradient"] ** 2},
           "groups": groups}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in out.items() if key != "groups"}, indent=1))
    for group, row in sorted(groups.items(), key=lambda kv: -kv[1]["gain"]):
        print(f"{group:20s} |g| {row['gradient_sq'] ** 0.5:7.3f} |d| {row['step_sq'] ** 0.5:7.3f} "
              f"|push| {row['push_sq'] ** 0.5:6.3f} gain {row['gain']:8.3f} ({row['gain'] / gain['step']:.3f} of the step's), "
              f"on the push {row['gain_on_push']:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
