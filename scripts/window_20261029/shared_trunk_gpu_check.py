"""The shared-trunk walk on the GPU (fix/shares-trunk-zeros, departure 28):
the heads-only arm's own direction from stage C's pass, read at theta_k and
two rungs in one walk on one trunk, against each rung read alone, on a few
held-out game-sides of stage 1, under the rounds' evaluator config (fp16,
the packed trunk). Prints the largest log-probability and class-mass
differences, the trunk reads and the walls. Run from the worktree under test:

    HIP_VISIBLE_DEVICES=1 <rocm python> C:/Users/momom2/wesnoth_runs/tools/shared_trunk_gpu_check.py --games 24
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tools"))
import numpy as np
import torch

from tools.pg_eval import EvalConfig, EvalContext
from tools.pg_positions import RecordIndex
from wesnoth_ai.pg_records import read_labels
from wesnoth_ai.pg_step import direction, norm

RUNS = Path("C:/Users/momom2/wesnoth_runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24, help="held-out games read (both sides of each)")
    ap.add_argument("--checkpoint", default="C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-run/training/checkpoints/own2.pt")
    ap.add_argument("--device", default="cuda", help="cpu for a smoke run (no float16 there)")
    a = ap.parse_args()
    s1 = RUNS / "selfplay_t1_local"
    rows = read_labels(s1 / "labels.jsonl")
    held = sorted({r.game for r in rows if r.heldout})[: a.games]
    keep = set(held)
    rows = [r for r in rows if not r.heldout or r.game in keep]
    index = RecordIndex(sorted(p for p in s1.glob("games_s*") if p.is_dir()))
    records = {g: index.path(g) for g in held}
    pass_ = torch.load(RUNS / "stage_c_local" / "pass.pt", map_location="cpu", weights_only=True)
    context = EvalContext(Path(a.checkpoint), records, rows, torch.device(a.device), EvalConfig(workers=6, fp16=True))
    d = direction(pass_, "heads_only", context.params())
    d = {n: t.to(device=context.device, dtype=torch.float32) for n, t in d.items()}   # as NetworkEvaluator does
    print(f"names {len(d)}, nonzero {sum(bool(torch.any(t)) for t in d.values())}, norm {norm(d):.4g}, "
          f"shares_trunk {context.shares_trunk(d)}", flush=True)
    stage_c = json.loads((RUNS / "stage_c_local" / "stage_c.json").read_text(encoding="utf-8"))
    rungs = sorted({float(r["s"]) for r in stage_c["ladder"]["rungs"] if r.get("s")})[:2] or [0.02, 0.04]
    steps = [0.0, *rungs]
    sides = list(context.sides)
    t0 = time.time()
    together = context.read(d, steps, sides)
    t_shared = time.time() - t0
    reads_shared = Counter(context.reads)
    t0 = time.time()
    alone = {s: context.read(d, [s], sides)[s] for s in steps}
    t_alone = time.time() - t0
    worst = {}
    for s in steps:
        ok = ~np.isnan(together[s].logp)
        worst[s] = (float(np.max(np.abs(together[s].logp[ok] - alone[s].logp[ok]))),
                    float(np.nanmax(np.abs(together[s].masses - alone[s].masses))), int(ok.sum()))
    moved = float(np.max(np.abs(together[steps[-1]].logp[ok] - together[0.0].logp[ok])))
    print(json.dumps({"steps": steps, "worst_logp_mass_decisions": worst, "rung_moves_logp_by": moved,
                      "reads_shared": dict(reads_shared), "reads_total": dict(context.reads),
                      "wall_shared_s": round(t_shared, 1), "wall_alone_s": round(t_alone, 1)}, indent=1), flush=True)
    context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
