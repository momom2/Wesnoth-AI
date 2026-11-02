"""Does the 16-decision window bias the pass's full-network gradient? (the
2026-11-02 21:45 review, item 1). Stage C's pass loss (its labels, its
corrections, theta_k = own2) on a few training games, computed twice on the
CPU in float32: windowed at 16 decisions as the pass runs, and in one window
per game-side by recomputation. The head maps must agree (their gradient
reads no memory: a built-in control); the trunk's agreement is the reading.
Reported per parameter group: the cosine and the relative norm of the
difference, raw and in stage C's reduced-AdamW metric (the trunk arms'
direction). Run from the worktree of the running code:

    python C:/Users/momom2/wesnoth_runs/tools/window_check.py --games 8 --threads 6 --out C:/Users/momom2/wesnoth_runs/window_check8.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tools"))
import torch

from tools.pg_pass import GradientPass, PassConfig, PassRefused
from tools.pg_positions import RecordIndex
from wesnoth_ai.pg_records import read_labels
from wesnoth_ai.pg_step import head_of, reduced_adamw, second_moment

RUNS = Path("C:/Users/momom2/wesnoth_runs")
S1 = RUNS / "selfplay_t1_local"
SC = RUNS / "stage_c_local"


def group(name: str) -> str:
    """The parameter's group: its head map, else the module path's first two parts."""
    head = head_of(name)
    if head:
        return "head:" + head
    bare = name[len("model."):] if name.startswith("model.") else name
    return ".".join(bare.split(".")[:2])


def compare(a, b, names):
    groups = {}
    for n in names:
        groups.setdefault(group(n), []).append(n)
    groups["ALL"] = list(names)
    groups["ALL_BUT_HEADS"] = [n for n in names if head_of(n) is None]
    out = {}
    for g, ns in sorted(groups.items()):
        x = torch.cat([a[n].double().flatten() for n in ns])
        y = torch.cat([b[n].double().flatten() for n in ns])
        nx, ny = float(x.norm()), float(y.norm())
        out[g] = {"params": int(x.numel()), "norm_windowed": nx, "norm_full": ny,
                  "cos": float(x @ y) / (nx * ny) if nx > 0 and ny > 0 else None,
                  "rel_diff": float((x - y).norm()) / ny if ny > 0 else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--checkpoint", default=str(Path.cwd() / "training" / "checkpoints" / "own2.pt"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    labels = read_labels(S1 / "labels.jsonl")
    train = sorted({r.game for r in labels if not r.heldout})
    games = set(train[:: max(1, len(train) // a.games)][: a.games])
    labels = [r for r in labels if r.game in games]
    index = RecordIndex(sorted(p for p in S1.glob("games_s*") if p.is_dir()))
    records = {g: index.path(g) for g in games}
    sc = json.loads((SC / "pass.json").read_text(encoding="utf-8"))["config"]
    c = json.loads((SC / "corrections.json").read_text(encoding="utf-8"))
    base = PassConfig(streams=sc["streams"], window=sc["window"], seed=sc["seed"], slots=sc["slots"],
                      workers=a.workers, prefetch=sc["prefetch"], procedure=sc["procedure"], floor_dtype="off",
                      alpha=float(c["alpha"]), gamma=float(c["gamma"]), luck_beta=tuple(float(b) for b in c["beta"]))
    result = {"games": sorted(games), "config": {"window": sc["window"], "streams": sc["streams"],
                                                  "alpha": base.alpha, "gamma": base.gamma}}
    totals = {}
    for label, cfg in (("windowed", replace(base, recompute=True)),
                       ("full", replace(base, window=10 ** 7, recompute=True))):
        t = time.time()
        try:
            r = GradientPass(Path(a.checkpoint), records, labels, torch.device("cpu"), cfg).run()
        except PassRefused as e:              # the replay limit on the CPU: the sums are still the pass's
            r = e.result
            result[label + "_refused"] = str(e)
        totals[label] = {n: r["total"][n].float() for n in r["total"]}
        result[label] = {"decisions": r["decisions"], "windows": r["windows"], "seconds": round(time.time() - t, 1),
                         "replay_kl": r["replay"]["kl"]}
        print(json.dumps(result[label]), f"{time.time() - t0:.0f} s", flush=True)
    names = sorted(totals["windowed"])
    result["raw"] = compare(totals["windowed"], totals["full"], names)
    pp = torch.load(SC / "pass.pt", map_location="cpu", weights_only=True)
    v = {n: t.float() for n, t in second_moment(pp).items()}
    result["reduced_adamw"] = compare(reduced_adamw(totals["windowed"], v), reduced_adamw(totals["full"], v), names)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k in ("raw", "reduced_adamw"):
        print(k, json.dumps({g: (round(x["cos"], 4) if x["cos"] is not None else None,
                                 round(x["rel_diff"], 4) if x["rel_diff"] is not None else None)
                             for g, x in result[k].items()}), flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
