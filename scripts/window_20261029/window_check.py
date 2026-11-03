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
from wesnoth_ai.pg_records import N_SPLITS, read_labels
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


def groups_of(names):
    groups = {"ALL_BUT_HEADS": [n for n in names if head_of(n) is None],
              "encoder.layers": [n for n in names if group(n) == "encoder.layers"],
              "slot_memory": [n for n in names if group(n).startswith("slot_memory")]}
    groups["trunk_rest"] = [n for n in groups["ALL_BUT_HEADS"] if n not in set(groups["encoder.layers"])
                            and n not in set(groups["slot_memory"])]
    return groups


def split_signal(sx, sy, names):
    """The expected windowed and full gradients' agreement from the splits:
    the cross-split sums S_xy = sum over splits a != b of <x_a, y_b> are
    unbiased for N_a N_b <E x, E y> whatever each game's own noise, so
    S_xy / sqrt(S_xx S_yy) estimates the cosine of the expected gradients
    (None when S_xx or S_yy reads <= 0: the signal unresolved). The share
    S_yy / |sum y|^2 is the part of the full total's square that is signal.
    Read from the splits' Gram matrices; `cos_se` is the delete-one-split
    jackknife's standard error of the cosine (None below 3 splits, or when a
    leave-one-out reading is unresolved)."""
    out = {}
    for g, ns in groups_of(names).items():
        if not ns:
            continue
        xs = [torch.cat([s[n].double().flatten() for n in ns]) for s in sx]
        ys = [torch.cat([s[n].double().flatten() for n in ns]) for s in sy]
        m = len(xs)
        gram = {k: torch.zeros(m, m, dtype=torch.float64) for k in ("xx", "yy", "xy")}
        for i in range(m):
            for j in range(m):
                gram["xy"][i, j] = float(xs[i] @ ys[j])
                if j >= i:
                    gram["xx"][i, j] = gram["xx"][j, i] = float(xs[i] @ xs[j])
                    gram["yy"][i, j] = gram["yy"][j, i] = float(ys[i] @ ys[j])

        def reading(keep):
            idx = torch.tensor(keep)
            s = {k: float(v[idx][:, idx].sum() - v[idx, idx].sum()) for k, v in gram.items()}
            return s, (s["xy"] / (s["xx"] * s["yy"]) ** 0.5 if s["xx"] > 0 and s["yy"] > 0 else None)
        s, cos = reading(list(range(m)))
        loo = [reading([j for j in range(m) if j != i])[1] for i in range(m)] if m >= 3 else []
        se = None
        if loo and all(c is not None for c in loo):
            mean = sum(loo) / m
            se = ((m - 1) / m * sum((c - mean) ** 2 for c in loo)) ** 0.5
        tx, ty = float(gram["xx"].sum()), float(gram["yy"].sum())
        out[g] = {"S_xx": s["xx"], "S_yy": s["yy"], "S_xy": s["xy"], "cos_signal": cos, "cos_se": se,
                  "splits": m,
                  "signal_share_full": s["yy"] / ty if ty > 0 else None,
                  "signal_share_windowed": s["xx"] / tx if tx > 0 else None}
    return out


def nonempty(splits):
    """The indices of the splits that hold a game (an empty split's sums are zeros)."""
    return [i for i, s in enumerate(splits["full"]) if any(bool(torch.any(t)) for t in s.values())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0, help="shift the stride's start: a disjoint subset")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--checkpoint", default=str(Path.cwd() / "training" / "checkpoints" / "own2.pt"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save", type=Path, default=None, help="the splits' sums of both gradients (a .pt), to pool runs")
    ap.add_argument("--resplit", action="store_true",
                    help="deal the games round-robin over the pass's splits (the labels' own splits can put a "
                         "small sample in two or three, which leaves the cross-split reading one product)")
    ap.add_argument("--pool", type=Path, nargs="*", default=[],
                    help="earlier runs' --save files, on games disjoint from this run's: their non-empty splits "
                         "join the cross-split reading")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    labels = read_labels(S1 / "labels.jsonl")
    train = sorted({r.game for r in labels if not r.heldout})
    games = set(train[a.offset:: max(1, len(train) // a.games)][: a.games])
    labels = [r for r in labels if r.game in games]
    if a.resplit:
        deal = {g: i % N_SPLITS for i, g in enumerate(sorted(games))}
        labels = [replace(r, split=deal[r.game]) for r in labels]
    pooled = {"windowed": [], "full": []}
    for p in a.pool:                          # read before the passes: a bad pool file fails in seconds, not hours
        d = torch.load(p, map_location="cpu", weights_only=True)
        shared = games & set(d["games"])
        if shared:
            raise SystemExit(f"{p} shares {len(shared)} games with this run: the splits must be independent")
        keep = nonempty(d["splits"])
        for label in pooled:
            pooled[label] += [d["splits"][label][i] for i in keep]
    index = RecordIndex(sorted(p for p in S1.glob("games_s*") if p.is_dir()))
    records = {g: index.path(g) for g in games}
    sc = json.loads((SC / "pass.json").read_text(encoding="utf-8"))["config"]
    c = json.loads((SC / "corrections.json").read_text(encoding="utf-8"))
    base = PassConfig(streams=sc["streams"], window=sc["window"], seed=sc["seed"], slots=sc["slots"],
                      workers=a.workers, prefetch=sc["prefetch"], procedure=sc["procedure"], floor_dtype="off",
                      alpha=float(c["alpha"]), gamma=float(c["gamma"]), luck_beta=tuple(float(b) for b in c["beta"]))
    result = {"games": sorted(games), "config": {"window": sc["window"], "streams": sc["streams"],
                                                  "alpha": base.alpha, "gamma": base.gamma}}
    totals, splits = {}, {}
    for label, cfg in (("windowed", replace(base, recompute=True)),
                       ("full", replace(base, window=10 ** 7, recompute=True))):
        t = time.time()
        try:
            r = GradientPass(Path(a.checkpoint), records, labels, torch.device("cpu"), cfg).run()
        except PassRefused as e:              # the replay limit on the CPU: the sums are still the pass's
            r = e.result
            result[label + "_refused"] = str(e)
        totals[label] = {n: r["total"][n].float() for n in r["total"]}
        splits[label] = [{n: s[n].float() for n in s} for s in r["splits"]]
        result[label] = {"decisions": r["decisions"], "windows": r["windows"], "seconds": round(time.time() - t, 1),
                         "replay_kl": r["replay"]["kl"]}
        print(json.dumps(result[label]), f"{time.time() - t0:.0f} s", flush=True)
    names = sorted(totals["windowed"])
    result["raw"] = compare(totals["windowed"], totals["full"], names)
    pp = torch.load(SC / "pass.pt", map_location="cpu", weights_only=True)
    v = {n: t.float() for n, t in second_moment(pp).items()}
    result["reduced_adamw"] = compare(reduced_adamw(totals["windowed"], v), reduced_adamw(totals["full"], v), names)
    result["split_counts"] = [sum(1 for n in names if bool(torch.any(s[n]))) > 0 for s in splits["full"]]
    keep = nonempty(splits)
    sw = [splits["windowed"][i] for i in keep] + pooled["windowed"]
    sf = [splits["full"][i] for i in keep] + pooled["full"]
    result["pooled"] = {"files": [str(p) for p in a.pool], "splits": len(pooled["full"])}
    result["signal_raw"] = split_signal(sw, sf, names)
    result["signal_reduced_adamw"] = split_signal([reduced_adamw(s, v) for s in sw],
                                                  [reduced_adamw(s, v) for s in sf], names)
    if a.save is not None:
        torch.save({"games": sorted(games), "splits": splits}, a.save)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k in ("signal_raw", "signal_reduced_adamw"):
        print(k, json.dumps({g: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in x.items()
                                 if kk in ("cos_signal", "cos_se", "signal_share_full", "signal_share_windowed")}
                             for g, x in result[k].items()}), flush=True)
    for k in ("raw", "reduced_adamw"):
        print(k, json.dumps({g: (round(x["cos"], 4) if x["cos"] is not None else None,
                                 round(x["rel_diff"], 4) if x["rel_diff"] is not None else None)
                             for g, x in result[k].items()}), flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
