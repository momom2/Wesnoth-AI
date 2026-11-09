"""Half B's first-order gain of each adopted step, split by decision class
(the 2026-11-06 21:43 review's suggestion 1). Read-only on wesnoth_runs;
Idle priority, 4 processes, no GPU.

For round K, the step's half-B reading at the adopted rung is, to first
order, the clustered mean over half-B game-sides of R * sum_t expm1(r_t),
r_t = log pi_new(a_t) - log pi_old(a_t). That sum splits exactly by the
class of decision t (attack, move, recruit, end_turn, refused), so the
class terms add to the total. The exact reading (R * expm1(sum_t r_t)) is
printed beside it, the reading runner.log logged for the round; it does not split.

Alignment check: the record's own logp (play time, fp16 serving) against
the cache's base logp (the pass, fp32) at the same positions.

    python -I scripts/window_20261029/class_split.py 4 13 > training/metrics/class_split_halfB.txt

Each round's result is kept in training/metrics/class_split_halfB.json as it
is read, and a round already there is not read again.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import psutil  # noqa: E402

RUN = "C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-run"
OUT = Path(__file__).resolve().parents[2] / "training" / "metrics" / "class_split_halfB.json"
SEED = 20261021
CLASSES = ("attack", "move", "recruit", "end_turn", "refused")


def _idle():
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    if RUN not in sys.path:
        sys.path.insert(0, RUN)


def _side_rows(path: str):
    """Per side: (class codes, record logp) of its decisions in play order."""
    _idle()
    from tools.pg_positions import read_record
    from wesnoth_ai.pg_records import decisions_of
    out = {}
    decs = decisions_of(read_record(Path(path)))
    for s in (1, 2):
        mine = [d for d in decs if d.side == s]
        cls = np.array([4 if d.refused else int(d.action[1]) for d in mine], dtype=np.int8)
        out[s] = (cls, np.array([d.logp for d in mine]))
    return out


def clustered(v, games):
    keys, inv = np.unique(np.array(games), return_inverse=True)
    pg = np.bincount(inv, weights=v, minlength=len(keys))
    cnt = np.bincount(inv, minlength=len(keys)).astype(float)
    m = pg.sum() / cnt.sum()
    n = len(keys)
    se = np.sqrt(n / (n - 1) * ((pg - cnt * m) ** 2).sum()) / cnt.sum()
    return float(m), float(se)


def one_round(k: int, pool) -> dict:
    from tools.pg_schedule import game_order
    root = Path("C:/Users/momom2/wesnoth_runs/rounds_local/round_%02d" % k)
    labels = [json.loads(x) for x in open(root / "play/labels.jsonl", encoding="utf-8") if x.strip()]
    held = {}
    for lab in labels:
        if lab["heldout"]:
            held.setdefault(lab["game"], {})[lab["side"]] = lab
    files = {}
    for shard in ("games_s00", "games_s01"):
        for p in (root / "play" / shard).glob("*.game.jsonl.gz"):
            files[p.name[:-len(".game.jsonl.gz")]] = p
    games = sorted(held)
    rows = dict(zip(games, pool.map(_side_rows, [str(files[g]) for g in games], chunksize=16)))
    cache = next((root / "eval_cache").iterdir())
    base = np.load(cache / "base_B.npz")["logp"]
    new_f = next(cache.glob("half_*_B.npz"))
    r = np.load(new_f)["logp"] - base
    start = 0
    B = []
    for gs in game_order(games, SEED):
        cls, rec_lp = rows[gs.file][gs.side]
        n = len(cls)
        lab = held[gs.file][gs.side]
        if lab["half"] == "B":
            B.append((gs.file, lab["R"], cls, rec_lp, start, start + n))
        start += n
    assert start == len(base), (start, len(base))
    gnames = [g for g, *_ in B]
    lp_rec = np.concatenate([x[3] for x in B])
    lp_base = np.concatenate([base[a:b] for *_, a, b in B])
    assert not np.isnan(lp_base).any() and not np.isnan(r[np.concatenate([np.arange(a, b) for *_, a, b in B])]).any()
    fin = np.isfinite(lp_rec)
    res = {"round": k, "file": new_f.name, "sides": len(B), "decisions": int(len(lp_base)),
           "align_corr": float(np.corrcoef(lp_rec[fin], lp_base[fin])[0, 1]),
           "align_mad": float(np.median(np.abs(lp_rec[fin] - lp_base[fin])))}
    fo_tot = np.array([R * np.expm1(r[a:b]).sum() for _, R, _, _, a, b in B])
    ex_tot = np.array([R * np.expm1(r[a:b].sum()) for _, R, _, _, a, b in B])
    res["fo"], res["exact"] = clustered(fo_tot, gnames), clustered(ex_tot, gnames)
    allcls = np.concatenate([x[2] for x in B])
    allr = np.concatenate([r[a:b] for *_, a, b in B])
    res["classes"] = {}
    for c, name in enumerate(CLASSES):
        v = np.array([R * np.expm1(r[a:b][cls == c]).sum() for _, R, cls, _, a, b in B])
        sel = allcls == c
        res["classes"][name] = {"fo": clustered(v, gnames), "share": float(sel.mean()),
                                "mean_dp": float(np.expm1(allr[sel]).mean()) if sel.any() else 0.0,
                                "rms_r": float(np.sqrt((allr[sel] ** 2).mean())) if sel.any() else 0.0}
    return res


def main():
    _idle()
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    with ProcessPoolExecutor(4, initializer=_idle) as pool:
        for k in range(lo, hi + 1):
            if str(k) in done:
                res = done[str(k)]
            else:
                res = one_round(k, pool)
                done[str(k)] = res
                OUT.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
            line = "round %2d: B fo %+.4f +- %.4f exact %+.4f +- %.4f | align r %.4f mad %.4f |" % (
                k, *res["fo"], *res["exact"], res["align_corr"], res["align_mad"])
            for name in CLASSES:
                c = res["classes"][name]
                line += " %s %+.4f +- %.4f (%.0f%%)" % (name, *c["fo"], 100 * c["share"])
            print(line, flush=True)


if __name__ == "__main__":
    main()
