"""Which held-out decisions carry a step's KL and its gain (stage C's own
readings; no forward).

Stage C's driver kept, per arm, log pi at theta_k + s d of every half-A
held-out decision's sampled action (its cache: half_<direction>_<s>_A.npz
beside theta_k's own, base.npz). The action a was sampled from pi_k, so

    delta = log pi_k(a) - log pi'(a),      pi' the law at theta_k + s d,

has mean KL(pi_k || pi') over the decisions: a one-sample reading, decision
by decision, of the KL each arm's step was sized on (the search read the
exact KL on a 5 % subsample of the game-sides), and the first-order gain is
the mean over game-sides of R sum_t (e^{-delta_t} - 1)
(wesnoth_ai/pg_step.py first_order_gain). Both are sums over decisions, so
both split by any property of a decision.

The decisions are laid out as the evaluator laid them (tools/pg_eval.py
EvalContext: the held-out games in tools/pg_schedule.py game_order, side 1
then side 2, each side's decisions in log order) from the records alone, and
the layout is checked three ways: its size is the readings', its half-A
decisions are the ones read, and each arm's first-order and exact gains come
out as stage_c.json has them.

Read per arm: the mean of delta and half its mean square with SEs clustered
by game, the spread of |delta|, and the parts of the KL and of the gain by
  logp      the logged log-probability of the sampled action,
  kind      attack, move, recruit, end_turn, and refused or not,
  legal     the number of legal actions,
  position  the decision's place in its game-side,
  rank      the decisions ranked by |delta|,
  side      the game-sides ranked by their own sum of delta^2;
and the decisions with the largest |delta|.

Last, every arm's KL and gain are split between the game-sides that carry
the full-network step's KL (the top shares by reduced_adamw's own sum of
delta^2) and the others, the bulk. A bulk part's gain over the root of its KL
part is what the arm would gain per root KL if the outliers' KL were not
there. delta is read at the sampled action, so a class picked by a side's own
delta leans a little on what was sampled: each split is read a second time
with every decision classed by its side's decisions of the other parity.

    python scripts/window_20261029/trunk_step_tail.py --layout <scratch>/heldout_layout.npz --out training/metrics/trunk_step_tail_20261110.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
RUNS = Path("C:/Users/momom2/wesnoth_runs")
STAGE1 = RUNS / "selfplay_t1_local"
STAGE_C = RUNS / "stage_c_local"
KINDS = ("attack", "move", "recruit", "end_turn")        # wesnoth_ai/pg_records.py ActionKey's kind
LOGP_EDGES = (-8.0, -5.0, -3.0, -1.0)
LEGAL_EDGES = (10, 30, 100, 300)
POSITION_EDGES = (50, 100, 200, 400)
RANK_SHARES = (0.001, 0.01, 0.1)
SIDE_SHARES = (0.01, 0.05, 0.2)
TOP = 30
BULK_BY = "reduced_adamw"                                # the arm whose outlier game-sides the bulk leaves out


def build_layout(seed: int) -> dict:
    """The held-out decisions in the evaluator's order, from the records."""
    from tools.pg_positions import RecordIndex, read_record
    from tools.pg_schedule import game_order
    from wesnoth_ai.pg_records import labelled_games, read_labels
    from wesnoth_ai.sequence_streams import unit_draw
    games = labelled_games(read_labels(STAGE1 / "labels.jsonl"), heldout=True)
    index = RecordIndex(sorted(q for q in STAGE1.glob("games_s*") if q.is_dir()))
    cols = {"kind": [], "refused": [], "legal": [], "logged": []}
    sides = {"game": [], "side": [], "start": [], "stop": [], "half": [], "R": [], "in_kl": []}
    start = 0
    for name in [g.file for g in game_order(list(games), seed) if g.side == 1]:
        per = {1: [], 2: []}
        for d in read_record(index.path(name))["decisions"]:
            per[int(d["side"])].append((int(d["action"][1]), bool(d["refused"]), int(d["n"]), float(d["logp"])))
        for side in (1, 2):
            row, n = games[name][side], len(per[side])
            for key, column in zip(cols, zip(*per[side])):
                cols[key].extend(column)
            for key, value in zip(sides, (name, side, start, start + n, row.half, row.R,
                                          unit_draw("pg_kl", seed, name, side))):
                sides[key].append(value)
            start += n
    return {"kind": np.array(cols["kind"], np.int8), "refused": np.array(cols["refused"], bool),
            "legal": np.array(cols["legal"], np.int32), "logged": np.array(cols["logged"], np.float32),
            "game": np.array(sides["game"]), "side": np.array(sides["side"], np.int8),
            "start": np.array(sides["start"], np.int64), "stop": np.array(sides["stop"], np.int64),
            "half": np.array(sides["half"]), "R": np.array(sides["R"], np.float64),
            "kl_draw": np.array(sides["in_kl"], np.float64)}


def clustered(per_game: np.ndarray, counts: np.ndarray) -> dict:
    """A mean (the games' sums over their counts) with its SE clustered by
    game, as wesnoth_ai/pg_step.py _clustered."""
    total = float(counts.sum())
    mean = float(per_game.sum()) / total
    n = int((counts > 0).sum())
    resid = (per_game - counts * mean)[counts > 0]
    return {"estimate": mean, "se": math.sqrt(n / (n - 1) * float((resid ** 2).sum())) / total}


def bins(values: np.ndarray, edges, name: str):
    """(group index per value, labels): below edges[0], between, at or past edges[-1]."""
    labels = ([f"{name} < {edges[0]:g}"] + [f"{edges[i]:g} <= {name} < {edges[i + 1]:g}" for i in range(len(edges) - 1)]
              + [f"{name} >= {edges[-1]:g}"])
    return np.searchsorted(np.asarray(edges), values, side="right"), labels


class Arm:
    """One reading of half A: delta at its decisions, and what splits it."""

    def __init__(self, layout: dict, read: np.ndarray, base: np.ndarray, logp: np.ndarray):
        self.at = np.flatnonzero(read)                                  # the decisions read, in layout order
        self.delta = (base - logp)[self.at]
        lengths = layout["stop"] - layout["start"]
        side_of = np.repeat(np.arange(len(lengths)), lengths)[self.at]
        self.side_of = side_of
        self.game_of = side_of // 2                                     # a game's two sides are adjacent
        self.n_games = len(lengths) // 2
        sides_read = np.unique(side_of)
        self.sides_read = sides_read
        self.side_count = np.bincount(sides_read // 2, minlength=self.n_games).astype(np.float64)
        self.decision_count = np.bincount(self.game_of, minlength=self.n_games).astype(np.float64)
        self.R = layout["R"]
        self.gain_term = np.expm1(-self.delta) * self.R[side_of]        # R (r_t - 1), by decision

    def per_decision(self, values: np.ndarray, where=None) -> dict:
        """The mean of `values` over all the decisions read, `where` zeroing
        the others (a part of the mean), clustered by game."""
        v = values if where is None else np.where(where, values, 0.0)
        return clustered(np.bincount(self.game_of, weights=v, minlength=self.n_games), self.decision_count)

    def per_side(self, values: np.ndarray, where=None) -> dict:
        """The mean over game-sides of each side's sum of `values`."""
        v = values if where is None else np.where(where, values, 0.0)
        return clustered(np.bincount(self.game_of, weights=v, minlength=self.n_games), self.side_count)

    def exact_gain(self) -> dict:
        sums = np.bincount(self.side_of, weights=self.delta, minlength=len(self.R))[self.sides_read]
        return clustered(np.bincount(self.sides_read // 2, weights=np.expm1(-sums) * self.R[self.sides_read],
                                     minlength=self.n_games), self.side_count)

    def split(self, group: np.ndarray, labels) -> dict:
        """Each group's share of the decisions and its parts of the KL (the
        mean of delta), of half the mean square and of the first-order gain."""
        out = {}
        for i, label in enumerate(labels):
            where = group == i
            if not where.any():
                continue
            out[label] = {"decisions": int(where.sum()), "share": float(where.mean()),
                          "kl_part": self.per_decision(self.delta, where),
                          "half_mean_square_part": float(0.5 * (self.delta[where] ** 2).sum() / len(self.delta)),
                          "median_abs_delta": float(np.median(np.abs(self.delta[where]))),
                          "gain_part": self.per_side(self.gain_term, where)}
        return out


def with_rate(parts: dict) -> dict:
    """`parts` (an Arm.split), each part with its gain over the root of its
    KL: the gain per root KL of a step that paid for this part's KL alone."""
    for part in parts.values():
        kl = part["kl_part"]["estimate"]
        part["gain_per_root_kl"] = part["gain_part"]["estimate"] / math.sqrt(kl) if kl > 0 else None
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", type=Path, required=True, help="the decisions' layout, built once and kept here")
    ap.add_argument("--seed", type=int, default=20261021, help="the evaluator's seed (tools/pg_eval.py EvalConfig)")
    ap.add_argument("--mine", type=Path, default=ROOT / "training/metrics/trunk_kl_by_module_20261110.json",
                    help="a float32 reading whose games are read here too")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:                                   # a reading, not a run: priority is best effort
        pass
    t0 = time.time()
    if a.layout.exists():
        with np.load(a.layout) as z:
            layout = {k: z[k] for k in z.files}
    else:
        layout = build_layout(a.seed)
        a.layout.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.layout, **layout)
    print(f"layout: {len(layout['start'])} game-sides, {int(layout['stop'][-1])} decisions  {time.time() - t0:.0f} s", flush=True)

    cache = next(p for p in (STAGE_C / "driver_cache").iterdir() if (p / "base.npz").exists())
    base = np.load(cache / "base.npz")["logp"]
    if len(base) != int(layout["stop"][-1]):
        raise SystemExit(f"the layout holds {int(layout['stop'][-1])} decisions and the readings {len(base)}")
    lengths = layout["stop"] - layout["start"]
    half_a = np.repeat(layout["half"] == "A", lengths)
    pos = np.concatenate([np.arange(n) for n in lengths])
    arms = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]
    by_step = {float(r["s"]).hex(): name for name, r in arms.items()}
    mine = json.loads(a.mine.read_text(encoding="utf-8")) if a.mine.exists() else {}
    out = {"cache": cache.name, "decisions": len(base), "game_sides": len(lengths),
           "replay": {"mean_logged_minus_base": float(np.mean(layout["logged"] - base)),
                      "half_mean_square": float(0.5 * np.mean((layout["logged"] - base) ** 2))},
           "readings": {}}

    held = {}
    for path in sorted(cache.glob("half_*_A.npz")):
        _, key, step, _ = path.stem.split("_")
        logp = np.load(path)["logp"]
        read = np.isfinite(logp)
        if not np.array_equal(read, half_a):
            raise SystemExit(f"{path.name}: the decisions read are not the layout's half A")
        arm = Arm(layout, read, base, logp)
        d, at = arm.delta, arm.at
        name = by_step.get(step, key)
        kl_file = cache / f"kl_{key}_{step}.json"
        r = {"direction": key, "arm": by_step.get(step), "s": float.fromhex(step),
             "subsample_kl": json.loads(kl_file.read_text(encoding="utf-8"))["value"] if kl_file.exists() else None,
             "decisions": len(d), "game_sides": len(arm.sides_read),
             "kl": arm.per_decision(d), "half_mean_square": arm.per_decision(0.5 * d * d),
             "first_order_gain": arm.per_side(arm.gain_term), "exact_gain": arm.exact_gain(),
             "abs_delta": {f"q{q:g}": float(np.quantile(np.abs(d), q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}}
        if by_step.get(step):
            r["stage_c"] = {k: arms[by_step[step]]["A"][k]["estimate"] for k in ("first_order", "exact")}
        order = np.argsort(-np.abs(d))
        rank = np.empty(len(d), np.int64)
        rank[order] = np.arange(len(d))
        cuts = [int(len(d) * share) for share in RANK_SHARES]
        kind = layout["kind"][at] + len(KINDS) * layout["refused"][at]
        in_kl = (layout["kl_draw"] < 0.05)[arm.side_of]
        r["by"] = {
            "logp": arm.split(*bins(layout["logged"][at], LOGP_EDGES, "logged logp")),
            "kind": arm.split(kind, list(KINDS) + [f"{k}, refused" for k in KINDS]),
            "legal": arm.split(*bins(layout["legal"][at], LEGAL_EDGES, "legal actions")),
            "position": arm.split(*bins(pos[at], POSITION_EDGES, "decision")),
            "rank": arm.split(np.searchsorted(cuts, rank, side="right"),
                              [f"top {RANK_SHARES[0]:g} by |delta|"]
                              + [f"{RANK_SHARES[i]:g} to {RANK_SHARES[i + 1]:g}" for i in range(len(cuts) - 1)]
                              + [f"past {RANK_SHARES[-1]:g}"]),
            "kl_subsample": arm.split(1 - in_kl.astype(np.int64), ["in the KL subsample", "outside it"]),
        }
        square = np.bincount(arm.side_of, weights=d * d, minlength=len(lengths))[arm.sides_read]
        plain = np.bincount(arm.side_of, weights=d, minlength=len(lengths))[arm.sides_read]
        by_square = np.argsort(-square)
        r["by"]["side"] = {f"top {share:g} of the game-sides by sum of delta^2": {
            "sum_square_share": float(square[by_square[:int(len(square) * share)]].sum() / square.sum()),
            "kl_share": float(plain[by_square[:int(len(square) * share)]].sum() / plain.sum())} for share in SIDE_SHARES}
        r["side_mean_delta"] = {f"q{q:g}": float(np.quantile(plain / lengths[arm.sides_read], q))
                                for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)}
        r["mine"] = {}
        for g in mine.get("games", []):
            for i in np.flatnonzero(layout["game"] == g):
                where = arm.side_of == i
                r["mine"][f"{g} side {int(layout['side'][i])}"] = (
                    {"decisions": int(where.sum()), "mean_delta": float(d[where].mean()),
                     "half_mean_square": float(0.5 * (d[where] ** 2).mean())} if where.any() else "half B: not read")
        r["top"] = [{"game": str(layout["game"][arm.side_of[i]]), "side": int(layout["side"][arm.side_of[i]]),
                     "decision": int(pos[at[i]]), "of": int(lengths[arm.side_of[i]]),
                     "kind": KINDS[layout["kind"][at[i]]], "refused": bool(layout["refused"][at[i]]),
                     "legal": int(layout["legal"][at[i]]), "logged_logp": float(layout["logged"][at[i]]),
                     "base_logp": float(base[at[i]]), "delta": float(d[i])} for i in order[:TOP]]
        out["readings"][f"{name} at {float.fromhex(step):.6g}"] = r
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{time.time() - t0:5.0f} s  {name:14s} s {float.fromhex(step):.4g}  KL {r['kl']['estimate']:.4g} +- {r['kl']['se']:.2g}"
              f"  first order {r['first_order_gain']['estimate']:.5f}  exact {r['exact_gain']['estimate']:.5f}"
              f"  stage C {r.get('stage_c')}", flush=True)
        held[name] = arm

    ref = held.get(BULK_BY)
    if ref is not None:
        out["bulk"] = {"ranked_by": BULK_BY, "outside": {}}
        parity = pos[ref.at] % 2                        # every arm read the same decisions
        square = [np.bincount(ref.side_of, weights=np.where(parity == h, ref.delta ** 2, 0.0), minlength=len(lengths))
                  for h in (0, 1)]

        def top(sq: np.ndarray, share: float) -> np.ndarray:
            """The game-sides in the top `share` by `sq`, as a mask."""
            mark = np.zeros(len(lengths), bool)
            mark[ref.sides_read[np.argsort(-sq[ref.sides_read])[:int(len(ref.sides_read) * share)]]] = True
            return mark

        for share in SIDE_SHARES:
            whole = top(square[0] + square[1], share)
            halves = [top(sq, share) for sq in square]
            row = out["bulk"]["outside"][f"the top {share:g}"] = {}
            for name, arm in held.items():
                crossed = np.where(parity == 0, halves[1][arm.side_of], halves[0][arm.side_of])
                row[name] = {how: with_rate(arm.split(mark.astype(np.int64), ["bulk", "outliers"]))
                             for how, mark in (("ranked on all its decisions", whole[arm.side_of]),
                                               ("ranked on the other parity's", crossed))}
                b = row[name]["ranked on the other parity's"]["bulk"]
                print(f"outside the top {share:<5g} {name:14s} KL part {b['kl_part']['estimate']:.4g}"
                      f"  gain part {b['gain_part']['estimate']:.5f} +- {b['gain_part']['se']:.5f}"
                      f"  per root KL {b['gain_per_root_kl']:.3f}", flush=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
