"""The pattern a trunk step's memory gap lies on, and the memory's course along
it (a reader of trunk_memory_mode.py's and trunk_memory_states.py's outputs;
CPU, seconds).

The pattern p is a unit vector of the memory's width: each side's gap from
trunk_memory_mode.py's arrays, averaged over the slots and over the decisions
past the tenth, normalized, and averaged over the sides. --pattern reads one
saved earlier instead, and --save-pattern keeps this one.

Per side of the mode reading: the cosine of the side's own gap pattern with p
and with the memory's mean vector, that pattern's share in the top six axes of
the memory's motion (the PCA of its slot-mean trajectory), the memory's
one-write change along p by kind of decision, and by turn the memory's
projection on p with theta_k's write on the gap's common part (the gate z, the
trunk's gain, the summed log growth).

Per side of the states reading: each lane's mean KL and its shares by law, and
by turn the projection on p of theta_k's memory, the stepped network's and the
free lane's, with each lane's mean KL.

    python scripts/window_20261029/trunk_memory_pattern.py --mode <json> --mode-arrays <dir> --states <json> --states-arrays <dir> --out training/metrics/trunk_memory_pattern_20261110.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KINDS = ("attack", "move", "recruit", "end_turn")
START = 10      # the pattern is read past the decisions that open the gap
AXES = 6
LANES = ("full", "weights", "memory", "free")
MEMORIES = (("reference", "memory_mean"), ("stepped", "full_mean"), ("free", "free_mean"))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-30)


def arrays_of(directory: Path, name: str):
    return np.load(directory / f"{name.replace(':', '_side')}.npz")


def rank_corr(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def kinds_in(kind) -> dict:
    return {name: int((kind == i).sum()) for i, name in enumerate(KINDS)}


def mode_side(side: dict, z, p) -> dict:
    """One side of trunk_memory_mode.py's reading against the pattern p."""
    c = {k: np.array(side["reading"][k]) for k in ("turn", "kind", "shadow", "gap", "gap_full", "gap_common_growth",
                                                    "gap_common_z", "gap_common_trunk_gain")}
    turn, kind, live = np.round(c["turn"]), c["kind"], ~c["shadow"].astype(bool)
    gap, memory = z["gap_mean"], z["memory_mean"]
    own = unit(gap[START:].mean(0))
    motion = memory[START:] - memory[START:].mean(0)
    _, sizes, axes = np.linalg.svd(motion, full_matrices=False)
    on_p = memory @ p
    change = np.diff(on_p)
    growth = np.log(c["gap_common_growth"])
    rows = []
    for u in np.unique(turn):
        m = turn == u
        rows.append({"turn": int(u), "decisions": int(m.sum()), **kinds_in(kind[m]),
                     "memory_on_p": float(on_p[m].mean()), "z": float(c["gap_common_z"][m].mean()),
                     "trunk_gain": float(c["gap_common_trunk_gain"][m].mean()),
                     "log_growth": float(growth[m & live].sum()),
                     "gap": float(c["gap"][m].mean()), "gap_full": float(c["gap_full"][m].mean())})
    return {"R": side["R"], "cos_with_p": float(own @ p),
            "cos_with_memory_mean": float(own @ unit(memory[START:].mean(0))),
            "share_in_top_axes": float(((axes[:AXES] @ own) ** 2).sum()),
            "axes_variance": (sizes[:AXES] ** 2 / (sizes ** 2).sum()).tolist(),
            "change_on_p_by_kind": {name: float(change[kind[:-1] == i].mean())
                                    for i, name in enumerate(KINDS) if (kind[:-1] == i).any()},
            "rank_corr_on_p_with_trunk_gain": rank_corr(on_p[live], c["gap_common_trunk_gain"][live]),
            "by_turn": rows}


def states_side(side: dict, z, p, laws) -> dict:
    """One side of trunk_memory_states.py's reading against the pattern p."""
    c = side["reading"]
    live = ~np.array(c["shadow"])
    turn, kind = np.round(np.array(c["turn"])), np.array(c["kind"])
    on_p = {name: z[key] @ p for name, key in MEMORIES}

    def mean(column, keep):
        rows = [x for x, k in zip(c[column], keep) if k and x is not None]
        return float(np.mean(rows)) if rows else None

    def by_law(lane):
        rows = np.array([x for x, k in zip(c[f"{lane}_by_law"], live) if k and x is not None], float)
        return dict(zip(laws, (rows.sum(0) / rows.sum()).tolist()))

    rows = []
    for u in np.unique(turn):
        m = turn == u
        rows.append({"turn": int(u), "decisions": int(m.sum()), **kinds_in(kind[m]),
                     **{f"{name}_on_p": float(v[m].mean()) for name, v in on_p.items()},
                     "size_full": mean("full_size", m),
                     **{f"kl_{lane}": mean(f"{lane}_kl", m & live) for lane in LANES}})
    return {"R": side["R"], "decisions": int(len(live)),
            "kl": {lane: mean(f"{lane}_kl", live) for lane in LANES},
            "same_top": {lane: mean(f"{lane}_same_top", live) for lane in LANES},
            "by_law": {lane: by_law(lane) for lane in ("full", "weights", "memory")},
            "peak_on_p": {name: float(v.max()) for name, v in on_p.items()},
            "by_turn": rows}


def show(name: str, rows, columns) -> None:
    print(f"\n=== {name}")
    print("  " + "  ".join(f"{column:>13s}" for column in columns))
    for row in rows:
        print("  " + "  ".join(f"{'-' if row[c] is None else format(row[c], '13.4g')}" if not isinstance(row[c], int)
                               else f"{row[c]:13d}" for c in columns))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", type=Path, help="trunk_memory_mode.py's JSON")
    ap.add_argument("--mode-arrays", type=Path, help="its --arrays directory")
    ap.add_argument("--states", type=Path, help="trunk_memory_states.py's JSON")
    ap.add_argument("--states-arrays", type=Path, help="its --arrays directory")
    ap.add_argument("--pattern", type=Path, help="a .npy unit vector to read against, for the mode reading's own")
    ap.add_argument("--save-pattern", type=Path, help="keep the pattern as a .npy")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.pattern is None and a.mode is None:
        ap.error("the pattern comes from --mode with --mode-arrays, or from --pattern")
    out = {}
    mode = json.loads(a.mode.read_text(encoding="utf-8")) if a.mode else None
    if a.pattern:
        p = np.load(a.pattern)
    else:
        p = unit(sum(unit(arrays_of(a.mode_arrays, name)["gap_mean"][START:].mean(0)) for name in mode["sides"]))
    if a.save_pattern:
        np.save(a.save_pattern, p)
    out["pattern"] = p.tolist()
    out["participation"] = float(1 / (p ** 4).sum())    # the cells that carry it, of the memory's width
    if mode:
        out["mode"] = {name: mode_side(side, arrays_of(a.mode_arrays, name), p) for name, side in mode["sides"].items()}
        for name, side in out["mode"].items():
            print(f"\n{name}: cos with p {side['cos_with_p']:+.3f}, share in the top {AXES} axes of the memory's motion "
                  f"{side['share_in_top_axes']:.3f}, one-write change on p by kind "
                  f"{ {k: round(v, 3) for k, v in side['change_on_p_by_kind'].items()} }")
            show(name, side["by_turn"], ("turn", "decisions", "attack", "recruit", "memory_on_p", "z", "trunk_gain",
                                          "log_growth", "gap", "gap_full"))
    if a.states:
        states = json.loads(a.states.read_text(encoding="utf-8"))
        out["states"] = {name: states_side(side, arrays_of(a.states_arrays, name), p, states["laws"])
                         for name, side in states["sides"].items() if "summary" in side}
        for name, side in out["states"].items():
            print(f"\n{name}: mean KL a decision { {k: f'{v:.3g}' for k, v in side['kl'].items() if v is not None} }, "
                  f"peak on p { {k: round(v, 2) for k, v in side['peak_on_p'].items()} }")
            show(name, side["by_turn"], ("turn", "decisions", "attack", "recruit", "reference_on_p", "stepped_on_p",
                                          "free_on_p", "size_full", "kl_full", "kl_weights", "kl_memory", "kl_free"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
