"""Does a step's own change of the policy go the way the memory's displacement
takes it? A reader of trunk_memory_states.py's per-decision columns.

While both changes are small the full lane's change of the logits is the
weights lane's plus the memory lane's, so at a decision

    (KL_full - KL_weights - KL_memory) / (2 sqrt(KL_weights KL_memory))

is the cosine of the two changes in the policy's own metric. It is read over a
side's first decisions, where the memory's part is still small.

Usage: trunk_memory_cross.py <states.json> [...] [--out record.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIRST = 48                       # the decisions read: the linear regime ends inside a side's first turns
SMALL, FLOOR = 3e-4, 1e-7        # the memory's KL below SMALL; both KLs above float32's floor
SEGMENTS = ((1, 6), (6, 12), (12, 24), (24, 48))


def column(reading: dict, name: str) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in reading[name]], float)


def read(record: dict) -> dict:
    pooled: dict[str, list] = {"cos": [], "dw": [], "dm": []}
    sides = {}
    for name, side in record["sides"].items():
        c = side["reading"]
        full, w, m = (column(c, f"{lane}_kl") for lane in ("full", "weights", "memory"))
        ok = ~np.array(c["shadow"]) & ~np.isnan(full + w + m) & (m < SMALL) & (m > FLOOR) & (w > FLOOR)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (full - w - m) / (2 * np.sqrt(w * m))
        rows = {}
        for a, b in SEGMENTS:
            seg = ok.copy()
            seg[:a] = False
            seg[b:] = False
            if seg.sum() >= 3:
                rows[f"[{a},{b})"] = {"decisions": int(seg.sum()), "cosine": float(np.median(cos[seg])),
                                      "memory_over_weights": float(np.median(m[seg] / w[seg]))}
        early = ok.copy()
        early[FIRST:] = False
        # the lanes' logp columns are already differences from the reference on its own memory
        for key, values in (("cos", cos), ("dw", column(c, "weights_logp")), ("dm", column(c, "memory_logp"))):
            pooled[key].append(values[early])
        sides[name] = rows
    cat = {key: np.concatenate(values) for key, values in pooled.items()}
    keep = ~np.isnan(cat["dw"] + cat["dm"])
    return {"direction": record["direction"], "s": record["s"], "decisions": int(len(cat["cos"])),
            "cosine": {"median": float(np.median(cat["cos"])), "q25": float(np.percentile(cat["cos"], 25)),
                       "q75": float(np.percentile(cat["cos"], 75)), "above_0": float(np.mean(cat["cos"] > 0))},
            "logged_logp_correlation": float(np.corrcoef(cat["dw"][keep], cat["dm"][keep])[0, 1]),
            "sides": sides}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("states", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = {"first": FIRST, "small": SMALL, "floor": FLOOR, "readings": {}}
    for path in a.states:
        reading = read(json.loads(path.read_text(encoding="utf-8")))
        out["readings"][path.name] = reading
        c = reading["cosine"]
        print(f"{path.name} ({reading['direction']}): {reading['decisions']} decisions, cosine median {c['median']:+.2f} "
              f"(quartiles {c['q25']:+.2f} {c['q75']:+.2f}; above 0 on {c['above_0']:.2f}); the logged action's logp "
              f"change, weights against memory: r {reading['logged_logp_correlation']:+.2f}")
        for name, rows in reading["sides"].items():
            print(f"  {name[-8:]}: " + "; ".join(f"{k} n {r['decisions']} cos {r['cosine']:+.2f} "
                                                 f"m/w {r['memory_over_weights']:.2f}" for k, r in rows.items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
