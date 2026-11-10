"""The memory's gap along the pattern, by direction (a reader of
trunk_memory_states.py's outputs with --arrays; CPU, seconds).

Per reading and side: the gap (the stepped network's memory minus the
reference's, slot means) projected on the pattern p and its cosine with p, at
a few decisions and at its peak; the gap's size at the peak over the
reference memory's; the turn each memory releases the pattern (the first
decision after the reference's peak on p where it reads under RELEASED); and
the reference's own level on p at turn 2 and its rank correlation with the
turn number.

The release turn is the first dip only. A pattern raised again after it is
not seen (826876:1's second hold, to about turn 14; 820061:2's from turn 5),
and a reading ends at trunk_memory_states.py's last decision (240 at most),
so None means "not within the decisions read", not "never".

    python scripts/window_20261029/trunk_memory_gap.py --pattern <p.npy> --reading pass=<states.json>,<arrays dir> [--reading ...] [--sides <name> ...] --out training/metrics/trunk_memory_gap_20261110.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

AT = (1, 10, 40, 100)       # the decisions read: one write, the free lane's seed, and two points of the growth
RELEASED = 1.0              # under this on p a memory counts as released (held reads +3 to +7)


def release(on_p: np.ndarray, turn: np.ndarray, top: int):
    after = np.where(on_p[top:] < RELEASED)[0]
    return float(turn[top + after[0]]) if len(after) else None


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(x):                                       # ties share their mean rank
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def read(p: np.ndarray, record: dict, arrays: Path, sides) -> dict:
    out = {}
    for name, side in record["sides"].items():
        if "reading" not in side or (sides and name not in sides):
            continue
        z = np.load(arrays / f"{name.replace(':', '_side')}.npz")
        turn = np.array(side["reading"]["turn"], float)
        M, stepped = z["memory_mean"].astype(float), z["full_mean"].astype(float)
        gap = stepped - M
        on_p, size = gap @ p, np.linalg.norm(gap, axis=1)
        cos = on_p / np.maximum(size, 1e-12)
        peak, top = int(np.argmax(np.abs(on_p))), int(np.argmax(M @ p))
        level = M @ p
        out[name] = {"decisions": len(on_p),
                     "reference": {"on_p_turn_2": float(level[turn == 2].mean()), "rank_corr_with_turn": rank_corr(level, turn)},
                     "at": {str(d): {"on_p": float(on_p[d]), "cosine": float(cos[d])} for d in AT if d < len(on_p)},
                     "peak": {"decision": peak, "turn": float(turn[peak]), "on_p": float(on_p[peak]),
                              "cosine": float(cos[peak]), "gap_over_memory": float(size[peak] / np.linalg.norm(M[peak]))},
                     "release_turn": {"reference": release(level, turn, top), "stepped": release(stepped @ p, turn, top)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pattern", type=Path, required=True, help="trunk_memory_pattern.py's saved p")
    ap.add_argument("--reading", action="append", required=True, metavar="LABEL=STATES.json,ARRAYS",
                    help="one trunk_memory_states.py reading and its --arrays directory")
    ap.add_argument("--sides", nargs="*", help="the sides to read (default: every side of each reading)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    p = np.load(a.pattern).astype(float)
    p /= np.linalg.norm(p)
    out = {"at": AT, "released": RELEASED, "readings": {}}
    for item in a.reading:
        label, paths = item.split("=", 1)
        states, arrays = paths.rsplit(",", 1)
        record = json.loads(Path(states).read_text(encoding="utf-8"))
        rows = read(p, record, Path(arrays), a.sides)
        out["readings"][label] = {"direction": record["direction"], "s": record["s"], "sides": rows}
        print(f"{label} ({record['direction']}, s {record['s']:.3g})")
        for name, row in rows.items():
            k, r, ref = row["peak"], row["release_turn"], row["reference"]
            print(f"  {name[-8:]}: " + " ".join(f"d{d} {v['on_p']:+.4f} (cos {v['cosine']:+.2f})" for d, v in row["at"].items())
                  + f"; peak {k['on_p']:+.3f} at d{k['decision']} turn {k['turn']:.0f} (cos {k['cosine']:+.2f}, "
                  f"|gap|/|M| {k['gap_over_memory']:.3f}); release turn: reference {r['reference']}, stepped {r['stepped']}; "
                  f"reference on p at turn 2 {ref['on_p_turn_2']:+.2f}, rank r with the turn {ref['rank_corr_with_turn']:+.2f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
