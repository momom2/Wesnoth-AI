"""Round 10, invent stance: R's count feedback, a second-order term L can see.

R (43 additive unit values, ridge 5,000, fitted here on all of batch 1's
different-type recruit near-ties, as the design seals it) switches toward the
same unit again and again (round9_simplify_critique_clusters.out: 0.58 of its
switches follow an earlier switch by the same side). In R's own play (no coins)
each earlier R switch is always taken, and each earlier coin that played the
runner-up where R does not switch is not, so the counts n0, n1 at a later R
switch differ from the coin world's. Batch 1's labels rise with d = n0 - n1
inside unit identity (round10_invent_recruit_rules.out section 7: +0.00267 per
unit of d, SE 0.00048). The first-order composition correction to R's welfare
in its own play is then slope x the mean shift of d at R's switches.

Model limits, stated: only the near-tie recruit decisions the batch logged move
the counts (recruit-versus-move coins and non-band recruits are ignored), and
the label depends on the history only through d at that slope. A reading, not a
gate. Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
ELO = 800 / math.log(10)
SIDE_GAMES = 2 * 60_000
RIDGE = 5_000.0
SLOPE, SLOPE_SE = 0.00267, 0.00048  # round10_invent_recruit_rules.out section 7


def load():
    cols = defaultdict(list)
    for m in MATCHES:
        d = json.loads(gzip.decompress((HERE / f"round10_invent_recruit_rows_{m}_p0.json.gz").read_bytes()))
        for k, v in d["rows"].items():
            cols[k].extend(v)
    return {k: np.array(v) for k, v in cols.items()}


def main():
    c = load()
    N = len(c["T"])
    u0 = np.array([f + "|" + u for f, u in zip(c["fac"], c["u0"])])
    u1 = np.array([f + "|" + u for f, u in zip(c["fac"], c["u1"])])
    names = sorted(set(u0) | set(u1))
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((N, len(names)))
    X[np.arange(N), [at[u] for u in u1]] += 1.0
    X[np.arange(N), [at[u] for u in u0]] -= 1.0
    beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(len(names)), X.T @ c["T"])
    R = X @ beta > 0
    print(f"rows {N}; R fitted on all rows switches {R.sum() / SIDE_GAMES:.2f} per side-game")

    order = np.lexsort((c["cmd"], c["side"], c["game"]))
    shift = np.zeros(N)
    later_same = 0
    key_prev = None
    delta = defaultdict(int)
    for i in order:
        key = (c["game"][i], c["side"][i])
        if key != key_prev:
            delta = defaultdict(int)
            key_prev = key
        shift[i] = delta[c["u0"][i]] - delta[c["u1"][i]]
        if R[i] and c["d"][i] == 0:      # R's play takes the switch the coin did not
            delta[c["u1"][i]] += 1
            delta[c["u0"][i]] -= 1
        elif not R[i] and c["d"][i] == 1:  # the coin played the runner-up; R's play does not
            delta[c["u0"][i]] += 1
            delta[c["u1"][i]] -= 1
        if R[i] and shift[i] != 0:
            later_same += 1
    s = shift[R]
    print(f"at R's switches: mean shift of d in R's play {s.mean():+.3f} (sd {s.std():.3f}); "
          f"share with a nonzero shift {np.mean(s != 0):.3f}")
    per_sg = SLOPE * s.sum() / SIDE_GAMES
    print(f"first-order composition correction to R's welfare: {per_sg:+.5f} z per side-game = "
          f"{per_sg * ELO:+.2f} Elo (slope SE alone: {SLOPE_SE * abs(s.sum()) / SIDE_GAMES * ELO:.2f} Elo)")
    for lab, m in (("side-turn 1", c["st"][R] == 1), ("side-turn >= 2", c["st"][R] >= 2)):
        print(f"  {lab}: {m.sum() / SIDE_GAMES:.2f} switches per side-game, mean shift {s[m].mean():+.3f}")


if __name__ == "__main__":
    main()
