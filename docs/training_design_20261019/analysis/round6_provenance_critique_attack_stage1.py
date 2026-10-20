"""Round 6, critique of the provenance stance: the attack-promotion slope without the knob
search's selection.

round6_provenance_attack_pairs.py fits the result's response to a positive attack offset on all
games of the 24 decode-knob vectors (training/metrics/decode_knobs_20261012). The search was
successive halving (search.jsonl: stage 1 gave every vector 100 games, seeds 200000 + 1000 v + i;
the best 8 got 300 more, the best 2 another 800). All 13 vectors with a positive attack offset
stopped at stage 1, so their readings are the stage-1 games of vectors selected for doing badly,
while the a <= 0 survivors' readings were diluted toward their truth by later games.

Here the same kinked fit (same covariates, same counting, functions imported from
round5_provenance_knob_switches.py) runs twice:
  A. all games (reproduces round6_provenance_attack_pairs.out);
  B. stage-1 games only (seed < 230000): every vector with 100 unselected games.
It also prints the S0(d) prediction (offset +1, e = -1.5, r = t = 0) from each fit.
numpy and the standard library only.
"""
import gzip
import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ks", HERE / "round5_provenance_knob_switches.py")
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)
STAGE1_MAX_SEED = 230_000


def read_games(tarball, max_seed=None):
    out = []
    with tarfile.open(tarball) as t:
        for member in t.getmembers():
            if not member.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(member).read()).decode().splitlines()[0])
            if max_seed is not None and int(rec["seed"]) >= max_seed:
                continue
            if str(rec.get("winner")) not in ("1", "2"):
                continue
            players = rec["players"]
            a = players["a"] if players["a"]["label"] != "own1" else players["b"]
            b = players["b"] if a is players["a"] else players["a"]
            counts, _ = ks.side_counts(rec["commands"])
            ca = counts.get(a["side"], dict.fromkeys(ks.CLASSES, 0))
            cb = counts.get(b["side"], dict.fromkeys(ks.CLASSES, 0))
            out.append((1.0 if int(rec["winner"]) == a["side"] else -1.0, [ca[k] - cb[k] for k in ks.CLASSES]))
    return out


def fit(max_seed, slope, knobs, name):
    rows = []
    for v in range(24):
        label = f"kv{v:02d}"
        games = read_games(ks.KNOBS / f"games_{label}.tar.gz", max_seed)
        z = np.array([g[0] for g in games])
        d = np.array([g[1] for g in games], float)
        rows.append((len(z), z.mean(), d[:, 0].mean(), d.sum(1).mean()) + knobs[label])
    n = np.array([r[0] for r in rows], float)
    z = np.array([r[1] for r in rows])
    att = np.array([r[2] for r in rows]) - slope[0] * z
    allx = np.array([r[3] for r in rows]) - slope.sum() * z
    e, a, rr, t = (np.array([r[k] for r in rows]) for k in (4, 5, 6, 7))
    X = np.column_stack([np.maximum(a, 0), np.minimum(a, 0), np.minimum(allx, 0), np.maximum(allx, 0), rr, t])
    c, s = ks.wls(X, z, n)
    disp = float(np.sum(n * (z - X @ c) ** 2) / (24 - X.shape[1]))
    s = s * np.sqrt(max(disp, 1.0))
    X1 = np.column_stack([np.ones(24), np.maximum(a, 0), np.minimum(a, 0), e + 1.5, rr, t])
    c1, _ = ks.wls(X1, att, n)
    print(f"{name}: {int(n.sum())} games, dispersion {disp:.2f}")
    print(f"  attack offset > 0: {c[0]:+.4f} +- {s[0]:.4f} z per nat; < 0: {c[1]:+.4f} +- {s[1]:.4f}")
    print(f"  extra attacks per nat of positive offset {c1[1]:+.2f}; per added attack {c[0] / c1[1]:+.4f}")
    print(f"  S0(d) prediction at offset +1 (e -1.5, r = t = 0, decision-count channel ignored): "
          f"{347.4 * c[0]:+.0f} +- {347.4 * s[0]:.0f} Elo")


def main():
    slope, _, _ = ks.reverse_slope()
    knobs = ks.knob_values()
    fit(None, slope, knobs, "A. all games")
    fit(STAGE1_MAX_SEED, slope, knobs, "B. stage-1 games only (unselected)")


if __name__ == "__main__":
    main()
