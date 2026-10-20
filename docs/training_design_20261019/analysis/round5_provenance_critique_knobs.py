"""Round 5, critique of the provenance stance: how robust is the knob fit behind Match A's
prediction and the "per decision added / removed" values?

Reads the decode-knob search's whole game records (training/metrics/decode_knobs_20261012/,
24 vectors and the confirmation of kv15 on fresh seeds) with the decision counting of
round5_provenance_knob_switches.py (imported, not copied).

Three issues the provenance script leaves open, each tested here:
  1. Selection. Promoted vectors carry their stage-1 (and stage-2) games, which were the basis
     of their promotion, at weights 400 or 1,200. Variant "stage 1 only" uses the 24 x 100
     games no selection touched; variant "with confirm" adds kv15's fresh 800 games.
  2. The winning-share correction subtracts 40 x the vector's OBSERVED mean z from the regressor
     of a regression whose response is that same z: the response's noise enters the regressor.
     Variant "fitted correction" subtracts 40 x the reduced form's fitted z instead.
  3. Match A and Match C need no decision count at all: the reduced form, z on
     (min(e + 1.5, 0), max(e + 1.5, 0), attack, recruit, tod), predicts them directly.

Stages are read from the seed (docs/decode_knobs_prereg_20261012.md): stage 1 at 200000 +
1000 i, stage 2 at 230000 + 1000 j, stage 3 at 240000 and 241000, the confirmation at 250000;
replacements add k x 1,000,000. numpy and the standard library only.
"""
import gzip
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ks", HERE / "round5_provenance_knob_switches.py")
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)

KNOBS = ks.KNOBS
WIN_SHARE = 40.0  # decisions per side-game per unit z, round5_provenance_knob_switches.out


def stage_of(seed):
    base = seed % 1_000_000
    if 200_000 <= base < 224_000:
        return 1
    if 230_000 <= base < 238_000:
        return 2
    if 240_000 <= base < 242_000:
        return 3
    if base >= 250_000:
        return 4
    return 0


def read(tarball):
    out = []
    with tarfile.open(tarball) as t:
        for m in t.getmembers():
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(m).read()).decode().splitlines()[0])
            if str(rec.get("winner")) not in ("1", "2"):
                continue
            players = rec["players"]
            if isinstance(players, str):
                import ast
                players = ast.literal_eval(players)
            a = players["a"] if players["a"]["label"] != "own1" else players["b"]
            b = players["b"] if a is players["a"] else players["a"]
            counts, _ = ks.side_counts(rec["commands"])
            ca = counts.get(a["side"], dict.fromkeys(ks.CLASSES, 0))
            cb = counts.get(b["side"], dict.fromkeys(ks.CLASSES, 0))
            out.append((stage_of(int(rec["seed"])), 1.0 if int(rec["winner"]) == a["side"] else -1.0,
                        float(sum(ca[k] - cb[k] for k in ks.CLASSES))))
    return out


def wls(X, y, w):
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    cov = np.linalg.inv((X * w[:, None]).T @ X)
    resid = y - X @ coef
    disp = float(np.sum(w * resid ** 2) / (len(y) - X.shape[1]))
    return coef, cov, disp


def fit(rows, label):
    n = np.array([r["n"] for r in rows], float)
    z = np.array([r["z"] for r in rows])
    x = np.array([r["extra"] for r in rows])
    e, a, rr, t = (np.array([r["knobs"][k] for r in rows]) for k in range(4))
    lo, hi = np.minimum(e + 1.5, 0.0), np.maximum(e + 1.5, 0.0)
    # reduced form: no decision counts
    R = np.column_stack([lo, hi, a, rr, t])
    c, cov, disp = wls(R, z, n)
    inf = max(disp, 1.0)
    se = np.sqrt(np.diag(cov) * inf)
    zA, seA = 1.5 * c[1], 1.5 * se[1]
    zC, seC = -c[0], se[0]
    print(f"\n[{label}] vectors {len(rows)}, games {int(n.sum())}, dispersion {disp:.2f} (SEs inflated by it when > 1)")
    print(f"  reduced form: z per unit of offset below -1.5 {c[0]:+.4f} +- {se[0]:.4f}, above {c[1]:+.4f} +- {se[1]:.4f}, "
          f"attack {c[2]:+.4f} +- {se[2]:.4f}")
    print(f"  predicted Match A (0 vs -1.5) z {zA:+.3f} +- {seA:.3f} ({347.4 * zA:+.0f} +- {347.4 * seA:.0f} Elo); "
          f"Match C (-2.5 vs -1.5) z {zC:+.3f} +- {seC:.3f} (measured -0.035 +- 0.035)")
    zhat = R @ c
    for name, corr in (("observed-z correction", WIN_SHARE * z), ("fitted-z correction", WIN_SHARE * zhat)):
        xc = x - corr
        X2 = np.column_stack([np.minimum(xc, 0.0), np.maximum(xc, 0.0), a, rr, t])
        c2, cov2, disp2 = wls(X2, z, n)
        se2 = np.sqrt(np.diag(cov2) * max(disp2, 1.0))
        print(f"  decision model, {name}: per decision removed {c2[0]:+.5f} +- {se2[0]:.5f}, "
              f"added {c2[1]:+.5f} +- {se2[1]:.5f}")


def main():
    knobs = ks.knob_values()
    per_vector = {}
    for v in range(24):
        label = f"kv{v:02d}"
        per_vector[label] = read(KNOBS / f"games_{label}.tar.gz")
    confirm = read(KNOBS / "games_confirm.tar.gz")
    print("games by stage, kv15:", {s: sum(1 for g in per_vector["kv15"] if g[0] == s) for s in (0, 1, 2, 3, 4)},
          "confirm:", {s: sum(1 for g in confirm if g[0] == s) for s in (0, 1, 2, 3, 4)})

    def row(label, games, knob_label=None):
        z = np.array([g[1] for g in games])
        x = np.array([g[2] for g in games])
        return {"label": label, "n": len(z), "z": z.mean(), "extra": x.mean(), "knobs": knobs[knob_label or label]}

    all_rows = [row(lab, g) for lab, g in per_vector.items()]
    fit(all_rows, "all knob games (the provenance script's data)")
    fit(all_rows + [row("confirm", confirm, "kv15")], "all knob games + kv15's fresh confirmation")
    s1 = [row(lab, [g for g in gs if g[0] == 1]) for lab, gs in per_vector.items()]
    fit(s1, "stage 1 only: 24 x 100 games, no selection")
    for lab in ("kv15", "kv10"):
        for s in (1, 2, 3):
            gs = [g for g in per_vector[lab] if g[0] == s]
            if gs:
                print(f"  {lab} stage {s}: {len(gs)} games, z {np.mean([g[1] for g in gs]):+.3f}")
    print(f"  confirm: {len(confirm)} games, z {np.mean([g[1] for g in confirm]):+.3f}")


if __name__ == "__main__":
    sys.exit(main())
