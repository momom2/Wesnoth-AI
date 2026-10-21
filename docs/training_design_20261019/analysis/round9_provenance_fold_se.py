"""Round 9, provenance stance: two checks on the cross-fitted welfare readings of
round9_provenance_recruit_cells.py and round9_provenance_unit_cells.py.

1. A between-fold SE. The scripts' SE, sqrt(sum T^2 over switched labels) / side
   games, assumes labels uncorrelated within a game (exact for T0, measured
   design effect 1.00 for T over all band decisions, but not measured within
   a subset). Folds hold whole games, so the spread of the 20 per-fold sums
   t_f gives an SE that allows any correlation within a game:
       SE_fold = sqrt(F/(F-1) * sum_f (t_f - mean t)^2) / side games.
   With 20 folds it is itself uncertain by about 1/sqrt(38) = 16%.
2. The selection-immune part of the recruit rule. The recruit partition was
   examined because recruit>recruit had section 7's largest typed mean. A rule
   that switches a share q of a group with mean m collects q*m whatever the
   cells; centring each fold's labels on the group's out-of-fold mean removes
   that part and leaves what the rule earns from the spread between cells,
   which the choice made on the mean did not touch:
       W_c = sum_f sum_{switched cells} (s_cf - m_{-f} n_cf) / side games.
Standard library and numpy only; reads the per-cell fold sums the two scripts
wrote (no game record is reopened).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = load("round9_provenance_recruit_cells")
uc = load("round9_provenance_unit_cells")
ELO, FOLDS = rc.ELO, rc.FOLDS


def fold_readings(s_rule, n_rule, s_read, ss_read, n_read, sg, rule, centre=False):
    t, var = np.zeros(FOLDS), 0.0
    for f in range(FOLDS):
        so, no = s_rule.sum(1) - s_rule[:, f], n_rule.sum(1) - n_rule[:, f]
        choose = rule(so, no)
        if centre:
            m = (s_read.sum() - s_read[:, f].sum()) / (n_read.sum() - n_read[:, f].sum())
            t[f] = (s_read[choose, f] - m * n_read[choose, f]).sum()
        else:
            t[f] = s_read[choose, f].sum()
        var += ss_read[choose, f].sum()
    w = t.sum() / sg
    se_label = math.sqrt(var) / sg
    se_fold = math.sqrt(FOLDS / (FOLDS - 1) * np.sum((t - t.mean()) ** 2)) / sg
    return w, se_label, se_fold


def show(name, w, se_l, se_f):
    print(f"  {name:62s} {w * ELO:+6.1f} Elo; SE from labels {se_l * ELO:4.1f}, between folds {se_f * ELO:4.1f}"
          f" (ratio {se_f / se_l:.2f})")


def main():
    cells, a, sg = rc.load()
    print("1-2. recruit>recruit, different units (round9_provenance_recruit_cells.py)")
    additive = lambda so, no: (no > 0) & (rc.additive_rule(cells, so, no, 2000.0) > 0)  # noqa: E731
    cellpos = lambda so, no: (no > 0) & (so > 0)  # noqa: E731
    for rname, rule in (("additive R, ridge 2000", additive), ("cell mean > 0", cellpos)):
        for read in ("T", "T0"):
            w = fold_readings(a["s_T"], a["n"], a[f"s_{read}"], a[f"ss_{read}"], a["n"], sg, rule)
            show(f"{rname}, fitted on T, read with {read}", *w)
        w = fold_readings(a["s_T"], a["n"], a["s_T"], a["ss_T"], a["n"], sg, rule, centre=True)
        show(f"{rname}, fitted on T, read with T centred on the group mean", *w)

    print("\n1. unit-keyed groups (round9_provenance_unit_cells.py)")
    _data, ucells, ua = uc.merged()
    usg = sum(d["side_games"] for d in _data)
    subs = np.array([c.split("|", 1)[0] for c in ucells])
    for gname, members in uc.GROUPS.items():
        if members is None:
            mask = ~np.isin(subs, ["recruit>recruit:other_unit", "recruit>recruit:same_unit"])
        else:
            mask = np.isin(subs, list(members))
        n = ua["n"][mask]
        for label in ("T", "T0"):
            s, ss = ua[f"s_{label}"][mask], ua[f"ss_{label}"][mask]
            for rname, rule in (("(a)", lambda so, no: (no > 0) & (so > 0)),
                                ("(b) >= 1,000", lambda so, no: (no >= 1000) & (so > 0))):
                w = fold_readings(s, n, s, ss, n, usg, rule)
                show(f"{gname[:34]:34s} [{label:2s}] {rname}", *w)


if __name__ == "__main__":
    main()
