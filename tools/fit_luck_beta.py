"""The luck coefficients beta of the coin labels' first batch
(docs/training_design_20261019/current_design.md 3.2):
the least-squares fit of z on the seven luck sums of tools/match_luck.py's
walk, an intercept beside them, over the decisive games of its rows.

    python tools/fit_luck_beta.py training/metrics/match_luck_20261019/rows.jsonl \\
        --out configs/luck_beta_20261019.json

On the 2026-10-19 walk (23 argmax matches, 19,466 games) it reads the
18,800 decisive games. numpy and the standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.match_luck import COLUMNS  # noqa: E402
from tools.match_luck_readout import fit_beta, load_rows, luck_matrix  # noqa: E402


def fit(rows: Sequence[dict]) -> dict:
    """beta and the intercept over the decisive, walked rows."""
    keep = [r for r in rows if r.get("decisive") and "error" not in r]
    z = np.array([float(r["z"]) for r in keep])
    X = luck_matrix(keep, COLUMNS)
    beta = fit_beta(z, X)
    return {"columns": list(COLUMNS), "beta": [float(b) for b in beta],
            "intercept": float(z.mean() - X.mean(axis=0) @ beta), "games": len(keep),
            "matches": sorted({r["match"] for r in keep}),
            "variance_ratio": float(np.var(z - X @ beta, ddof=1) / np.var(z, ddof=1))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = fit(load_rows(args.rows))
    out["source"] = {"rows": args.rows.as_posix(),
                     "sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
                     "fit": "least squares of z on the luck sums, an intercept beside them, decisive games"}
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{out['games']} decisive games of {len(out['matches'])} matches; beta "
          + ", ".join(f"{c} {b:+.5g}" for c, b in zip(out["columns"], out["beta"]))
          + f"; Var(z - beta . S) / Var(z) {out['variance_ratio']:.4f} (in sample)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
