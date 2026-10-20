"""Round 5, critique of the challenge stance: can one look-ahead match split its gain by decision kind?

Reuses round5_challenge_lookahead.py's readers (telemetry from the two recorded look-ahead matches,
the leave-one-match-out corrected result G). Per match, regresses G on the game's flips by the
kind of the raw decision they replaced (attack, move, recruit, end_turn), no intercept (J = 0 for
the base side, performance-difference lemma), and reports each kind's coefficient with its OLS SE,
plus the correlation of each kind's flip count with the game's total decisions (flip counts move
with game length, so the per-kind coefficients are not causal even before noise).

numpy and the standard library only. Run: python round5_challenge_critique_kinds.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("la", HERE / "round5_challenge_lookahead.py")
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)
OUT = HERE / "round5_challenge_critique_kinds.out"


def main():
    rows = [json.loads(line) for line in open(la.ROWS)]
    G = la.corrected_results([r for r in rows if r.get("decisive")])
    lines = []
    for match, tar in la.TARS.items():
        tel = la.read_telemetry(tar)
        dec = [r for r in rows if r["match"] == match and r.get("decisive") and r["game"] in tel]
        g = np.array([G[r["game"]] for r in dec])
        X = np.array([[tel[r["game"]]["flips_by_kind"][k] for k in la.KINDS] for r in dec], float)
        n_dec = np.array([tel[r["game"]]["decisions"] for r in dec], float)
        coef, *_ = np.linalg.lstsq(X, g, rcond=None)
        resid = g - X @ coef
        s2 = resid @ resid / (len(g) - X.shape[1])
        cov = s2 * np.linalg.pinv(X.T @ X)
        lines.append(f"[{match}] games {len(g)}")
        for k, c, v, col in zip(la.KINDS, coef, np.diag(cov), X.T):
            rho = np.corrcoef(col, n_dec)[0, 1] if col.std() > 0 else float("nan")
            lines.append(f"  {k:8s} flips/side-game {col.mean():5.2f}  coef {c:+.4f} +- {np.sqrt(v):.4f}  "
                         f"corr(flips, decisions) {rho:+.2f}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
