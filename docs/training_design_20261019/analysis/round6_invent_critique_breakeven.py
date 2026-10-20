"""Round 6 (critique of the invent stance): the kept ratio r at which dice-priced innovations
first save an S1 batch, under round6_invent_dpi_power.py's own model and functions (the round-5
synthesis power model, inputs unchanged, r replaced). Standard library only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_simplify_power as P  # noqa: E402
import round6_invent_dpi_power as D  # noqa: E402


def main():
    base = P.read_inputs()
    for mu in (0.0, -0.0018):
        target = D.S.floor_sd_match(D.with_r(base, D.R_NOW), mu, D.D_EFF, D.BATCHES, 0.25)
        rows = []
        for k in range(732, 499, -4):
            r = k / 1000
            rows.append((r, D.batches_to_floor(D.with_r(base, r), mu, target)))
        first = {}
        for r, b in rows:
            first.setdefault(b, r)
        print(f"mu {mu:+.4f}: today's floor {target:.4f}; largest r giving each batch count: "
              + ", ".join(f"{b} batches at r <= {r:.3f}" for b, r in sorted(first.items(), reverse=True)))


if __name__ == "__main__":
    main()
