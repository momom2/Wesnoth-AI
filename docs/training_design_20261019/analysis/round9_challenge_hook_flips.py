"""Round 9, challenge stance: what one coherent seat-level change does to play,
read from S0(a)'s 6,400 games (seed section 7): own1 against itself, attack
logits +1 on the seat a per-game coin chose (D = 1 hooks player A, P = 1/2).

The coin is randomized per game, so for any per-seat count c,
    E[c_A | D = 1] - E[c_A | D = 0]
is the hook's causal effect on that seat's count, and the same contrast on the
other seat's count is the hook's effect on the opponent. Read per side-game:
attacks, moves, recruits, side-turns, actions per side-turn.

Then the hooked seat's result: its mean z over the decisive games (the readout's
-0.118, seed section 7) divided by the decisions the hook changed per side-game
gives the mean advantage per changed decision at margins up to 1 nat,
against the coin band's in-band attack-promoting types (readout_b_offline.txt:
move>attack -0.0031 +- 0.0032, end_turn>attack -0.0135 +- 0.0154).
The hook barely changes the attack count; it makes units attack before moving
(an attack ends the unit's movement), so the changed decisions are counted by
the moves the hooked seat loses. That count misses changes that keep the move
count (a reordering that still moves), so the per-decision figure is an upper
bound on the magnitude of the mean per changed decision.

Inputs (read-only): training/metrics/near_tie_coins_20261020/games_a.tar.gz,
rows_a.jsonl. Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "training" / "metrics" / "near_tie_coins_20261020"
ELO_PER_Z = 347.4  # 800 / ln 10 / 2 ... the design's Elo per unit of mean z (current_design.md)


def per_side_counts(commands):
    """Counts per side (1, 2) of attack, move, recruit commands and side-turns."""
    counts = {1: dict(attack=0, move=0, recruit=0, turns=0), 2: dict(attack=0, move=0, recruit=0, turns=0)}
    side = None
    for c in commands:
        k = c[0]
        if k == "init_side":
            side = int(c[1])
            counts[side]["turns"] += 1
        elif side is not None and k in ("attack", "move", "recruit"):
            counts[side][k] += 1
    return counts


def main():
    rows = {}
    with open(REC / "rows_a.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["game"]] = r
    recs = []
    with tarfile.open(REC / "games_a.tar.gz", "r:*") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]
            if not (m.isfile() and name.endswith(".game.jsonl.gz")):
                continue
            game = name[: -len(".game.jsonl.gz")]
            row = rows.get(game)
            if row is None or not row.get("decisive"):
                continue
            rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode("utf-8").splitlines()[0])
            d = int(rec["coins"]["attack"]["d"])
            side_a = int(row["side_a"])
            cnt = per_side_counts(rec["commands"])
            a, b = cnt[side_a], cnt[3 - side_a]
            recs.append((d, int(row["z"]), a, b))
    n = len(recs)
    d = np.array([r[0] for r in recs])
    z = np.array([r[1] for r in recs], dtype=float)
    print(f"decisive games read: {n}; hooked A in {int(d.sum())}")
    hooked_z = np.where(d == 1, z, -z)
    mz, sz = hooked_z.mean(), hooked_z.std(ddof=1) / math.sqrt(n)
    print(f"hooked seat's mean z {mz:+.4f} +- {sz:.4f} = {mz * ELO_PER_Z:+.1f} +- {sz * ELO_PER_Z:.1f} Elo")

    def contrast(key, seat):
        v = np.array([(r[2] if seat == "A" else r[3])[key] for r in recs], dtype=float)
        x1, x0 = v[d == 1], v[d == 0]
        diff = x1.mean() - x0.mean()
        se = math.sqrt(x1.var(ddof=1) / len(x1) + x0.var(ddof=1) / len(x0))
        return x0.mean(), diff, se

    print("\nper side-game counts: unhooked level (seat A when D = 0), hook's effect on the hooked seat, on its opponent")
    effects = {}
    for key in ("attack", "move", "recruit", "turns"):
        base, own, own_se = contrast(key, "A")
        # the opponent of a hooked A is B; B's count when D = 1 minus when D = 0
        _, opp, opp_se = contrast(key, "B")
        effects[key] = (base, own, own_se, opp, opp_se)
        print(f"  {key:8s} level {base:7.2f}; hooked seat {own:+.3f} +- {own_se:.3f}; opponent {opp:+.3f} +- {opp_se:.3f}")
    acts = np.array([r[2]["attack"] + r[2]["move"] + r[2]["recruit"] for r in recs], dtype=float)
    turns = np.array([max(r[2]["turns"], 1) for r in recs], dtype=float)
    per = acts / turns
    print(f"  actions per side-turn, seat A: D=0 {per[d == 0].mean():.3f}, D=1 {per[d == 1].mean():.3f}")

    # The hook moves decisions toward attacking now. A unit that attacks cannot move
    # afterwards, so the decisions it changes show up as moves lost (the attack count
    # barely moves): count changed decisions by the moves the hooked seat lost.
    lost, lost_se = -effects["move"][1], effects["move"][2]
    extra, extra_se = effects["attack"][1], effects["attack"][2]
    print(f"\nhooked seat: {lost:.2f} +- {lost_se:.2f} fewer moves, {extra:+.2f} +- {extra_se:.2f} attacks per side-game")
    per_change = mz / lost
    rel = math.sqrt((sz / mz) ** 2 + (lost_se / lost) ** 2)
    pc_se = abs(per_change) * rel
    print(f"mean advantage per changed decision (moves lost as the count): {per_change:+.4f} +- {pc_se:.4f} z")
    inband = -4.6 / ELO_PER_Z / 4.12
    inband_se = 4.4 / ELO_PER_Z / 4.12
    print(f"in-band attack-promoting coin switches: {inband:+.4f} +- {inband_se:.4f} z per switch "
          "(-4.6 +- 4.4 Elo over 4.12 switches per side-game, round9_challenge_coherence.out section 4)")
    gap = per_change - inband
    gap_se = math.sqrt(pc_se ** 2 + inband_se ** 2)
    print(f"per-decision difference, hook (margins up to 1 nat) minus in-band (margins under 0.35): "
          f"{gap:+.4f} +- {gap_se:.4f} (z {gap / gap_se:+.2f})")


if __name__ == "__main__":
    main()
