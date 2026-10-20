"""Round 5 synthesis: what a learned luck corrector at fight children would cost, from own1's records.

A luck term L_k(Q) = Q(realised child) - sum_o p_k(o) Q(child o) with a learned Q (own1's value
head, a critic) needs one network forward per distinct child state of every fight, on top of the
game's own decision forwards. Generation is bound by network forwards on the shared inference
server (seed section 2), so the cost multiple of a game is
    m = (decision forwards + child forwards) / decision forwards.

Inputs (read-only):
- training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz: own1 self-play, 800
  games. Per game: decisions (commands move/attack/recruit/recall/end_turn, both sides) and attacks.
  The records also hold, for a few fights per game, the enumerated outcome tables of the
  defender's counter-weapon choice (outcomes[...]["counter_weapon"]["tables"]); their row counts
  are the support sizes of those fights' laws. That subsample is the fights whose counter weapon
  was a choice, not a random fight: its support size is stated as such.
- the variance ratio the free columns keep with the setup baseline in own1 self-play
  (round5_provenance_match_variance.out: gen_own1_selfplay 0.730, own64_selfplay 0.734).

Output: decisions and attacks per game, the support-size distribution on the subsample, the cost
multiple m, and the break-even variance ratio r_C / m below which a learned corrector carries more
label information per dollar than the free columns (information per dollar ~ 1 / (r * cost)).
numpy and the standard library only.
"""
import gzip
import json
import re
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARBALL = ROOT / "training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz"
DECISION_KINDS = {"move", "attack", "recruit", "recall", "end_turn"}


def read_games():
    decisions, attacks, supports = [], [], []
    with tarfile.open(TARBALL) as t:
        for m in t.getmembers():
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(m).read()).decode().splitlines()[0])
            cmds = rec["commands"]
            decisions.append(sum(1 for c in cmds if c[0] in DECISION_KINDS))
            attacks.append(sum(1 for c in cmds if c[0] == "attack"))
            for entry in rec.get("outcomes", {}).values():
                cw = entry.get("counter_weapon")
                if not cw:
                    continue
                table = cw["tables"].get(str(cw["chosen"]))
                if table:
                    supports.append(len(table))
    return np.array(decisions, float), np.array(attacks, float), np.array(supports, float)


def columns_ratio():
    text = (HERE / "round5_provenance_match_variance.out").read_text()
    both = dict(re.findall(r"^\s+(\S+)\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)$", text, re.M))
    return (float(both["gen_own1_selfplay"]) + float(both["own64_selfplay"])) / 2


def main():
    dec, att, sup = read_games()
    r_c = columns_ratio()
    print(f"games {len(dec)}; decisions per game {dec.mean():.1f} (both sides); attacks per game {att.mean():.1f}")
    q = np.percentile(sup, [10, 50, 90])
    print(f"support size on the counter-weapon subsample: fights {len(sup)}, mean {sup.mean():.2f}, "
          f"quantiles 10/50/90 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}")
    print(f"columns + setup keep r_C = {r_c:.3f} in own1 self-play")
    for s in (4.0, sup.mean(), 12.0):
        m = (dec.mean() + att.mean() * s) / dec.mean()
        print(f"  support {s:5.2f}: child forwards per game {att.mean() * s:6.0f}, cost multiple m {m:.2f}, "
              f"a learned corrector pays iff it keeps under r_C / m = {r_c / m:.3f}")


if __name__ == "__main__":
    main()
