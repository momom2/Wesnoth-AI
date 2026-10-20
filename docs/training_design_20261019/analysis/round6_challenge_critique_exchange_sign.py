"""Round 6, critique of the challenge stance: the sign of a fight's expected exchange.

The challenge's decision score gives a non-attack 0 and an attack X(a) = E_law[psi(o)], psi
linear in the outcome (HP lost by each fighter as a fraction of its maximum, deaths, ...). If
most attacks a network ranks near the top have a favourable expected exchange, X(a) > 0 for most
of them, and the real arm (argmax X) switches mostly toward attacks while the anti arm switches
mostly away from them: the two arms' switch sets then differ in pair type.

Proxy measured here, on the played fights of `own1` self-play whose defender's counter weapon
was a choice (the only fights with strike tables on disk; a subsample, not a random one):
the share of fights whose expected exchange is favourable to the attacker, under three plain
weightings of the outcome (absolute HP, HP as a fraction of the fighter's starting HP, and kill
probability). Starting HP is read as the largest HP in the fighter's outcome rows (the outcome in
which it is never hit, present whenever a hit chance is below 1).

Input (read-only): training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz
numpy and the standard library only.
"""
import gzip
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TARBALL = ROOT / "training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz"


def fight_rows():
    with tarfile.open(TARBALL) as t:
        for m in t.getmembers():
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(m).read()).decode().splitlines()[0])
            for entry in rec.get("outcomes", {}).values():
                cw = entry.get("counter_weapon")
                if not cw:
                    continue
                table = cw["tables"].get(str(cw["chosen"]))
                if table:
                    yield table


def exchange(table):
    p = np.array([row[-1] for row in table], dtype=float)
    p /= p.sum()
    att = np.array([row[0] for row in table], dtype=float)
    dfn = np.array([row[1] for row in table], dtype=float)
    att0, def0 = att.max(), dfn.max()
    lost_att, lost_def = att0 - p @ att, def0 - p @ dfn
    kill_def, kill_att = p @ (dfn <= 0), p @ (att <= 0)
    return lost_def - lost_att, lost_def / def0 - lost_att / att0, kill_def - kill_att


def main():
    ex = np.array([exchange(t) for t in fight_rows()])
    print(f"fights {len(ex)} (counter-weapon subsample of own1 self-play, 800 games)")
    for name, col in zip(("absolute HP", "HP fraction", "kill probability"), ex.T):
        print(f"  expected exchange favourable to the attacker ({name}): share {np.mean(col > 0):.3f}; "
              f"unfavourable {np.mean(col < 0):.3f}; median {np.median(col):+.3f}")


if __name__ == "__main__":
    sys.exit(main())
