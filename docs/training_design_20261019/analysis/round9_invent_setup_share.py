"""Round 9, invent stance: what twins share without any dice coupling.

Twins on one seed share the exact setup: the cell (map, side of A, both
factions) and also both leaders, which the design's cell baseline does not
see. This script measures (1) the leader share of Var(z - beta . S) within
(faction, side) groups, by the method of moments on game-clustered leader
means (between-leader variance minus its sampling noise), and (2) whether any
seed is shared between two match records on disk (twins never existed).

Inputs: games_b00 and games_b01 of training/metrics/near_tie_coins_20261020 (setup
leaders from the game records, z and S from rows_<M>.jsonl; batch 1 = own1
coin self-play), beta as round9_invent_cells.py. Seeds: every games_*.tar.gz
under training/metrics (member names only) and match_luck_20261019/rows.jsonl.
Usage: python round9_invent_setup_share.py leaders|seeds (each run under a minute)
Standard library and numpy only.
"""
import gzip
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MET = ROOT / "training" / "metrics"
REC = MET / "near_tie_coins_20261020"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
BETA = (0.0011885942625018315, 0.00028509220077075834, -0.00048074045056809554,
        -0.11838146825305246, 0.603040987021951, 0.0541244809103413, 0.022519783857895107)


def leader_share():
    rows = []
    for match in ("b00", "b01"):
        walk = {}
        with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("decisive") and "S" in r:
                    walk[r["game"]] = r
        with tarfile.open(REC / f"games_{match}.tar.gz", "r:gz") as tf:
            for m in tf:
                if not m.name.endswith(".game.jsonl.gz"):
                    continue
                game = Path(m.name).name[: -len(".game.jsonl.gz")]
                w = walk.get(game)
                if w is None:
                    continue
                rec = json.loads(gzip.decompress(tf.extractfile(m).read()).decode())
                setup = rec["setup"]
                side_a = int(rec["players"]["a"]["side"])
                side_b = 3 - side_a
                resid = int(w["z"]) - sum(b * float(w["S"][c]) for b, c in zip(BETA, COLUMNS))
                rows.append((resid, side_a, setup[f"faction{side_a}"], setup[f"leader{side_a}"],
                             setup[f"faction{side_b}"], setup[f"leader{side_b}"]))
    resid = np.array([r[0] for r in rows])
    total = resid.var()
    print(f"games {len(rows)}; Var(z - beta.S) {total:.4f}")
    share = 0.0
    for who, (fi, li) in (("A", (2, 3)), ("B", (4, 5))):
        groups = defaultdict(list)
        for r in rows:
            groups[(r[1], r[fi])].append((r[li], r[0]))
        excess = 0.0
        for key, items in groups.items():
            by_leader = defaultdict(list)
            for leader, x in items:
                by_leader[leader].append(x)
            n = len(items)
            mean = np.mean([x for _, x in items])
            # method of moments: sum_l n_l (m_l - m)^2 / n estimates var_between + (L - 1) s^2 / n
            between = sum(len(v) * (np.mean(v) - mean) ** 2 for v in by_leader.values()) / n
            s2 = np.mean([np.var(v, ddof=1) for v in by_leader.values() if len(v) > 1])
            excess += n * (between - (len(by_leader) - 1) * s2 / n)
        part = excess / len(rows) / total
        share += part
        print(f"  leader of side {who} within (side of A, faction): share {part:+.4f} of Var(z - beta.S)")
    print(f"  both leaders: {share:+.4f} (method of moments; SE not computed, noise term removed)")


def seed_overlap():
    seeds = {}
    for path in sorted(MET.glob("*/games_*.tar.gz")):
        s = set()
        with tarfile.open(path, "r:gz") as tf:
            for name in tf.getnames():
                m = re.search(r"_s[12]_(\d+)\.(json|game\.jsonl\.gz)$", name)
                if m:
                    s.add(int(m.group(1)))
        seeds[str(path.relative_to(MET))] = s
    walk = defaultdict(set)
    with open(MET / "match_luck_20261019" / "rows.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            walk["walk:" + r["match"]].add(int(r["seed"]))
    # the walk re-reads archived games, so walk rows and tarballs are compared separately
    for label, group in (("archived tarballs", seeds), ("walked matches", walk)):
        count = Counter()
        for s in group.values():
            count.update(s)
        shared = sum(1 for v in count.values() if v > 1)
        print(f"{label}: {len(group)} match records, {len(count)} seeds, seeds in more than one record: {shared}")


def main(part):
    (leader_share if part == "leaders" else seed_overlap)()


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
