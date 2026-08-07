"""Self-consistency check for raw replays: OOS-corrupt recordings
(2026-08-06 class).

A recording is provably broken -- independent of any simulator
assumption -- when its own recorded early-game spend exceeds its own
recorded starting gold plus a deliberately GENEROUS income bound:

    spend(side, turns 1..2) > gold_0(side) + base_income
                              + N_VILLAGES_BOUND * village_gold

Unit costs come from the pinned 1.18.4 stats; everything else is read
from the file. The engine itself OOS-errors on these files
(user-verified in the viewer on CotB 14385); six were found and
deleted from the 2026-08-06 full-corpus sweep (5 more small-shortfall
files were NOT provable and stay under investigation -- this check is
a corruption PROOF, not a divergence heuristic).

Usage:
    python tools/check_replay_consistency.py <file.bz2> [...]
Prints one line per file: OK / SELF-INCONSISTENT <detail>.
Importable: `check_gold_consistency(path) -> Optional[str]`
(None = passes; string = proof of inconsistency).
"""
from __future__ import annotations

import bz2
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Generous village bound: no 2p ladder map lets one side hold more
# villages by turn 2 (most allow 2-4; Cynsaun's 30+ villages are not
# reachable by turn 2 either).
N_VILLAGES_BOUND = 10
BASE_INCOME_BOUND = 2 * 2          # two turn-2 incomes of base 2


def check_gold_consistency(path: Path) -> Optional[str]:
    from tools.replay_dataset import _stats_for
    try:
        t = bz2.open(path, "rb").read().decode("utf-8", errors="replace")
    except OSError as e:
        return f"unreadable: {e}"
    mp = t.find("[multiplayer]")
    golds = re.findall(r'^\s*gold="?(-?\d+)"?', t[:mp if mp > 0 else len(t)],
                       re.M)
    vg_m = re.search(r'mp_village_gold="?(\d+)', t)
    village_gold = int(vg_m.group(1)) if vg_m else 2
    body = t[t.find("[replay]"):]
    side, turn = 0, 0
    spend = {1: 0, 2: 0}
    for m in re.finditer(r'\[(init_side|recruit)\]', body):
        if m.group(1) == "init_side":
            sn = re.search(r'side_number="?(\d)', body[m.end():m.end() + 60])
            s = int(sn.group(1)) if sn else 0
            if s == 1:
                turn += 1
                if turn > 2:
                    break
            side = s
        elif turn <= 2 and side in (1, 2):
            ty = re.search(r'type="([^"]+)"', body[m.end():m.end() + 120])
            if ty:
                spend[side] += int(_stats_for(ty.group(1)).get("cost", 0)
                                   or 0)
    bound_extra = BASE_INCOME_BOUND + N_VILLAGES_BOUND * village_gold
    for s in (1, 2):
        gold0 = int(golds[s - 1]) if len(golds) >= s else 100
        if spend[s] > gold0 + bound_extra:
            return (f"side {s} spends {spend[s]} by turn 2 on recorded "
                    f"start gold {gold0} (+{bound_extra} income bound)")
    return None


def main(argv) -> int:
    bad = 0
    for a in argv[1:]:
        for p in sorted(Path(".").glob(a)) or [Path(a)]:
            verdict = check_gold_consistency(p)
            if verdict:
                bad += 1
                print(f"SELF-INCONSISTENT {p}: {verdict}")
            else:
                print(f"OK {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
