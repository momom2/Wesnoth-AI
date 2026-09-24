"""Reading the census rows back, and the existing combinations over them."""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = r"C:/Users/amaur/Desktop/Perso/projects/Wesnoth_AI_xod"
HERE = Path(__file__).parent
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/tools")
os.chdir(ROOT)

from tools.combat_dominance import Combo, Dim, admits  # noqa: E402

LOOSEST = Combo(True, 0.15, "literal", True)


def games(pattern="rows_*.jsonl.gz"):
    out = []
    for path in sorted(glob.glob(str(HERE / pattern))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
    return out


def dims_of(r):
    return [Dim(n, k, s, e, dist, bool(g)) for n, k, s, e, dist, g in r["dims"]]


def admitted(r, combo):
    if r["flags"].get("class_block"):
        return False
    return admits(dims_of(r), r["tier"], combo)
