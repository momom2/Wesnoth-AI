"""Labels of the first N training games of the pilot, all in split 0, so one
split's 2N game-sides fill the pass's streams."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = [json.loads(line) for line in (d / "labels_pilot.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
train = list(dict.fromkeys(r["game"] for r in rows if not r["heldout"]))
print("training games", len(train), "heldout rows", sum(r["heldout"] for r in rows))
for n in (16, 32):
    keep = set(train[:n])
    out = [dict(r, split=0) for r in rows if r["game"] in keep]
    (d / f"labels_{n}_one_split.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(n, "games ->", len(out), "rows")
