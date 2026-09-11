"""How many pairs one imitation epoch trains on, from a pre-encoded
corpus and the dataset manifest: winner-side pairs (the mover read
from global feature 1, as the trainer reads it) plus the expected
value-only states (value_states_per_game on loser-side pairs).

Usage: python scripts/count_epoch_pairs.py ENCODED_DIR DATASET_DIR [value_states_per_game]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.preencode_corpus import read_record, record_path  # noqa: E402

enc = Path(sys.argv[1])
dataset = Path(sys.argv[2])
k_value = int(sys.argv[3]) if len(sys.argv) > 3 else 16
rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
train_rows = [r for r in rows if not r.get("holdout")]
print("manifest rows", len(rows), "training", len(train_rows), "keys", sorted(train_rows[0].keys()), flush=True)
totals = Counter()
for i, r in enumerate(train_rows):
    name = r.get("file") or r.get("name") or r.get("path")
    winner = r.get("winner") or r.get("winner_side")
    try:
        pairs = read_record(record_path(enc, Path(name).name))
    except Exception as e:  # noqa: BLE001
        totals["file_errors"] += 1
        if totals["file_errors"] <= 3:
            print("error", name, repr(e), flush=True)
        continue
    n = len(pairs)
    win = sum(1 for raw, _ai in pairs if (1 if raw.global_feats[1] < 0 else 2) == winner)
    totals["pairs"] += n
    totals["winner_side"] += win
    totals["loser_side"] += n - win
    totals["value_only_expected"] += min(k_value, n - win)
    if i % 2000 == 0:
        print(i, dict(totals), flush=True)
totals["epoch_pairs_expected"] = totals["winner_side"] + totals["value_only_expected"]
print("DONE", dict(totals), flush=True)
