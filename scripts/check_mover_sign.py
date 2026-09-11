"""Does the trainer read the same mover from a pre-encoded record as
from the replay it came from? For N training games: the record's
pairs (global feature 1, the trainer's mover) against the streamed
GameStates' current_side, in the same order.

Usage: python scripts/check_mover_sign.py ENCODED_DIR DATASET_DIR [N]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.preencode_corpus import read_record, record_path  # noqa: E402
from tools.supervised_train import _pair_stream_serial  # noqa: E402

enc = Path(sys.argv[1])
dataset = Path(sys.argv[2])
n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
train_rows = [r for r in rows if not r.get("holdout")][:n_games]
for r in train_rows:
    name = Path(r["file"]).name
    pairs = read_record(record_path(enc, name))
    streamed = [(item[1].global_info.current_side, item[2].action_type)
                for item in _pair_stream_serial([dataset / name]) if item[0] == "pair"]
    rec = [(1 if raw.global_feats[1] < 0 else 2, ai.action_type) for raw, ai in pairs]
    same_len = len(rec) == len(streamed)
    agree = sum(1 for a, b in zip(rec, streamed) if a == b)
    print(f"{name}: winner {r['winner_side']}, record pairs {len(rec)}, streamed {len(streamed)}, "
          f"same length {same_len}, mover+type agree {agree}/{min(len(rec), len(streamed))}, "
          f"record movers {sum(1 for m, _ in rec if m == 1)} side-1 / {sum(1 for m, _ in rec if m == 2)} side-2", flush=True)
