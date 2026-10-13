"""The sequence trainer's test pass (tests/test_sequence_train.py,
tests/test_sequence_speed.py): three small games written from scratch,
extracted into a corpus and pre-encoded by the script run by path, and the
trainer's arguments for a tiny network over them."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


def synthetic_dataset(tmp_path: Path) -> Path:
    """Three small games extracted into a corpus directory with its
    manifest; the third is held out."""
    from helpers.synthetic_replay import move, turn, two_sides, write_replay
    from tools.replay_extract import extract_replay
    rides = [[(3, 5), (3, 6)], [(3, 5), (4, 5)], [(3, 5), (3, 4)]]
    dataset = tmp_path / "corpus"
    dataset.mkdir()
    rows = []
    for i, ride in enumerate(rides):
        commands = [*turn(1, move(1, ride)), *turn(2, move(2, [(11, 4), (11, 5)])),
                    *turn(1), *turn(2)]
        rec = extract_replay(write_replay(tmp_path / f"g{i}.bz2",
                                          two_sides(extra1=[("Cavalryman", 3, 5, False)]), commands))
        name = f"g{i}.json.gz"
        with gzip.open(dataset / name, "wt", encoding="utf-8") as f:
            json.dump(rec, f)
        rows.append({"file": name, "winner_side": 1 + i % 2, "winner_actions": 3,
                     "holdout": i == 2, "corpus_version": 4})
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return dataset


def pass_arguments(tmp_path: Path):
    """The three games pre-encoded by the script run by path, as the box
    runs it (its worker processes pickle the records, which this process
    must be able to read), and the trainer's arguments for a tiny network."""
    dataset = synthetic_dataset(tmp_path)
    sequences = tmp_path / "sequences"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / "tools" / "preencode_sequences.py"), "--dataset", str(dataset),
                    "--out", str(sequences), "--workers", "1", "--log-level", "WARNING"],
                   cwd=root, check=True, timeout=600)
    return ["--sequences", str(sequences), "--dataset", str(dataset), "--device", "cpu",
            "--streams", "2", "--window", "3", "--warmup-steps", "2", "--checkpoint-every", "1000000",
            "--log-level", "WARNING", "--probe-ks", "0,8", "--signal-every", "3",
            "--d-model", "32", "--num-layers", "1", "--num-heads", "2", "--d-ff", "64"]
