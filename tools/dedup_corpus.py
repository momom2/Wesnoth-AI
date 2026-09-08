"""Remove the redundant copies of a match from an existing imitation
corpus: the same game saved at two turns, or uploaded twice, appears
under two file names, and one copy of a holdout game in the training
split leaks it. tools/build_imitation_dataset.py applies the same rule
(tools/replay_dataset.match_key, build_imitation_dataset.dedup_rows)
at extraction time; this pass repairs a corpus built before it.

Per cluster the copy with the most commands survives; it becomes
holdout when any copy was; the others move to `<dataset>_duplicates/`
and are listed in duplicates.jsonl with their survivor. The manifest
and the value index are rewritten. A second run is a no-op.

Usage:
    python tools/dedup_corpus.py --dataset replays_dataset_imitation --workers 8
"""
import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_imitation_dataset import dedup_rows  # noqa: E402
from tools.replay_dataset import match_key  # noqa: E402


def _key_of(path_str: str):
    with gzip.open(path_str, "rt", encoding="utf-8") as f:
        return os.path.basename(path_str), match_key(json.load(f))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def dedup_corpus(dataset: Path, workers: int = 4, log=print):
    manifest_path = dataset / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    t0 = time.time()
    keys = {}
    with Pool(workers) as pool:
        for i, (name, key) in enumerate(pool.imap_unordered(
                _key_of, [str(dataset / r["file"]) for r in rows], chunksize=50), 1):
            keys[name] = key
            if i % 2000 == 0:
                log(f"{i}/{len(rows)} keyed, {time.time() - t0:.0f} s")
    for r in rows:
        r["match_key"] = keys[r["file"]]
    kept, dropped = dedup_rows(rows)
    dup_dir = dataset.parent / (dataset.name + "_duplicates")
    for r in dropped:
        src = dataset / r["file"]
        if src.exists():
            dup_dir.mkdir(parents=True, exist_ok=True)
            os.replace(src, dup_dir / r["file"])
    kept.sort(key=lambda r: r["file"])
    write_jsonl(manifest_path, kept)
    dup_path = dataset / "duplicates.jsonl"
    previous = read_jsonl(dup_path) if dup_path.exists() else []
    write_jsonl(dup_path, previous + dropped)
    index_path = dataset / "value_corpus_index.jsonl"
    if index_path.exists():
        gone = {r["file"] for r in dropped}
        write_jsonl(index_path, [r for r in read_jsonl(index_path) if r["file"] not in gone])
    was_holdout = {r["file"] for r in rows if r.get("holdout")}
    promoted = sum(1 for r in kept if r.get("holdout") and r["file"] not in was_holdout)
    log(f"kept {len(kept)} games, moved {len(dropped)} duplicates to {dup_dir.name}, "
        f"{promoted} survivors promoted to holdout, {time.time() - t0:.0f} s")
    return kept, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=ROOT / "replays_dataset_imitation")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    dedup_corpus(args.dataset, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
