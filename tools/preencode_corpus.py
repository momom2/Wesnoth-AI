"""Pre-encode the imitation corpus once, so training streams tensors
instead of replaying every game through the simulator on every run.

For each game of the dataset manifest, the same work the encode
workers do at training time (`replay_dataset.iter_replay_pairs` +
`encoder.encode_raw` against a frozen vocab) runs here once, and the
whole game's pair list -- exactly the ("file", pairs) message a
worker would send -- is stored as one compressed pickle:
`OUT/<game>.pairs.zpkl`. `supervised_train.py --preencoded OUT` then
reads those files in its usual seeded file order; the recipe (which
pairs, in which order, with which weights) is unchanged because only
the "replay this file" step is replaced.

The vocab (unit types, factions) and the hex basis are part of the
encoding: `--vocab-from CKPT` takes the checkpoint's dicts (what a
warm start trains with), and `preencoded_manifest.json` records a
fingerprint of them; the trainer refuses a corpus whose fingerprint
is not its encoder's.

Idempotent and resumable: existing records are skipped, so a killed
pass continues where it stopped. Progress is logged on the run.

Usage (a box, one pass over the 17k games; ~5 pairs/s per worker):
    python tools/preencode_corpus.py --dataset replays_dataset_imitation \\
        --out replays_dataset_imitation_encoded \\
        --vocab-from training/checkpoints/seed.pt --workers 60
"""
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

log = logging.getLogger("preencode")

RECORD_SUFFIX = ".pairs.zpkl"
MANIFEST_NAME = "preencoded_manifest.json"


def vocab_fingerprint(type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                      relevant_set: bool) -> str:
    """One string for (unit vocab, faction vocab, hex basis): the
    encoding depends on nothing else that varies between runs."""
    h = hashlib.sha1()
    h.update(json.dumps(sorted(type_to_id.items())).encode("utf-8"))
    h.update(b"|")
    h.update(json.dumps(sorted(faction_to_id.items())).encode("utf-8"))
    h.update(b"|relset=%d" % int(bool(relevant_set)))
    return h.hexdigest()


def vocab_from_checkpoint(path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return dict(ck.get("unit_type_to_id", {})), dict(ck.get("faction_to_id", {}))


def record_path(out_dir: Path, gz_name: str) -> Path:
    return out_dir / (gz_name + RECORD_SUFFIX)


def write_record(path: Path, pairs: List) -> None:
    """Atomic: the whole game's (RawEncoded, ActionIndices) list, one
    pickle, zlib level 1 (the hex flag arrays are mostly zeros)."""
    blob = zlib.compress(pickle.dumps(pairs, protocol=pickle.HIGHEST_PROTOCOL), 1)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)


def read_record(path: Path) -> List:
    return pickle.loads(zlib.decompress(path.read_bytes()))


def encode_game(gz_path: Path, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                relevant_set: bool) -> List:
    """The encode worker's per-file work (tools/encode_worker.py)."""
    from tools.replay_dataset import iter_replay_pairs
    from wesnoth_ai.encoder import encode_raw
    pairs = []
    for state, ai in iter_replay_pairs(gz_path, relevant_set=relevant_set):
        pairs.append((encode_raw(state, type_to_id=type_to_id, faction_to_id=faction_to_id,
                                 relevant_set=relevant_set), ai))
    return pairs


_W: Dict = {}


def _worker_init(type_to_id, faction_to_id, relevant_set, out_dir):
    _W.update(type_to_id=type_to_id, faction_to_id=faction_to_id,
              relevant_set=relevant_set, out_dir=Path(out_dir))


def _worker_encode(gz_path_str: str):
    gz_path = Path(gz_path_str)
    dst = record_path(_W["out_dir"], gz_path.name)
    if dst.exists():
        return gz_path.name, None, "exists"
    try:
        pairs = encode_game(gz_path, _W["type_to_id"], _W["faction_to_id"], _W["relevant_set"])
    except Exception as e:  # noqa: BLE001 - one bad game must not stop the pass
        return gz_path.name, None, f"error: {type(e).__name__}: {e}"[:200]
    write_record(dst, pairs)
    return gz_path.name, len(pairs), "ok"


def load_manifest(out_dir: Path) -> Optional[Dict]:
    p = out_dir / MANIFEST_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=ROOT / "replays_dataset_imitation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-from", type=Path, required=True,
                    help="Checkpoint whose unit/faction vocab the encoding uses "
                         "(the trainer's --init-from / --resume checkpoint).")
    ap.add_argument("--relevant-set-hexes", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="first N manifest games")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    type_to_id, faction_to_id = vocab_from_checkpoint(args.vocab_from)
    fp = vocab_fingerprint(type_to_id, faction_to_id, args.relevant_set_hexes)
    args.out.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(args.out)
    if existing and existing.get("fingerprint") != fp:
        raise SystemExit(f"{args.out} was encoded with another vocab or basis "
                         f"({existing.get('fingerprint')} against {fp}); use another --out")
    rows = [json.loads(line) for line in
            (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()][:args.limit]
    files = [str(args.dataset / r["file"]) for r in rows]
    counts: Dict[str, int] = dict((existing or {}).get("pairs_per_file", {}))
    errors: Dict[str, str] = dict((existing or {}).get("errors", {}))
    t0 = time.time()
    done = skipped = 0

    def flush():
        args.out.joinpath(MANIFEST_NAME).write_text(json.dumps({
            "fingerprint": fp, "vocab_from": str(args.vocab_from),
            "relevant_set_hexes": bool(args.relevant_set_hexes),
            "dataset": str(args.dataset), "n_files": len(counts),
            "n_pairs": int(sum(counts.values())), "pairs_per_file": counts,
            "errors": errors, "unit_types": len(type_to_id), "factions": len(faction_to_id),
        }, indent=0), encoding="utf-8")

    log.info("pre-encoding %d games with %d workers into %s (vocab %d types, %d factions, "
             "relevant set %s)", len(files), args.workers, args.out, len(type_to_id),
             len(faction_to_id), args.relevant_set_hexes)
    with mp.get_context("spawn").Pool(
            args.workers, initializer=_worker_init,
            initargs=(type_to_id, faction_to_id, args.relevant_set_hexes, str(args.out))) as pool:
        for i, (name, n, status) in enumerate(pool.imap_unordered(_worker_encode, files,
                                                                    chunksize=4), 1):
            if status == "ok":
                counts[name] = n
                done += 1
            elif status == "exists":
                skipped += 1
                if name not in counts:
                    counts[name] = len(read_record(record_path(args.out, name)))
            else:
                errors[name] = status
            if i % 200 == 0 or i == len(files):
                el = time.time() - t0
                log.info("%d/%d files (%d new, %d existing, %d errors), %d pairs, %.0f s, "
                         "%.1f pairs/s", i, len(files), done, skipped, len(errors),
                         sum(counts.values()), el, sum(counts.values()) / max(el, 1e-9))
                flush()
    flush()
    log.info("PREENCODE_DONE %d files, %d pairs, %d errors in %.0f s", len(counts),
             sum(counts.values()), len(errors), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
