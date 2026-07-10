"""Pre-encode a human-corpus anchor cache for rehearsal training.

Samples games from the value-corpus index, reconstructs + encode_raw's
them in parallel (same producer as the value fine-tune), and pickles a
flat list of (RawEncoded, z, moves_left) tuples. sim_self_play's
--human-anchor-file loads this once and runs value-only rehearsal
steps against it every iteration — the anti-forgetting anchor for the
value head (2026-07-10: self-play alone eroded human-corpus late-game
AUC 0.88 -> 0.60 in ~80 iterations).

The cache is checkpoint-independent (RawEncoded depends only on the
frozen vocab, which every campaign checkpoint shares — verified by
the vocab-sharing invariant tests). Rebuild only if the encoder
feature layout or the corpus changes.

Usage (on the training box; ~64 workers, a few minutes):
    python tools/build_human_anchor.py \
        --out replays_dataset/human_anchor.pkl \
        [--games 2000] [--stride 8] [--workers 16] [--seed 7]
        [--skip-games 0]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("build_human_anchor")

_G = {}


def _init(dataset_dir, t2i, f2i, stride):
    _G["dir"] = Path(dataset_dir)
    _G["t2i"] = t2i
    _G["f2i"] = f2i
    _G["stride"] = stride


def _one(row):
    from tools.value_corpus import game_raw_experiences
    try:
        return game_raw_experiences(
            _G["dir"] / row["file"], row["winner"],
            type_to_id=_G["t2i"], faction_to_id=_G["f2i"],
            stride=_G["stride"],
            rng=random.Random(row["game_id"].__hash__() & 0xFFFF))
    except Exception as e:                      # noqa: BLE001
        log.debug(f"skip {row['file']}: {e}")
        return []


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--out", type=Path,
                    default=Path("replays_dataset/human_anchor.pkl"))
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--skip-games", type=int, default=0,
                    help="Skip the first N rows after the seed-0 "
                         "shuffle (reserve them, e.g. as probe "
                         "holdout).")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from transformer_policy import TransformerPolicy
    # Vocab comes from any policy construction (frozen, shared across
    # all campaign checkpoints).
    net = TransformerPolicy(d_model=32, num_layers=1, num_heads=4,
                            d_ff=64)
    t2i = dict(net._encoder.unit_type_to_id)
    f2i = dict(net._encoder.faction_to_id)
    del net

    index = args.dataset_dir / "value_corpus_index.jsonl"
    rows = [json.loads(l) for l in index.open(encoding="utf-8")]
    random.Random(0).shuffle(rows)      # the shared corpus shuffle
    rows = rows[args.skip_games:]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.games]
    log.info(f"encoding {len(rows)} games "
             f"(stride {args.stride}, {args.workers} workers)")

    out: List = []
    with Pool(args.workers, initializer=_init,
              initargs=(args.dataset_dir, t2i, f2i, args.stride)) as p:
        for i, recs in enumerate(p.imap_unordered(_one, rows, 8), 1):
            out.extend(recs)
            if i % 250 == 0:
                log.info(f"  {i}/{len(rows)} games, {len(out)} pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    zpos = sum(1 for _, z, _ in out if z > 0)
    log.info(f"wrote {args.out}: {len(out)} pairs "
             f"(z +{zpos} / -{len(out) - zpos}) "
             f"{args.out.stat().st_size / 2**20:.0f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
