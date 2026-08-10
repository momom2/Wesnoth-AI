"""Policy-head human-anchor rehearsal (F1, user ruling 2026-08-10).

The value-head anchor (`--human-anchor-file`, tools/build_human_anchor)
protects V(s) from self-play forgetting. This module extends the same
rehearsal pattern to the FOUR POLICY HEADS: each training iteration
runs a few extra gradient updates of the imitation objective (hard
human action, four-head CE via supervised_train._loss_parts_for_output
-- exactly the BC loss, NOT a search target) on a fixed pre-encoded
pool of winner-side human pairs.

RLPD-shaped: rehearse the offline data during online RL so the
imitation prior (holdout CE 3.102 at handoff t0) survives the
self-referential self-play distillation target.

DEFAULT OFF. Ruling: leg 1 of the tier-b handoff runs A1
(--distill-prior-discount 0.9) as its ONE prior-protection arm for
attribution; this anchor is leg 2's arm if A1 fails the CE observable
(human-holdout policy CE flat across the leg + RCA probe at leg end).

Cache build (box or laptop; needs replays_dataset_imitation/):
    python tools/policy_anchor.py --out replays_dataset/policy_anchor.pkl \
        [--games 500] [--stride 4] [--seed 7]

Then:  tools/sim_self_play.py --human-anchor-policy-file <pkl>

Holdout discipline: manifest-holdout games are EXCLUDED from the
cache -- rehearsing them would contaminate the very probe the
observable is measured on.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("policy_anchor")

CACHE_VERSION = 1


# ---------------------------------------------------------------------
# Rehearsal step (called per iteration by sim_self_play)
# ---------------------------------------------------------------------

def anchor_policy_step(trainer, pairs: List) -> Dict[str, float]:
    """One gradient step of the imitation objective over `pairs`
    (list of (RawEncoded, ActionIndices)). Chunked by
    train_batch_size like step_value_from_raw; one optimizer.step()
    over the whole batch (losses are /N).

    Reuses supervised_train's per-sample four-head CE so the anchor
    loss IS the BC objective -- same weights, same guards, no
    legality mask. Value head gets no gradient here (the value anchor
    is a separate call; keeps per-head attribution clean)."""
    import torch
    from tools.supervised_train import _loss_parts_for_output

    model, encoder = trainer.model, trainer.encoder
    dev = trainer.device or next(model.parameters()).device
    model.train()
    encoder.train()
    N = len(pairs)
    B = max(1, trainer.config.train_batch_size)
    trainer.optimizer.zero_grad()
    total = 0.0
    actor_sum = 0.0
    actor_n = 0
    for start in range(0, N, B):
        chunk = pairs[start:start + B]
        encoded = encoder.encode_from_raw_batch(
            [p[0] for p in chunk], device=dev)
        outputs = model.forward_batch(encoded)
        chunk_losses = []
        for (_, ai), output in zip(chunk, outputs):
            parts = _loss_parts_for_output(output, ai, dev)
            chunk_losses.append(parts.total)
            if parts.actor_fired:
                actor_sum += float(parts.actor.item())
                actor_n += 1
        loss = torch.stack(chunk_losses).sum() / N
        total += float(loss.item())
        loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(encoder.parameters()),
        trainer.config.grad_clip))
    trainer.optimizer.step()
    model.eval()
    encoder.eval()
    return {"policy_ce": total,
            "actor_ce": (actor_sum / actor_n) if actor_n else float("nan"),
            "grad_norm": grad_norm}


def load_policy_anchor(path: Path) -> List:
    """Load and validate a policy-anchor cache; returns the pair list."""
    with Path(path).open("rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or blob.get("version") != CACHE_VERSION:
        raise ValueError(
            f"{path}: not a policy-anchor cache (want version "
            f"{CACHE_VERSION}, got "
            f"{blob.get('version') if isinstance(blob, dict) else type(blob)})")
    return blob["pairs"]


# ---------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------

def build_cache(dataset_dir: Path, out: Path, *, games: int,
                stride: int, seed: int,
                type_to_id: dict, faction_to_id: dict) -> int:
    """Sample winner-side (RawEncoded, ActionIndices) pairs from
    non-holdout imitation games into a pickled cache. Returns the
    number of pairs written."""
    from wesnoth_ai.encoder import encode_raw
    from tools.replay_dataset import iter_replay_pairs

    man_path = dataset_dir / "manifest.jsonl"
    rows = [json.loads(ln) for ln in man_path.open(encoding="utf-8")]
    pool = [r for r in rows if not r["holdout"] and r["winner_actions"] > 0]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:games]
    log.info(f"sampling {len(pool)} games (holdout excluded) "
             f"stride={stride}")

    pairs = []
    for i, r in enumerate(pool):
        gz = dataset_dir / r["file"]
        winner = int(r["winner_side"])
        offset = rng.randrange(stride) if stride > 1 else 0
        k = 0
        try:
            for state, ai in iter_replay_pairs(gz):
                if state.global_info.current_side != winner:
                    continue
                k += 1
                if (k - 1) % stride != offset:
                    continue
                raw = encode_raw(state, type_to_id=type_to_id,
                                 faction_to_id=faction_to_id)
                pairs.append((raw, ai))
        except Exception as e:                      # noqa: BLE001
            log.warning(f"skip {gz.name}: {e!r}")
        if (i + 1) % 50 == 0:
            log.info(f"  {i + 1}/{len(pool)} games, {len(pairs)} pairs")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION,
                     "meta": {"games": len(pool), "stride": stride,
                              "seed": seed, "winners_only": True,
                              "holdout_excluded": True},
                     "pairs": pairs}, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"wrote {len(pairs)} pairs -> {out}")
    return len(pairs)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset_imitation"))
    ap.add_argument("--out", type=Path,
                    default=Path("replays_dataset/policy_anchor.pkl"))
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--stride", type=int, default=4,
                    help="Keep every stride-th winner-side pair "
                         "(random per-game offset).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # Frozen vocab: same seeding path as supervised_train (unit_stats
    # pre-seed) so RawEncoded ids match every campaign checkpoint.
    from wesnoth_ai.encoder import GameStateEncoder
    from tools.supervised_train import _seed_vocab_from_unit_stats
    enc = GameStateEncoder(d_model=32)
    _seed_vocab_from_unit_stats(
        enc, args.dataset_dir.parent / "unit_stats.json")
    n = build_cache(args.dataset_dir, args.out, games=args.games,
                    stride=args.stride, seed=args.seed,
                    type_to_id=enc.unit_type_to_id,
                    faction_to_id=enc.faction_to_id)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
