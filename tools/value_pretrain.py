"""Value-head pre-training on the human-replay corpus.

Fine-tunes a checkpoint's value (+ moves-left) heads on human games
with clean win/loss labels (tools/build_value_corpus.py), holding out
whole games for evaluation. Human experiences carry no visit counts,
so `step_mcts` trains value/ml only -- the policy head receives no
direct gradient (the shared trunk DOES move unless --freeze-trunk).

Experiment protocol (2026-07-08): the self-play value head plateaued
~1 nat above the state-blind floor on fresh games and got WORSE on
human states as self-play progressed (4.27 -> 6.27 CE vs floor 0.69).
This script answers: can clean human labels teach a value function
that generalizes? Success = held-out human CE well below the ~0.69
marginal floor... (floor is ln2 for balanced +-1 labels; a
state-reading head must go BELOW it).

Usage:
    python tools/value_pretrain.py \
        --checkpoint-in <ckpt.pt> --checkpoint-out human_value.pt \
        [--epochs 4] [--stride 6] [--holdout-games 300]
        [--batch 256] [--lr 5e-5] [--freeze-trunk]
        [--limit-games N]   # smoke
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("value_pretrain")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--checkpoint-in", type=Path, required=True)
    ap.add_argument("--checkpoint-out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--holdout-games", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--value-label-smoothing", type=float, default=0.02)
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="Train ONLY the value/moves-left heads (trunk "
                         "+ policy heads untouched; safest for resuming "
                         "self-play, weakest value features).")
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    import json
    import torch
    from transformer_policy import TransformerPolicy
    from tools.value_corpus import game_experiences

    raw = torch.load(args.checkpoint_in, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    policy = TransformerPolicy(
        d_model=a["d_model"], num_layers=a["num_layers"],
        num_heads=a["num_heads"], d_ff=a["d_ff"],
        aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")))
    policy.load_checkpoint(args.checkpoint_in)
    trainer = policy._trainer
    trainer.config.value_label_smoothing = args.value_label_smoothing
    # Human samples carry no policy target; entropy/policy terms are
    # inert. Keep value at full weight.
    trainer.config.value_coef = 1.0

    if args.freeze_trunk:
        n_frozen = 0
        for name, p in policy._model.named_parameters():
            if not (name.startswith("value_head")
                    or name.startswith("moves_left_head")):
                p.requires_grad_(False)
                n_frozen += 1
        # Rebuild the optimizer over the remaining trainable params
        # (the loaded optimizer state indexes ALL params).
        trainer.optimizer = torch.optim.AdamW(
            [p for p in policy._model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=trainer.config.weight_decay)
        log.info(f"trunk frozen ({n_frozen} tensors); value/ml heads "
                 f"only")
    else:
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr

    # ---- game-level split (whole games; no state leakage) ----------
    index = args.dataset_dir / "value_corpus_index.jsonl"
    rows = [json.loads(l) for l in index.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit_games:
        rows = rows[:args.limit_games]
    holdout_rows = rows[:args.holdout_games]
    train_rows = rows[args.holdout_games:]
    log.info(f"{len(train_rows)} train games, {len(holdout_rows)} "
             f"held-out games")

    def load_exps(row, stride):
        return game_experiences(args.dataset_dir / row["file"],
                                row["winner"], stride=stride, rng=rng)

    # Fixed held-out probe (sampled once; ~2k states).
    probe = []
    for row in holdout_rows:
        if len(probe) >= 2048:
            break
        try:
            probe.extend(load_exps(row, max(args.stride, 8)))
        except Exception as e:                  # noqa: BLE001
            log.debug(f"holdout skip {row['file']}: {e}")
    probe = probe[:2048]
    m0 = trainer.eval_value_metrics(probe)
    log.info(f"BEFORE: holdout ce={m0['ce']:.4f} "
             f"pred_entropy={m0['pred_entropy']:.4f} "
             f"floor={m0['marginal_ce_floor']:.4f} (n={len(probe)})")

    best_ce = m0["ce"]
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        t0 = time.time()
        n_pairs = n_batches = 0
        batch: List = []
        vloss_sum = vloss_n = 0.0
        for row in train_rows:
            try:
                batch.extend(load_exps(row, args.stride))
            except Exception as e:              # noqa: BLE001
                log.debug(f"skip {row['file']}: {e}")
                continue
            while len(batch) >= args.batch:
                chunk, batch = batch[:args.batch], batch[args.batch:]
                stats = trainer.step_mcts(chunk)
                vloss_sum += stats.value_loss; vloss_n += 1
                n_pairs += len(chunk); n_batches += 1
                if n_batches % 50 == 0:
                    log.info(f"  epoch {epoch}: {n_batches} batches, "
                             f"{n_pairs} pairs, train_v="
                             f"{vloss_sum / max(1, vloss_n):.4f}")
        if batch:
            stats = trainer.step_mcts(batch)
            n_pairs += len(batch)
        m = trainer.eval_value_metrics(probe)
        log.info(f"epoch {epoch}: {n_pairs} pairs in "
                 f"{time.time() - t0:.0f}s | holdout ce={m['ce']:.4f} "
                 f"pred_entropy={m['pred_entropy']:.4f} "
                 f"(floor {m['marginal_ce_floor']:.4f})")
        if m["ce"] < best_ce:
            best_ce = m["ce"]
            policy.save_checkpoint(args.checkpoint_out)
            log.info(f"  new best -> saved {args.checkpoint_out}")
    log.info(f"done. best holdout ce={best_ce:.4f} "
             f"(started {m0['ce']:.4f}, floor "
             f"~{m0['marginal_ce_floor']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
