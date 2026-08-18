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


def _load_worker(task):
    """Pool worker: reconstruct one game -> capped experience list.
    Top-level (spawn-picklable); each call seeds its own rng from the
    task so results are order-independent under imap_unordered.
    Loading REPLAYS the whole game (~4 s single-threaded), which
    made one epoch over 16,824 games an 18-hour wall -- the 48-core
    box was idle while one process replayed games (2026-08-17)."""
    dataset_dir, fname, winner, stride, cap, seed = task
    import random as _random
    from tools.value_corpus import game_experiences
    rng = _random.Random(seed)
    try:
        exps = game_experiences(Path(dataset_dir) / fname, winner,
                                stride=stride, rng=rng)
    except Exception:                                   # noqa: BLE001
        return []
    if cap and len(exps) > cap:
        exps = rng.sample(exps, cap)
    return exps


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
    ap.add_argument("--loader-jobs", type=int, default=8,
                    help="Parallel game-reconstruction workers "
                         "(loading replays whole games; 1 job = "
                         "~18 h/epoch over the full corpus).")
    ap.add_argument("--max-states-per-game", type=int, default=8,
                    help="Cap on training states drawn per game "
                         "(AFTER stride thinning; random subsample). "
                         "AlphaGo hygiene (A3, 2026-08-17): all of a "
                         "game's states share ONE outcome bit, so "
                         "the head's effective sample size is games, "
                         "not states -- many states per game just "
                         "replays the same bit. 0 = uncapped "
                         "(legacy).")
    ap.add_argument("--linear-probe", action="store_true",
                    help="Q1 diagnostic arm: replace the value head "
                         "with a single fresh Linear(d_model -> "
                         "atoms), freeze everything else, train, "
                         "report -- do NOT save a checkpoint. Read "
                         "against the full-head arm: linear ~ full "
                         "=> the trunk's features are the cap on "
                         "value discrimination, and head work can't "
                         "fix it.")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="Train ONLY the value/moves-left heads (trunk "
                         "+ policy heads untouched; safest for resuming "
                         "self-play, weakest value features).")
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--probe-states", type=int, default=1024,
                    help="Held-out probe size. Each probe state is a "
                         "deep-copied GameState held in RAM for the "
                         "whole run -- 2048 of them OOM-killed the "
                         "first overnight run on the laptop "
                         "(silently; Windows gives no traceback).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    import json
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
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

    if args.linear_probe:
        # Q1 arm: a fresh single-linear head on frozen features.
        import torch.nn as nn
        from wesnoth_ai.model import VALUE_N_ATOMS
        d = a["d_model"]
        torch.manual_seed(args.seed)
        policy._model.value_head = nn.Sequential(
            nn.Linear(d, VALUE_N_ATOMS))
        args.freeze_trunk = True
        log.info(f"LINEAR-PROBE arm: value_head := Linear({d} -> "
                 f"{VALUE_N_ATOMS}); checkpoint saving disabled")
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
        # Freeze-integrity receipt (A3 acceptance: trunk param hashes
        # bit-identical; with the policy head frozen, frozen-state
        # p(end_turn) trivially cannot move). Verified at exit.
        def _frozen_sig():
            import hashlib
            h = hashlib.sha256()
            for name, p in sorted(
                    policy._model.named_parameters()):
                if not p.requires_grad:
                    h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        frozen_sig0 = _frozen_sig()
    else:
        frozen_sig0 = None

        def _frozen_sig():
            return None
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr

    # ---- game-level split (whole games; no state leakage) ----------
    index = args.dataset_dir / "value_corpus_index.jsonl"
    rows = [json.loads(ln) for ln in index.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit_games:
        rows = rows[:args.limit_games]
    holdout_rows = rows[:args.holdout_games]
    train_rows = rows[args.holdout_games:]
    log.info(f"{len(train_rows)} train games, {len(holdout_rows)} "
             f"held-out games")

    def load_exps(row, stride, cap=None):
        exps = game_experiences(args.dataset_dir / row["file"],
                                row["winner"], stride=stride, rng=rng)
        cap = args.max_states_per_game if cap is None else cap
        if cap and len(exps) > cap:
            exps = rng.sample(exps, cap)
        return exps

    def iter_exps_parallel(rows, stride, epoch):
        """Yield per-game experience lists from a worker pool (order-
        free). Per-(epoch, game) seeds keep subsampling deterministic
        for a given --seed regardless of arrival order."""
        tasks = [(str(args.dataset_dir), r["file"], r["winner"],
                  stride, args.max_states_per_game,
                  args.seed * 1_000_003 + epoch * 131 + i)
                 for i, r in enumerate(rows)]
        if args.loader_jobs <= 1:
            for t in tasks:
                yield _load_worker(t)
            return
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.loader_jobs) as pool:
            yield from pool.imap_unordered(_load_worker, tasks,
                                           chunksize=8)

    # Fixed held-out probe (sampled once, parallel-loaded). Probe
    # stride is high so the probe spans MANY games at few states.
    probe = []
    for exps in iter_exps_parallel(holdout_rows, max(args.stride, 16),
                                   epoch=-1):
        probe.extend(exps)
        if len(probe) >= args.probe_states:
            break
    probe = probe[:args.probe_states]
    m0 = trainer.eval_value_metrics(probe)
    log.info(f"BEFORE: holdout ce={m0['ce']:.4f} "
             f"value_auc={m0.get('value_auc', float('nan')):.4f} "
             f"pred_entropy={m0['pred_entropy']:.4f} "
             f"floor={m0['marginal_ce_floor']:.4f} (n={len(probe)})")

    # A3 (2026-08-17): outcome AUC is the acceptance-gate metric
    # (level discrimination -- what the launch gate reads); best
    # checkpoint is selected on it, CE logged alongside.
    best_ce = m0["ce"]
    best_auc = m0.get("value_auc", float("nan"))
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        t0 = time.time()
        n_pairs = n_batches = 0
        batch: List = []
        vloss_sum = vloss_n = 0.0
        for exps in iter_exps_parallel(train_rows, args.stride,
                                       epoch=epoch):
            batch.extend(exps)
            while len(batch) >= args.batch:
                chunk, batch = batch[:args.batch], batch[args.batch:]
                stats = trainer.step_mcts(chunk)
                vloss_sum += stats.value_loss
                vloss_n += 1
                n_pairs += len(chunk)
                n_batches += 1
                if n_batches % 50 == 0:
                    log.info(f"  epoch {epoch}: {n_batches} batches, "
                             f"{n_pairs} pairs, train_v="
                             f"{vloss_sum / max(1, vloss_n):.4f}")
        if batch:
            stats = trainer.step_mcts(batch)
            n_pairs += len(batch)
        m = trainer.eval_value_metrics(probe)
        auc = m.get("value_auc", float("nan"))
        log.info(f"epoch {epoch}: {n_pairs} pairs in "
                 f"{time.time() - t0:.0f}s | holdout ce={m['ce']:.4f} "
                 f"value_auc={auc:.4f} "
                 f"pred_entropy={m['pred_entropy']:.4f} "
                 f"(floor {m['marginal_ce_floor']:.4f})")
        best_ce = min(best_ce, m["ce"])
        import math as _math
        if not _math.isnan(auc) and (
                _math.isnan(best_auc) or auc > best_auc):
            best_auc = auc
            if not args.linear_probe:
                policy.save_checkpoint(args.checkpoint_out)
                log.info(f"  new best AUC -> saved "
                         f"{args.checkpoint_out}")
    if frozen_sig0 is not None:
        sig1 = _frozen_sig()
        if sig1 != frozen_sig0:
            log.error("FREEZE VIOLATION: frozen parameters changed "
                      "during training -- the saved head is NOT a "
                      "pure head fit. Do not use this checkpoint.")
            return 1
        log.info("freeze integrity verified: frozen params "
                 "bit-identical")
    log.info(f"done. best holdout value_auc={best_auc:.4f} "
             f"ce={best_ce:.4f} (started auc="
             f"{m0.get('value_auc', float('nan')):.4f} "
             f"ce={m0['ce']:.4f}, floor "
             f"~{m0['marginal_ce_floor']:.2f})"
             + ("  [linear-probe arm: nothing saved]"
                if args.linear_probe else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
