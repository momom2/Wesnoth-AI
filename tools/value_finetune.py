"""Unfrozen value-head fine-tune on the human corpus, parallelized.

The frozen-trunk experiment (tools/value_pretrain.py) cured the value
head's confident-wrongness but plateaued at the floor with only weak
board reading (late-game AUC ~0.68) -- the frozen self-play trunk's
features cap it. This script unfreezes the FULL trunk (+ value/ml
heads) so the network can learn win-predictive features, and tests
whether that lifts discrimination toward a genuinely strong value head.

Parallelism: the cost is CPU game-reconstruction, not GPU. Worker
processes reconstruct + encode_raw games (RawEncoded is the picklable
worker->trainer boundary) so a many-core box is saturated; the main
process does encode_from_raw_batch + forward/backward on the GPU.
encode-at-sample-time also avoids the deepcopy that OOM'd the frozen
run.

Policy heads get NO gradient (human data has no visit counts; only
value + moves-left losses flow). The trunk moves; self-play re-aligns
the policy later.

    python tools/value_finetune.py \
        --checkpoint-in <human_or_campaign.pt> \
        --checkpoint-out training/checkpoints/human_value_full.pt \
        --workers 30 --epochs 3 --stride 8 --lr 2e-5 \
        --limit-games 2500 --holdout-games 300 [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("value_finetune")

_W: dict = {}


def _init_worker(dataset_dir, type_to_id, faction_to_id, stride, mlnorm):
    _W["dir"] = Path(dataset_dir)
    _W["t2i"] = type_to_id
    _W["f2i"] = faction_to_id
    _W["stride"] = stride
    _W["mlnorm"] = mlnorm


def _work(task):
    file, winner, seed = task
    import random as _r
    from tools.value_corpus import game_raw_experiences
    try:
        return game_raw_experiences(
            _W["dir"] / file, winner,
            type_to_id=_W["t2i"], faction_to_id=_W["f2i"],
            stride=_W["stride"], rng=_r.Random(seed),
            moves_left_norm=_W["mlnorm"])
    except Exception as e:                      # noqa: BLE001
        return ("ERR", file, str(e)[:120])


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=Path("replays_dataset"))
    ap.add_argument("--checkpoint-in", type=Path, required=True)
    ap.add_argument("--checkpoint-out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--probe-stride", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=256,
                    help="Gradient-step batch (states per optimizer "
                         "step).")
    ap.add_argument("--fwd-chunk", type=int, default=8,
                    help="forward_batch sub-chunk (attention is O(S^2) "
                         "per full-map state; keep small: 8 on CPU, "
                         "16-32 on a 24GB GPU).")
    ap.add_argument("--value-label-smoothing", type=float, default=0.02)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--holdout-games", type=int, default=300)
    ap.add_argument("--probe-states", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=40,
                    help="Run the held-out probe every N gradient "
                         "steps (mid-epoch) so the run yields a CE "
                         "curve, not just epoch endpoints. 0 = only "
                         "at epoch boundaries.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import torch
    from transformer_policy import TransformerPolicy
    from tools.mcts_policy import MOVES_LEFT_NORM_TURNS
    from tools.value_corpus import game_raw_experiences

    dev = torch.device(args.device)
    raw = torch.load(args.checkpoint_in, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    policy = TransformerPolicy(
        device=dev, d_model=a["d_model"], num_layers=a["num_layers"],
        num_heads=a["num_heads"], d_ff=a["d_ff"],
        aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")))
    policy.load_checkpoint(args.checkpoint_in)
    trainer = policy._trainer
    trainer.config.value_label_smoothing = args.value_label_smoothing
    trainer.config.value_coef = 1.0
    trainer.config.train_batch_size = args.fwd_chunk   # attention chunk
    # Full unfreeze: fresh optimizer over ALL params at the gentle
    # fine-tune LR (no stale self-play momentum, no frozen groups).
    for p in policy._model.parameters():
        p.requires_grad_(True)
    trainer.optimizer = torch.optim.AdamW(
        policy._model.parameters(), lr=args.lr,
        weight_decay=trainer.config.weight_decay)
    log.info(f"full-trunk fine-tune | lr={args.lr} | device={dev} | "
             f"workers={args.workers}")

    t2i = dict(policy._encoder.unit_type_to_id)
    f2i = dict(policy._encoder.faction_to_id)

    rows = [json.loads(l) for l in
            (args.dataset_dir / "value_corpus_index.jsonl")
            .open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit_games:
        rows = rows[:args.limit_games]
    holdout_rows = rows[:args.holdout_games]
    train_rows = rows[args.holdout_games:]
    log.info(f"{len(train_rows)} train games, {len(holdout_rows)} held-out")

    # Fixed held-out probe (serial; RawEncoded are compact, no OOM).
    probe_raw, probe_z = [], []
    for r in holdout_rows:
        if len(probe_raw) >= args.probe_states:
            break
        try:
            for rw, z, _ in game_raw_experiences(
                    args.dataset_dir / r["file"], r["winner"],
                    type_to_id=t2i, faction_to_id=f2i,
                    stride=args.probe_stride, rng=rng):
                probe_raw.append(rw); probe_z.append(z)
        except Exception:                       # noqa: BLE001
            continue
    probe_raw, probe_z = probe_raw[:args.probe_states], probe_z[:args.probe_states]
    m0 = trainer.eval_value_metrics_from_raw(probe_raw, probe_z)
    log.info(f"BEFORE: holdout ce={m0['ce']:.4f} "
             f"pred_entropy={m0['pred_entropy']:.4f} "
             f"floor={m0['marginal_ce_floor']:.4f} (n={len(probe_raw)})")

    import multiprocessing as mp
    best = m0["ce"]

    # Running global step count, so the mid-epoch held-out curve is
    # monotonic across epochs (not reset each epoch).
    step_ctr = {"n": 0}

    def _run_epoch(epoch, produce):
        """produce: iterable yielding per-game result lists."""
        br, bz, bm = [], [], []
        n_pairs = n_batches = errs = 0
        vsum = 0.0
        t0 = time.time()
        for res in produce:
            if res and isinstance(res, tuple) and res[0] == "ERR":
                errs += 1
                continue
            for rw, z, ml in res:
                br.append(rw); bz.append(z); bm.append(ml)
                if len(br) >= args.batch:
                    st = trainer.step_value_from_raw(br, bz, bm)
                    vsum += st["value_loss"]; n_batches += 1
                    step_ctr["n"] += 1
                    n_pairs += len(br)
                    if n_batches % 50 == 0:
                        log.info(f"  epoch {epoch}: {n_batches} batches, "
                                 f"{n_pairs} pairs, "
                                 f"train_v={vsum / n_batches:.4f}")
                    # Mid-epoch held-out curve point.
                    if (args.eval_every
                            and step_ctr["n"] % args.eval_every == 0):
                        me = trainer.eval_value_metrics_from_raw(
                            probe_raw, probe_z)
                        log.info(f"  [curve] step {step_ctr['n']} "
                                 f"(epoch {epoch}): holdout "
                                 f"ce={me['ce']:.4f} "
                                 f"ent={me['pred_entropy']:.4f}")
                    br, bz, bm = [], [], []
        if br:
            trainer.step_value_from_raw(br, bz, bm); n_pairs += len(br)
        return n_pairs, errs, time.time() - t0

    def _epoch_tasks(epoch):
        rng.shuffle(train_rows)
        return [(r["file"], r["winner"], args.seed + epoch * 100000 + i)
                for i, r in enumerate(train_rows)]

    def _finish_epoch(epoch, n_pairs, errs, dt):
        nonlocal best
        m = trainer.eval_value_metrics_from_raw(probe_raw, probe_z)
        log.info(f"epoch {epoch}: {n_pairs} pairs in {dt:.0f}s "
                 f"({errs} game errs) | holdout ce={m['ce']:.4f} "
                 f"pred_entropy={m['pred_entropy']:.4f} "
                 f"(floor {m['marginal_ce_floor']:.4f})")
        if m["ce"] < best:
            best = m["ce"]
            policy.save_checkpoint(args.checkpoint_out)
            log.info(f"  new best -> saved {args.checkpoint_out}")

    if args.workers <= 1:
        # Serial fallback (local validation; avoids the Windows-spawn
        # Pool). Reconstruct in-process.
        _init_worker(str(args.dataset_dir), t2i, f2i, args.stride,
                     MOVES_LEFT_NORM_TURNS)
        for epoch in range(args.epochs):
            np_, er, dt = _run_epoch(
                epoch, (_work(t) for t in _epoch_tasks(epoch)))
            _finish_epoch(epoch, np_, er, dt)
    else:
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(str(args.dataset_dir), t2i, f2i,
                                args.stride, MOVES_LEFT_NORM_TURNS)) as pool:
            for epoch in range(args.epochs):
                np_, er, dt = _run_epoch(
                    epoch, pool.imap_unordered(_work, _epoch_tasks(epoch),
                                               chunksize=1))
                _finish_epoch(epoch, np_, er, dt)
    log.info(f"done. best holdout ce={best:.4f} (started {m0['ce']:.4f}, "
             f"floor ~{m0['marginal_ce_floor']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
