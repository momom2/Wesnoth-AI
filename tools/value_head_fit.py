"""A3 value-head fit on CACHED frozen-trunk features (user directive
2026-08-17: the load->forward pipeline is the cost; pay it once).

The frozen-trunk arms only ever train heads that read the 384-d
global token (`model.value_head(global_ctx)`, model.py:419), so the
corpus is loaded and forwarded through the trunk EXACTLY ONCE per
trunk: every sampled state's global_ctx vector is cached to disk
(~400 MB for 270k states), and all head fits -- full MLP arm,
linear-probe arm, every epoch -- train on the cache at zero further
trunk forwards. The v4 matrix re-loaded and re-forwarded the same
17k games 16 times (4 arms x 4 epochs, ~47 min each); the cache
does it twice (once per trunk).

Per-epoch sampling diversity is kept by caching a SUPERSET
(--cache-states-per-game, default 16) and drawing
--train-states-per-game (default 8, the AlphaGo-hygiene cap) fresh
per epoch from the cache. Heads other than value keep their loaded
weights (moves-left is not refit here; value is the gate
objective).

Arms:
  --arm full    fine-tune the checkpoint's value-head MLP; best-AUC
                head is written into a copy of the checkpoint.
  --arm linear  Q1 diagnostic: fresh Linear(d_model -> atoms) on
                the same cache; REPORT ONLY, nothing saved.
                linear ~ full  =>  the trunk is the cap.

Usage:
    python tools/value_head_fit.py --checkpoint-in CKPT
        --checkpoint-out OUT.pt [--arm full|linear]
        [--dataset-dir replays_dataset_imitation]
        [--cache-dir /workspace/a3/cache] [--epochs 8] ...
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

log = logging.getLogger("value_head_fit")


def build_cache(policy, rows, dataset_dir: Path, stride: int,
                cap: int, seed: int, loader_jobs: int, batch: int,
                device) -> dict:
    """Load games (parallel), forward once through the frozen trunk,
    return {feats [N,d], z [N], game_idx [N]} on CPU."""
    import torch
    from tools.value_pretrain import _load_worker
    from wesnoth_ai.encoder import encode_raw

    model = policy._trainer.model
    encoder = policy._trainer.encoder
    model.eval()
    feats_buf: List = []

    def _hook(_m, inp, _out):
        feats_buf.append(inp[0].detach().cpu())

    handle = model.value_head.register_forward_hook(_hook)
    tasks = [(str(dataset_dir), r["file"], r["winner"], stride, cap,
              seed * 1_000_003 + i) for i, r in enumerate(rows)]
    feats, zs, gidx = [], [], []
    pend_states, pend_z, pend_g = [], [], []

    def _flush():
        if not pend_states:
            return
        with torch.no_grad():
            for e in pend_states:
                encoder.register_names(e.game_state)
            raws = [encode_raw(e.game_state,
                               type_to_id=encoder.unit_type_to_id,
                               faction_to_id=encoder.faction_to_id,
                               relevant_set=getattr(
                                   encoder, "relevant_set_hexes",
                                   False))
                    for e in pend_states]
            encoded = encoder.encode_from_raw_batch(raws)
            feats_buf.clear()
            model.forward_batch(encoded)
        feats.append(torch.cat(feats_buf, dim=0))
        zs.extend(pend_z)
        gidx.extend(pend_g)
        pend_states.clear()
        pend_z.clear()
        pend_g.clear()

    t0 = time.time()
    n_games = 0

    def _consume(gi, exps):
        nonlocal n_games
        n_games += 1
        for e in exps:
            pend_states.append(e)
            pend_z.append(float(e.z))
            pend_g.append(gi)
        if len(pend_states) >= batch:
            _flush()
        if n_games % 1000 == 0:
            log.info(f"  cache: {n_games}/{len(rows)} games, "
                     f"{len(zs) + len(pend_z)} states, "
                     f"{time.time() - t0:.0f}s")

    if loader_jobs <= 1:
        for gi, t in enumerate(tasks):
            _consume(gi, _load_worker(t))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(loader_jobs) as pool:
            # imap (ordered) so game_idx aligns with rows order.
            for gi, exps in enumerate(pool.imap(_load_worker, tasks,
                                                chunksize=8)):
                _consume(gi, exps)
    _flush()
    handle.remove()
    if not feats:
        raise SystemExit("cache build produced no states")
    return {"feats": torch.cat(feats, dim=0),
            "z": torch.tensor(zs, dtype=torch.float32),
            "game_idx": torch.tensor(gidx, dtype=torch.int64)}


def eval_head(head, feats, z, atoms, device, batch=4096) -> dict:
    """ce / value_auc / pred_entropy / floor on cached features --
    the same math as trainer.eval_value_metrics, head-only."""
    import torch
    from wesnoth_ai.trainer import _project_returns_to_atoms
    head.eval()
    evs, ces, ent = [], [], 0.0
    with torch.no_grad():
        for lo in range(0, len(z), batch):
            f = feats[lo:lo + batch].to(device)
            zt = z[lo:lo + batch].to(device)
            logits = head(f)
            logp = torch.log_softmax(logits, dim=-1)
            ces.append(-(_project_returns_to_atoms(zt, atoms)
                         * logp).sum(dim=-1).cpu())
            ent += float(-(logp.exp() * logp).sum().item())
            evs.append((logp.exp() * atoms).sum(dim=-1).cpu())
        marginal = _project_returns_to_atoms(
            z.to(device), atoms).mean(dim=0)
        floor = float(-(marginal
                        * marginal.clamp_min(1e-9).log()).sum().item())
    ce_all = torch.cat(ces)
    ev = torch.cat(evs)
    pos, neg = ev[z > 0], ev[z < 0]
    if len(pos) and len(neg):
        gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
        eq = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
        auc = float((gt + 0.5 * eq).item()) / (len(pos) * len(neg))
    else:
        auc = float("nan")
    return {"ce": float(ce_all.mean().item()), "value_auc": auc,
            "pred_entropy": ent / len(z), "floor": floor}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-in", type=Path, required=True)
    ap.add_argument("--checkpoint-out", type=Path, required=True)
    ap.add_argument("--arm", choices=("full", "linear"),
                    default="full")
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset_imitation"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("training/value_feature_cache"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--cache-states-per-game", type=int, default=16)
    ap.add_argument("--train-states-per-game", type=int, default=8)
    ap.add_argument("--holdout-games", type=int, default=300)
    ap.add_argument("--probe-states", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--loader-jobs", type=int, default=8)
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cpu", "cuda"))
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    import json
    import torch
    import torch.nn as nn
    from wesnoth_ai.trainer import _categorical_value_loss
    from wesnoth_ai.transformer_policy import TransformerPolicy

    if args.device == "cpu":
        dev = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but unavailable")
        dev = torch.device("cuda")
    else:
        dev = (torch.device("cuda") if torch.cuda.is_available()
               else torch.device("cpu"))

    raw = torch.load(args.checkpoint_in, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    step = int(raw.get("decision_step", 0))
    policy = TransformerPolicy(
        device=(dev if dev.type == "cuda" else None),
        d_model=a["d_model"], num_layers=a["num_layers"],
        num_heads=a["num_heads"], d_ff=a["d_ff"],
        aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")))
    policy.load_checkpoint(args.checkpoint_in)
    model = policy._trainer.model
    atoms = model._value_atoms.to(dev)

    # ---- split (identical rule to value_pretrain) ------------------
    index = args.dataset_dir / "value_corpus_index.jsonl"
    rows = [json.loads(ln) for ln in index.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit_games:
        rows = rows[:args.limit_games]
    holdout_rows = rows[:args.holdout_games]
    train_rows = rows[args.holdout_games:]
    log.info(f"{len(train_rows)} train games, {len(holdout_rows)} "
             f"held-out games; device={dev}")

    # ---- caches (built once per trunk; keyed by trunk+sampling) ----
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    key = (f"{args.checkpoint_in.stem}_{step}_s{args.seed}"
           f"_st{args.stride}_c{args.cache_states_per_game}"
           f"_g{len(rows)}")
    tr_path = args.cache_dir / f"{key}_train.pt"
    pr_path = args.cache_dir / f"{key}_probe.pt"
    if tr_path.exists() and pr_path.exists():
        log.info(f"cache hit: {tr_path.name}")
        train_c = torch.load(tr_path, weights_only=False)
        probe_c = torch.load(pr_path, weights_only=False)
    else:
        log.info("building probe cache...")
        probe_c = build_cache(policy, holdout_rows, args.dataset_dir,
                              max(args.stride, 16),
                              args.train_states_per_game, args.seed,
                              args.loader_jobs, args.batch, dev)
        n = min(len(probe_c["z"]), args.probe_states)
        probe_c = {k: v[:n] for k, v in probe_c.items()}
        log.info(f"building train cache ({len(train_rows)} games)...")
        train_c = build_cache(policy, train_rows, args.dataset_dir,
                              args.stride,
                              args.cache_states_per_game, args.seed,
                              args.loader_jobs, args.batch, dev)
        torch.save(train_c, tr_path)
        torch.save(probe_c, pr_path)
        log.info(f"cached: {len(train_c['z'])} train / "
                 f"{len(probe_c['z'])} probe states -> {tr_path.name}")

    # ---- head ------------------------------------------------------
    import copy
    if args.arm == "linear":
        torch.manual_seed(args.seed)
        head = nn.Linear(a["d_model"], len(atoms)).to(dev)
        log.info(f"LINEAR-PROBE arm: Linear({a['d_model']} -> "
                 f"{len(atoms)}); nothing will be saved")
    else:
        head = copy.deepcopy(model.value_head).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    pf, pz = probe_c["feats"], probe_c["z"]
    m0 = eval_head(head, pf, pz, atoms, dev)
    log.info(f"BEFORE: ce={m0['ce']:.4f} "
             f"value_auc={m0['value_auc']:.4f} "
             f"floor={m0['floor']:.4f} (n={len(pz)})")

    # Group cached state indices per game for the per-epoch draw.
    by_game: dict = {}
    for i, g in enumerate(train_c["game_idx"].tolist()):
        by_game.setdefault(g, []).append(i)
    tf, tz = train_c["feats"], train_c["z"]
    best_auc = m0["value_auc"]
    best_state = None
    for epoch in range(args.epochs):
        erng = random.Random(args.seed * 977 + epoch)
        idx = []
        for g, lst in by_game.items():
            take = (erng.sample(lst, args.train_states_per_game)
                    if len(lst) > args.train_states_per_game else lst)
            idx.extend(take)
        erng.shuffle(idx)
        head.train()
        t0 = time.time()
        tot = n_b = 0.0
        for lo in range(0, len(idx), args.batch):
            sel = idx[lo:lo + args.batch]
            f = tf[sel].to(dev)
            zt = tz[sel].to(dev)
            # _categorical_value_loss returns a SUM over the batch
            # (its trainer call sites divide by total game weight);
            # normalize here or the effective lr scales with batch.
            loss = _categorical_value_loss(head(f), zt,
                                           atoms) / len(sel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n_b += 1
        m = eval_head(head, pf, pz, atoms, dev)
        log.info(f"epoch {epoch}: {len(idx)} states in "
                 f"{time.time() - t0:.0f}s train_v={tot / n_b:.4f} | "
                 f"holdout ce={m['ce']:.4f} "
                 f"value_auc={m['value_auc']:.4f}")
        import math as _math
        if not _math.isnan(m["value_auc"]) and (
                best_state is None or _math.isnan(best_auc)
                or m["value_auc"] > best_auc):
            best_auc = m["value_auc"]
            best_state = copy.deepcopy(head.state_dict())

    if args.arm == "full" and best_state is not None:
        # Freeze receipt: only value_head tensors change.
        import hashlib
        def _sig():
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                if not name.startswith("value_head"):
                    h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        sig0 = _sig()
        model.value_head.load_state_dict(best_state)
        if _sig() != sig0:
            log.error("FREEZE VIOLATION: non-head params changed")
            return 1
        policy.save_checkpoint(args.checkpoint_out)
        log.info(f"saved best-AUC head -> {args.checkpoint_out}")
    log.info(f"done. best holdout value_auc={best_auc:.4f} "
             f"(started {m0['value_auc']:.4f})"
             + ("  [linear arm: nothing saved]"
                if args.arm == "linear" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
