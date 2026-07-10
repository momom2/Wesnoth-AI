"""Value-head discrimination probe: phase-bucketed AUC on human games.

The reusable form of the 2026-07-08/09 diagnosis probes (previously
re-typed as heredocs). For each checkpoint, samples states from human
corpus games and reports, per game phase (opening turns 1-8 / midgame
9-16 / endgame 17+):

  - AUC: P(random winning-state E[V] > random losing-state E[V]).
    0.5 = blind to the board, 1.0 = perfect. The monotonic
    open<mid<late gradient is the fingerprint of a real evaluator.
  - mean E[V] on winner-to-move vs loser-to-move states.

Reference points (common 300-game holdout, 2026-07-09):
  campaign self-play head : late ~0.50 (confidently wrong)
  frozen-trunk pretrain   : late 0.583
  full-2500 fine-tune     : late 0.818
  allgames fine-tune      : late 0.890

Usage:
    python tools/probe_value_head.py CKPT [CKPT2 ...]
        [--dataset-dir replays_dataset] [--games 150]
        [--skip-games N]   # skip the first N index rows after the
                           # seed-0 shuffle (e.g. training subset)
        [--stride 15] [--seed 99]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return cov / (va ** 0.5 * vb ** 0.5)


def auc(win: List[float], loss: List[float], n=20000) -> float:
    if not win or not loss:
        return float("nan")
    r = random.Random(7)
    hits = 0.0
    for _ in range(n):
        a, b = r.choice(win), r.choice(loss)
        hits += 1.0 if a > b else (0.5 if a == b else 0.0)
    return hits / n


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--skip-games", type=int, default=0)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--material-corr", action="store_true",
                    help="Also report per-phase Pearson r of E[V] "
                         "with the state's CURRENT material margin vs "
                         "with the actual outcome z. A draw-tiebreak-"
                         "degenerate head (trained mostly on material-"
                         "z draws) shows r_material >> r_outcome.")
    args = ap.parse_args(argv[1:])

    import torch
    from transformer_policy import TransformerPolicy
    from tools.value_corpus import game_experiences

    rows = [json.loads(l) for l in
            (args.dataset_dir / "value_corpus_index.jsonl")
            .open(encoding="utf-8")]
    # Same seed-0 shuffle every training run uses, so --skip-games can
    # exclude a training subset deterministically.
    random.Random(0).shuffle(rows)
    rows = rows[args.skip_games:]
    random.Random(args.seed).shuffle(rows)
    sample = rows[:args.games]
    print(f"probing {len(sample)} games "
          f"(skip={args.skip_games}, stride={args.stride})")

    def bucket(turn):
        return ("open(1-8)" if turn <= 8
                else "mid(9-16)" if turn <= 16 else "late(17+)")

    for ck in args.checkpoints:
        raw = torch.load(ck, map_location="cpu", weights_only=False)
        a = raw["arch"]
        p = TransformerPolicy(
            device=torch.device("cpu"),
            d_model=a["d_model"], num_layers=a["num_layers"],
            num_heads=a["num_heads"], d_ff=a["d_ff"],
            aux_score=bool(raw.get("aux_score")),
            moves_left=bool(raw.get("moves_left")))
        p.load_checkpoint(ck)
        m, enc = p._model, p._encoder
        m.eval()
        atoms = m._value_atoms
        buckets = {"open(1-8)": ([], []), "mid(9-16)": ([], []),
                   "late(17+)": ([], [])}
        # per-phase (ev, z, material) triples for --material-corr
        tri = {k: [] for k in buckets}
        mat_cfg = None
        if args.material_corr:
            from tools.draw_tiebreak import (DrawTiebreakConfig,
                                             material_margin)
            mat_cfg = DrawTiebreakConfig(cap=0.3)
        with torch.no_grad():
            for r in sample:
                try:
                    exps = game_experiences(
                        args.dataset_dir / r["file"], r["winner"],
                        stride=args.stride, rng=random.Random(3))
                except Exception:               # noqa: BLE001
                    continue
                for e in exps:
                    enc.register_names(e.game_state)
                    out = m(enc.encode(e.game_state))
                    ev = float((torch.softmax(
                        out.value_logits.squeeze(), -1) * atoms).sum())
                    bk = bucket(e.game_state.global_info.turn_number)
                    w, l = buckets[bk]
                    (w if e.z > 0 else l).append(ev)
                    if mat_cfg is not None:
                        side = e.game_state.global_info.current_side
                        mm = material_margin(e.game_state, side,
                                             mat_cfg)
                        tri[bk].append((ev, e.z, mm))
        print(f"== {ck.name} (step="
              f"{raw.get('decision_step', '?')}) ==")
        for name, (w, l) in buckets.items():
            if w and l:
                print(f"  {name:11} n={len(w) + len(l):5}  "
                      f"E[V]win={st.mean(w):+.3f}  "
                      f"E[V]loss={st.mean(l):+.3f}  "
                      f"AUC={auc(w, l):.3f}")
        if mat_cfg is not None:
            for name, rows_ in tri.items():
                if len(rows_) > 10:
                    evs = [t[0] for t in rows_]
                    print(f"  {name:11} corr(E[V], outcome)="
                          f"{pearson(evs, [t[1] for t in rows_]):+.3f}"
                          f"  corr(E[V], material)="
                          f"{pearson(evs, [t[2] for t in rows_]):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
