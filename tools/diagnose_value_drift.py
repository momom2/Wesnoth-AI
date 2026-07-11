"""Value-drift attribution: WHICH loss stream erodes the value head?

From one checkpoint, runs K isolated gradient-step segments -- one
per loss stream -- and measures the damage each does to the head's
human-corpus discrimination (CE + AUC on a pre-encoded anchor
sample). Converts "trunk drift" into a per-stream table.

Arms (each restarts from the SAME checkpoint):
  policy-only    visit-count distillation, value/aux/ml coefs = 0
  value-only     decisive-state value loss (visit counts stripped ->
                 policy term is zero), aux/ml coefs = 0
  aux-only       visit counts stripped, value/ml = 0, aux on
  ml-only        visit counts stripped, value/aux = 0, moves-left on
  all-on         production mix (reference)
  anchor-only    step_value_from_raw on human batches (control: the
                 rehearsal itself should IMPROVE the probe)

Training data: self-play experiences from a spool/pickle file
(--experiences), e.g. captured by ladder_anatomy or a worker run.

Usage (on the training box):
    python tools/diagnose_value_drift.py --checkpoint CKPT.pt
        --experiences EXPS.pkl --anchor /workspace/human_anchor.pkl
        [--steps 48] [--minibatch 128] [--probe-states 768]
"""

from __future__ import annotations

import argparse
import copy
import logging
import pickle
import random
import statistics as st
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("diagnose_value_drift")


def _auc(pairs, n=20000):
    win = [v for v, z in pairs if z > 0]
    loss = [v for v, z in pairs if z < 0]
    if not win or not loss:
        return float("nan")
    r = random.Random(7)
    h = 0.0
    for _ in range(n):
        a, b = r.choice(win), r.choice(loss)
        h += 1.0 if a > b else (0.5 if a == b else 0.0)
    return h / n


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--experiences", type=Path, required=True,
                    help="Pickle of MCTSExperiences (self-play).")
    ap.add_argument("--anchor", type=Path, required=True,
                    help="Pre-encoded human anchor pickle.")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--probe-states", type=int, default=768)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from transformer_policy import TransformerPolicy

    with args.experiences.open("rb") as f:
        pool = pickle.load(f)
    with args.anchor.open("rb") as f:
        anchor = pickle.load(f)
    rng = random.Random(args.seed)
    probe = rng.sample(anchor, min(args.probe_states, len(anchor)))
    probe_raws = [t[0] for t in probe]
    probe_zs = [t[1] for t in probe]
    log.info(f"{len(pool)} self-play experiences, "
             f"{len(probe)} probe states")

    raw = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    def fresh_policy():
        p = TransformerPolicy(
            device=device, d_model=a["d_model"],
            num_layers=a["num_layers"], num_heads=a["num_heads"],
            d_ff=a["d_ff"], aux_score=bool(raw.get("aux_score")),
            moves_left=bool(raw.get("moves_left")))
        p.load_checkpoint(args.checkpoint)
        # production-run loss shape
        p._trainer.config.value_coef = 1.0
        p._trainer.config.value_label_smoothing = 0.02
        p._trainer.config.draw_value_weight = 0.0
        return p

    def probe_metrics(p):
        m = p._trainer.eval_value_metrics_from_raw(probe_raws, probe_zs)
        vals = p._trainer.values_from_raw(probe_raws)
        return {"ce": m["ce"],
                "auc": _auc(list(zip(vals, probe_zs)))}

    def strip_visits(exps):
        out = []
        for e in exps:
            c = copy.copy(e)
            c.visit_counts = []
            out.append(c)
        return out

    ARMS = {
        "policy-only": dict(strip=False, vc=0.0, ac=0.0, mc=0.0),
        "value-only":  dict(strip=True,  vc=1.0, ac=0.0, mc=0.0),
        "aux-only":    dict(strip=True,  vc=0.0, ac=0.15, mc=0.0),
        "ml-only":     dict(strip=True,  vc=0.0, ac=0.0, mc=0.1),
        "all-on":      dict(strip=False, vc=1.0, ac=0.15, mc=0.1),
        "anchor-only": dict(anchor=True),
    }

    base_p = fresh_policy()
    m0 = probe_metrics(base_p)
    log.info(f"BASELINE: ce={m0['ce']:.4f} auc={m0['auc']:.4f}")
    del base_p
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, cfg in ARMS.items():
        p = fresh_policy()
        arm_rng = random.Random(args.seed + 1)
        if cfg.get("anchor"):
            for _ in range(args.steps):
                sample = arm_rng.sample(
                    anchor, min(args.minibatch, len(anchor)))
                p._trainer.step_value_from_raw(
                    [t[0] for t in sample], [t[1] for t in sample],
                    [t[2] for t in sample])
        else:
            p._trainer.config.value_coef = cfg["vc"]
            p._trainer.config.aux_coef = cfg["ac"]
            p._trainer.config.moves_left_coef = cfg["mc"]
            for _ in range(args.steps):
                sample = arm_rng.sample(
                    pool, min(args.minibatch, len(pool)))
                if cfg["strip"]:
                    sample = strip_visits(sample)
                p._trainer.step_mcts(sample)
        m = probe_metrics(p)
        log.info(f"{name:12} ce={m['ce']:.4f} ({m['ce'] - m0['ce']:+.4f})"
                 f"  auc={m['auc']:.4f} ({m['auc'] - m0['auc']:+.4f})"
                 f"  per-step dAUC={1000 * (m['auc'] - m0['auc']) / args.steps:+.3f}m")
        del p
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
