"""The sequence trainer's command line (tools/sequence_train.py, whose
docstring describes the flags' effects)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tools.sequence_setup import ARCH, DEFAULTS


def parse_args(argv=None):
    """The trainer's command line (tools/sequence_train.py)."""
    ap = argparse.ArgumentParser(description="Train the parity-memory recipe on pre-encoded game "
                                             "sequences (tools/sequence_train.py).")
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="the corpus directory with manifest.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from --out")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start a new pass from this checkpoint's weights and optimizer state")
    ap.add_argument("--decay-from", type=float, default=None,
                    help="hold the learning rate until this fraction of the pass, then lower it linearly "
                         "to 0 at its end; the checkpoint there is kept as <out>.stable.pt")
    ap.add_argument("--pass-positions", type=int, default=None,
                    help="end the pass after this many positions; the rate's schedule runs over them")
    ap.add_argument("--anneal-rule", type=float, default=None,
                    help="after every probe, stop when one more epoch at the peak rate would lower the "
                         "fitted holdout loss by this much or less (tools/lr_law.py; exit 6, or 7 to look)")
    ap.add_argument("--law-points", type=Path, default=None,
                    help="the earlier passes' probe points the anneal rule fits with this pass's")
    ap.add_argument("--law-key", default="k64", help="the probe's slot reading the anneal rule fits")
    ap.add_argument("--initial-areas", default=None,
                    help="with --init-from a checkpoint that does not carry them: its rate history's areas "
                         "as JSON (tools/lr_law.py replay --areas-at)")
    ap.add_argument("--own-sequences", type=Path, default=None,
                    help="the memory player's own game-sides (tools/own_targets.py), trained beside the corpus's")
    ap.add_argument("--own-streams", type=int, default=None,
                    help="with --own-sequences: the streams own game-sides take (default: half of --streams)")
    ap.add_argument("--own-policy-weight", type=float, default=1.0,
                    help="the weight of an own position's soft-target policy loss")
    ap.add_argument("--init-other-records", action="store_true",
                    help="with --init-from: a checkpoint whose records another core phase or corpus version built "
                         "(its vocabularies, encoding and observation epoch must be this run's)")
    ap.add_argument("--probe-at-start", action="store_true", help="probe the starting weights before training")
    ap.add_argument("--imitation-config", type=Path, default=Path("configs/imitation.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="no bfloat16 autocast on CUDA")
    ap.add_argument("--packed-trunk", action="store_true",
                    help="train on the packed trunk: the real tokens only, flash varlen attention on CUDA under "
                         "bfloat16; the packed trunk has no dropout (the padded trunk's layers drop at 1e-4), so "
                         "it trains another network than the padded trunk does")
    ap.add_argument("--activation-budget-gb", type=float, default=0.0,
                    help="keep the activations of as many of a window's last time steps as fit in this many GiB "
                         "instead of recomputing them in the backward pass (0: every step recomputed)")
    ap.add_argument("--activation-bytes-per-token", type=float, default=None,
                    help="the activation bytes a trunk token keeps, measured (default: the estimate of "
                         "tools/sequence_timing.py for the trunk and architecture)")
    ap.add_argument("--window-batched", action="store_true",
                    help="embed a window's positions at once and run the heads and losses once over its time steps' "
                         "trunk outputs; per time step only the packed trunk and the memory's write run "
                         "(tools/sequence_batched.py): other rounding than the stepped forward, so a network it "
                         "trains needs its own acceptance; with --packed-trunk")
    ap.add_argument("--columns", action="store_true",
                    help="read the records' columns (tools/columnize_sequences.py writes them beside the pickles): "
                         "the same batches, without unpickling a Python object per position")
    ap.add_argument("--max-positions", type=int, default=None, help="stop after this many (a smoke run)")
    ap.add_argument("--log-seconds", type=float, default=60.0)
    for key in ("streams", "window", "warmup_steps", "value_states_per_game", "seed", "probe_every",
                "barrier_positions", "checkpoint_every", "signal_every"):
        ap.add_argument("--" + key.replace("_", "-"), type=int, default=DEFAULTS[key])
    for key, value in ARCH.items():
        ap.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    for key in ("lr", "weight_decay", "grad_clip", "belief_weight"):
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=DEFAULTS[key])
    ap.add_argument("--probe-ks", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=DEFAULTS["probe_ks"])
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)
