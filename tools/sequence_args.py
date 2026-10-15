"""The sequence trainer's command line (tools/sequence_train.py, whose
docstring describes the flags' effects)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tools.sequence_setup import ARCH, DEFAULTS


def budget_value(text: str):
    """--activation-budget-gb: a size in GiB, or `auto`."""
    if text == "auto":
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a size in GiB or auto") from None


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
    ap.add_argument("--activation-budget-gb", type=budget_value, default=0.0,
                    help="keep the activations of as many of a window's last time steps as fit in this many GiB "
                         "instead of recomputing them in the backward pass (0: every step recomputed); auto: as "
                         "many as the CUDA device leaves, measured in the run (tools/sequence_memory_plan.py)")
    ap.add_argument("--memory-margin-gb", type=float, default=1.0,
                    help="with --activation-budget-gb auto: the device memory left free besides the measured peak")
    ap.add_argument("--activation-bytes-per-token", type=float, default=None,
                    help="the activation bytes a trunk token keeps, measured (default: the estimate of "
                         "tools/sequence_timing.py for the trunk and architecture)")
    ap.add_argument("--window-batched", action="store_true",
                    help="embed a window's positions at once and run the heads and losses once over its time steps' "
                         "trunk outputs; per time step only the packed trunk and the memory's write run "
                         "(tools/sequence_batched.py): other rounding than the stepped forward, so a network it "
                         "trains needs its own acceptance; with --packed-trunk")
    ap.add_argument("--trunk-bf16", action="store_true",
                    help="hold the packed trunk's residual stream and LayerNorms in bfloat16 (on the CPU too) instead "
                         "of float32 (wesnoth_ai/trunk_modes.py): other arithmetic, its own acceptance; with "
                         "--packed-trunk")
    ap.add_argument("--fused-relu", action="store_true",
                    help="run the packed trunk's first feed-forward linear, its bias and the ReLU in one cuBLASLt "
                         "call (wesnoth_ai/trunk_modes.py): the same arithmetic on the CPU, possibly another GEMM "
                         "algorithm on CUDA; with --packed-trunk, not with --compiled-trunk")
    ap.add_argument("--compiled-trunk", nargs="?", const="inductor", default=None, metavar="BACKEND",
                    help="compile each packed trunk layer's out-projection, norms and feed-forward with "
                         "torch.compile (BACKEND, default inductor), the token count dynamic "
                         "(wesnoth_ai/trunk_modes.py): other rounding, its own acceptance; with --packed-trunk")
    ap.add_argument("--fp8-trunk", action="store_true",
                    help="run the packed trunk's linear layers' matrix products on float8 operands with per-tensor "
                         "scales, forward and backward (wesnoth_ai/fp8_linear.py): other arithmetic, its own "
                         "acceptance; with --packed-trunk, not with --fused-relu; on CUDA of compute capability 8.9 "
                         "or more, emulated on the CPU")
    ap.add_argument("--train-every", type=int, default=1,
                    help="train one window in this many; the others run the forward only, carrying the memory and "
                         "counting toward the pass's positions (1: every window trains)")
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


def check_args(args) -> None:
    """SystemExit for arguments the trainer cannot run together."""
    if args.resume and args.init_from is not None:
        raise SystemExit("--resume continues --out's own pass; --init-from starts a new one: not both")
    if args.decay_from is not None and not 0.0 <= args.decay_from < 1.0:
        raise SystemExit(f"--decay-from {args.decay_from}: a fraction of the pass, from 0 up to 1")
    if args.pass_positions is not None and args.pass_positions <= 0:
        raise SystemExit(f"--pass-positions {args.pass_positions}: a pass of at least one position")
    if args.initial_areas is not None and args.init_from is None:
        raise SystemExit("--initial-areas gives the areas of the checkpoint --init-from starts from")
    if args.init_other_records and args.init_from is None:
        raise SystemExit("--init-other-records qualifies --init-from")
    if args.own_streams is not None and args.own_sequences is None:
        raise SystemExit("--own-streams gives the streams of --own-sequences")
    if args.anneal_rule is not None and (args.law_points is None or not args.law_points.is_file()):
        raise SystemExit(f"--anneal-rule fits the earlier passes' probes: --law-points {args.law_points} "
                         f"is not a file")
    if args.anneal_rule is not None and args.law_key not in {f"k{k}" for k in args.probe_ks}:
        raise SystemExit(f"--law-key {args.law_key}: the probe reads {', '.join(f'k{k}' for k in args.probe_ks)}")
    if args.packed_trunk and args.fp32 and torch.device(args.device).type == "cuda":
        raise SystemExit("--packed-trunk runs flash varlen attention, which takes bfloat16: not with --fp32")
    if args.window_batched and not args.packed_trunk:
        raise SystemExit("--window-batched runs the packed trunk's layout: with --packed-trunk")
    variants = [flag for flag, on in (("--trunk-bf16", args.trunk_bf16), ("--fused-relu", args.fused_relu),
                                      ("--compiled-trunk", args.compiled_trunk is not None),
                                      ("--fp8-trunk", args.fp8_trunk)) if on]
    if variants and not args.packed_trunk:
        raise SystemExit(f"{' '.join(variants)}: variants of the packed trunk, with --packed-trunk")
    if args.fused_relu and args.compiled_trunk is not None:
        raise SystemExit("--fused-relu is an eager call; --compiled-trunk fuses the ReLU itself: not both")
    if args.fused_relu and args.fp8_trunk:
        raise SystemExit("--fused-relu's epilogue belongs to a bfloat16 product, which --fp8-trunk runs in fp8: "
                         "not both")
    if args.activation_budget_gb != "auto" and args.activation_budget_gb < 0:
        raise SystemExit(f"--activation-budget-gb {args.activation_budget_gb}: a size, 0 or more, or auto")
    if args.memory_margin_gb < 0:
        raise SystemExit(f"--memory-margin-gb {args.memory_margin_gb}: a size, 0 or more")
    if args.train_every < 1:
        raise SystemExit(f"--train-every {args.train_every}: one window in 1 or more")
