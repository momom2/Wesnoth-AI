"""Standalone OFFLINE validation of the detector training signal.

Runs the Tier-1 advisor (docs/detector_training_signal.md) over recorded
games with a REAL value net (a checkpoint) and reports the delta_v
distribution + fire-rates -- BEFORE wiring anything into self-play. It
answers the questions that decide whether the signal is worth wiring:

  - how often does a Tier-1 certificate fire per side-turn?
  - how often is a finding JUDGEABLE (reconstruction didn't bail)? -- this
    is the signal-coverage number (a tracked risk in BACKLOG).
  - what does the model's OWN value net think of the proposed reorder?
    delta_v > 0 = it agrees (would learn it); delta_v <= 0 = it would
    IGNORE (the deliberate-play case the whole design is built around).

Usage:
  python -m tools.validate_advisor --checkpoint training/checkpoints/tier_a_campaign_final.pt
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.swap_detector import load_side_turns                  # noqa: E402
from tools.detector_advisor import advice_signals, model_value_fn  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy       # noqa: E402


def load_value_fn(ckpt_path: Path, device: str = "cpu"):
    """Build a TransformerPolicy at the checkpoint's saved arch, load its
    weights, and return a value_fn over its (eval-mode) net."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = raw.get("arch", {}) or {}
    kw = {k: int(arch[k]) for k in ("d_model", "num_layers", "num_heads", "d_ff")
          if k in arch}
    policy = TransformerPolicy(
        device=device,
        aux_score=bool(raw.get("aux_score", False)),
        moves_left=bool(raw.get("moves_left", False)),
        **kw)
    policy.load_checkpoint(Path(ckpt_path))
    policy._model.eval()
    return model_value_fn(policy._model, policy._encoder), kw


def _pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline advisor validation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bundles",
                    default="training/validate_exports/hf_bundles")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"loading value net from {args.checkpoint} ...", flush=True)
    value_fn, arch = load_value_fn(Path(args.checkpoint))
    print(f"  arch={arch or 'defaults'}", flush=True)

    pat = args.bundles
    if Path(pat).is_dir():
        pat = str(Path(pat) / "*.tar")
    paths = sorted(glob.glob(pat))
    if args.limit:
        paths = paths[:args.limit]
    print(f"scanning {len(paths)} bundle(s) ...", flush=True)

    sigs = []
    n_turns = n_games = 0
    for bp in paths:
        saw = False
        for st in load_side_turns(Path(bp)):
            saw = True
            n_turns += 1
            new = advice_signals(st, value_fn)
            for s in new:                       # print inline (sleep-robust)
                dv = "None" if s.delta_v is None else f"{s.delta_v:+.4f}"
                print(f"    [{s.motif}] g={s.game_id} t{s.turn} s{s.side} "
                      f"{s.attacker_pos}->{s.defender_pos} dv={dv}", flush=True)
            sigs.extend(new)
        if saw:
            n_games += 1
        print(f"  {Path(bp).name}: turns={n_turns} findings={len(sigs)}",
              flush=True)

    total = len(sigs)
    judged = [s for s in sigs if s.delta_v is not None]
    dvs = [s.delta_v for s in judged]
    pos = [d for d in dvs if d > 1e-6]
    neg = [d for d in dvs if d < -1e-6]

    print("\n=== advisor offline validation ===")
    print(f"games:                 {n_games}")
    print(f"side-turns:            {n_turns}")
    print(f"Tier-1 findings:       {total}  "
          f"({total / n_turns:.3f}/side-turn)" if n_turns else "")
    print(f"judgeable (dv!=None):  {len(judged)}  ({_pct(len(judged), total)}"
          f" -- reconstruction coverage)")
    if dvs:
        print(f"delta_v  mean={statistics.mean(dvs):+.4f}  "
              f"median={statistics.median(dvs):+.4f}  "
              f"min={min(dvs):+.4f}  max={max(dvs):+.4f}")
        print(f"  dv > 0 (net AGREES -> would learn):  {len(pos)}  "
              f"({_pct(len(pos), len(judged))})")
        print(f"  dv < 0 (net IGNORES -> deliberate?): {len(neg)}  "
              f"({_pct(len(neg), len(judged))})")
    by_motif = {}
    for s in sigs:
        by_motif.setdefault(s.motif, []).append(s)
    print("\nby motif:")
    for m, ss in by_motif.items():
        j = [s.delta_v for s in ss if s.delta_v is not None]
        p = len([d for d in j if d > 1e-6])
        print(f"  {m:20s} fires={len(ss):3d} judged={len(j):3d} "
              f"dv>0={p:3d} mean_dv={statistics.mean(j):+.4f}"
              if j else f"  {m:20s} fires={len(ss):3d} judged=0")

    print("\nexamples (top |delta_v|):")
    for s in sorted(judged, key=lambda s: -abs(s.delta_v))[:12]:
        print(f"  g={s.game_id} t{s.turn} s{s.side} [{s.motif}] "
              f"{s.attacker_pos}->{s.defender_pos} dv={s.delta_v:+.4f} "
              f"{ {k: v for k, v in s.gain_vector.items() if v != '='} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
