#!/usr/bin/env python3
"""Model cost study (docs/plan_20260904.md step 1.4): token budget and
forward cost of one checkpoint on the benchmark states under three
encodings, without any model change:

  full     the production full-board encoding
  relset   the relevant-set hex subset (encoder.encode_raw(relevant_set=True))
  hexN     the full encoding with the hex stream cut to its first N tokens
           (synthetic: the cost curve against the hex count alone)

Per row: token counts per state (hex / unit / recruit / total: mean, p50,
p90, max), padding ratio of the batches, median ms per batch of B leaves
through model.forward_batch (bf16 on CUDA, batches cut from the
length-sorted list as tools/bench_pipeline.seam_costs does), the leaves
per second that median implies (the mean-based rate, total leaves over
total time, is `leaves_per_s_mean` in the JSON: the full-board row's
longest batches lift it apart), the analytic GFLOP per leaf at the mean
length (linears + attention, docs/gpu_forward_design_20260904.md
section 0) and the FLOP ceiling at the 4090's dense bf16 peak.

The relset row runs the checkpoint's own weights on the subset: its
timing is valid, its outputs are not (the seed was trained full-board).

Usage (box):
  python tools/bench_model_cost.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \\
      --device cuda --hex-tokens 300 600 900 --out bench_model_cost.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.bench_pipeline import DEFAULT_DATASET, DEFAULT_MANIFEST, load_states, n_tokens

log = logging.getLogger("bench_model_cost")

# RTX 4090 dense bf16 tensor-core peak (docs/gpu_forward_design_20260904.md, section 0).
PEAK_TFLOPS_4090 = 165.0


def quantile(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def trunk_arch(model) -> Dict[str, int]:
    """(d_model, layers, d_ff) of a WesnothModel, read from its modules."""
    layer0 = model.encoder.layers[0]
    return {"d_model": model.d_model, "layers": len(model.encoder.layers),
            "d_ff": layer0.linear1.out_features}


def gflop_per_leaf(tokens: float, arch: Dict[str, int]) -> float:
    """Linears (QKV, out, FFN: 2 FLOP per weight per token) plus
    attention (QK^T and AV: 4 L^2 d per layer)."""
    d, n, dff = arch["d_model"], arch["layers"], arch["d_ff"]
    return (n * 2 * (4 * d * d + 2 * d * dff) * tokens + n * 4 * d * tokens * tokens) / 1e9


def encode_states(policy, states, relevant_set: bool) -> list:
    import torch
    from wesnoth_ai.encoder import encode_raw
    enc = policy._inference_encoder
    out = []
    with torch.no_grad():
        for gs, _ in states:
            enc.register_names(gs)
            raw = encode_raw(gs, type_to_id=enc.unit_type_to_id,
                             faction_to_id=enc.faction_to_id, relevant_set=relevant_set,
                             fog_hides_enemy_villages=bool(getattr(enc, "fog_hides_enemy_villages", False)),
                             terrain_multi_hot=bool(getattr(enc, "terrain_multi_hot", False)))
            out.append(enc.encode_from_raw(raw))
    return out


def truncate_hexes(encoded, n_hex: int):
    """The same state with only its first n_hex hex tokens (row-major
    slot order, so the kept hexes are the top rows of the map)."""
    keep = encoded.hex_positions[:n_hex]
    return dataclasses.replace(
        encoded, hex_tokens=encoded.hex_tokens[:, :n_hex], hex_positions=keep,
        pos_to_hex={(p.x, p.y): i for i, p in enumerate(keep)})


def token_stats(encoded_list) -> Dict[str, float]:
    streams = {"hex": [e.hex_tokens.size(1) for e in encoded_list],
               "unit": [e.unit_tokens.size(1) for e in encoded_list],
               "recruit": [e.recruit_tokens.size(1) for e in encoded_list],
               "total": [n_tokens(e) for e in encoded_list]}
    out: Dict[str, float] = {}
    for name, v in streams.items():
        out[f"{name}_mean"] = statistics.fmean(v)
        out[f"{name}_p50"] = statistics.median(v)
        out[f"{name}_p90"] = quantile(v, 0.9)
        out[f"{name}_max"] = max(v)
    return out


def time_batches(model, encoded_list, batch: int, sync) -> Dict[str, float]:
    """Median and mean ms per forward_batch call over consecutive batches
    of the length-sorted states (two warm-up calls first)."""
    import torch
    ordered = sorted(encoded_list, key=n_tokens)
    batches = [ordered[i:i + batch] for i in range(0, len(ordered) - batch + 1, batch)]
    pad = statistics.fmean(batch * n_tokens(b[-1]) / sum(n_tokens(e) for e in b)
                           for b in batches)
    with torch.no_grad():
        for b in batches[:2]:
            model.forward_batch(b)
            sync()
        ms: List[float] = []
        for b in batches:
            t0 = time.perf_counter()
            model.forward_batch(b)
            sync()
            ms.append((time.perf_counter() - t0) * 1000.0)
    return {"n_batches": len(batches), "pad_ratio": pad,
            "ms_median": statistics.median(ms), "ms_mean": statistics.fmean(ms)}


def measure_row(name: str, model, encoded_list, batch: int, sync, arch) -> dict:
    row = {"row": name, "batch": batch, **token_stats(encoded_list)}
    row.update(time_batches(model, encoded_list, batch, sync))
    # The rate the table's own ms column implies (2026-09-05 review:
    # derived from the mean next to a median, the two columns of the
    # full-board row disagreed by 9%).
    row["leaves_per_s"] = 1000.0 * batch / row["ms_median"]
    row["leaves_per_s_mean"] = 1000.0 * batch / row["ms_mean"]
    row["gflop_per_leaf"] = gflop_per_leaf(row["total_mean"], arch)
    row["ceiling_leaves_per_s"] = 1000.0 * PEAK_TFLOPS_4090 / row["gflop_per_leaf"]
    log.info("%s: tokens %.0f, %.1f ms/batch, %.0f leaves/s", name, row["total_mean"],
             row["ms_median"], row["leaves_per_s"])
    return row


def markdown_table(rows: List[dict]) -> str:
    lines = ["| row | hex mean | hex p90 | tokens mean | p50 | p90 | max | pad | "
             "ms/batch (median) | leaves/s | GFLOP/leaf | ceiling @165T |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['row']} | {r['hex_mean']:.0f} | {r['hex_p90']:.0f} | {r['total_mean']:.0f} "
            f"| {r['total_p50']:.0f} | {r['total_p90']:.0f} | {r['total_max']:.0f} "
            f"| {r['pad_ratio']:.2f} | {r['ms_median']:.1f} | {r['leaves_per_s']:.0f} "
            f"| {r['gflop_per_leaf']:.1f} | {r['ceiling_leaves_per_s']:.0f} |")
    return "\n".join(lines)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--states-json", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--states", type=int, default=None, help="Use only the first N states.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hex-tokens", type=int, nargs="*", default=[300, 600, 900],
                    help="Synthetic rows: hex stream cut to N tokens.")
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import torch
    from tools.eval_players import _load_policy
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible")
    bf16 = cuda if args.infer_bf16 is None else args.infer_bf16
    device = torch.device("cuda" if cuda else "cpu")
    # forward_batch runs eager whatever --infer-compile says
    # (docs/gpu_forward_design_20260904.md section 3.1), so compile is off.
    policy = _load_policy(args.checkpoint, device, label="cost", infer_bf16=bf16,
                          infer_compile=False)
    model = policy._inference_model
    arch = trunk_arch(model)
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    states = load_states(args.states_json, args.dataset, args.states)
    log.info("loaded %d states; trunk %s", len(states), arch)

    full = encode_states(policy, states, relevant_set=False)
    rows = [measure_row("full", model, full, args.batch, sync, arch),
            measure_row("relset", model, encode_states(policy, states, relevant_set=True),
                        args.batch, sync, arch)]
    for n_hex in args.hex_tokens:
        cut = [truncate_hexes(e, n_hex) for e in full]
        rows.append(measure_row(f"hex{n_hex}", model, cut, args.batch, sync, arch))

    report = markdown_table(rows)
    print(report)
    if args.out:
        result = {"checkpoint": str(args.checkpoint), "device": args.device,
                  "infer_bf16": bf16, "torch": torch.__version__, "arch": arch,
                  "n_states": len(states), "rows": rows}
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
