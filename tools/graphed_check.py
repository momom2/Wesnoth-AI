#!/usr/bin/env python3
"""The graphed serve path against the eager one, on a checkpoint and a GPU.

The CPU tests run the static-shape body without a graph
(tests/test_graphed_serve.py); the capture and the replay need CUDA. This
check serves the same batches of played positions (tools/random_positions.py),
each with a memory state, through the eager server and through the graphed
one (`--graphed`'s caps, `graphed_serve.eval_caps`), and compares per leaf:
the legal actions (the same set, in the same order), the chosen action at
argmax, the priors, the value and the new memory state. bf16 batched
numerics differ with the padded shapes, so the priors and the states agree
to a tolerance and a near tie may flip the argmax; the legal sets may not
differ. It also times each batch size both ways (median host wall per
call, the captures excluded).

    python tools/graphed_check.py --checkpoint own1.pt --device cuda --out graphed_check.json

Exit 0 when every batch was served from a graph and the comparisons hold,
8 when one does not hold (the graphed arms of a benchmark are skipped).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

EXIT_FAIL = 8


def _servers(spec: Path, device: torch.device, max_batch: int):
    """(eager InferenceServer, graphed InferenceServer, GraphedServe, the
    policy) as tools/eval_inference_server.py builds them."""
    from tools.eval_players import _load_policy
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import GraphedServe, eval_caps
    from wesnoth_ai.packed_trunk import check_packed_trunk_supported, flash_varlen_applies
    cuda = device.type == "cuda"
    policy = _load_policy(spec, device, label="check", infer_bf16=cuda, infer_compile=False)
    model, encoder = policy._inference_model, policy._inference_encoder
    if cuda and flash_varlen_applies(device, torch.bfloat16):
        check_packed_trunk_supported(model.encoder)
        model.infer_packed_trunk = True
    gserve = GraphedServe(model, encoder, device, caps=eval_caps(max_batch))
    kw = dict(device=device, output_device=torch.device("cpu"), autocast_bf16=cuda, packed_embed=True)
    return (InferenceServer(model, encoder, **kw), InferenceServer(model, encoder, graphed=gserve, **kw),
            gserve, policy)


def _leaves(policy, n_games: int, slots: int, seed: int):
    """(RawEncoded, PackedMasks, MemoryState) for harvested positions; a
    third of them a game-side's first decision, the rest a random state."""
    from tools.inference_seam import RemoteEncoder
    from tools.random_positions import harvest
    from wesnoth_ai.memory import MemoryState
    enc = policy._inference_encoder
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id,
                         relevant_set=bool(enc.relevant_set_hexes), server_priors=True,
                         fog_hides_enemy_villages=bool(enc.fog_hides_enemy_villages),
                         terrain_multi_hot=bool(enc.terrain_multi_hot),
                         observation_parity=bool(enc.observation_parity),
                         relevant_set_version=int(enc.relevant_set_version))
    rng = np.random.default_rng(seed)
    d = int(policy._inference_model.d_model)
    out = []
    for g in range(n_games):
        for gs in harvest(seed + g, decisions=500, every=20, max_turns=20):
            e = renc.encode(gs)
            if not slots:
                out.append((e._raw, e._masks))
                continue
            state = None if len(out) % 3 == 0 else (0.5 * rng.standard_normal((slots, d))).astype(np.float32)
            out.append((e._raw, e._masks, MemoryState(slots, state)))
    return out


def _compare(ref, got) -> Dict[str, float]:
    """Per batch: legal sets equal, argmax equal, the largest prior, value
    and memory differences."""
    same_legal = same_argmax = 0
    d_prior = d_value = d_memory = 0.0
    for r, g in zip(ref, got):
        a, b = r.legal_compact, g.legal_compact
        legal = all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("actor", "kind", "target", "weapon"))
        same_legal += legal
        if legal and len(a.prior):
            same_argmax += int(np.argmax(a.prior) == np.argmax(b.prior))
            d_prior = max(d_prior, float(np.max(np.abs(a.prior - b.prior))))
        elif legal:
            same_argmax += 1
        d_value = max(d_value, float((r.value - g.value).abs().max()))
        if r.memory is not None:
            d_memory = max(d_memory, float((r.memory - g.memory).abs().max()))
    return {"leaves": len(ref), "same_legal": same_legal, "same_argmax": same_argmax,
            "max_prior_diff": d_prior, "max_value_diff": d_value, "max_memory_diff": d_memory}


def _median_ms(fn, device, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        times.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(times)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=40, help="the server's max batch (the caps)")
    ap.add_argument("--batch-sizes", default="1,4,9,16,20,32,40")
    ap.add_argument("--games", type=int, default=3, help="games the positions are harvested from")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--min-argmax", type=float, default=0.95)
    ap.add_argument("--max-value-diff", type=float, default=0.05)
    ap.add_argument("--max-memory-diff", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv[1:])
    device = torch.device(args.device)
    eager, graphed, gserve, policy = _servers(args.checkpoint, device, args.max_batch)
    slots = int(getattr(policy._inference_model, "memory_slots", 0) or 0)
    leaves = _leaves(policy, args.games, slots, args.seed)
    sizes = [int(b) for b in args.batch_sizes.split(",") if int(b) <= min(args.max_batch, len(leaves))]
    rows, totals = [], {"leaves": 0, "same_legal": 0, "same_argmax": 0}
    worst = {"max_prior_diff": 0.0, "max_value_diff": 0.0, "max_memory_diff": 0.0}
    with torch.no_grad():
        for B in sizes:
            for start in range(0, len(leaves) - B + 1, max(B, 1)):
                batch = leaves[start:start + B]
                c = _compare(eager.infer_batch(batch), graphed.infer_batch(batch))
                for k in totals:
                    totals[k] += c[k]
                for k in worst:
                    worst[k] = max(worst[k], c[k])
            batch = leaves[:B]
            rows.append({"batch": B,
                         "eager_ms": round(_median_ms(lambda: eager.infer_batch(batch), device, args.repeats), 3),
                         "graphed_ms": round(_median_ms(lambda: graphed.infer_batch(batch), device, args.repeats), 3)})
            print(json.dumps(rows[-1]), flush=True)
    summary = gserve.summary()
    n = max(1, totals["leaves"])
    checks = {
        "every batch graphed": not summary["fallbacks"] and (summary["graphs"] or device.type != "cuda"),
        "legal sets equal": totals["same_legal"] == totals["leaves"],
        "argmax agreement": totals["same_argmax"] / n >= args.min_argmax,
        "value": worst["max_value_diff"] <= args.max_value_diff,
        "memory": worst["max_memory_diff"] <= args.max_memory_diff,
    }
    result = {"pass": all(checks.values()), "checks": checks, "totals": totals, "worst": worst,
              "argmax_agreement": totals["same_argmax"] / n, "timing": rows, "graphed_serve": summary,
              "positions": len(leaves), "memory_slots": slots, "device": device.type}
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("pass", "checks", "argmax_agreement", "worst")}), flush=True)
    return 0 if result["pass"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main(sys.argv))
