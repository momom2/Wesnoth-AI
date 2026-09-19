#!/usr/bin/env python3
"""Where one serve batch's time goes, and what a CUDA graph would leave.

The inference server's cost per batch is mostly fixed: 13.9 ms of
device-stream time at a mean batch of 6.2 against 15.0 at 7.6 on the
eval path (docs/box_specs.md 2026-09-13), 16 ms at 17.5 leaves on the
pool, and the pool's serve thread logs 16 ms of HOST time per batch
launching it (encode 2.7, forward 8.2, priors 5.3) against 0.14 ms
waiting for the device. That reads as launch-bound: the stream's span
is the host feeding it kernels one by one. This tool measures that on
a fixed batch of real positions:

  1. the production call (`InferenceServer.infer_batch` with masks),
     host wall per batch and the seam's own stage seconds;
  2. under torch.profiler: how many kernels one batch launches and how
     long they keep the device busy, against the batch's wall;
  3. the forward alone (packed embed -> trunk -> heads -> fp32), eager
     against the same forward captured once in a torch.cuda.CUDAGraph
     and replayed: the launch overhead a static-shape serve path
     would remove;
  4. the priors stage alone (mask staging, masked softmaxes,
     compaction, the one readback).

    python tools/bench_serve_graph.py --checkpoint training/checkpoints/relset.pt \\
        --device cuda --batch-sizes 8,16 --states 32 --out OUT.json

On cpu the graph rows are skipped (a smoke of the harness only).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

log = logging.getLogger("bench_serve_graph")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, device: torch.device, repeats: int, warmup: int = 5) -> Dict[str, float]:
    """Median and min host wall per call, the device drained after each."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": statistics.median(times), "min_ms": min(times), "n": repeats}


def load_pairs(policy, manifest: Path, dataset: Path, limit: int, device: torch.device):
    """(RawEncoded, PackedMasks) per bench state, as the actor ships them."""
    from tools.bench_pipeline import load_states
    from tools.inference_seam import build_light_encoded
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.server_priors import pack_masks
    enc = policy._inference_encoder
    pairs = []
    for gs, _scenario in load_states(manifest, dataset, limit):
        enc.register_names(gs)
        raw = encode_raw(gs, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id,
                         relevant_set=bool(getattr(enc, "relevant_set_hexes", False)),
                         fog_hides_enemy_villages=bool(getattr(enc, "fog_hides_enemy_villages", False)),
                         terrain_multi_hot=bool(getattr(enc, "terrain_multi_hot", False)))
        light = build_light_encoded(raw, torch.device("cpu"))
        pairs.append((raw, pack_masks(light, gs)))
    return pairs


class _Forward:
    """The device half of the packed-embed serve path on one fixed
    batch: token rows and index arrays staged once, then the gather,
    the kind embedding, the trunk, the heads and the fp32 cast -- what
    `WesnothModel.forward_embedded` runs after `build_packed_layout`."""

    def __init__(self, model, encoder, raws, device: torch.device, bf16: bool):
        from wesnoth_ai.model import ActorKind, TokenKind
        from wesnoth_ai.packed_trunk import build_packed_layout
        self.model, self.device, self.bf16 = model, device, bf16
        with torch.no_grad():
            self.streams = encoder.encode_from_raw_embedded(raws, device=device)
        sizes = self.streams.sizes
        self.sizes = sizes
        self.U_max, self.R_max, self.H_max = (max(s[i] for s in sizes) for i in (0, 1, 2))
        self.packed = bool(getattr(model, "infer_packed_trunk", False)) and device.type == "cuda" and bf16
        if self.packed:
            self.layout = build_packed_layout(sizes, self.H_max, self.U_max, self.R_max,
                                              TokenKind, ActorKind, source="streams")
            self.index = self.layout.to_device(device)
        _sync(device)

    def __call__(self):
        model = self.model
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            if self.packed:
                x = (self.streams.tokens.index_select(0, self.index.src)
                     + model.token_kind_embed(self.index.kind))
                out = model._packed_trunk_heads(x, self.index, self.layout, self.sizes,
                                                self.H_max, self.U_max, self.R_max)
            else:
                out = model.forward_embedded(self.streams)
            return out.float32() if self.bf16 else out


def capture_graph(fwd: _Forward):
    """The forward captured once; returns (graph, its outputs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd()
    return g, out


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0


def profile_batch(server, pairs, device: torch.device) -> Dict[str, object]:
    """One production batch under the profiler: kernel launches, the
    device's busy time, the top kernels, against the call's wall."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    server.infer_batch(pairs, stats={})
    _sync(device)
    t0 = time.perf_counter()
    with profile(activities=acts) as prof:
        server.infer_batch(pairs, stats={})
        _sync(device)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    dev_type = torch.autograd.DeviceType.CUDA
    rows = prof.key_averages()
    kern = [r for r in rows if getattr(r, "device_type", None) == dev_type]

    def _self_dev(r):
        for k in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(r, k, None)
            if v is not None:
                return float(v)
        return 0.0

    launches = int(sum(r.count for r in kern))
    busy_us = float(sum(_self_dev(r) for r in kern))
    top = sorted(kern, key=_self_dev, reverse=True)[:12]
    return {
        "wall_ms": wall_ms, "kernel_launches": launches, "device_busy_ms": busy_us / 1000.0,
        "busy_fraction": (busy_us / 1000.0 / wall_ms) if wall_ms else None,
        "top_kernels": [{"name": r.key[:90], "count": int(r.count), "self_ms": _self_dev(r) / 1000.0}
                        for r in top],
    }


def bench_batch(policy, server, pairs, device: torch.device, bf16: bool, repeats: int,
                graph: bool) -> Dict[str, object]:
    from wesnoth_ai.server_priors import start_priors
    model, encoder = policy._inference_model, policy._inference_encoder
    raws = [r for r, _ in pairs]
    packs = [m for _, m in pairs]
    B = len(pairs)
    tokens = sum(len(r.hex_xs) + len(r.unit_xs) + len(r.recruit_xs) + 2 for r in raws)
    row: Dict[str, object] = {"batch": B, "tokens": tokens}

    stats: Dict[str, float] = {}
    row["infer_batch"] = _time_ms(lambda: server.infer_batch(pairs, stats=stats), device, repeats)
    n = repeats + 5
    row["seam_stages_ms"] = {k[2:]: 1000.0 * v / n for k, v in stats.items() if k.startswith("t_")}
    if "gpu_ms" in stats:
        row["seam_stages_ms"]["stream_span"] = stats["gpu_ms"] / n

    # The same call through the static-shape path (wesnoth_ai/graphed_serve):
    # on CUDA the bucket's graph replays, on CPU its body runs eagerly.
    if graph:
        from tools.inference_seam import InferenceServer
        from wesnoth_ai.graphed_serve import Caps, GraphedServe
        try:
            gserve = GraphedServe(model, encoder, device, caps=Caps(b_cap=B))
            gserver = InferenceServer(model, encoder, device=device,
                                      output_device=torch.device("cpu"), autocast_bf16=bf16,
                                      packed_embed=True, graphed=gserve)
            row["infer_batch_graphed"] = _time_ms(lambda: gserver.infer_batch(pairs), device, repeats)
            row["graphed_summary"] = gserve.summary()
            ref = server.infer_batch(pairs)
            got = gserver.infer_batch(pairs)
            row["graphed_max_abs_diff"] = {
                "value": max(_max_abs(r.value, g.value) for r, g in zip(ref, got)),
                "prior": max(float(np.abs(r.legal_compact.prior - g.legal_compact.prior).max())
                             if r.legal_compact.prior.shape == g.legal_compact.prior.shape else float("inf")
                             for r, g in zip(ref, got)),
                "same_actions": all(
                    np.array_equal(r.legal_compact.actor, g.legal_compact.actor)
                    and np.array_equal(r.legal_compact.target, g.legal_compact.target)
                    for r, g in zip(ref, got))}
        except Exception as e:                       # noqa: BLE001 -- the record says why
            row["infer_batch_graphed"] = None
            row["graphed_error"] = f"{type(e).__name__}: {e}"
            log.warning("graphed serve failed: %s", row["graphed_error"])

    row["encode_embedded"] = _time_ms(
        lambda: encoder.encode_from_raw_embedded(raws, device=device), device, repeats)

    fwd = _Forward(model, encoder, raws, device, bf16)
    row["forward_packed"] = fwd.packed
    row["forward_eager"] = _time_ms(fwd, device, repeats)
    with torch.no_grad():
        padded = fwd()
    _sync(device)
    extras = [padded.value, padded.value_logits, padded.cliffness]
    row["priors"] = _time_ms(lambda: start_priors(padded, packs, extras).finish(), device, repeats)

    if graph and device.type == "cuda":
        try:
            g, out = capture_graph(fwd)
            row["forward_graph"] = _time_ms(g.replay, device, repeats)
            g.replay()
            _sync(device)
            row["graph_max_abs_diff"] = {
                "actor_logits": _max_abs(out.actor_logits, padded.actor_logits),
                "target_logits": _max_abs(out.target_logits, padded.target_logits),
                "value": _max_abs(out.value, padded.value)}
        except Exception as e:                       # noqa: BLE001 -- the record says why
            row["forward_graph"] = None
            row["graph_error"] = f"{type(e).__name__}: {e}"
            log.warning("graph capture failed: %s", row["graph_error"])
    row["profile"] = profile_batch(server, pairs, device)
    return row


def threaded_check(model, encoder, device: torch.device, bf16: bool, pairs, threads: int,
                   rounds: int) -> Dict[str, object]:
    """`threads` serve threads, each with its own GraphedServe through the
    seam's factory, serving distinct batches concurrently for `rounds`
    rounds; every reply is compared with the eager seam's on the same
    batch (actions identical, priors and values within bf16 noise). The
    pool's two serve threads returned empty action lists on 2026-09-14
    where the single-threaded eval server was right; this is the
    reproduction."""
    import threading
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    eager = InferenceServer(model, encoder, device=device, output_device=torch.device("cpu"),
                            autocast_bf16=bf16, packed_embed=True)
    per = max(1, len(pairs) // threads)
    batches = [pairs[i * per:(i + 1) * per] for i in range(threads)]
    refs = [eager.infer_batch(b) for b in batches]
    gserver = InferenceServer(
        model, encoder, device=device, output_device=torch.device("cpu"), autocast_bf16=bf16,
        packed_embed=True,
        graphed=lambda: GraphedServe(model, encoder, device, caps=Caps(b_cap=per)))
    bad = {"empty": 0, "actions": 0, "prior": 0, "value": 0, "errors": []}
    done = [0] * threads

    from wesnoth_ai.leaf_wire import pack_request, unpack_request

    def worker(t):
        st: Dict[str, float] = {}
        try:
            for r in range(rounds):
                # Alternate the batch so buckets and buffers get reused;
                # the leaves go through the pool's wire format (views
                # into one request buffer) and the serve loop's stats.
                # Batch sizes cycle from 1 to the cap: the pool serves a
                # lone root leaf as often as a full batch.
                n = 1 + (r % per)
                b = unpack_request(pack_request(batches[(t + r) % threads][:n]))
                ref = refs[(t + r) % threads][:n]
                outs = gserver.infer_batch(b, stats=st)
                for o, e in zip(outs, ref):
                    if o.legal_compact.prior.shape[0] == 0 and e.legal_compact.prior.shape[0] > 0:
                        bad["empty"] += 1
                    elif not (np.array_equal(o.legal_compact.actor, e.legal_compact.actor)
                              and np.array_equal(o.legal_compact.target, e.legal_compact.target)):
                        bad["actions"] += 1
                    elif float(np.abs(o.legal_compact.prior - e.legal_compact.prior).max()) > 2e-2:
                        bad["prior"] += 1
                    if float((o.value - e.value).abs().max()) > 2e-2:
                        bad["value"] += 1
                done[t] += 1
        except Exception as ex:                          # noqa: BLE001
            bad["errors"].append(f"thread {t}: {type(ex).__name__}: {ex}")

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return {"threads": threads, "rounds": rounds, "leaves_per_batch": per, "done": done,
            "bad": bad, "summary": gserver.graphed_summary()}


def replay_dump(path: Path, model, encoder, device: torch.device, bf16: bool) -> int:
    """The dumped batch through the eager seam, then the static body run
    eagerly (graphs off), then the graph; shapes and outputs printed."""
    import pickle
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    pairs = pickle.load(open(path, "rb"))
    raws = [r for r, _ in pairs]
    sizes = [(r.unit_xs.shape[0], r.recruit_type_ids.shape[0], r.hex_xs.shape[0]) for r in raws]
    print("batch", len(pairs), "sizes (U, R, H)", sizes)
    for r, m in pairs[:3]:
        print("  raw fields", {k: (tuple(v.shape), str(v.dtype), int(v.min()) if v.size else None,
                                  int(v.max()) if v.size else None)
                               for k, v in vars(r).items() if isinstance(v, np.ndarray) and v.dtype.kind in "iu"})
        print("  masks", m.n_units, m.n_recruits, m.n_hexes, m.attack_valid.shape,
              None if m.attack_bias is None else m.attack_bias.shape)
    eager = InferenceServer(model, encoder, device=device, output_device=torch.device("cpu"),
                            autocast_bf16=bf16, packed_embed=True)
    ref = eager.infer_batch(pairs)
    print("eager ok:", [int(o.legal_compact.prior.shape[0]) for o in ref])
    for graphs in (False, True):
        g = GraphedServe(model, encoder, device, caps=Caps(b_cap=max(16, len(pairs))), graphs=graphs)
        srv = InferenceServer(model, encoder, device=device, output_device=torch.device("cpu"),
                              autocast_bf16=bf16, packed_embed=True, graphed=g)
        try:
            outs = srv.infer_batch(pairs)
            same = all(np.array_equal(o.legal_compact.actor, e.legal_compact.actor)
                       and np.array_equal(o.legal_compact.target, e.legal_compact.target)
                       for o, e in zip(outs, ref))
            print(f"graphs={graphs}: served {g.served}, fallbacks {g.fallbacks}, same actions {same}, "
                  f"entries {[int(o.legal_compact.prior.shape[0]) for o in outs]}")
        except Exception as e:                       # noqa: BLE001
            print(f"graphs={graphs}: FAILED {type(e).__name__}: {e}")
            return 1
    return 0


def markdown(res: Dict[str, object]) -> str:
    lines = ["| batch | tokens | infer_batch ms | encode ms | forward eager ms | forward graph ms | priors ms | launches | device busy ms | busy of wall |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        g = r.get("forward_graph")
        p = r["profile"]
        graph_ms = f"{g['median_ms']:.2f}" if g else "-"
        busy = (f"{p['device_busy_ms']:.2f} | {p['busy_fraction']:.2f}"
                if p.get("busy_fraction") is not None else "- | -")
        lines.append(
            f"| {r['batch']} | {r['tokens']} | {r['infer_batch']['median_ms']:.2f} "
            f"| {r['encode_embedded']['median_ms']:.2f} | {r['forward_eager']['median_ms']:.2f} "
            f"| {graph_ms} | {r['priors']['median_ms']:.2f} | {p['kernel_launches']} | {busy} |")
    for r in res["rows"]:
        st = r.get("seam_stages_ms") or {}
        if st:
            lines.append(f"\nbatch {r['batch']} seam stages, ms per call: "
                         + ", ".join(f"{k} {v:.2f}" for k, v in st.items()))
        gi = r.get("infer_batch_graphed")
        if gi:
            lines.append(f"batch {r['batch']} infer_batch through graphed serve: "
                         f"{gi['median_ms']:.2f} ms (min {gi['min_ms']:.2f}); "
                         f"{r.get('graphed_summary')}; diff vs eager {r.get('graphed_max_abs_diff')}")
        if r.get("graphed_error"):
            lines.append(f"batch {r['batch']} graphed serve: {r['graphed_error']}")
        if r.get("graph_max_abs_diff"):
            lines.append(f"batch {r['batch']} graph vs eager max abs diff: {r['graph_max_abs_diff']}")
        if r.get("graph_error"):
            lines.append(f"batch {r['batch']} graph capture: {r['graph_error']}")
        lines.append(f"batch {r['batch']} top kernels: "
                     + "; ".join(f"{k['name'][:40]} x{k['count']} {k['self_ms']:.2f} ms"
                                 for k in r["profile"]["top_kernels"][:6]))
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "bench_states.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "replays_dataset_imitation")
    ap.add_argument("--states", type=int, default=32, help="positions loaded (the largest batch)")
    ap.add_argument("--batch-sizes", default="8,16")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--replay-dump", type=Path, default=None,
                    help="a batch the seam dumped on a graphed failure (WESNOTH_GRAPHED_DUMP): "
                         "serve it eagerly, then through the static body eagerly and through "
                         "a graph, comparing outputs (run with CUDA_LAUNCH_BLOCKING=1 to name "
                         "the failing kernel)")
    ap.add_argument("--threads", type=int, default=0,
                    help="also run the threaded check: N serve threads with their own graphs")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from tools.eval_sim import _load_policy
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.packed_trunk import check_packed_trunk_supported, flash_varlen_applies

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no cuda device")
    cuda = device.type == "cuda"
    torch.set_num_threads(4)
    policy = _load_policy(args.checkpoint, device, label="bench_serve_graph", infer_bf16=cuda)
    model, encoder = policy._inference_model, policy._inference_encoder
    packed = False
    if cuda and flash_varlen_applies(device, torch.bfloat16):
        check_packed_trunk_supported(model.encoder)
        model.infer_packed_trunk = True
        packed = True
    server = InferenceServer(model, encoder, device=device, output_device=torch.device("cpu"),
                             autocast_bf16=cuda, packed_embed=packed)
    if args.replay_dump is not None:
        return replay_dump(args.replay_dump, model, encoder, device, cuda)
    sizes = [int(s) for s in args.batch_sizes.split(",") if s]
    pairs = load_pairs(policy, args.manifest, args.dataset, max(max(sizes), args.states), device)
    log.info("%d positions, packed trunk %s, bf16 %s", len(pairs), packed, cuda)
    rows = [bench_batch(policy, server, pairs[:B], device, cuda, args.repeats, not args.no_graph)
            for B in sizes]
    threaded = None
    if args.threads > 0:
        threaded = threaded_check(model, encoder, device, cuda, pairs, args.threads, args.rounds)
        log.info("threaded check: %s", threaded)
    res = {"rows": rows, "threaded": threaded, "env": {
        "checkpoint": args.checkpoint.name, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__, "packed_trunk": packed, "bf16": cuda,
        "relevant_set": bool(getattr(encoder, "relevant_set_hexes", False)),
        "repeats": args.repeats}}
    print(markdown(res))
    if threaded is not None:
        print("threaded check:", json.dumps(threaded))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
