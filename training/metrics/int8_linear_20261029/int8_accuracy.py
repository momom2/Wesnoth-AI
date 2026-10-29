"""own2's logits on the 200 bench positions with int8 linears, against the
padded fp32 trunk, beside fp16's and bf16's distances from it: the reading of
docs/int8_linear_prereg_20261029.md's gate (README.md beside this file).

Paths, each over the positions in the manifest's order, batches of 24 (the
last 8), every position with the 64-slot initial memory, eval mode, no grad:
  fp32  the padded trunk, no autocast (the reference R);
  bf16  the padded trunk under a bf16 autocast;
  fp16  the packed trunk under an fp16 autocast (_run_hip_fp16_trunk, the
        serving path);
  int8  the fp16 path with the four linears of every layer on int8
        (wesnoth_ai/int8_trunk.py), arm P; with --diagnostics, also each
        linear class alone on int8;
  int8_S  with --arm-s, arm S: P with SmoothQuant's migration (alpha 0.5),
        the activation ranges read by a float32 pass over 200 calibration
        positions off the bench set (`calibration_states`).
Statistics per path against R over the real slots (PaddedOutput.sample):
E (the largest |X - R| over the largest |R|) per head, actor argmax
disagreements, RMS error over R's RMS, the mean total variation of the
actor softmax, and E for the belief logits and the new memory.

    set HIP_VISIBLE_DEVICES=1
    python int8_accuracy.py --root <repo> --dataset <replays_dataset_imitation> --out accuracy.json [--diagnostics]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HEADS = ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits")
EXTRA = ("belief_logits", "memory")
BATCH = 24
LINEAR_CLASSES = ("qkv", "out", "ff1", "ff2")
CALIBRATION_N = 200


def statistics(path_out, ref_out):
    """The prereg's statistics of one path against the reference."""
    import torch
    out = {}
    for h in HEADS + EXTRA:
        diff = scale = 0.0
        se = sr = n = 0.0
        for x, r in zip(path_out, ref_out):
            xv, rv = x.get(h), r.get(h)
            if rv is None or xv is None:
                continue
            fin = torch.isfinite(rv)
            if not torch.equal(fin, torch.isfinite(xv)):
                raise RuntimeError(f"{h}: the finite entries differ from the reference's")
            if not fin.any():
                continue
            d = (xv - rv)[fin]
            diff = max(diff, d.abs().max().item())
            scale = max(scale, rv[fin].abs().max().item())
            se += d.double().pow(2).sum().item()
            sr += rv[fin].double().pow(2).sum().item()
            n += d.numel()
        if n:
            out[h] = {"E": diff / scale, "max_abs": diff, "scale": scale,
                      "rms_rel": (se / max(sr, 1e-300)) ** 0.5}
    disagree, tv = 0, []
    for x, r in zip(path_out, ref_out):
        xa, ra = x["actor_logits"].flatten(), r["actor_logits"].flatten()
        disagree += int(xa.argmax().item() != ra.argmax().item())
        tv.append(0.5 * (torch.softmax(xa.double(), 0) - torch.softmax(ra.double(), 0)).abs().sum().item())
    out["actor_argmax_disagreements"] = disagree
    out["actor_tv_mean"] = sum(tv) / len(tv)
    out["actor_tv_max"] = max(tv)
    return out


def gate(arm, bf16):
    """The prereg's gate: E_h(arm) <= E_h(bf16) on the five heads and the
    actor argmax disagreements no more than bf16's."""
    checks = {h: {"int8": arm[h]["E"], "bf16": bf16[h]["E"], "pass": arm[h]["E"] <= bf16[h]["E"]}
              for h in HEADS}
    checks["actor_argmax_disagreements"] = {
        "int8": arm["actor_argmax_disagreements"], "bf16": bf16["actor_argmax_disagreements"],
        "pass": arm["actor_argmax_disagreements"] <= bf16["actor_argmax_disagreements"]}
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def calibration_states(dataset: Path, bench_manifest: Path, out: Path, reconstruct_boundary):
    """Arm S's calibration positions (the prereg): the first 200 entries of
    tools/bench_pipeline.build_state_manifest(dataset, 400, seed=2) whose
    game file is not one of the bench set's, rebuilt; their manifest is
    written to `out`."""
    import gzip
    from tools.bench_pipeline import build_state_manifest
    bench = {e["file"] for e in json.loads(bench_manifest.read_text(encoding="utf-8"))["states"]}
    man = build_state_manifest(dataset, 400, seed=2)
    entries = [e for e in man["states"] if e["file"] not in bench][:CALIBRATION_N]
    if len(entries) < CALIBRATION_N:
        raise RuntimeError(f"only {len(entries)} calibration positions off the bench set")
    out.write_text(json.dumps({"dataset": man["dataset"], "seed": man["seed"], "drawn": len(man["states"]),
                               "excluded_bench_files": len(bench), "states": entries}, indent=1) + "\n",
                   encoding="utf-8")
    states = []
    for e in entries:
        with gzip.open(dataset / e["file"], "rt", encoding="utf-8") as f:
            data = json.load(f)
        res = reconstruct_boundary(data, e["cut_turn"])
        if res is None or res[1] != e["begin_side"]:
            raise RuntimeError(f"calibration state failed to reconstruct: {e}")
        states.append(res[0])
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", type=int, default=None, help="the first N bench positions (default all)")
    ap.add_argument("--diagnostics", action="store_true", help="also each linear class alone on int8")
    ap.add_argument("--arm-s", action="store_true",
                    help="also arm S (SmoothQuant, alpha 0.5), calibrated on 200 positions off the bench "
                         "set; its calibration manifest is written beside --out")
    args = ap.parse_args()
    sys.path.insert(0, str(args.root))
    from tools.bench_states import load_states, reconstruct_boundary
    manifest = args.root / "configs" / "bench_states.json"
    t0 = time.perf_counter()
    states = [gs for gs, _ in load_states(manifest, args.dataset, args.states)]
    calibration = []
    if args.arm_s:
        calibration = calibration_states(args.dataset, manifest, args.out.parent / "calibration_states.json",
                                         reconstruct_boundary)
    t_states = time.perf_counter() - t0

    import torch
    from tools.eval_players import _load_policy
    from wesnoth_ai.hip_int8 import hip_int8_applies
    from wesnoth_ai.int8_trunk import (
        LINEAR_CLASSES as CLASSES, Int8TrunkWeights, record_activation_ranges, run_int8_trunk, smoothing_factors,
    )
    assert tuple(CLASSES) == LINEAR_CLASSES
    device = torch.device("cuda")
    if not hip_int8_applies(device):
        raise RuntimeError("the int8 kernels do not serve this device")
    policy = _load_policy(args.root / "training" / "checkpoints" / "own2.pt", device, label="own2")
    encoder, model = policy._inference_encoder, policy._inference_model
    model.eval()
    with torch.no_grad():
        encs = [encoder.encode(gs) for gs in states]
        cal_encs = [encoder.encode(gs) for gs in calibration]
    k = model.memory_slots
    int8_weights = Int8TrunkWeights.build(model.encoder, device, model._weights_version)
    fp16_layers = model._weight_copy(torch.float16, torch.float32, device).layers
    fused = model._run_hip_fp16_trunk
    calls = []

    def trunk_for(classes, weights=int8_weights):
        if classes is None:
            return lambda x, index: calls.append("fp16") or fused(x, index)
        return lambda x, index: calls.append(classes) or run_int8_trunk(
            x, index, weights, int8_classes=classes, fp16_layers=fp16_layers)

    def per_sample(out):
        rows = []
        for s in out.float32().samples():
            row = {h: getattr(s, h).detach().float().cpu() for h in HEADS}
            row["belief_logits"] = None if s.belief_logits is None else s.belief_logits.detach().float().cpu()
            row["memory"] = None if s.memory is None else s.memory.detach().float().cpu()
            rows.append(row)
        return rows

    def run(path, classes=None, weights=int8_weights, trunk=None, positions=None):
        positions = encs if positions is None else positions
        rows = []
        calls.clear()
        model._run_hip_fp16_trunk = trunk or trunk_for(classes, weights)
        t = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(positions), BATCH):
                chunk = positions[i:i + BATCH]
                mem = [model.initial_memory(k).to(device) for _ in chunk]
                if path == "fp32":
                    with torch.autocast("cuda", enabled=False):
                        out = model.forward_padded(chunk, autocast_bf16=False, packed=False, memory=mem)
                elif path == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model.forward_padded(chunk, autocast_bf16=False, packed=False, memory=mem)
                else:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = model.forward_padded(chunk, autocast_bf16=False, packed=True, memory=mem)
                rows.extend(per_sample(out))
        torch.cuda.synchronize()
        if path == "packed" and trunk is None and len(calls) != (len(positions) + BATCH - 1) // BATCH:
            raise RuntimeError(f"the packed fp16 forward did not take the fused loop ({len(calls)} calls)")
        model._run_hip_fp16_trunk = fused
        return rows, time.perf_counter() - t

    ref, t_ref = run("fp32")
    results, walls = {}, {"fp32": t_ref}
    for name, path, classes in (("bf16", "bf16", None), ("fp16", "packed", None),
                                ("int8", "packed", LINEAR_CLASSES)):
        rows, walls[name] = run(path, classes)
        results[name] = statistics(rows, ref)
    if args.diagnostics:
        for c in LINEAR_CLASSES:
            rows, walls[f"int8_{c}_only"] = run("packed", (c,))
            results[f"int8_{c}_only"] = statistics(rows, ref)
    rec = {"positions": len(encs), "batch": BATCH, "memory_slots": k,
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "reference": "padded trunk, float32, no autocast",
           "paths": results, "gate_P": gate(results["int8"], results["bf16"])}
    if args.arm_s:
        ranges = []
        _, walls["calibration"] = run("packed", positions=cal_encs, trunk=lambda x, index: record_activation_ranges(
            x, index, model.encoder, ranges))
        factors, floored = smoothing_factors(model.encoder, ranges, alpha=0.5)
        s_weights = Int8TrunkWeights.build(model.encoder, device, model._weights_version, smooth=factors)
        rows, walls["int8_S"] = run("packed", LINEAR_CLASSES, weights=s_weights)
        results["int8_S"] = statistics(rows, ref)
        if args.diagnostics:
            for c in LINEAR_CLASSES:
                rows, walls[f"int8_S_{c}_only"] = run("packed", (c,), weights=s_weights)
                results[f"int8_S_{c}_only"] = statistics(rows, ref)
        rec["gate_S"] = gate(results["int8_S"], results["bf16"])
        rec["calibration"] = {"positions": len(cal_encs), "alpha": 0.5, "floored_channels": floored,
                              "manifest": "calibration_states.json",
                              "factor_range": [min(float(f.min()) for lf in factors for f in lf.values()),
                                               max(float(f.max()) for lf in factors for f in lf.values())]}
    rec["walls_s"] = {kk: round(v, 2) for kk, v in walls.items()}
    rec["states_s"] = round(t_states, 1)
    args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
