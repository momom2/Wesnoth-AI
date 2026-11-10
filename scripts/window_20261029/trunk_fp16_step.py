"""Whether float16's weight rounding makes the trunk step's KL (CPU, float32
arithmetic; Idle priority).

The servers of self-play and of the matches, and stage C's evaluator, run the
trunk's linears on a float16 copy of the weights (wesnoth_ai/model.py
_run_hip_fp16_trunk; wesnoth_ai/packed_trunk.py _layer_sources: the layer
norms stay float32). A float16 keeps 11 significant bits, and stage C's
full-network step moves a trunk weight by a small part of its last bit, so the
copy keeps most weights where they were and moves the others by a whole bit.
This script reads the step as that copy takes it:

    delta_q(s) = Q16(theta_k + s d) - Q16(theta_k)   on the rounded tensors,
                 s d                                 elsewhere,

Q16 a cast to float16 and back and d stage C's reduced_adamw direction (its
pass.pt), on the evaluator's float32 path (tools/pg_eval.py EvalContext.kl, a
few held-out games, every game-side), so the rounding of the weights is the
one thing that differs from the float32 reading of s d:

  kl       the mean per-decision KL(pi_k || pi at theta_k + delta) of s d and
           of delta_q(s), s at each of --multiples of stage C's step, the
           rounded tensors the packed trunk's linears; at stage C's step also
           with every linear map rounded (what a float16 autocast reads); the
           heads-only direction at its own step, rounded and not; and
           Q16(theta_k) - theta_k, the copy's own distance from theta_k.
  tensors  by kind of tensor at stage C's step: the share of weights the
           rounding moves, |delta_q|^2 over |s d|^2, and what that ratio is
           if a weight's place inside its last bit is uniform (a step x on a
           grid of pitch u, f the fractional part of |x| / u, has
           E delta_q^2 = x^2 + u^2 f (1 - f)).
  gain     the first-order gain the rounded step keeps: <g_B, delta_q of half
           A's direction> over <g_B, s d_A>, the two halves of the splits both
           ways, with the rounding's standard error under the same model.

If the rounding is what stage C's float16 evaluator read, delta_q's KL at
stage C's step is near that evaluator's (stage_c.json, about 0.00134) and the
part above the float32 reading grows as s, not as s^2.

    python scripts/window_20261029/trunk_fp16_step.py --out training/metrics/trunk_fp16_step_20261110.json
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FP16_MIN_EXPONENT = -24   # a float16's subnormals are spaced 2**-24 apart
FP16_BITS = 11            # significant bits of a normal float16


def q16(t):
    """`t` as a float16 holds it."""
    return t.half().float()


def fp16_pitch(t):
    """The spacing of float16 values at each entry of `t`."""
    import torch
    _, exponent = torch.frexp(t.abs())                  # |t| = m 2**e, m in [0.5, 1)
    return torch.pow(2.0, (exponent - FP16_BITS).clamp_min(FP16_MIN_EXPONENT).float())


def rounded_step(theta, d, s, rounded):
    """delta_q(s) (module docstring): `rounded` names the tensors a float16
    copy holds."""
    return {n: (q16(theta[n] + s * d[n]) - q16(theta[n])) if n in rounded else s * d[n] for n in d}


def linear_names(context):
    """The parameters the network's linear maps read: every nn.Linear's, and
    the attention's in-projections."""
    import torch
    out = set()
    for prefix, module in (("model.", context.model), ("encoder.", context.encoder)):
        for name, m in module.named_modules():
            if isinstance(m, torch.nn.Linear):
                out.update(f"{prefix}{name}.{leaf}" for leaf, _ in m.named_parameters(recurse=False))
            elif isinstance(m, torch.nn.MultiheadAttention):
                out.update(f"{prefix}{name}.{leaf}" for leaf in ("in_proj_weight", "in_proj_bias"))
    return out


def kind_of(name: str) -> str:
    """`name` without its layer number."""
    return ".".join(part for part in name.split(".") if not part.isdigit())


def rounding_model(x, pitch):
    """sum of E delta_q^2 - x^2 = u^2 f (1 - f) over the entries (module
    docstring), as a tensor of the entries' terms."""
    f = (x.abs() / pitch).frac()
    return pitch.pow(2) * f * (1 - f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=4, help="held-out games the KL reads")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--multiples", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.5],
                    help="the steps read, in units of stage C's reduced_adamw step")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as error:                          # a reading, not a run: priority is best effort
        print(f"WARNING: not at Idle priority ({error!r})", flush=True)
    import torch
    torch.set_num_threads(a.threads)
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import STAGE_C, held_out_context
    from wesnoth_ai.pg_step import direction, head_of, pooled_gradient

    t0 = time.time()
    p = load_pass(STAGE_C / "pass.pt")
    names = list(p["names"])
    arms = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]
    s_full, s_heads = arms["reduced_adamw"]["s"], arms["heads_only"]["s"]
    d = direction(p, "reduced_adamw")
    context, games = held_out_context(a.games)
    theta = context.params()
    linears = linear_names(context) & set(names)
    packed = {n for n in linears if n.startswith("model.encoder.layers.")}
    heads = {n for n in names if head_of(n)}
    out = {"games": games, "game_sides": len(context.sides), "s_full": s_full, "s_heads": s_heads,
           "stage_c_kl": {arm: arms[arm]["kl"] for arm in ("reduced_adamw", "heads_only")},
           "rounded": {k: {"tensors": len(v), "weights": sum(theta[n].numel() for n in v)}
                       for k, v in (("packed", packed), ("linears", linears), ("heads", heads))}}

    def write():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---- offline: what the rounding does to the step, by kind of tensor -----
    stats = collections.defaultdict(lambda: collections.defaultdict(float))
    for n in sorted(packed):
        x = s_full * d[n]
        pitch = fp16_pitch(theta[n])
        dq = q16(theta[n] + x) - q16(theta[n])
        k = stats[kind_of(n)]
        k["weights"] += x.numel()
        k["moved"] += float((dq != 0).sum())
        k["abs_theta"] += float(theta[n].abs().sum())
        k["step_over_pitch"] += float((x.abs() / pitch).sum())
        k["step2"] += float(x.double().pow(2).sum())
        k["rounded2"] += float(dq.double().pow(2).sum())
        k["model2"] += float((x.double().pow(2) + rounding_model(x, pitch).double()).sum())
    total = collections.defaultdict(float)
    for k in stats.values():
        for key, value in k.items():
            total[key] += value
    out["tensors"] = {
        kind: {"weights": int(k["weights"]), "mean_abs_theta": k["abs_theta"] / k["weights"],
               "mean_step_over_pitch": k["step_over_pitch"] / k["weights"], "moved_share": k["moved"] / k["weights"],
               "rounded_over_step_norm2": k["rounded2"] / k["step2"], "model_over_step_norm2": k["model2"] / k["step2"]}
        for kind, k in [*sorted(stats.items()), ("all packed", total)]}
    K = len(p["splits"])
    halves = (list(range(K // 2)), list(range(K // 2, K)))
    out["gain"] = []
    for mine, other in (halves, halves[::-1]):
        d_a = direction(p, "reduced_adamw", gradient=pooled_gradient(p, mine))
        g_b = pooled_gradient(p, other)
        plain = kept = variance = 0.0
        for n in packed:
            x = s_full * d_a[n]
            g = g_b[n].double()
            plain += float((g * x.double()).sum())
            kept += float((g * (q16(theta[n] + x) - q16(theta[n])).double()).sum())
            variance += float((g.pow(2) * rounding_model(x, fp16_pitch(theta[n])).double()).sum())
        out["gain"].append({"direction_splits": mine, "gradient_splits": other, "plain": plain, "rounded": kept,
                            "rounded_over_plain": kept / plain, "rounding_se_over_plain": math.sqrt(variance) / abs(plain)})
    write()
    print(json.dumps({"rounded": out["rounded"], "tensors": out["tensors"], "gain": out["gain"]}, indent=1),
          f"{time.time() - t0:.0f} s", flush=True)

    # ---- the KL readings ---------------------------------------------------
    out["kl"] = {}

    def read(label, delta, steps=(1.0,)):
        kl = context.kl(delta, list(steps))
        out["kl"][label] = kl[1.0] if tuple(steps) == (1.0,) else {repr(s): kl[s] for s in steps}
        print(f"{time.time() - t0:7.0f} s  {label:28s} KL {out['kl'][label]}", flush=True)
        write()

    steps = [m * s_full for m in a.multiples]
    read("float32, by step", d, steps)
    for m in sorted(a.multiples, key=lambda m: abs(m - 1.0)):          # stage C's own step first
        read(f"packed rounded at {m:g}", rounded_step(theta, d, m * s_full, packed))
    read("linears rounded at 1", rounded_step(theta, d, s_full, linears))
    d_heads = {n: (d[n] if n in heads else torch.zeros_like(d[n])) for n in names}
    read("heads float32", d_heads, [s_heads])
    read("heads rounded", rounded_step(theta, d_heads, s_heads, heads))
    read("theta_k's own rounding", {n: (q16(theta[n]) - theta[n]) if n in packed else torch.zeros_like(theta[n])
                                    for n in names})
    context.close()

    plain = out["kl"]["float32, by step"]
    out["summary"] = {
        "by_multiple": {f"{m:g}": {"float32": plain[repr(m * s_full)], "rounded": out["kl"][f"packed rounded at {m:g}"],
                                   "rounding_part_over_multiple":
                                       (out["kl"][f"packed rounded at {m:g}"] - plain[repr(m * s_full)]) / m}
                        for m in a.multiples},
        "stage_c_fp16_kl": arms["reduced_adamw"]["kl"],
    }
    write()
    print(json.dumps(out["summary"], indent=1), f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
