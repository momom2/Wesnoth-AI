"""The match test's candidates: `own2` stepped along stage C's pass, written
as checkpoints (docs/trunk_step_match_prereg_20261111.md on main; CPU,
float32, a minute or two).

  heads     theta_k + s_H d on the head maps alone: stage C's heads_only arm
  heads<x>  the head maps alone at stage C's ladder rung of x times its KL
  full<m>   theta_k + m s_F d: stage C's full-network (reduced_adamw) arm at
            m times its step
  mix<m>    the head maps at s_H and every other parameter at m s_F

d is stage C's reduced_adamw direction (its pass.pt), and s_F and s_H are its
own steps (stage_c.json: each arm sized to the mean KL). A checkpoint is the
source's with its weights replaced and the step recorded, as tools/pg_eval.py
write_candidate's. One that already exists is not rewritten: its weights must
equal the ones built here. --out/candidates.json lists each candidate's
SHA-256, its two steps and the norm of what moved, on the head maps and
elsewhere, over the norm the direction gives it (1 up to float32 rounding).

    python scripts/window_20261029/trunk_step_candidates.py --heads --full 3 8 --out C:/Users/momom2/wesnoth_runs/trunk_step_match_20261111/candidates
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NORM_RTOL = 1e-6    # the direction's norms against stage C's record: the same float32 sums
MOVED_RTOL = 1e-2   # what moved against step * |d|: theta + s d rounds each weight to float32


def steps(a, s_f: float, s_h: float, rungs: dict) -> dict:
    """{label: (the head maps' step, every other parameter's step)}."""
    out = {}
    if a.heads:
        out["heads"] = (s_h, 0.0)
    for mult in a.heads_rung:
        if mult not in rungs:
            raise SystemExit(f"stage C's ladder has no rung at {mult:g} eps: {sorted(rungs)}")
        out[f"heads{mult:g}"] = (rungs[mult], 0.0)
    for m in a.full:
        out[f"full{m:g}"] = (m * s_f, m * s_f)
    for m in a.mix:
        out[f"mix{m:g}"] = (s_h, m * s_f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heads", action="store_true", help="stage C's heads-only step")
    ap.add_argument("--heads-rung", type=float, nargs="*", default=[], metavar="MULT",
                    help="the head maps alone at stage C's ladder rung of MULT times its KL (heads<MULT>)")
    ap.add_argument("--full", type=float, nargs="*", default=[], metavar="M",
                    help="the full-network step at M times stage C's")
    ap.add_argument("--mix", type=float, nargs="*", default=[], metavar="M",
                    help="the head maps at their own step, the rest at M times stage C's full step")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True, help="the candidates' directory")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from tools.eval_provenance import file_sha256
    from tools.pg_pass import load_network
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import CHECKPOINT, STAGE_C
    from wesnoth_ai.param_groups import named_model_parameters
    from wesnoth_ai.pg_step import direction, head_of

    record = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))
    s_f, s_h = (float(record["arms"][arm]["s"]) for arm in ("reduced_adamw", "heads_only"))
    base_sha = file_sha256(CHECKPOINT)
    if base_sha != record["pass"]["checkpoint_sha256"]:
        raise SystemExit(f"{CHECKPOINT} is not the checkpoint of stage C's pass ({base_sha[:12]})")
    d = direction(load_pass(STAGE_C / "pass.pt"), "reduced_adamw")

    def size(tensors, heads: bool) -> float:
        return math.sqrt(sum(float(t.double().pow(2).sum()) for n, t in tensors if bool(head_of(n)) == heads))

    d_norm = {heads: size(d.items(), heads) for heads in (True, False)}
    for arm, got in (("reduced_adamw", math.hypot(d_norm[True], d_norm[False])), ("heads_only", d_norm[True])):
        want = float(record["arms"][arm]["norm"])
        if abs(got - want) > NORM_RTOL * want:
            raise SystemExit(f"the direction's {arm} norm is {got!r}, stage C's record says {want!r}")

    encoder, model = load_network(CHECKPOINT, torch.device("cpu"))
    named = named_model_parameters(model, encoder)
    base = {n: p.detach().clone() for n, p in named}
    unknown = sorted(set(d) - set(base))
    if unknown:
        raise SystemExit(f"the direction names parameters the network lacks: {unknown[:5]}")
    a.out.mkdir(parents=True, exist_ok=True)
    out = {"base": str(CHECKPOINT), "base_sha256": base_sha, "pass": str(STAGE_C / "pass.pt"), "s_full": s_f,
           "s_heads": s_h, "direction_norm": {"heads": d_norm[True], "rest": d_norm[False]}, "candidates": {}}
    if (a.out / "candidates.json").exists():            # an earlier run's candidates stay listed
        before = json.loads((a.out / "candidates.json").read_text(encoding="utf-8"))
        if before.get("base_sha256") == base_sha:
            out["candidates"] = before["candidates"]
    if a.heads_rung and record["ladder"]["arm"] != "heads_only":
        raise SystemExit(f"stage C's ladder is on the {record['ladder']['arm']} arm, not the head maps'")
    rungs = {float(r["multiplier"]): float(r["s"]) for r in record["ladder"]["rungs"]}
    for label, (s_head, s_rest) in steps(a, s_f, s_h, rungs).items():
        with torch.no_grad():
            for n, p in named:
                step = s_head if head_of(n) else s_rest
                p.copy_(base[n] + step * d[n] if n in d and step else base[n])
        model.weights_changed()
        moved = {heads: size(((n, p.detach() - base[n]) for n, p in named), heads) for heads in (True, False)}
        ratio = {}
        for heads, step in ((True, s_head), (False, s_rest)):
            want = step * d_norm[heads]
            ratio["heads" if heads else "rest"] = moved[heads] / want if want else None
            if abs(moved[heads] - want) > MOVED_RTOL * want:
                raise SystemExit(f"{label}: {'the head maps' if heads else 'the rest'} moved {moved[heads]!r}, "
                                 f"the step gives {want!r}")
        states = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "encoder_state": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}}
        path = a.out / f"{label}.pt"
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True)
            for key, state in states.items():
                if set(old[key]) != set(state) or any(not torch.equal(old[key][k], v) for k, v in state.items()):
                    raise SystemExit(f"{path} exists and its {key} is not this candidate's")
        else:
            ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            ck.update(states)
            ck["pg_candidate"] = {"base_sha256": base_sha, "step": s_rest, "head_step": s_head, "arm": label}
            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save(ck, tmp)
            os.replace(tmp, path)
        out["candidates"][label] = {"path": str(path), "sha256": file_sha256(path), "head_step": s_head,
                                    "step": s_rest, "moved": {"heads": moved[True], "rest": moved[False]},
                                    "moved_over_expected": ratio}
        (a.out / "candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(label, json.dumps(out["candidates"][label]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
