"""One Adam step along a pass's gradient (the simple loop,
docs/simple_loop_prereg_20261117.md; the ruling is item 6 of
docs/training_dynamics_theory_20261111.md, "The user's decisions").

    theta_{k+1} = theta_k + lr * m_hat / (sqrt(v_hat) + eps)

torch.optim.Adam's update (Kingma and Ba 2015) with `maximize`, since the
pass's `total` is the ascent direction (tools/pg_game_pass.py): one
learning rate for every parameter, no weight decay, no clipping, no
schedule. The optimizer's state, the two moments of every parameter and
the step count, is carried from step to step in a file of its own, which
names the checkpoint it was left at: a state of another checkpoint is
refused.

Refused (exit 3, nothing written): a gradient that is not finite, a
gradient of another checkpoint or of other parameters, a state that does
not continue from the checkpoint.

Writes the stepped checkpoint (the source's fields with its weights
replaced, and `pg_step` recording the step), the optimizer's state, and
`<out stem>.step.json`: the gradient's norm and the update's, in all and
per parameter group (wesnoth_ai/param_groups.py).

    python tools/pg_adam_step.py --checkpoint pi_k.pt --grad grad.pt --state adam_k.pt \\
        --lr 1e-5 --out pi_k1.pt --state-out adam_k1.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.pg_pass import load_network  # noqa: E402
from wesnoth_ai.param_groups import GROUP_ORDER, group_of, named_model_parameters  # noqa: E402

log = logging.getLogger("pg_adam_step")

EXIT_REFUSED = 3
# Adam's defaults (Kingma and Ba 2015; torch.optim.Adam).
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


class StepRefused(ValueError):
    """The step's inputs do not belong together, or the gradient is not finite."""


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def group_norms(tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """The norm of `tensors` per parameter group, in the logs' order."""
    squares: Dict[str, float] = {}
    for n, t in tensors.items():
        squares[group_of(n)] = squares.get(group_of(n), 0.0) + float((t.double() ** 2).sum())
    return {g: squares[g] ** 0.5 for g in GROUP_ORDER if g in squares}


def adam_step(checkpoint: Path, grad_path: Path, state_path: Optional[Path], lr: float, out: Path,
              state_out: Path, beta1: float = BETA1, beta2: float = BETA2, eps: float = EPS) -> Dict:
    """The step of the module docstring; its record (step.json's dict).
    Raises `StepRefused` before anything is written."""
    from tools.eval_provenance import file_sha256
    if not lr > 0:
        raise StepRefused(f"the learning rate must be positive, got {lr}")
    sha = file_sha256(checkpoint)
    grad = torch.load(grad_path, map_location="cpu", weights_only=True)
    if grad.get("checkpoint_sha256") != sha:
        raise StepRefused(f"the gradient is of checkpoint {str(grad.get('checkpoint_sha256'))[:12]}, "
                          f"not {sha[:12]}")
    cpu = torch.device("cpu")
    encoder, model = load_network(checkpoint, cpu)
    named = named_model_parameters(model, encoder)
    names = [n for n, _ in named]
    if list(grad["names"]) != names:
        raise StepRefused("the gradient's parameters are not the checkpoint's")
    total = grad["total"]
    for n, p in named:
        if tuple(total[n].shape) != tuple(p.shape):
            raise StepRefused(f"{n}: the gradient's shape {tuple(total[n].shape)} is not {tuple(p.shape)}")
        if not bool(torch.isfinite(total[n]).all()):
            raise StepRefused(f"{n}: the gradient is not finite")
    optimizer = torch.optim.Adam([p for _, p in named], lr=lr, betas=(beta1, beta2), eps=eps, maximize=True)
    step = 0
    if state_path is not None:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("checkpoint_sha256") != sha:
            raise StepRefused(f"the optimizer's state was left at checkpoint "
                              f"{str(state.get('checkpoint_sha256'))[:12]}, not {sha[:12]}")
        if list(state["names"]) != names:
            raise StepRefused("the optimizer's state is of other parameters")
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:          # this step's settings, not the saved ones
            group.update(lr=lr, betas=(beta1, beta2), eps=eps)
        step = int(state["step"])
    before = {n: p.detach().clone() for n, p in named}
    for n, p in named:
        p.grad = total[n].to(p.dtype)
    optimizer.step()
    step += 1
    update = {n: p.detach() - before[n] for n, p in named}
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    ck["model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck["encoder_state"] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    ck["pg_step"] = {"base_sha256": sha, "step": step, "lr": float(lr), "betas": [float(beta1), float(beta2)],
                     "eps": float(eps)}
    _atomic_save(ck, out)
    out_sha = file_sha256(out)
    _atomic_save({"names": names, "optimizer": optimizer.state_dict(), "step": step, "checkpoint_sha256": out_sha},
                 state_out)
    record = {"step": step, "lr": float(lr), "betas": [float(beta1), float(beta2)], "eps": float(eps),
              "base_sha256": sha, "checkpoint_sha256": out_sha,
              "grad_norm": float(torch.sqrt(sum((t.double() ** 2).sum() for t in total.values()))),
              "update_norm": float(torch.sqrt(sum((t.double() ** 2).sum() for t in update.values()))),
              "grad_norms": group_norms(total), "update_norms": group_norms(update),
              "parameters": int(sum(p.numel() for _, p in named))}
    summary = out.with_name(out.stem + ".step.json")
    tmp = summary.with_name(summary.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, summary)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="theta_k, the network the pass differentiated")
    ap.add_argument("--grad", type=Path, required=True, help="the pass's grad.pt")
    ap.add_argument("--state", type=Path, default=None, help="the optimizer's state at theta_k (none: a first step)")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--beta1", type=float, default=BETA1)
    ap.add_argument("--beta2", type=float, default=BETA2)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--out", type=Path, required=True, help="theta_{k+1}")
    ap.add_argument("--state-out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        r = adam_step(args.checkpoint, args.grad, args.state, args.lr, args.out, args.state_out,
                      args.beta1, args.beta2, args.eps)
    except StepRefused as e:
        log.error("PG_ADAM_STEP_REFUSED %s", e)
        return EXIT_REFUSED
    log.info("PG_ADAM_STEP_DONE step %d at lr %g: gradient norm %.4g, update norm %.4g; %s",
             r["step"], r["lr"], r["grad_norm"], r["update_norm"], r["checkpoint_sha256"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
