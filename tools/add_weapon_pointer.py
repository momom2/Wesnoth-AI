"""Give a checkpoint the pointer by which a weapon's score reads its target,
once (docs/blindspot_audit_20261125.md, gap 3; wesnoth_ai/model.py,
`weapon_target`).

A network without it scores a unit's weapon from the unit's own token, the
same whichever enemy it strikes. One with it adds a pointer's term: one
query per weapon slot from the attacker's token against one key from the
target hex's token (`weapon_q_proj`, `weapon_k_proj`). This tool adds the
two maps to a checkpoint: the key's map at zero, the query's at
`nn.Linear`'s own initial law drawn from a generator seeded with `--seed`.
With the keys at zero the term is zero, so the network computes exactly
what the source computed, until training moves the new weights. It writes
the source's fields with the weights added, `weapon_target` set to the
pointer's width, the running code's observation epoch, and
`weapon_pointer_added` naming the source and the seed.

With `--state`, the simple loop's optimizer state (tools/pg_adam_step.py)
of the source is carried over: the new parameters are named where the
network lists them, after the model's other parameters and before the
encoder's, with no moments, and the encoder's parameters keep their own
under their new indices. Adam then meets the new parameters as ones it has
never stepped and gives them its ordinary update at the loop's one learning
rate (tools/widen_unit_rows.py says why new entries in an old parameter
would not get that).

Refused (exit 3, nothing written): a checkpoint that already has the
pointer, or a state that does not continue from the checkpoint or does not
name the checkpoint's parameters.

    python tools/add_weapon_pointer.py --checkpoint pi.pt --out pi_pointer.pt \\
        [--state adam.pt --state-out adam_pointer.pt] [--seed 20261125]
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

from wesnoth_ai.checkpoint_structure import saved_structure  # noqa: E402
from wesnoth_ai.constants import OBSERVATION_EPOCH, WEAPON_TARGET_RANK  # noqa: E402

log = logging.getLogger("add_weapon_pointer")

EXIT_REFUSED = 3
SEED = 20261125
# In the checkpoint's model_state, in the order the network registers them.
NEW_WEIGHTS = ("weapon_q_proj.weight", "weapon_q_proj.bias", "weapon_k_proj.weight")
PARAMETERS = tuple("model." + n for n in NEW_WEIGHTS)        # in the optimizer state's names
PER_ACTOR_HEAD = "weapon_head.2.weight"                      # [weapon slots, d_model]


class PointerRefused(ValueError):
    """The checkpoint or the state is not what this tool extends."""


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def pointer_weights(d_model: int, slots: int, rank: int, seed: int, dtype) -> Dict[str, torch.Tensor]:
    """The two maps' first weights: the query's as `nn.Linear` draws its
    own (weight and bias uniform within 1 / sqrt(d_model)), from a
    generator seeded with `seed`; the key's zero."""
    g = torch.Generator().manual_seed(int(seed))
    bound = d_model ** -0.5
    weight = (torch.rand(slots * rank, d_model, generator=g, dtype=torch.float64) * 2 - 1) * bound
    bias = (torch.rand(slots * rank, generator=g, dtype=torch.float64) * 2 - 1) * bound
    return {NEW_WEIGHTS[0]: weight.to(dtype), NEW_WEIGHTS[1]: bias.to(dtype),
            NEW_WEIGHTS[2]: torch.zeros(rank, d_model, dtype=dtype)}


def add_to_checkpoint(ck: Dict, base_sha256: str, seed: int = SEED, rank: int = WEAPON_TARGET_RANK) -> Dict:
    """`ck` as a checkpoint whose weapon score reads the target (a new
    dict; the source's weights are shared with `ck`)."""
    if saved_structure(ck)["weapon_target"] or any(n in ck["model_state"] for n in NEW_WEIGHTS):
        raise PointerRefused("the checkpoint already has the weapon pointer")
    head = ck["model_state"][PER_ACTOR_HEAD]
    slots, d_model = int(head.shape[0]), int(head.shape[1])
    out = dict(ck)
    # The new weights go last, where the network registers them.
    out["model_state"] = {**ck["model_state"], **pointer_weights(d_model, slots, rank, seed, head.dtype)}
    out["weapon_target"] = int(rank)
    out["observation_epoch"] = int(OBSERVATION_EPOCH)
    out["weapon_pointer_added"] = {"base_sha256": base_sha256, "rank": int(rank), "seed": int(seed)}
    return out


def add_to_state(state: Dict, base_sha256: str, out_sha256: str) -> Dict:
    """The optimizer state of the source checkpoint as the new one's: the
    pointer's parameters named after the model's others, without moments
    (Adam builds them at its first step), and every later parameter's
    moments under its new index."""
    if state.get("checkpoint_sha256") != base_sha256:
        raise PointerRefused("the optimizer's state does not continue from the checkpoint")
    names: List[str] = list(state["names"])
    if any(p in names for p in PARAMETERS):
        raise PointerRefused("the optimizer's state already holds the weapon pointer")
    at = sum(1 for n in names if n.startswith("model."))
    if any(not n.startswith("model.") for n in names[:at]) or any(not n.startswith("encoder.") for n in names[at:]):
        raise PointerRefused("the optimizer's state does not name the model's parameters, then the encoder's")
    optimizer = dict(state["optimizer"])
    groups = [dict(g) for g in optimizer["param_groups"]]
    if len(groups) != 1 or list(groups[0]["params"]) != list(range(len(names))):
        raise PointerRefused("the optimizer's state is not one group over the named parameters in order")
    shift = len(PARAMETERS)
    groups[0]["params"] = list(range(len(names) + shift))
    optimizer["param_groups"] = groups
    optimizer["state"] = {(int(i) if int(i) < at else int(i) + shift): moments
                          for i, moments in state["optimizer"]["state"].items()}
    out = dict(state)
    out["names"] = names[:at] + list(PARAMETERS) + names[at:]
    out["optimizer"] = optimizer
    out["checkpoint_sha256"] = out_sha256
    return out


def add_pointer(checkpoint: Path, out: Path, state: Optional[Path] = None, state_out: Optional[Path] = None,
                seed: int = SEED) -> Dict:
    """The tool (module docstring); its record. Raises `PointerRefused`
    before anything is written."""
    from tools.eval_provenance import file_sha256
    from tools.pg_pass import load_network
    from wesnoth_ai.param_groups import named_model_parameters
    if (state is None) != (state_out is None):
        raise PointerRefused("--state and --state-out go together")
    base_sha = file_sha256(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    new = add_to_checkpoint(ck, base_sha, seed)
    saved_state = None
    if state is not None:
        saved_state = torch.load(state, map_location="cpu", weights_only=True)
        add_to_state(saved_state, base_sha, "")              # refuse before the checkpoint is written
    _atomic_save(new, out)
    out_sha = file_sha256(out)
    record = {"base_sha256": base_sha, "checkpoint_sha256": out_sha, "rank": int(new["weapon_target"]),
              "seed": int(seed), "parameters": list(PARAMETERS), "observation_epoch": int(OBSERVATION_EPOCH),
              "state": None}
    if saved_state is not None:
        new_state = add_to_state(saved_state, base_sha, out_sha)
        encoder, model = load_network(out, torch.device("cpu"))
        if new_state["names"] != [n for n, _ in named_model_parameters(model, encoder)]:
            out.unlink()
            raise PointerRefused("the optimizer's state does not name the new checkpoint's parameters in order")
        _atomic_save(new_state, state_out)
        record["state"] = {"step": int(saved_state["step"]), "sha256": file_sha256(state_out)}
    summary = out.with_name(out.stem + ".pointer.json")
    tmp = summary.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, summary)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--state-out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED, help="of the query map's first weights")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        record = add_pointer(args.checkpoint, args.out, args.state, args.state_out, args.seed)
    except PointerRefused as e:
        log.error("refused: %s", e)
        return EXIT_REFUSED
    with_state = "" if record["state"] is None else f", optimizer state at step {record['state']['step']}"
    log.info("added the weapon pointer (width %d) to %s -> %s%s", record["rank"], args.checkpoint, args.out,
             with_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
