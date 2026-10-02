"""The parity-memory recipe's structural checkpoint keys
(docs/parity_memory_design_20260929.md, "Model interface").

Three top-level keys, beside `relevant_set_hexes` and the other flags:

- `observation_parity` (bool): the encoder reads the parity observation and
  the network has its sighting stream and belief head. Sizes parameters.
- `memory_slots` (int): the learned memory's slot count, 0 for none. Sizes
  parameters.
- `relevant_set_version` (int): which relevant set the hex stream holds.
  Data flow, like the fog gate.

A checkpoint without them is obs8's lineage: False, 0 and 1.
"""
from __future__ import annotations

from typing import Dict

STRUCTURE_DEFAULTS: Dict[str, object] = {
    "observation_parity": False,
    "memory_slots": 0,
    "relevant_set_version": 1,
}


def checkpoint_structure(model, encoder) -> Dict[str, object]:
    """The keys a checkpoint of this model and encoder records."""
    if bool(model.observation_parity) != bool(encoder.observation_parity):
        raise ValueError(f"the model (observation_parity={model.observation_parity}) and the "
                         f"encoder (observation_parity={encoder.observation_parity}) read "
                         f"different observations")
    return {"observation_parity": bool(encoder.observation_parity),
            "memory_slots": int(model.memory_slots),
            "relevant_set_version": int(encoder.relevant_set_version)}


def saved_structure(ckpt: Dict) -> Dict[str, object]:
    """The keys a checkpoint records, absent ones at obs8's values."""
    return {key: type(default)(ckpt.get(key, default))
            for key, default in STRUCTURE_DEFAULTS.items()}


def refuse_other_structure(ckpt: Dict, model, encoder, name: str) -> None:
    """Raise when the checkpoint's parameter-sizing keys differ from the
    modules it is loaded into: they cannot be reconciled by a partial load
    (build the policy with the checkpoint's own, as
    tools/eval_players.peek_checkpoint_arch does)."""
    saved = saved_structure(ckpt)
    built = checkpoint_structure(model, encoder)
    for key in ("observation_parity", "memory_slots"):
        if saved[key] != built[key]:
            raise RuntimeError(f"{name} was trained with {key}={saved[key]!r}; this policy is "
                               f"built with {built[key]!r}. Rebuild it with the checkpoint's own.")
