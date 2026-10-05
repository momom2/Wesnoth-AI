"""The network's parameter groups, and the per-group norms of what a
training step applies.

Every trained parameter belongs to one group: the first whose predicate
claims its namespaced name ("model." or "encoder."), else the trunk. The
trainers log, at every step, each group's gradient norm before the clip
and the norm of the update the optimizer made (`StepNorms`), read off the
step itself; the signal probes (tools/signal_telemetry.py) and the offline
profiler (signal_profiler/) group the same way.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:                     # torch is imported where it is used: the
    import torch                      # loops read the group names without it

PARAM_GROUPS = (
    ("encoder",        lambda n: n.startswith("encoder.")),
    ("value_head",     lambda n: n.startswith(("model.value_head", "model.material_proj"))),
    ("actor_head",     lambda n: n.startswith("model.actor_head")),
    ("type_head",      lambda n: n.startswith("model.type_head")),
    ("target_proj",    lambda n: n.startswith("model.target_")),
    ("weapon_head",    lambda n: n.startswith("model.weapon_head")),
    ("gbc_heads",      lambda n: n.startswith("model.gbc_heads")),
    ("aux_ml",         lambda n: n.startswith(("model.aux_score_head", "model.moves_left"))),
    ("belief_head",    lambda n: n.startswith("model.belief_head")),
    # The learned memory (wesnoth_ai/memory.py): its initial state, the
    # embedding of each slot it reads, and its gated write.
    ("memory_initial", lambda n: n.startswith("model.slot_memory.initial")),
    ("memory_slots",   lambda n: n.startswith("model.slot_memory.slot_embed")),
    ("memory_write",   lambda n: n.startswith(("model.slot_memory.gate", "model.slot_memory.candidate"))),
)
MEMORY_GROUPS = ("memory_initial", "memory_slots", "memory_write")
# The order the logs list the groups in.
GROUP_ORDER = ("encoder", "trunk", "actor_head", "type_head", "target_proj", "weapon_head", "value_head",
               "belief_head", "gbc_heads", "aux_ml") + MEMORY_GROUPS


def group_of(name: str) -> str:
    """The group of a namespaced parameter name."""
    for group, claims in PARAM_GROUPS:
        if claims(name):
            return group
    return "trunk"


def named_model_parameters(model: torch.nn.Module,
                           encoder: Optional[torch.nn.Module]) -> List[Tuple[str, torch.nn.Parameter]]:
    """The trained parameters, namespaced as PARAM_GROUPS expects."""
    named = [("model." + n, p) for n, p in model.named_parameters()]
    if encoder is not None:
        named += [("encoder." + n, p) for n, p in encoder.named_parameters()]
    return [(n, p) for n, p in named if p.requires_grad]


class StepNorms:
    """Per group, the L2 norm of a step's gradient (`gradients`, after the
    backward passes and before the clip) and of its update (`updates`, the
    parameters' change across the optimizer step that `before_update`
    opens). Each reading is one transfer from the device."""

    def __init__(self, named: Sequence[Tuple[str, torch.nn.Parameter]]):
        groups: Dict[str, List[torch.nn.Parameter]] = {}
        for name, p in named:
            groups.setdefault(group_of(name), []).append(p)
        self.groups = {g: groups[g] for g in GROUP_ORDER if g in groups}
        self._before: Optional[Dict[int, torch.Tensor]] = None

    def gradients(self) -> Dict[str, float]:
        return self._norms(lambda p: None if p.grad is None else p.grad.detach())

    def before_update(self) -> None:
        self._before = {id(p): p.detach().clone() for ps in self.groups.values() for p in ps}

    def updates(self) -> Dict[str, float]:
        before, self._before = self._before, None
        if before is None:
            raise RuntimeError("StepNorms.updates without before_update")
        return self._norms(lambda p: p.detach() - before[id(p)])

    def _norms(self, tensor_of) -> Dict[str, float]:
        import torch
        sums = []
        for params in self.groups.values():
            parts = [t.float().pow(2).sum() for t in map(tensor_of, params) if t is not None]
            sums.append(torch.stack(parts).sum() if parts else torch.zeros((), device=params[0].device))
        return dict(zip(self.groups, torch.stack(sums).sqrt().tolist()))


def memory_share(norms: Dict[str, float]) -> Optional[float]:
    """The memory's share of a step's squared norm (its gradient's or its
    update's); None for a network without a memory or a zero step."""
    if not any(g in norms for g in MEMORY_GROUPS):
        return None
    total = sum(v * v for v in norms.values())
    if total <= 0:
        return None
    return sum(norms[g] ** 2 for g in MEMORY_GROUPS if g in norms) / total
