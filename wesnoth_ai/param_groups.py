"""The network's parameter groups, and the per-group norms of a training
step's gradient.

Every trained parameter belongs to one group: the first whose predicate
claims its namespaced name ("model." or "encoder."), else the trunk. The
trainers log, at every step, each group's gradient norm before the clip,
read off the clip's own computation (`GradientGroups`); the signal probes
(tools/signal_telemetry.py) and the offline profiler (signal_profiler/)
group the same way.
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


def named_as(params: Sequence["torch.nn.Parameter"], model, encoder) -> List[Tuple[str, "torch.nn.Parameter"]]:
    """`params` in their own order, each with its namespaced name: a clip
    that sums the per-tensor norms in the trainer's order keeps its total
    to the bit."""
    names = {id(p): n for n, p in named_model_parameters(model, encoder)}
    return [(names[id(p)], p) for p in params]


class GradientGroups:
    """Gradient clipping that also reads each group's gradient norm.

    `clip` is `torch.nn.utils.clip_grad_norm_` over the named parameters,
    split in two: the norm of every gradient tensor (the one pass over the
    gradients the clip makes anyway), then the scaling by the total
    (`_scale_to_bound`). Each group's norm is summed from the same
    per-tensor norms, so the reading costs no pass of its own: a sum over
    a few hundred numbers and one transfer from the device."""

    def __init__(self, named: Sequence[Tuple[str, torch.nn.Parameter]]):
        index = [group_of(name) for name, _ in named]
        present = [g for g in GROUP_ORDER if g in set(index)]
        groups = {g: i for i, g in enumerate(present)}
        self.groups: Tuple[str, ...] = tuple(present)
        self._params = [p for _, p in named]
        self._group = [groups[g] for g in index]
        self._index_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    def clip(self, max_norm: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Clip the gradients to `max_norm` (in place); the total norm before
        the clip, as clip_grad_norm_ returns it, and each group's."""
        import torch
        held = [i for i, p in enumerate(self._params) if p.grad is not None]
        if not held:
            return torch.zeros(()), {}
        with torch.no_grad():
            norms = torch.stack(torch._foreach_norm([self._params[i].grad for i in held]))
            total = torch.linalg.vector_norm(norms)
            _scale_to_bound([self._params[i].grad for i in held], max_norm, total)
            key = tuple(held)
            index = self._index_cache.get(key)
            if index is None:
                index = torch.tensor([self._group[i] for i in held], device=norms.device)
                self._index_cache[key] = index
            squares = torch.zeros(len(self.groups), device=norms.device, dtype=torch.float32)
            squares.index_add_(0, index, norms.float().pow(2))
        return total, dict(zip(self.groups, squares.sqrt().tolist()))


def _scale_to_bound(grads: List["torch.Tensor"], max_norm: float, total: "torch.Tensor") -> None:
    """Scale the gradients in place so their total norm is at most
    `max_norm`, exactly as `clip_grad_norm_` scales them after taking the
    norm (PyTorch 2.5 inlines it; 2.6 names it `clip_grads_with_norm_`,
    which the boxes' 2.5 image lacks)."""
    import torch
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    by_device: Dict["torch.device", List["torch.Tensor"]] = {}
    for g in grads:
        by_device.setdefault(g.device, []).append(g)
    for device, group in by_device.items():
        torch._foreach_mul_(group, coef.to(device))


def memory_share(norms: Dict[str, float]) -> Optional[float]:
    """The memory's share of a step's squared gradient norm (the same before
    and after the clip, which scales every group alike); None for a network
    without a memory or a zero gradient."""
    if not any(g in norms for g in MEMORY_GROUPS):
        return None
    total = sum(v * v for v in norms.values())
    if total <= 0:
        return None
    return sum(norms[g] ** 2 for g in MEMORY_GROUPS if g in norms) / total
