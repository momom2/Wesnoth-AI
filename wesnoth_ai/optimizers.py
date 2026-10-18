"""The sequence trainer's optimizers (tools/sequence_train.py `--optimizer`).

Each parameter takes one role, read off its namespaced name and module
(`parameter_roles`):
  - matrix: the trunk's 2-D weights (each layer's attention in- and
    out-projections and its two feed-forward matrices), the weights Muon and
    SOAP take;
  - no_decay: every 1-D parameter (biases, LayerNorm gains and biases, the
    encoder's end-turn token), every embedding table, and the memory's
    learned initial state and slot embedding (wesnoth_ai/memory.py);
  - decay: the other weights (the heads', the encoder's projections, the
    memory's write).

The candidates:
  adamw          torch's AdamW over every parameter, weight decay on all of
                 them, betas (0.9, 0.999): the trainer as it was, to the bit;
  adamw_hygiene  AdamW, betas (0.9, 0.95), no weight decay on the no_decay
                 role;
  muon           Muon (wesnoth_ai/muon.py) on the matrix role, the query, key
                 and value of each in-projection orthogonalized apart;
                 adamw_hygiene on the rest;
  soap           SOAP (wesnoth_ai/soap.py) on the matrix role, adamw_hygiene
                 on the rest;
  schedulefree   schedule-free AdamW (A. Defazio, X. A. Yang, H. Mehta,
                 K. Mishchenko, A. Khaled and A. Cutkosky, "The Road Less
                 Scheduled", NeurIPS 2024; the `schedulefree` package, 1.4.1)
                 with adamw_hygiene's betas and roles.

How each meets the trainer's schedule: the gradient clip scales the
gradient every candidate reads; the learning rate the trainer sets on every
group each step (its warm-up, its cooldown) is every candidate's rate, Muon's
included (its RMS matching makes AdamW's rate its own). Schedule-free
replaces the schedule after the warm-up: its iterate average does what a
cooldown does, so it takes no `--decay-from` and no anneal rule; the
trainer's linear warm-up is the package's own (`warmup_steps` left at 0, the
same ramp set from outside), and the package weighs each step of its average
by the square of the largest rate so far, so the warm-up's steps count
little. Its weights are read at its average x, not at the point y its
gradients are taken: `evaluation_point` puts x in the parameters for the
probe and the checkpoint, which holds x as its weights and y beside the
optimizer's state, so a resume continues from y exactly.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from wesnoth_ai.muon import MOMENTUM, NS_STEPS, Muon
from wesnoth_ai.param_groups import group_of
from wesnoth_ai.soap import BETAS as SOAP_BETAS
from wesnoth_ai.soap import PRECONDITION_EVERY, SOAP

OPTIMIZERS = ("adamw", "adamw_hygiene", "muon", "soap", "schedulefree")
HYGIENE_BETAS = (0.9, 0.95)
MATRIX, DECAY, NO_DECAY = "matrix", "decay", "no_decay"
# The memory's state rows: its learned initial memory and its slot embedding.
MEMORY_ROWS = ("model.slot_memory.initial", "model.slot_memory.slot_embed.weight")
# An attention in-projection stacks the query, key and value matrices.
IN_PROJECTION = "self_attn.in_proj_weight"
IN_PROJECTION_BLOCKS = 3
NamedParams = Sequence[Tuple[str, torch.nn.Parameter]]


@dataclass(frozen=True)
class OptimizerChoice:
    """The optimizer a run trains with and its own settings (the training
    meta records `describe()`, and a resume must match it)."""
    name: str = "adamw"
    precondition_every: Optional[int] = None     # soap: None reads PRECONDITION_EVERY

    def describe(self) -> Dict:
        """{"name": ...} and the settings that make this optimizer what it is."""
        if self.name == "adamw":
            return {"name": "adamw"}
        out = {"name": self.name, "betas": list(HYGIENE_BETAS)}
        if self.name == "muon":
            out.update(momentum=MOMENTUM, nesterov=True, ns_steps=NS_STEPS, in_projection_blocks=IN_PROJECTION_BLOCKS)
        if self.name == "soap":
            out.update(soap_betas=list(SOAP_BETAS), precondition_every=self.precondition_every or PRECONDITION_EVERY)
        return out


def embedding_names(modules: Dict[str, torch.nn.Module]) -> set:
    """The namespaced names of every embedding table's weight."""
    return {f"{space}.{name}.weight" for space, module in modules.items()
            for name, m in module.named_modules() if isinstance(m, torch.nn.Embedding)}


def parameter_roles(named: NamedParams, modules: Dict[str, torch.nn.Module]) -> List[str]:
    """Each parameter's role (module docstring), in `named`'s order."""
    embeddings = embedding_names(modules)
    roles = []
    for name, p in named:
        if p.dim() < 2 or name in embeddings or name in MEMORY_ROWS:
            roles.append(NO_DECAY)
        elif p.dim() == 2 and group_of(name) == "trunk":
            roles.append(MATRIX)
        else:
            roles.append(DECAY)
    return roles


def _hygiene_groups(named: NamedParams, roles: Sequence[str], weight_decay: float,
                    take: Sequence[str]) -> List[Dict]:
    """AdamW's groups of the parameters whose role is in `take`: decay and
    no_decay (the matrix role decays), in the trainer's order."""
    decayed = [p for (_, p), r in zip(named, roles) if r in take and r != NO_DECAY]
    kept = [p for (_, p), r in zip(named, roles) if r in take and r == NO_DECAY]
    return [g for g in ({"params": decayed, "weight_decay": weight_decay, "role": "decay"},
                        {"params": kept, "weight_decay": 0.0, "role": "no_decay"}) if g["params"]]


def build_optimizer(choice: OptimizerChoice, named: NamedParams, modules: Dict[str, torch.nn.Module],
                    lr: float, weight_decay: float):
    """The optimizer of `choice` over `named` (the trainer's parameters in
    its order; `modules` {"model": ..., "encoder": ...} name the embeddings)."""
    params = [p for _, p in named]
    if choice.name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    roles = parameter_roles(named, modules)
    if choice.name == "adamw_hygiene":
        return torch.optim.AdamW(_hygiene_groups(named, roles, weight_decay, (MATRIX, DECAY, NO_DECAY)),
                                 lr=lr, betas=HYGIENE_BETAS)
    if choice.name == "schedulefree":
        return ScheduleFreeAdamW(_hygiene_groups(named, roles, weight_decay, (MATRIX, DECAY, NO_DECAY)), lr)
    rest = torch.optim.AdamW(_hygiene_groups(named, roles, weight_decay, (DECAY, NO_DECAY)), lr=lr,
                             betas=HYGIENE_BETAS)
    matrices = [(n, p) for (n, p), r in zip(named, roles) if r == MATRIX]
    if choice.name == "muon":
        split = [p for n, p in matrices if n.endswith(IN_PROJECTION)]
        whole = [p for n, p in matrices if not n.endswith(IN_PROJECTION)]
        groups = [g for g in ({"params": split, "blocks": IN_PROJECTION_BLOCKS}, {"params": whole, "blocks": 1})
                  if g["params"]]
        return CompositeOptimizer({"muon": Muon(groups, lr=lr, weight_decay=weight_decay), "adamw": rest})
    if choice.name == "soap":
        every = choice.precondition_every or PRECONDITION_EVERY
        soap = SOAP([p for _, p in matrices], lr=lr, weight_decay=weight_decay, precondition_every=every)
        return CompositeOptimizer({"soap": soap, "adamw": rest})
    raise ValueError(f"unknown optimizer {choice.name!r}: one of {', '.join(OPTIMIZERS)}")


class CompositeOptimizer:
    """Optimizers over disjoint parameters stepped as one: their groups in
    one list (the trainer sets every group's rate), their states in one
    mapping (the telemetry reads a parameter's), their state dicts by name."""

    def __init__(self, parts: Dict[str, torch.optim.Optimizer]):
        self.parts = dict(parts)

    @property
    def param_groups(self) -> List[Dict]:
        return [g for opt in self.parts.values() for g in opt.param_groups]

    @property
    def state(self) -> "MergedState":
        return MergedState([opt.state for opt in self.parts.values()])

    def step(self) -> None:
        for opt in self.parts.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.parts.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict:
        return {name: opt.state_dict() for name, opt in self.parts.items()}

    def load_state_dict(self, state: Dict) -> None:
        if set(state) != set(self.parts):
            raise ValueError(f"a state of the parts {sorted(state)} for an optimizer of {sorted(self.parts)}")
        for name, opt in self.parts.items():
            opt.load_state_dict(state[name])


class MergedState(Mapping):
    """The states of disjoint optimizers read as one mapping, without the
    entry an optimizer's defaultdict would create for a parameter it does
    not hold."""

    def __init__(self, states: Sequence[Mapping]):
        self.states = list(states)

    def __getitem__(self, key):
        for state in self.states:
            if key in state:
                return state[key]
        raise KeyError(key)

    def __iter__(self):
        for state in self.states:
            yield from state

    def __len__(self) -> int:
        return sum(len(state) for state in self.states)


class ScheduleFreeAdamW:
    """`schedulefree.AdamWScheduleFree` kept at its training point y, except
    inside `evaluation_point()`, which puts its average x in the parameters
    and restores y exactly after (the package's own `eval()` then `train()`
    is a round trip of two interpolations, exact only up to rounding)."""

    def __init__(self, groups: List[Dict], lr: float):
        import schedulefree
        self.inner = schedulefree.AdamWScheduleFree(groups, lr=lr, betas=HYGIENE_BETAS, warmup_steps=0)
        self.inner.train()
        self._params = [p for g in self.inner.param_groups for p in g["params"]]
        self._kept: Optional[List[torch.Tensor]] = None

    @property
    def param_groups(self) -> List[Dict]:
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state

    def step(self) -> None:
        if self._kept is not None:
            raise RuntimeError("a schedule-free step at the evaluation point")
        self.inner.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    @contextmanager
    def evaluation_point(self):
        if self._kept is not None:
            yield
            return
        with torch.no_grad():
            self._kept = [p.detach().clone() for p in self._params]
            self.inner.eval()
        try:
            yield
        finally:
            with torch.no_grad():
                self.inner.train()
                for p, y in zip(self._params, self._kept):
                    p.copy_(y)
            self._kept = None

    def state_dict(self) -> Dict:
        """The package's state and the training point y (the parameters,
        or what `evaluation_point` keeps while it holds x)."""
        y = self._kept if self._kept is not None else [p.detach() for p in self._params]
        return {"schedulefree": self.inner.state_dict(), "train_point": list(y)}

    def load_state_dict(self, state: Dict) -> None:
        """The package's state, then y in the parameters (which a checkpoint's
        weights left at x)."""
        self.inner.load_state_dict(state["schedulefree"])
        with torch.no_grad():
            self.inner.train()
            for p, y in zip(self._params, state["train_point"]):
                p.copy_(y)


def evaluation_point(opt):
    """The block in which the parameters hold the weights an evaluation and
    a checkpoint read: schedule-free's average, every other candidate's
    parameters as they are."""
    return opt.evaluation_point() if isinstance(opt, ScheduleFreeAdamW) else nullcontext()


def state_bytes(opt) -> int:
    """The bytes of an optimizer's state tensors."""
    return sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))

