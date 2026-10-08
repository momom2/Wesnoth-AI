"""The step-1 critic (docs/selfplay_program_20261008.md, "Step 1"): the
reference's network read for its value only.

`WesnothModel(observation_parity=True, memory_slots=0, aux_score=True)` with
the parity recipe's encoder (`wesnoth_ai.critic_data.ENCODING`). Two heads
train, nothing else does:

  value  the C51 head on the game's outcome signed to the side to move, by
         the trainer's categorical loss (`trainer._categorical_value_loss`);
  aux    the network's `aux_score_head` (one linear unit) on the mover's HP
         margin at its next turn start, divided by AUX_SCALE, by squared
         error over the positions that have one.

The aux head reads the global token the value head reads, with its
gradient: the critic has no policy for an auxiliary signal to disturb, so
the 2026-09-01 detach (the model's own `aux_score` output reads a detached
token) does not apply here, and an aux head that could not reach the trunk
would leave the critic's value unchanged. The token is taken off the value
head's input by a forward hook, as tools/turn_value.py on branch
exp/turn-value read it. The policy and belief heads still run in the
forward and get no loss; `trained_parameters` leaves them out of the
optimizer.

A full critic starts from the reference (`load_reference`): its encoder
whole, its network without the memory, which a critic does not have; the
small critic starts from scratch at SMALL_ARCH.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from wesnoth_ai.param_groups import group_of

# The reference's architecture (tools/sequence_train.ARCH), and a quarter of
# its width and depth with the same width per attention head.
FULL_ARCH = {"d_model": 384, "num_layers": 8, "num_heads": 12, "d_ff": 1536}
SMALL_ARCH = {"d_model": 96, "num_layers": 2, "num_heads": 3, "d_ff": 384}
# HP per unit of the aux head's target: a mid-game margin is tens of HP, a
# decided game a few hundred, so the target sits within a few units of 0.
AUX_SCALE = 100.0
# The groups the critic trains (wesnoth_ai.param_groups); "aux_ml" holds the
# aux head.
TRAINED_GROUPS = ("encoder", "trunk", "value_head", "aux_ml")
# The memory's parameters, which a full critic drops from the reference.
MEMORY_PREFIX = "slot_memory."
AUX_KEYS = ("aux_score_head.weight", "aux_score_head.bias")


def build_critic(arch: Dict[str, int], type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                 device: torch.device):
    """(encoder, model) of a critic, untrained."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    encoder = GameStateEncoder(d_model=arch["d_model"], relevant_set_hexes=True,
                               fog_hides_enemy_villages=True, terrain_multi_hot=True,
                               observation_parity=True, relevant_set_version=2,
                               unit_type_to_id=dict(type_to_id), faction_to_id=dict(faction_to_id)).to(device)
    encoder.freeze_vocab()
    model = WesnothModel(observation_parity=True, memory_slots=0, aux_score=True, **arch).to(device)
    return encoder, model


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_vocab(path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return dict(ck["unit_type_to_id"]), dict(ck["faction_to_id"])


def load_reference(encoder, model, path: Path) -> Dict[str, object]:
    """Load a reference checkpoint (parity3's recipe) into a full critic:
    the encoder whole, the network with `strict=False`. Returns exactly
    what the partial load left out (keys missing from the checkpoint, keys
    the critic has no place for) and the file's SHA-256. Anything beyond
    the aux head missing and the memory unexpected raises: a critic must
    not start from a half-loaded network under the reference's name."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if dict(ck.get("unit_type_to_id", {})) != dict(encoder.unit_type_to_id) \
            or dict(ck.get("faction_to_id", {})) != dict(encoder.faction_to_id):
        raise ValueError(f"{path}: its vocabularies differ from the critic's")
    if not ck.get("observation_parity") or int(ck.get("relevant_set_version", 1)) != 2:
        raise ValueError(f"{path} is not a parity-recipe checkpoint")
    arch = ck.get("arch")
    if arch is not None and {k: int(arch[k]) for k in FULL_ARCH} != {k: model_arch(model)[k] for k in FULL_ARCH}:
        raise ValueError(f"{path} has architecture {arch}; the critic has {model_arch(model)}")
    encoder.load_state_dict(ck["encoder_state"], strict=True)
    result = model.load_state_dict(ck["model_state"], strict=False)
    missing, unexpected = sorted(result.missing_keys), sorted(result.unexpected_keys)
    if set(missing) - set(AUX_KEYS) or any(not k.startswith(MEMORY_PREFIX) for k in unexpected):
        raise ValueError(f"{path}: an unexpected partial load (missing {missing}, unexpected {unexpected})")
    return {"path": str(path), "sha256": file_sha256(Path(path)), "missing": missing,
            "unexpected": unexpected}


def model_arch(model) -> Dict[str, int]:
    layer = model.encoder.layers[0]
    return {"d_model": int(model.d_model), "num_layers": len(model.encoder.layers),
            "num_heads": int(layer.self_attn.num_heads), "d_ff": int(layer.linear1.out_features)}


def trained_parameters(model, encoder) -> List[Tuple[str, torch.nn.Parameter]]:
    """The critic's trained parameters, namespaced as wesnoth_ai.param_groups
    names them: the encoder, the trunk, the value head and the aux head."""
    from wesnoth_ai.param_groups import named_model_parameters
    return [(n, p) for n, p in named_model_parameters(model, encoder) if group_of(n) in TRAINED_GROUPS]


# ---------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------

def critic_forward(encoder, model, raws: Sequence, device: torch.device,
                   autocast: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(value logits [B, K], aux prediction [B]) of a batch of encodings,
    both float32."""
    captured: List[torch.Tensor] = []
    hook = model.value_head.register_forward_hook(lambda _m, inp, _out: captured.append(inp[0]))
    try:
        with torch.autocast(device.type, dtype=autocast or torch.bfloat16, enabled=autocast is not None):
            streams = encoder.embed_staged(encoder.stage_raws(list(raws), device=device))
            out = model.forward_embedded(streams)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape[0] != len(raws):
        raise RuntimeError(f"captured {len(captured)} global batches for {len(raws)} positions")
    aux = model.aux_score_head(captured[0].float()).squeeze(-1)
    return out.value_logits.float(), aux


def expected_value(model, value_logits: torch.Tensor) -> torch.Tensor:
    """The value head's mean over its atoms, [B]."""
    return (torch.softmax(value_logits.float(), dim=-1) * model._value_atoms).sum(dim=-1)


def critic_losses(model, value_logits: torch.Tensor, aux_pred: torch.Tensor, z: torch.Tensor,
                  aux_target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The batch-mean value cross-entropy and the mean squared aux error
    over the positions with an aux target (0 when none has one)."""
    from wesnoth_ai.trainer import _categorical_value_loss
    value = _categorical_value_loss(value_logits, z, model._value_atoms) / max(1, z.numel())
    have = torch.isfinite(aux_target)
    target = torch.where(have, aux_target / AUX_SCALE, torch.zeros_like(aux_target))
    squares = torch.where(have, (aux_pred - target) ** 2, torch.zeros_like(aux_pred))
    aux = squares.sum() / max(1, int(have.sum()))
    return {"value": value, "aux": aux}


@torch.no_grad()
def critic_values(encoder, model, raws: Sequence, device: torch.device, batch: int = 64,
                  autocast: Optional[torch.dtype] = None) -> List[float]:
    """The critic's expected value of each encoding, for its side to move."""
    model.eval()
    encoder.eval()
    out: List[float] = []
    for start in range(0, len(raws), batch):
        logits, _aux = critic_forward(encoder, model, raws[start:start + batch], device, autocast)
        out.extend(float(v) for v in expected_value(model, logits).cpu())
    return out


def load_critic(path: Path, device: torch.device):
    """(encoder, model, checkpoint metadata) of a critic checkpoint written
    by tools/critic_train.py."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    encoder, model = build_critic(dict(ck["arch"]), ck["unit_type_to_id"], ck["faction_to_id"], device)
    encoder.load_state_dict(ck["encoder_state"], strict=True)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    encoder.eval()
    meta = {k: v for k, v in ck.items() if k not in ("model_state", "encoder_state", "optimizer_state")}
    return encoder, model, meta


def nan_to_none(x: float) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else float(x)
