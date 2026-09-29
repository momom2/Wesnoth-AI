"""Hand-built RawEncoded records and closed-form weights for the model tests.

`hand_raw` builds one decision's RawEncoded from a seed, at obs8's widths or
at the parity observation's (docs/parity_memory_design_20260929.md), so the
embedding and trunk paths can be driven without a simulator or a Rust
wheel. `fill_closed_form` sets every parameter of a module from a formula of
its name, so a stored reference output depends on the forward math alone,
not on torch's random initialization. `legacy_reference` computes the
reference that tests/data/legacy_model_reference.json pins: obs8's
architecture at a tiny width, its state_dict keys, its outputs on three
records through the single-sample, padded and packed paths, and the
parameters a seeded construction draws.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional

import numpy as np
import torch

from wesnoth_ai.classes import Position
from wesnoth_ai.encoder import (
    GLOBAL_FEAT_DIM, GLOBAL_FEAT_DIM_PARITY, MAX_FACTIONS, NUM_HEX_DYNAMIC_FLAGS,
    NUM_HEX_DYNAMIC_FLAGS_PARITY, NUM_HEX_MODIFIERS, NUM_TERRAINS, NUM_TERRAINS_PARITY,
    SIGHT_FEAT_DIM, UNIT_FEAT_DIM, UNIT_FEAT_DIM_PARITY, RawEncoded,
)

D, LAYERS, HEADS, FF = 16, 1, 2, 32
OUTPUT_FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
                 "value_logits")
CONSTRUCTION_SEED = 20260929
N_TYPES = 40          # unit type ids drawn below this


def hand_raw(seed: int, U: int, R: int, H: int, *, parity: bool = False, S: int = 0,
             faction_probs: Optional[np.ndarray] = None) -> RawEncoded:
    """One decision's record: H hexes, U units, R recruits and, with
    `parity`, S sightings and a faction posterior (uniform over the six
    default factions unless `faction_probs` is given). Terrain ids are
    multi-hot masks, as an encoder with `terrain_multi_hot` reads them."""
    g = np.random.default_rng(seed)
    unit_dim = UNIT_FEAT_DIM_PARITY if parity else UNIT_FEAT_DIM
    dyn_dim = NUM_HEX_DYNAMIC_FLAGS_PARITY if parity else NUM_HEX_DYNAMIC_FLAGS
    glob_dim = GLOBAL_FEAT_DIM_PARITY if parity else GLOBAL_FEAT_DIM
    terrains = NUM_TERRAINS_PARITY if parity else NUM_TERRAINS

    def ints(n, hi):
        return g.integers(0, hi, size=n).astype(np.int64)

    def floats(*shape):
        return g.random(shape).astype(np.float32)

    hex_xs, hex_ys = ints(H, 30), ints(H, 30)
    unit_xs, unit_ys = ints(U, 30), ints(U, 30)
    raw = RawEncoded(
        hex_positions=[Position(int(x), int(y)) for x, y in zip(hex_xs, hex_ys)],
        hex_xs=hex_xs, hex_ys=hex_ys,
        hex_terrain_ids=g.integers(1, 2 ** terrains, size=H).astype(np.int64),
        hex_modifier_flags=(floats(H, NUM_HEX_MODIFIERS) > 0.7).astype(np.float32),
        hex_dynamic_flags=(floats(H, dyn_dim) > 0.6).astype(np.float32),
        unit_positions=[Position(int(x), int(y)) for x, y in zip(unit_xs, unit_ys)],
        unit_ids=[f"u{seed}_{i}" for i in range(U)],
        unit_is_ours=(floats(U) > 0.5).astype(np.float32),
        unit_type_ids=ints(U, N_TYPES), unit_side_ids=ints(U, 3),
        unit_xs=unit_xs, unit_ys=unit_ys, unit_feats=floats(U, unit_dim),
        recruit_types=[f"type{i}" for i in range(R)],
        recruit_is_ours=np.ones(R, dtype=np.float32),
        recruit_type_ids=ints(R, N_TYPES), recruit_side_ids=np.zeros(R, dtype=np.int64),
        recruit_xs=ints(R, 30), recruit_ys=ints(R, 30), recruit_feats=floats(R, unit_dim),
        global_feats=floats(glob_dim), our_faction_id=int(g.integers(0, 7)),
        their_faction_id=int(g.integers(0, 7)), material=float(g.random()),
    )
    if parity:
        if faction_probs is None:
            faction_probs = np.zeros(MAX_FACTIONS, dtype=np.float32)
            faction_probs[1:7] = 1.0 / 6.0
        raw.their_faction_probs = np.asarray(faction_probs, dtype=np.float32)
        raw.sight_type_ids = ints(S, N_TYPES)
        raw.sight_xs, raw.sight_ys = ints(S, 30), ints(S, 30)
        raw.sight_feats = floats(S, SIGHT_FEAT_DIM)
    return raw


def fill_closed_form(*modules: torch.nn.Module) -> None:
    """Every parameter from a formula of its name and index: a sine over
    the elements with a phase from the name's CRC, scaled by the fan-in
    for matrices. Buffers are left alone."""
    with torch.no_grad():
        for module in modules:
            for name, p in module.named_parameters():
                n = p.numel()
                phase = (zlib.crc32(name.encode()) % 997) / 97.0
                scale = 1.0 / math.sqrt(p.shape[-1]) if p.dim() >= 2 else 0.5
                values = scale * torch.sin(0.61 * torch.arange(n, dtype=torch.float64) + phase)
                p.copy_(values.to(p.dtype).view_as(p))


def legacy_records() -> List[RawEncoded]:
    """Three obs8 records of different sizes, one without recruits."""
    return [hand_raw(1, U=3, R=2, H=9), hand_raw(2, U=1, R=0, H=5),
            hand_raw(3, U=4, R=1, H=12)]


def _output_lists(out) -> Dict[str, list]:
    return {f: getattr(out, f).detach().double().tolist() for f in OUTPUT_FIELDS}


def legacy_pair():
    """obs8's encoder and model at the tiny width, closed-form weights."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    encoder = GameStateEncoder(d_model=D, terrain_multi_hot=True).eval()
    model = WesnothModel(d_model=D, num_layers=LAYERS, num_heads=HEADS, d_ff=FF).eval()
    fill_closed_form(encoder, model)
    return encoder, model


def construction_fingerprint() -> Dict[str, List[float]]:
    """(sum, sum of squares) of every parameter a seeded construction of
    obs8's encoder then model draws."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(CONSTRUCTION_SEED)
    encoder = GameStateEncoder(d_model=D, terrain_multi_hot=True)
    model = WesnothModel(d_model=D, num_layers=LAYERS, num_heads=HEADS, d_ff=FF)
    out = {}
    for prefix, module in (("encoder.", encoder), ("model.", model)):
        for name, p in module.named_parameters():
            t = p.detach().double()
            out[prefix + name] = [float(t.sum()), float((t * t).sum())]
    return out


def legacy_reference() -> Dict[str, object]:
    """Everything tests/data/legacy_model_reference.json records."""
    encoder, model = legacy_pair()
    raws = legacy_records()
    with torch.no_grad():
        singles = [_output_lists(model(encoder.encode_from_raw(r))) for r in raws]
        padded = model.forward_embedded(encoder.encode_from_raw_embedded(raws), packed=False)
        packed = model.forward_streams(*encoder.encode_from_raw_padded(raws), packed=True)
    return {
        "dims": [D, LAYERS, HEADS, FF],
        "model_keys": {k: list(v.shape) for k, v in model.state_dict().items()},
        "encoder_keys": {k: list(v.shape) for k, v in encoder.state_dict().items()},
        "single": singles,
        "padded": [_output_lists(o) for o in padded.samples()],
        "packed": [_output_lists(o) for o in packed.samples()],
        "construction": construction_fingerprint(),
    }
