"""Encode a GameState into torch tensors for the policy/value network.

Five token streams, all projected to a common ``d_model`` so a
downstream transformer can attend across them uniformly:

- **hexes**   — one per on-board map hex (terrain + modifiers + pos)
- **units**   — one per unit visible to the acting side
- **recruits**— one per (side, recruit-type) offer
- **global**  — a single token summarizing turn/gold/villages
- **end_turn**— a single learned-parameter sentinel

The **acting side** (``current_side``) is the frame of reference: the
same raw GameState encodes differently depending on whose turn it is.
"ours" vs "theirs" is resolved here; the model downstream doesn't need
to know about side IDs.

No batching for Phase 3.1. Everything has a leading batch-dim of 1.
Phase 3.2 will pad and batch when the trainer needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from classes import (
    Alignment,
    GameState,
    Position,
    Terrain,
    TerrainModifiers,
    Unit,
)


# ---------------------------------------------------------------------
# Sizes of the embedding tables. Generous defaults; tune if needed.
# Changing any of these requires a retrain (embedding shapes change).
# ---------------------------------------------------------------------

MAX_MAP_SIZE    = 128   # covers every mainline MP map with headroom
MAX_UNIT_TYPES  = 200   # # distinct unit type names we can embed
NUM_TERRAINS    = max(Terrain) + 1       # 14 enum values today
NUM_ALIGNMENTS  = max(Alignment) + 1     # 4
NUM_SIDE_CODES  = 2     # 0 = ours (current_side), 1 = theirs

# Per-hex multi-hot: (village, keep, castle).
NUM_HEX_MODIFIERS = 3

# Per-unit numerical features. Order MATTERS — changing it requires
# a retrain (the Linear weights are positional).
#   0: max_hp / HP_NORM
#   1: current_hp / max_hp
#   2: max_moves / MOVES_NORM
#   3: current_moves / max(max_moves, 1)
#   4: max_exp / EXP_NORM
#   5: current_exp / max(max_exp, 1)
#   6: cost / COST_NORM
#   7: is_leader flag
#   8: has_attacked flag
UNIT_NUMERIC_FEATS    = 9
UNIT_ALIGNMENT_ONEHOT = NUM_ALIGNMENTS       # 4 one-hot
UNIT_FEAT_DIM         = UNIT_NUMERIC_FEATS + UNIT_ALIGNMENT_ONEHOT  # 13

# Global features:
#   0: turn / TURN_NORM
#   1: current_side normalized to [-1, 1]
#   2: our_gold / GOLD_NORM
#   3: our_base_income / INCOME_NORM
#   4: our_villages / VILLAGES_NORM
#   5: their_villages / VILLAGES_NORM
GLOBAL_FEAT_DIM = 6

# Normalization divisors. Rough upper bounds for each quantity.
HP_NORM       = 80.0
MOVES_NORM    = 10.0
EXP_NORM      = 150.0
COST_NORM     = 80.0
GOLD_NORM     = 500.0
INCOME_NORM   = 50.0
VILLAGES_NORM = 30.0
TURN_NORM     = 60.0


@dataclass
class EncodedState:
    """Tensors + side-info a model and action sampler need.

    All tensors have a leading batch dim = 1. The ``*_positions`` /
    ``*_ids`` / ``*_types`` plain-Python lists are parallel to the
    seq dim of their token tensors — used to translate model output
    indices back into game-space actions.
    """

    hex_tokens:     torch.Tensor           # [1, H, d_model]
    hex_positions:  List[Position]         # len H
    # Note: we only emit on-board hexes, so no hex_mask is needed in
    # Phase 3.1. When Phase 3.2 pads to a fixed H across a batch, add
    # a mask.

    unit_tokens:    torch.Tensor           # [1, U, d_model]
    unit_is_ours:   torch.Tensor           # [1, U] float {0.0, 1.0}
    unit_positions: List[Position]         # len U
    unit_ids:       List[str]              # len U — Wesnoth's unit.id

    recruit_tokens: torch.Tensor           # [1, R, d_model]
    recruit_is_ours: torch.Tensor          # [1, R] float
    recruit_types:  List[str]              # len R — e.g., "Dwarvish Fighter"

    global_token:   torch.Tensor           # [1, 1, d_model]
    end_turn_token: torch.Tensor           # [1, 1, d_model]


class GameStateEncoder(nn.Module):
    """Learned embedder: GameState → EncodedState."""

    def __init__(
        self,
        d_model: int = 128,
        unit_type_to_id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Sharing this dict with StateConverter keeps Unit.name_id and
        # recruit-type indices in lockstep — otherwise a unit known to
        # the converter as id=3 would get a different id here and hit
        # a different row of unit_type_embed than its recruit entry.
        self.unit_type_to_id: Dict[str, int] = (
            unit_type_to_id if unit_type_to_id is not None else {}
        )

        # --- hex embeddings -------------------------------------------
        self.terrain_embed  = nn.Embedding(NUM_TERRAINS, d_model)
        self.modifier_proj  = nn.Linear(NUM_HEX_MODIFIERS, d_model, bias=False)

        # --- shared position embeddings --------------------------------
        self.pos_x_embed = nn.Embedding(MAX_MAP_SIZE, d_model)
        self.pos_y_embed = nn.Embedding(MAX_MAP_SIZE, d_model)

        # --- unit embeddings ------------------------------------------
        self.unit_type_embed = nn.Embedding(MAX_UNIT_TYPES, d_model)
        self.unit_feat_proj  = nn.Linear(UNIT_FEAT_DIM, d_model)
        self.side_embed      = nn.Embedding(NUM_SIDE_CODES, d_model)

        # --- global ---------------------------------------------------
        self.global_proj = nn.Linear(GLOBAL_FEAT_DIM, d_model)

        # --- end_turn sentinel ----------------------------------------
        # Small init so it doesn't dominate the softmax at step 0.
        self.end_turn_token = nn.Parameter(torch.randn(d_model) * 0.02)

    # ----- public API --------------------------------------------------

    def encode(self, game_state: GameState) -> EncodedState:
        """Build an EncodedState for one GameState."""
        device = next(self.parameters()).device
        current_side = game_state.global_info.current_side

        hex_tokens, hex_positions = self._encode_hexes(game_state, device)
        unit_tokens, unit_is_ours, unit_positions, unit_ids = (
            self._encode_units(game_state, current_side, device)
        )
        recruit_tokens, recruit_is_ours, recruit_types = (
            self._encode_recruits(game_state, current_side, device)
        )
        global_token = self._encode_global(game_state, current_side, device)

        return EncodedState(
            hex_tokens=hex_tokens,
            hex_positions=hex_positions,
            unit_tokens=unit_tokens,
            unit_is_ours=unit_is_ours,
            unit_positions=unit_positions,
            unit_ids=unit_ids,
            recruit_tokens=recruit_tokens,
            recruit_is_ours=recruit_is_ours,
            recruit_types=recruit_types,
            global_token=global_token,
            end_turn_token=self.end_turn_token.view(1, 1, -1),
        )

    # ----- internals ---------------------------------------------------

    def _name_id(self, name: str) -> int:
        """Map a unit-type name to a stable small int, growing on demand.

        Clamped to ``MAX_UNIT_TYPES - 1`` so an out-of-vocabulary type
        doesn't blow up the embedding table. Collisions on overflow are
        acceptable — the AI just treats rare types as "the overflow
        type". Retrain with a bigger table if this starts to bite.
        """
        if name not in self.unit_type_to_id:
            self.unit_type_to_id[name] = len(self.unit_type_to_id)
        idx = self.unit_type_to_id[name]
        return min(idx, MAX_UNIT_TYPES - 1)

    def _clamp_pos(self, v: int) -> int:
        return max(0, min(v, MAX_MAP_SIZE - 1))

    def _encode_hexes(self, game_state: GameState, device):
        # Row-major sort so hex_positions has a predictable order.
        hexes = sorted(
            game_state.map.hexes,
            key=lambda h: (h.position.y, h.position.x),
        )
        positions = [h.position for h in hexes]
        if not hexes:
            return (
                torch.zeros(1, 0, self.d_model, device=device),
                positions,
            )

        xs = torch.tensor(
            [self._clamp_pos(p.x) for p in positions],
            device=device, dtype=torch.long,
        )
        ys = torch.tensor(
            [self._clamp_pos(p.y) for p in positions],
            device=device, dtype=torch.long,
        )
        pos_emb = self.pos_x_embed(xs) + self.pos_y_embed(ys)  # [H, d]

        terr_ids = torch.tensor(
            [_first_terrain_id(h.terrain_types) for h in hexes],
            device=device, dtype=torch.long,
        )
        terr_emb = self.terrain_embed(terr_ids)  # [H, d]

        mod_flags = torch.tensor(
            [_modifier_flags(h.modifiers) for h in hexes],
            device=device, dtype=torch.float32,
        )  # [H, 3]
        mod_emb = self.modifier_proj(mod_flags)  # [H, d]

        tokens = (pos_emb + terr_emb + mod_emb).unsqueeze(0)  # [1, H, d]
        return tokens, positions

    def _encode_units(self, game_state: GameState, current_side: int, device):
        units = sorted(
            game_state.map.units,
            key=lambda u: (u.position.y, u.position.x, u.id),
        )
        if not units:
            return (
                torch.zeros(1, 0, self.d_model, device=device),
                torch.zeros(1, 0, device=device, dtype=torch.float32),
                [], [],
            )

        positions = [u.position for u in units]
        ids = [u.id for u in units]
        is_ours = [1.0 if u.side == current_side else 0.0 for u in units]

        # Unit-type embedding: reuse Unit.name_id (assigned by the
        # converter) when possible; fall back to our own registry for
        # recruit tokens that might not have appeared as units yet.
        type_ids = torch.tensor(
            [
                min(u.name_id if u.name_id < MAX_UNIT_TYPES
                    else self._name_id(u.name), MAX_UNIT_TYPES - 1)
                for u in units
            ],
            device=device, dtype=torch.long,
        )
        # Keep the shared dict in sync so recruit tokens match.
        for u in units:
            if u.name not in self.unit_type_to_id:
                self.unit_type_to_id[u.name] = u.name_id
        type_emb = self.unit_type_embed(type_ids)  # [U, d]

        side_ids = torch.tensor(
            [0 if u.side == current_side else 1 for u in units],
            device=device, dtype=torch.long,
        )
        side_emb = self.side_embed(side_ids)  # [U, d]

        xs = torch.tensor([self._clamp_pos(u.position.x) for u in units],
                          device=device, dtype=torch.long)
        ys = torch.tensor([self._clamp_pos(u.position.y) for u in units],
                          device=device, dtype=torch.long)
        pos_emb = self.pos_x_embed(xs) + self.pos_y_embed(ys)  # [U, d]

        feats = torch.tensor(
            [_unit_features(u) for u in units],
            device=device, dtype=torch.float32,
        )  # [U, UNIT_FEAT_DIM]
        feat_emb = self.unit_feat_proj(feats)  # [U, d]

        tokens = (type_emb + side_emb + pos_emb + feat_emb).unsqueeze(0)
        is_ours_t = torch.tensor(is_ours, device=device,
                                 dtype=torch.float32).unsqueeze(0)
        return tokens, is_ours_t, positions, ids

    def _encode_recruits(self, game_state: GameState, current_side: int, device):
        entries = []  # (name, side)
        for side_idx, side_info in enumerate(game_state.sides, start=1):
            for name in side_info.recruits:
                entries.append((name, side_idx))

        if not entries:
            return (
                torch.zeros(1, 0, self.d_model, device=device),
                torch.zeros(1, 0, device=device, dtype=torch.float32),
                [],
            )

        recruit_types = [name for name, _ in entries]
        is_ours = [1.0 if s == current_side else 0.0 for _, s in entries]

        type_ids = torch.tensor(
            [self._name_id(name) for name, _ in entries],
            device=device, dtype=torch.long,
        )
        type_emb = self.unit_type_embed(type_ids)

        side_ids = torch.tensor(
            [0 if s == current_side else 1 for _, s in entries],
            device=device, dtype=torch.long,
        )
        side_emb = self.side_embed(side_ids)

        # Recruit tokens have no position — they live in the keep
        # abstractly. They also don't have the per-unit numerical
        # features (HP/moves/...), because no instance exists yet.
        tokens = (type_emb + side_emb).unsqueeze(0)
        is_ours_t = torch.tensor(is_ours, device=device,
                                 dtype=torch.float32).unsqueeze(0)
        return tokens, is_ours_t, recruit_types

    def _encode_global(self, game_state: GameState, current_side: int, device):
        gi = game_state.global_info
        sides = game_state.sides
        us_idx = current_side - 1
        them_idx = 1 - us_idx if len(sides) == 2 else us_idx  # 2p assumption

        our_gold         = sides[us_idx].current_gold if 0 <= us_idx < len(sides) else 0
        our_income       = sides[us_idx].base_income  if 0 <= us_idx < len(sides) else 0
        our_villages     = sides[us_idx].nb_villages_controlled if 0 <= us_idx < len(sides) else 0
        their_villages   = sides[them_idx].nb_villages_controlled if 0 <= them_idx < len(sides) else 0

        feats = torch.tensor([[
            gi.turn_number / TURN_NORM,
            (current_side - 1.5) * 2.0,   # 1 → -1, 2 → +1
            our_gold / GOLD_NORM,
            our_income / INCOME_NORM,
            our_villages / VILLAGES_NORM,
            their_villages / VILLAGES_NORM,
        ]], device=device, dtype=torch.float32)

        emb = self.global_proj(feats)  # [1, d_model]
        return emb.unsqueeze(0)        # [1, 1, d_model]


# ---------------------------------------------------------------------
# Plain-python helpers (no torch) — keep them out of the module so
# they're easy to unit-test.
# ---------------------------------------------------------------------

def _first_terrain_id(terrain_types) -> int:
    """Pick one terrain id per hex; Hex.terrain_types can have several.

    Priority: VILLAGE > CASTLE > special-unwalkable > first.
    """
    if not terrain_types:
        return Terrain.FLAT.value
    # Prefer a "building" terrain if present.
    for pref in (Terrain.VILLAGE, Terrain.CASTLE):
        if pref in terrain_types:
            return pref.value
    return next(iter(terrain_types)).value


def _modifier_flags(modifiers) -> List[float]:
    return [
        1.0 if TerrainModifiers.VILLAGE in modifiers else 0.0,
        1.0 if TerrainModifiers.KEEP    in modifiers else 0.0,
        1.0 if TerrainModifiers.CASTLE  in modifiers else 0.0,
    ]


def _unit_features(u: Unit) -> List[float]:
    max_hp = max(u.max_hp, 1)
    max_mv = max(u.max_moves, 1)
    max_xp = max(u.max_exp, 1)

    numeric = [
        u.max_hp / HP_NORM,
        u.current_hp / max_hp,
        u.max_moves / MOVES_NORM,
        u.current_moves / max_mv,
        u.max_exp / EXP_NORM,
        u.current_exp / max_xp,
        u.cost / COST_NORM,
        1.0 if u.is_leader else 0.0,
        1.0 if u.has_attacked else 0.0,
    ]
    alignment_onehot = [0.0] * NUM_ALIGNMENTS
    alignment_onehot[u.alignment.value] = 1.0
    return numeric + alignment_onehot
