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

import numpy as np
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
# # distinct unit type names we can embed in one model. Default 200
# is enough for the full default-era roster (~50 units across 6
# factions) plus typical custom-era expansions. Overflow:
# `register_names` (encoder.py) clamps the (200+1)-th type seen to
# id 199, aliasing it with whatever happened to land at id 199 first
# -- a silent data-quality issue rather than an error. Watch the
# encoder's overflow log on first epoch of supervised training; if
# it fires, re-train with a larger MAX_UNIT_TYPES (changing it
# requires a fresh model -- the embedding row count is baked in).
MAX_UNIT_TYPES  = 200
MAX_FACTIONS    = 32    # default era has 6; supervised corpus adds a
                        # handful ("Custom", era-specific, "") — 32
                        # leaves room for growth.
NUM_TERRAINS    = max(Terrain) + 1       # 14 enum values today
NUM_ALIGNMENTS  = max(Alignment) + 1     # 4
NUM_SIDE_CODES  = 2     # 0 = ours (current_side), 1 = theirs


# Pre-seed the faction vocab so the 6 default-era factions always
# land on the same embedding row regardless of whether the first
# state we see happens to be, e.g., a Rebels vs Undead supervised
# replay (which would otherwise bind Rebels→0 and never re-map it).
# Empty string is reserved for "unknown/unset" → id 0.
_DEFAULT_FACTIONS = [
    "", "Drakes", "Knalgan Alliance", "Rebels",
    "Loyalists", "Northerners", "Undead",
]

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

# Normalization divisors. Re-exported from `constants.py` so era
# mods can override them in one place; see the comment block in
# constants.py for scale rationale.
from constants import (
    HP_NORM, MOVES_NORM, EXP_NORM, COST_NORM,
    GOLD_NORM, INCOME_NORM, VILLAGES_NORM, TURN_NORM,
)


import logging
log = logging.getLogger("encoder")


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
    # `(x, y) -> j` map. Cached once at encode time so callers like
    # action_sampler._build_legality_masks don't rebuild it per
    # decision. Saves a few ms on busy mid-game states with ~250
    # visible hexes; matters for MCTS rollouts.
    pos_to_hex:     "Dict[tuple, int]"     # mapping over hex_positions
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


# ---------------------------------------------------------------------
# Two-phase encoding split
# ---------------------------------------------------------------------
#
# encode() does two distinct things: (1) walk the GameState graph and
# build lists of integer indices and float feature vectors, (2) feed
# those through nn.Embedding/nn.Linear to produce learned tensors.
#
# Phase (1) is pure Python and dominates step time when the model
# itself is fast (e.g. on a CUDA GPU). Splitting it out as a free
# function — `encode_raw` — that takes the vocab dicts read-only lets
# worker processes do (1) ahead of time, while the main process keeps
# (2) where the trainable parameters live.
#
# `RawEncoded` carries the result of (1): plain Python lists, ints,
# floats, strings, and Position named tuples. It pickles cheaply, so
# crossing a multiprocessing.Queue boundary is fast.
#
# Backwards compat: GameStateEncoder.encode(game_state) still exists
# and behaves identically — it grows vocab on demand and runs both
# phases in sequence.
#
# Vocab discipline for workers: when the encoder's vocab is frozen
# (e.g., shared with workers), `encode_raw` falls back to the
# overflow bucket (MAX_*-1) for unseen names rather than mutating
# the dict. New names then collide there until the next training run
# pre-seeds the vocab.

@dataclass
class RawEncoded:
    """Pure-Python representation of an encoded GameState.

    No torch, no nn parameters — picklable for cross-process transport.
    Convert to an `EncodedState` via `GameStateEncoder.encode_from_raw`.

    Bulk per-hex / per-unit / per-recruit fields are numpy arrays so
    the trainer-side `encode_from_raw` can use `torch.from_numpy`
    (~3 µs per call, no copy on CPU) instead of `torch.tensor(list, ...)`
    (~500 µs per call, includes a Python-side list → C-array conversion
    and a fresh allocation). Workers pay the np.asarray cost off the
    critical path (it's hidden behind whatever GPU work the main thread
    is doing on the previous batch).

    `*_positions` and `*_ids` / `*_types` stay as Python objects:
    they're only used by the action sampler downstream as plain Python
    addresses into game-state, never as tensor inputs.
    """

    # Per-hex stream (variable H).
    hex_positions:      List["Position"]
    hex_xs:             np.ndarray          # int64 [H]
    hex_ys:             np.ndarray          # int64 [H]
    hex_terrain_ids:    np.ndarray          # int64 [H]
    hex_modifier_flags: np.ndarray          # float32 [H, NUM_HEX_MODIFIERS]

    # Per-unit stream (variable U).
    unit_positions:  List["Position"]
    unit_ids:        List[str]
    unit_is_ours:    np.ndarray             # float32 [U]
    unit_type_ids:   np.ndarray             # int64 [U]; clamped to MAX-1
    unit_side_ids:   np.ndarray             # int64 [U]; 0 = ours, 1 = theirs
    unit_xs:         np.ndarray             # int64 [U]
    unit_ys:         np.ndarray             # int64 [U]
    unit_feats:      np.ndarray             # float32 [U, UNIT_FEAT_DIM]

    # Per-recruit stream (variable R). Each recruit is treated as a
    # phantom unit at its leader's keep, so the actor head can
    # discriminate recruit options by cost / hp / alignment / etc.
    # rather than only by type+side. Without these the sampler
    # collapses all of "our" recruits onto a single embedding cluster
    # and never learns which to pick. recruit_xs/ys = leader's keep
    # of the recruit's side; recruit_feats = `_unit_features` of a
    # full-HP / 0-MP / 0-XP / non-leader phantom of that type.
    recruit_types:    List[str]
    recruit_is_ours:  np.ndarray            # float32 [R]
    recruit_type_ids: np.ndarray            # int64 [R]
    recruit_side_ids: np.ndarray            # int64 [R]
    recruit_xs:       np.ndarray            # int64 [R]
    recruit_ys:       np.ndarray            # int64 [R]
    recruit_feats:    np.ndarray            # float32 [R, UNIT_FEAT_DIM]

    # Global features.
    global_feats:     np.ndarray            # float32 [GLOBAL_FEAT_DIM]
    our_faction_id:   int
    their_faction_id: int


class GameStateEncoder(nn.Module):
    """Learned embedder: GameState → EncodedState."""

    def __init__(
        self,
        d_model: int = 128,
        unit_type_to_id: Optional[Dict[str, int]] = None,
        faction_to_id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.d_model = d_model

        # We maintain our OWN name→id map. StateConverter also maintains
        # one (Unit.name_id comes from it), but we intentionally ignore
        # that here — an encoder-only dict means a recruit-string
        # "Dwarvish Fighter" and a unit of the same name always hit the
        # same embedding row, even if the converter saw them in a
        # different order. Optional init arg lets checkpoints restore
        # the dict at load time.
        self.unit_type_to_id: Dict[str, int] = (
            unit_type_to_id if unit_type_to_id is not None else {}
        )

        # Faction vocab. Pre-seeded with the 6 default factions plus
        # "" (unknown). Growing on demand for custom/era-specific
        # factions encountered during supervised training. Load-time
        # restore from checkpoint keeps id assignments stable.
        if faction_to_id is not None:
            self.faction_to_id: Dict[str, int] = dict(faction_to_id)
        else:
            self.faction_to_id = {f: i for i, f in enumerate(_DEFAULT_FACTIONS)}

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

        # --- faction embeddings ---------------------------------------
        # Separate embed tables for "our" and "their" faction so the
        # model can learn a conditioning distinction (Drakes-as-us
        # plays differently than Drakes-as-them). Small init so
        # untrained faction tokens don't swamp the global_token.
        self.our_faction_embed   = nn.Embedding(MAX_FACTIONS, d_model)
        self.their_faction_embed = nn.Embedding(MAX_FACTIONS, d_model)
        nn.init.normal_(self.our_faction_embed.weight,   std=0.02)
        nn.init.normal_(self.their_faction_embed.weight, std=0.02)

        # --- end_turn sentinel ----------------------------------------
        # Small init so it doesn't dominate the softmax at step 0.
        self.end_turn_token = nn.Parameter(torch.randn(d_model) * 0.02)

    # ----- public API --------------------------------------------------

    def encode(self, game_state: GameState) -> EncodedState:
        """Build an EncodedState for one GameState.

        Convenience entry point: registers any new names into the
        encoder's vocab, runs `encode_raw` against the (now-grown)
        dicts, and finalizes the tensors via `encode_from_raw`. This
        is the call self-play and tests use.
        """
        self.register_names(game_state)
        raw = encode_raw(
            game_state,
            type_to_id=self.unit_type_to_id,
            faction_to_id=self.faction_to_id,
        )
        return self.encode_from_raw(raw)

    def register_names(self, game_state: GameState) -> None:
        """Grow `unit_type_to_id` / `faction_to_id` for any name we
        haven't seen before. Pure dict mutation, no torch — safe to
        call before the encoder's parameters are touched.

        Workers should NOT call this: their dicts are read-only views.

        Capacity: when `unit_type_to_id` reaches `MAX_UNIT_TYPES - 1`
        slots, every additional new type gets clamped to id
        `MAX_UNIT_TYPES - 1` in `encode_from_raw`. The clamp is
        silent on the encode path; we surface it here at registration
        time with a warning so it's visible during pretrain
        (when new types are most likely to appear). At inference,
        `register_names` is generally not called -- the trainer
        freezes the vocab after pretrain and worker processes only
        consume the saved dict.
        """
        type_to_id = self.unit_type_to_id
        cap = MAX_UNIT_TYPES
        for u in game_state.map.units:
            if u.name not in type_to_id:
                if len(type_to_id) >= cap:
                    if not getattr(self, "_warned_type_overflow", False):
                        log.warning(
                            f"unit_type vocab full (MAX_UNIT_TYPES={cap}); "
                            f"new types like {u.name!r} will alias id "
                            f"{cap - 1}. Pre-seed via "
                            f"tools/scrape_unit_stats.py + load on encoder "
                            f"init, OR re-train with a larger MAX_UNIT_TYPES."
                        )
                        self._warned_type_overflow = True
                    # Don't add the new entry; the encode path's
                    # `min(get(name, overflow), overflow)` clamp does
                    # the right thing without a phantom dict slot.
                    continue
                type_to_id[u.name] = len(type_to_id)
        faction_to_id = self.faction_to_id
        for s in game_state.sides:
            if s.faction not in faction_to_id:
                if len(faction_to_id) >= MAX_FACTIONS:
                    if not getattr(self, "_warned_faction_overflow", False):
                        log.warning(
                            f"faction vocab full (MAX_FACTIONS={MAX_FACTIONS}); "
                            f"new faction {s.faction!r} aliasing.")
                        self._warned_faction_overflow = True
                    continue
                faction_to_id[s.faction] = len(faction_to_id)
            for r in s.recruits:
                if r not in type_to_id:
                    if len(type_to_id) >= cap:
                        # Already warned above on the unit pass; quiet
                        # here to avoid log-spam on a single state.
                        continue
                    type_to_id[r] = len(type_to_id)

    def encode_from_raw(
        self,
        raw: RawEncoded,
        *,
        device: Optional[torch.device] = None,
    ) -> EncodedState:
        """Finalize a `RawEncoded` into an `EncodedState`.

        This is the half that touches learned parameters — embeddings
        and linear projections. It must run on the process that owns
        the encoder's parameters (the trainer / inference main).

        Hot-path note: bulk fields go through `torch.from_numpy(...)`
        (~3 µs, zero-copy on CPU) plus an optional `.to(device)` for
        the GPU case (single coalesced memcpy per array). The previous
        version used `torch.tensor(python_list, ...)` which paid an
        extra ~500 µs per call iterating the Python list. With ~12
        such calls per pair the savings dominate the trainer's
        main-thread budget once workers prefetch encode_raw.
        """
        if device is None:
            device = next(self.parameters()).device
        d = self.d_model
        # `non_blocking=True` lets the H2D copy overlap with whatever
        # the device was already doing. Harmless on CPU (no-op).
        nb = device.type != "cpu"

        def _to_dev(arr_or_tensor):
            t = torch.from_numpy(arr_or_tensor)
            if device.type != "cpu":
                t = t.to(device, non_blocking=nb)
            return t

        # ---- hexes ----
        H = raw.hex_xs.shape[0]
        if H == 0:
            hex_tokens = torch.zeros(1, 0, d, device=device)
        else:
            hx = _to_dev(raw.hex_xs)
            hy = _to_dev(raw.hex_ys)
            ht = _to_dev(raw.hex_terrain_ids)
            hm = _to_dev(raw.hex_modifier_flags)
            hex_tokens = (
                self.pos_x_embed(hx)
                + self.pos_y_embed(hy)
                + self.terrain_embed(ht)
                + self.modifier_proj(hm)
            ).unsqueeze(0)  # [1, H, d]

        # ---- units ----
        U = raw.unit_xs.shape[0]
        if U == 0:
            unit_tokens  = torch.zeros(1, 0, d, device=device)
            unit_is_ours = torch.zeros(1, 0, device=device, dtype=torch.float32)
        else:
            ut = _to_dev(raw.unit_type_ids)
            us_ids = _to_dev(raw.unit_side_ids)
            ux = _to_dev(raw.unit_xs)
            uy = _to_dev(raw.unit_ys)
            uf = _to_dev(raw.unit_feats)
            unit_tokens = (
                self.unit_type_embed(ut)
                + self.side_embed(us_ids)
                + self.pos_x_embed(ux)
                + self.pos_y_embed(uy)
                + self.unit_feat_proj(uf)
            ).unsqueeze(0)  # [1, U, d]
            unit_is_ours = _to_dev(raw.unit_is_ours).unsqueeze(0)

        # ---- recruits ----
        R = raw.recruit_type_ids.shape[0]
        if R == 0:
            recruit_tokens  = torch.zeros(1, 0, d, device=device)
            recruit_is_ours = torch.zeros(1, 0, device=device, dtype=torch.float32)
        else:
            rt = _to_dev(raw.recruit_type_ids)
            rs = _to_dev(raw.recruit_side_ids)
            rx = _to_dev(raw.recruit_xs)
            ry = _to_dev(raw.recruit_ys)
            rf = _to_dev(raw.recruit_feats)
            # Recruit token now mirrors the unit token's structure: type
            # + side + position (leader's keep) + per-unit features.
            # Sharing `unit_feat_proj` and the position embeds with real
            # units lets the actor head treat "would-be Dwarvish
            # Fighter at our keep" the same way as "Dwarvish Fighter
            # standing here", so the recruit decision conditions on
            # the same numeric features the model uses everywhere
            # else. Closes the under-specification gap that drove
            # "model never recruits" in earlier supervised eval.
            recruit_tokens = (
                self.unit_type_embed(rt)
                + self.side_embed(rs)
                + self.pos_x_embed(rx)
                + self.pos_y_embed(ry)
                + self.unit_feat_proj(rf)
            ).unsqueeze(0)  # [1, R, d]
            recruit_is_ours = _to_dev(raw.recruit_is_ours).unsqueeze(0)

        # ---- global ----
        # 6-element float vector + two ints — small enough that
        # torch.tensor scalar paths are fine; from_numpy on a 6-element
        # array would be a wash.
        gf = torch.from_numpy(raw.global_feats).unsqueeze(0)
        if device.type != "cpu":
            gf = gf.to(device, non_blocking=nb)
        emb = self.global_proj(gf)
        our_fid  = torch.tensor([raw.our_faction_id],  device=device, dtype=torch.long)
        them_fid = torch.tensor([raw.their_faction_id], device=device, dtype=torch.long)
        emb = emb + self.our_faction_embed(our_fid) + self.their_faction_embed(them_fid)
        global_token = emb.unsqueeze(0)  # [1, 1, d]

        # Pre-compute (x, y) -> hex index map for downstream sampler
        # legality checks. Saves rebuilding it per-decision in
        # action_sampler._build_legality_masks.
        pos_to_hex = {
            (p.x, p.y): j for j, p in enumerate(raw.hex_positions)
        }

        return EncodedState(
            hex_tokens=hex_tokens,
            hex_positions=raw.hex_positions,
            pos_to_hex=pos_to_hex,
            unit_tokens=unit_tokens,
            unit_is_ours=unit_is_ours,
            unit_positions=raw.unit_positions,
            unit_ids=raw.unit_ids,
            recruit_tokens=recruit_tokens,
            recruit_is_ours=recruit_is_ours,
            recruit_types=raw.recruit_types,
            global_token=global_token,
            end_turn_token=self.end_turn_token.view(1, 1, -1),
        )


# ---------------------------------------------------------------------
# encode_raw — phase-1 of encoding. Pure Python, vocab read-only.
# ---------------------------------------------------------------------

def _clamp_pos(v: int) -> int:
    return max(0, min(v, MAX_MAP_SIZE - 1))


def _lookup_id(name: str, table: Dict[str, int], maxn: int) -> int:
    """Read-only vocab lookup. Out-of-vocab → overflow bucket (maxn-1).

    Mirrors the clamping behavior of the old `_name_id` / `_faction_id`
    methods, but never mutates the dict — safe to call from worker
    processes that share a frozen view of the vocab.
    """
    return min(table.get(name, maxn - 1), maxn - 1)


def encode_raw(
    game_state: GameState,
    *,
    type_to_id: Dict[str, int],
    faction_to_id: Dict[str, int],
) -> RawEncoded:
    """Build a `RawEncoded` from a GameState using read-only vocab.

    Self-contained: no torch, no nn modules, no GPU. The result is
    picklable, so workers can call this and ship results back to the
    trainer over a multiprocessing queue.

    The caller is responsible for keeping `type_to_id` / `faction_to_id`
    in sync between workers and the encoder owning the embedding tables
    — typically by pre-seeding before spawning workers and never
    growing during training.

    Output bulk fields are numpy arrays (int64 / float32) so the
    trainer's `encode_from_raw` can wrap them with `torch.from_numpy`
    in zero-copy O(1) time. The np.asarray calls here run in the
    worker process and are hidden behind the main thread's GPU work.
    """
    current_side = game_state.global_info.current_side
    sides = game_state.sides
    MAP_LIMIT = MAX_MAP_SIZE - 1   # avoid attribute lookup in tight loops

    # ---- hexes (drop fogged) ----
    fog_set = {(p.x, p.y) for p in game_state.map.fog}
    if fog_set:
        hexes_iter = (
            h for h in game_state.map.hexes
            if (h.position.x, h.position.y) not in fog_set
        )
    else:
        hexes_iter = iter(game_state.map.hexes)
    hexes = sorted(hexes_iter, key=lambda h: (h.position.y, h.position.x))

    hex_positions = [h.position for h in hexes]
    H = len(hex_positions)

    if H == 0:
        hex_xs_np = np.empty(0, dtype=np.int64)
        hex_ys_np = np.empty(0, dtype=np.int64)
        hex_terrain_ids_np = np.empty(0, dtype=np.int64)
        hex_modifier_flags_np = np.empty((0, NUM_HEX_MODIFIERS), dtype=np.float32)
    else:
        # Inline the clamp + terrain/modifier extraction in tight loops
        # to skip per-call Python-frame overhead. _clamp_pos/_first_terrain_id
        # are still defined as standalone helpers for clarity / unit tests;
        # we just don't call them per-hex.
        hex_xs_np          = np.empty(H, dtype=np.int64)
        hex_ys_np          = np.empty(H, dtype=np.int64)
        hex_terrain_ids_np = np.empty(H, dtype=np.int64)
        hex_modifier_flags_np = np.zeros((H, NUM_HEX_MODIFIERS), dtype=np.float32)
        terrain_village = Terrain.VILLAGE
        terrain_castle  = Terrain.CASTLE
        terrain_flat_v  = Terrain.FLAT.value
        mod_village = TerrainModifiers.VILLAGE
        mod_keep    = TerrainModifiers.KEEP
        mod_castle  = TerrainModifiers.CASTLE
        for i, h in enumerate(hexes):
            p = h.position
            hex_xs_np[i] = 0 if p.x < 0 else (MAP_LIMIT if p.x > MAP_LIMIT else p.x)
            hex_ys_np[i] = 0 if p.y < 0 else (MAP_LIMIT if p.y > MAP_LIMIT else p.y)
            tt = h.terrain_types
            if not tt:
                hex_terrain_ids_np[i] = terrain_flat_v
            elif terrain_village in tt:
                hex_terrain_ids_np[i] = terrain_village.value
            elif terrain_castle in tt:
                hex_terrain_ids_np[i] = terrain_castle.value
            else:
                hex_terrain_ids_np[i] = next(iter(tt)).value
            mods = h.modifiers
            if mod_village in mods: hex_modifier_flags_np[i, 0] = 1.0
            if mod_keep    in mods: hex_modifier_flags_np[i, 1] = 1.0
            if mod_castle  in mods: hex_modifier_flags_np[i, 2] = 1.0

    # ---- units ----
    units = sorted(
        game_state.map.units,
        key=lambda u: (u.position.y, u.position.x, u.id),
    )
    U = len(units)
    unit_positions = [u.position for u in units]
    unit_ids       = [u.id for u in units]

    unit_is_ours_np  = np.empty(U, dtype=np.float32)
    unit_type_ids_np = np.empty(U, dtype=np.int64)
    unit_side_ids_np = np.empty(U, dtype=np.int64)
    unit_xs_np       = np.empty(U, dtype=np.int64)
    unit_ys_np       = np.empty(U, dtype=np.int64)
    unit_feats_np    = np.empty((U, UNIT_FEAT_DIM), dtype=np.float32)
    type_overflow    = MAX_UNIT_TYPES - 1
    for i, u in enumerate(units):
        is_ours = u.side == current_side
        unit_is_ours_np[i]  = 1.0 if is_ours else 0.0
        unit_side_ids_np[i] = 0 if is_ours else 1
        unit_type_ids_np[i] = min(
            type_to_id.get(u.name, type_overflow), type_overflow
        )
        ux, uy = u.position.x, u.position.y
        unit_xs_np[i] = 0 if ux < 0 else (MAP_LIMIT if ux > MAP_LIMIT else ux)
        unit_ys_np[i] = 0 if uy < 0 else (MAP_LIMIT if uy > MAP_LIMIT else uy)
        unit_feats_np[i] = _unit_features(u)

    # ---- recruits ----
    # Pre-compute each side's leader keep position so recruit tokens
    # carry a meaningful "where would I spawn?" coordinate. If a side
    # has no leader (it's been killed but the side isn't formally
    # eliminated), fall back to (0, 0) -- the recruit is unreachable
    # anyway, the legality mask will hide it.
    side_leader_xy: Dict[int, Tuple[int, int]] = {}
    for u in game_state.map.units:
        if u.is_leader and u.side not in side_leader_xy:
            side_leader_xy[u.side] = (u.position.x, u.position.y)
    recruit_types: List[str] = []
    recruit_is_ours: List[float] = []
    recruit_type_ids: List[int]  = []
    recruit_side_ids: List[int]  = []
    recruit_xs: List[int] = []
    recruit_ys: List[int] = []
    recruit_feats_rows: List[np.ndarray] = []
    for side_idx, side_info in enumerate(sides, start=1):
        is_ours = side_idx == current_side
        lx, ly = side_leader_xy.get(side_idx, (0, 0))
        lx_clamped = 0 if lx < 0 else (MAP_LIMIT if lx > MAP_LIMIT else lx)
        ly_clamped = 0 if ly < 0 else (MAP_LIMIT if ly > MAP_LIMIT else ly)
        for name in side_info.recruits:
            recruit_types.append(name)
            recruit_is_ours.append(1.0 if is_ours else 0.0)
            recruit_type_ids.append(
                min(type_to_id.get(name, type_overflow), type_overflow)
            )
            recruit_side_ids.append(0 if is_ours else 1)
            recruit_xs.append(lx_clamped)
            recruit_ys.append(ly_clamped)
            recruit_feats_rows.append(_recruit_features_for(name))
    recruit_is_ours_np  = np.asarray(recruit_is_ours,  dtype=np.float32)
    recruit_type_ids_np = np.asarray(recruit_type_ids, dtype=np.int64)
    recruit_side_ids_np = np.asarray(recruit_side_ids, dtype=np.int64)
    recruit_xs_np       = np.asarray(recruit_xs,       dtype=np.int64)
    recruit_ys_np       = np.asarray(recruit_ys,       dtype=np.int64)
    if recruit_feats_rows:
        recruit_feats_np = np.stack(recruit_feats_rows, axis=0)
    else:
        recruit_feats_np = np.zeros((0, UNIT_FEAT_DIM), dtype=np.float32)

    # ---- global ----
    gi = game_state.global_info
    us_idx = current_side - 1
    them_idx = 1 - us_idx if len(sides) == 2 else us_idx  # 2p assumption
    our_gold       = sides[us_idx].current_gold if 0 <= us_idx < len(sides) else 0
    our_income     = sides[us_idx].base_income  if 0 <= us_idx < len(sides) else 0
    our_villages   = sides[us_idx].nb_villages_controlled if 0 <= us_idx < len(sides) else 0
    their_villages = sides[them_idx].nb_villages_controlled if 0 <= them_idx < len(sides) else 0

    global_feats_np = np.array([
        gi.turn_number / TURN_NORM,
        (current_side - 1.5) * 2.0,   # 1 → -1, 2 → +1
        our_gold       / GOLD_NORM,
        our_income     / INCOME_NORM,
        our_villages   / VILLAGES_NORM,
        their_villages / VILLAGES_NORM,
    ], dtype=np.float32)

    our_fac  = sides[us_idx].faction   if 0 <= us_idx   < len(sides) else ""
    them_fac = sides[them_idx].faction if 0 <= them_idx < len(sides) else ""
    our_faction_id   = _lookup_id(our_fac,  faction_to_id, MAX_FACTIONS)
    their_faction_id = _lookup_id(them_fac, faction_to_id, MAX_FACTIONS)

    return RawEncoded(
        hex_positions=hex_positions,
        hex_xs=hex_xs_np,
        hex_ys=hex_ys_np,
        hex_terrain_ids=hex_terrain_ids_np,
        hex_modifier_flags=hex_modifier_flags_np,
        unit_positions=unit_positions,
        unit_ids=unit_ids,
        unit_is_ours=unit_is_ours_np,
        unit_type_ids=unit_type_ids_np,
        unit_side_ids=unit_side_ids_np,
        unit_xs=unit_xs_np,
        unit_ys=unit_ys_np,
        unit_feats=unit_feats_np,
        recruit_types=recruit_types,
        recruit_is_ours=recruit_is_ours_np,
        recruit_type_ids=recruit_type_ids_np,
        recruit_side_ids=recruit_side_ids_np,
        recruit_xs=recruit_xs_np,
        recruit_ys=recruit_ys_np,
        recruit_feats=recruit_feats_np,
        global_feats=global_feats_np,
        our_faction_id=our_faction_id,
        their_faction_id=their_faction_id,
    )


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


# ---------------------------------------------------------------------
# Recruit phantom-unit features
# ---------------------------------------------------------------------
# A "recruit option" doesn't have a Unit instance until it's spawned,
# but we want the same feature vector shape `_unit_features` produces
# so the model treats recruits and on-board units consistently. Build a
# phantom feature vector from the unit-stats DB (scraped from
# wesnoth_src). HP / moves / xp / cost / alignment all come from the
# stats; current_* fields are spawn defaults (full HP, 0 MP since
# spawn turn, 0 XP); is_leader=False, has_attacked=False.

_RECRUIT_STATS_CACHE: Dict[str, np.ndarray] = {}
_FALLBACK_RECRUIT_STATS = {
    "hitpoints": 33, "moves": 5, "experience": 50, "cost": 14,
    "alignment": "neutral",
}


def _alignment_value(name: str) -> int:
    """Map an alignment string to the Alignment enum's int value.
    Mirrors what classes.Alignment uses."""
    n = (name or "neutral").lower()
    # Order matches classes.Alignment: NEUTRAL, LAWFUL, CHAOTIC, LIMINAL.
    table = {"neutral": 0, "lawful": 1, "chaotic": 2, "liminal": 3}
    return table.get(n, 0)


def _recruit_features_for(unit_type: str) -> np.ndarray:
    """Return a [UNIT_FEAT_DIM] float32 phantom feature vector for
    a recruit option of `unit_type`. Cached to avoid re-reading the
    stats DB on every encoding call."""
    cached = _RECRUIT_STATS_CACHE.get(unit_type)
    if cached is not None:
        return cached
    # Lazy import: tools.replay_dataset already loads the unit DB on
    # first access; reuse it rather than re-parsing the JSON.
    try:
        from tools.replay_dataset import _stats_for, _load_unit_db
        _load_unit_db()
        stats = _stats_for(unit_type)
    except Exception:
        stats = _FALLBACK_RECRUIT_STATS
    max_hp = float(stats.get("hitpoints", 33))
    max_mv = float(stats.get("moves", 5))
    max_xp = float(stats.get("experience", 50))
    cost   = float(stats.get("cost", 14))
    align  = _alignment_value(stats.get("alignment", "neutral"))

    numeric = [
        max_hp / HP_NORM,
        1.0,                      # current_hp = max on spawn
        max_mv / MOVES_NORM,
        0.0,                      # current_moves = 0 on spawn turn
        max_xp / EXP_NORM,
        0.0,                      # current_exp = 0
        cost / COST_NORM,
        0.0,                      # is_leader = False
        0.0,                      # has_attacked = False
    ]
    alignment_onehot = [0.0] * NUM_ALIGNMENTS
    alignment_onehot[align] = 1.0
    out = np.asarray(numeric + alignment_onehot, dtype=np.float32)
    _RECRUIT_STATS_CACHE[unit_type] = out
    return out
