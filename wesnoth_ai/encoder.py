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

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


import numpy as np
import torch
import torch.nn as nn

from wesnoth_ai.packed_trunk import EmbeddedStreams, FlatLayout

from wesnoth_ai.classes import (
    Alignment,
    GameState,
    Position,
    SideInfo,
    Terrain,
    TerrainModifiers,
    Unit,
    opponent_of,
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
NUM_SIDE_CODES  = 3     # 0 = ours, 1 = theirs, 2 = neutral
# (2 = scenery/statues: non-player sides and petrified units --
#  visible board furniture, not combatants; 2026-07-11)


# Pre-seeded faction vocab. Re-exported from constants.py so era
# mods can override in one place. Empty string is reserved for
# "unknown/unset" -> id 0.
from wesnoth_ai.constants import DEFAULT_FACTIONS as _DEFAULT_FACTIONS  # noqa: E402 -- re-export point documented above
from wesnoth_ai.material import material_of_units  # noqa: E402
from wesnoth_ai.visibility import (  # noqa: E402
    relevant_hexes_in_slot_order, hexes_in_slot_order, own_recruit_types,
                        visible_units_in_slot_order, is_scenery_unit)


def pad_legacy_encoder_state(encoder_state: dict, encoder) -> dict:
    """Zero-pad legacy encoder tensors that grew in 2026-07-11's
    observation upgrade (dynamic_flag_proj [d,1]->[d,3] for village
    ownership; side_embed [2,d]->[3,d] for neutral scenery), so every
    loader -- not just TransformerPolicy.load_checkpoint -- can
    resume old checkpoints. `strict=False` does NOT tolerate shape
    mismatches (adversarial review 2026-07-11, verified), so any
    direct `encoder.load_state_dict(ckpt["encoder_state"], ...)`
    MUST route its state through this helper first. Returns a new
    dict; the input is not mutated."""
    out = dict(encoder_state)
    dfw = out.get("dynamic_flag_proj.weight")
    cur = encoder.dynamic_flag_proj.weight
    if (dfw is not None and dfw.shape[1] < cur.shape[1]):
        # Pad with the SOURCE's width on the non-growing axis
        # (project round-2 C2: requiring shape[0] equality made the
        # shim inert exactly when d_model grows -- net2net's primary
        # case -- leaving the new observation slots at random init;
        # transfer_state_dict handles the d_model axis afterwards).
        pad = torch.zeros(dfw.shape[0], cur.shape[1] - dfw.shape[1],
                          dtype=dfw.dtype)
        out["dynamic_flag_proj.weight"] = torch.cat([dfw, pad], dim=1)
    sew = out.get("side_embed.weight")
    cur_se = encoder.side_embed.weight
    if (sew is not None and sew.shape[0] < cur_se.shape[0]):
        pad = torch.zeros(cur_se.shape[0] - sew.shape[0],
                          sew.shape[1], dtype=sew.dtype)
        out["side_embed.weight"] = torch.cat([sew, pad], dim=0)
    # The global features grew from 6 to 8 with the time-of-day terms
    # (2026-09-22). A zero pad makes an old checkpoint ignore them, so
    # it plays exactly the game it was trained on.
    gpw = out.get("global_proj.weight")
    cur_gp = encoder.global_proj.weight
    if gpw is not None and gpw.shape[1] < cur_gp.shape[1]:
        pad = torch.zeros(gpw.shape[0], cur_gp.shape[1] - gpw.shape[1],
                          dtype=gpw.dtype)
        out["global_proj.weight"] = torch.cat([gpw, pad], dim=1)
    return out


def repair_optimizer_state_shapes(optimizer, log=None) -> int:
    """Zero-pad optimizer moment tensors whose parameter grew since
    the checkpoint was saved. Companion to `pad_legacy_encoder_state`:
    the load shim pads the WEIGHTS, but a restored Adam state still
    carries old-shaped exp_avg/exp_avg_sq, and the first `step()`
    crashes on the broadcast (observed 2026-07-11 on the A4000
    relaunch: "output with shape [256, 1] doesn't match the broadcast
    shape [256, 3]" inside adam's _multi_tensor path). Call AFTER
    `optimizer.load_state_dict`. Old dims keep their momentum; new
    dims start at zero, matching the zero-padded weights. Skips
    scalars (Adam's `step` counter) and exact-shape entries. Returns
    the number of tensors repaired."""
    n = 0
    for p, s in optimizer.state.items():
        for k, t in list(s.items()):
            if (isinstance(t, torch.Tensor) and t.ndim == p.ndim
                    and t.shape != p.shape
                    and all(a <= b for a, b in zip(t.shape, p.shape))):
                new = torch.zeros_like(p)
                new[tuple(slice(0, d) for d in t.shape)] = t
                s[k] = new
                n += 1
                if log is not None:
                    log.info(f"padded optimizer state '{k}' "
                             f"{tuple(t.shape)} -> {tuple(p.shape)}")
    return n

# Per-hex STATIC multi-hot: (village, keep, castle). Static = doesn't
# change during a game (or only changes via village-capture, which
# is captured by the village bit being side-agnostic).
NUM_HEX_MODIFIERS = 3

# Per-hex DYNAMIC features that DO change within a turn / decision.
# Separate from the static modifiers so:
#   (a) old checkpoints' modifier_proj (3-input Linear) loads
#       unchanged via `strict=False`, while the new dynamic_flag_proj
#       initializes fresh;
#   (b) future per-hex dynamic flags (e.g. "attacked-from last turn",
#       "ZoC'd by us", ...) can extend NUM_HEX_DYNAMIC_FLAGS without
#       breaking either projection.
#
# Currently:
#   0: recruit_rejected -- this hex bounced a recruit attempt this
#                          turn (cleared at init_side). See the
#                          legality-mask contract in CLAUDE.md.
#   1: village_ours     -- this village hex is owned by the side to
#                          move. ALWAYS shown when true (you know
#                          your own villages even in fog).
#   2: village_theirs   -- this village hex is owned by the opponent
#                          AND its hex is currently visible to the
#                          side to move (a hex it sees, or fog
#                          is off). A fogged enemy-owned village has
#                          BOTH flags 0 = appears neutral (user spec
#                          2026-07-11; deliberate small deviation
#                          from Wesnoth's stale last-seen display,
#                          which would need per-side memory).
#   Neutral / non-village hexes carry 0/0.
#
# Moves have no rejection flag: a move onto a hex a hidden unit holds
# stops next to it and reveals it (pathfind_sim.walk_move_path).
# (Historical: the flag count was held at 1 for a while for
# checkpoint compatibility of dynamic_flag_proj; it has been 3
# since the fog/ZoC flags landed -- pad_legacy_encoder_state
# handles old checkpoints.)
NUM_HEX_DYNAMIC_FLAGS = 3

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
# Per-game GLOBAL features. Order MATTERS (the Linear is positional).
#   0: turn number / TURN_NORM
#   1: side to move, -1 or +1
#   2: our gold / GOLD_NORM
#   3: our income / INCOME_NORM
#   4: our villages / VILLAGES_NORM
#   5: their villages / VILLAGES_NORM
#   6: this turn's lawful bonus / LAWFUL_BONUS_NORM
#   7: next turn's lawful bonus / LAWFUL_BONUS_NORM
#
# 6 and 7 landed 2026-09-22. Until then the network could not see the
# time of day at all, while combat applied its bonus
# (rust/wesnoth_core/src/combat.rs `combat_modifier`), so a lawful
# unit's damage swung by half for reasons the policy could not
# observe. The turn number is not a stand-in: two pool scenarios start
# at second watch and four minis roll a random start, so the same turn
# number means different phases on different maps.
#
# BOTH this turn's and the next turn's, because the bonus alone cannot
# tell dawn from dusk -- both are 0 -- and they are strategically
# opposite: at dawn the lawful side is about to get stronger, at dusk
# the chaotic side is. One float cannot express that.
#
# Old checkpoints load through `pad_legacy_encoder_state`, which
# zero-pads `global_proj.weight`, so they observe exactly what they
# observed before.
GLOBAL_FEAT_DIM = 8
# Wesnoth's lawful bonus is -25, 0 or +25 under every schedule in the
# pool; dividing by 25 puts the feature in [-1, 1] like the side term.
LAWFUL_BONUS_NORM = 25.0

# Normalization divisors. Re-exported from `constants.py` so era
# mods can override them in one place; see the comment block in
# constants.py for scale rationale.
from wesnoth_ai.constants import (  # noqa: E402 -- re-export point documented above
    HP_NORM, MOVES_NORM, EXP_NORM, COST_NORM,
    GOLD_NORM, INCOME_NORM, VILLAGES_NORM, TURN_NORM,
)


import logging  # noqa: E402 -- follows the module's feature-table prelude
log = logging.getLogger("encoder")

# Serializes the append-only vocab growth in `register_names`. Training
# and inference encoders SHARE the same `unit_type_to_id`/`faction_to_id`
# dict objects (transformer_policy wires them by reference), and MCTS
# leaf expansion can register names from worker threads while the
# trainer reads the same dicts -- an unguarded `dict[name] = len(dict)`
# could then race (two threads claim the same id, or a reader sees a
# half-updated dict). A process-wide lock is enough: cross-PROCESS
# actors (the actor pool) hold their own dict copies shipped via the
# vocab snapshot, so they never share this state. See CLAUDE.md "Vocab".
_VOCAB_LOCK = threading.Lock()


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

    # Cached numpy view of recruit_is_ours. Populated alongside the
    # tensor field at encode time; the sampler reads this rather
    # than doing `.detach().cpu().numpy()` per decision. Saves the
    # device hop (~25 µs on GPU, ~5 µs on CPU) at the sampler's
    # innermost loop, which fires once per recruit-eligible state.
    # Default `None` keeps backwards compatibility -- callers that
    # build EncodedState by hand still work; the sampler falls back
    # to the device hop when this field is missing.
    recruit_is_ours_np: Optional[np.ndarray] = None

    # Visible-unit id set (optimization #3, 2026-06-14). The encoder
    # only emits VISIBLE units (units_visible_to), so the ids in
    # `unit_ids` ARE the fog-visible set for `current_side` -- exactly
    # what action_sampler._build_legality_masks needs to decide which
    # enemies are hidden. Stashing it here lets the sampler skip a
    # SECOND units_visible_to() call per decision (it was computing
    # the identical set independently). Keyed by stable `u.id` (not
    # python id()), so it stays correct even if the EncodedState is
    # paired with a deep-copied GameState. `None` => sampler falls
    # back to recomputing (hand-built EncodedState / tests), so
    # behavior is unchanged there.
    visible_unit_ids: Optional[frozenset] = None

    # True when the hex stream is the RELEVANT SUBSET rather than the whole
    # board (see encode_raw's `relevant_set`). Consumers that resolve a raw
    # board position to a slot index MUST treat a miss as a hard error in
    # this mode: under the full board a miss means off-board, but under the
    # subset it would mean an action the mask offered has no token to point
    # at -- i.e. a silently shrunken action space.
    hex_subset: bool = False
    # Material from the mover's side ([1, 1] float32), the value head's
    # optional input (wesnoth_ai/material.py). None on states built by
    # hand; a `value_material` model refuses to run without it.
    material: Optional[torch.Tensor] = None

    # The side's observation computed once per decision by the Rust
    # kernel (wesnoth_ai/observe.py): the seen hexes, unit visibility,
    # the reach-context flags and the recruit row. The legality mask
    # builder reads it instead of rebuilding the same sets; None on
    # the Python path.
    observation: Optional[object] = None


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
    hex_dynamic_flags:  np.ndarray          # float32 [H, NUM_HEX_DYNAMIC_FLAGS]

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
    # Material from the mover's side over the visible units
    # (wesnoth_ai/material.py); the value head reads it when the model
    # is built with `value_material`.
    material: float = 0.0

    # The side's observation from the Rust kernel (see EncodedState);
    # arrays only, so the record stays picklable. None on the Python path.
    observation: Optional[object] = None

    # True when the hex stream is the RELEVANT SUBSET rather than the whole
    # board (see encode_raw's `relevant_set`). Consumers that resolve a raw
    # board position to a slot index MUST treat a miss as a hard error in
    # this mode: under the full board a miss means off-board, but under the
    # subset it would mean an action the mask offered has no token to point
    # at -- i.e. a silently shrunken action space.
    hex_subset: bool = False


# torch's CUDA caching allocator rounds every block to this many
# bytes, so a tensor that starts a fresh block is aligned to it; the
# coalesced batch buffer puts each field on the same boundary.
_ALLOCATOR_ALIGNMENT = 512


class GameStateEncoder(nn.Module):
    """Learned embedder: GameState → EncodedState."""

    def __init__(
        self,
        d_model: int = 128,
        unit_type_to_id: Optional[Dict[str, int]] = None,
        faction_to_id: Optional[Dict[str, int]] = None,
        relevant_set_hexes: bool = False,
        fog_hides_enemy_villages: bool = False,
        terrain_multi_hot: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        # The hex's terrain as its full SET from the engine's aliases
        # (Hex.terrain_mask), embedded as a multi-hot over the same
        # table, instead of one class picked by enum ordinal (which
        # read a forested plain as plain on 86% of the Ladder pool's
        # forest hexes). Rides the checkpoint like the fog gate: on for
        # a fresh network, a checkpoint's own setting on load, so the
        # reference player's lineage keeps its observations.
        self.terrain_multi_hot = bool(terrain_multi_hot)
        # Opt-in relevant-hex stream (see encode_raw). Default OFF: this
        # changes the ACTION SPACE's index basis, so a checkpoint or replay
        # buffer built under one setting is meaningless under the other.
        self.relevant_set_hexes = bool(relevant_set_hexes)
        # Global feature 5 under fog: the enemy villages the mover can
        # see instead of the true count (rides the checkpoint).
        self.fog_hides_enemy_villages = bool(fog_hides_enemy_villages)

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
        # Bit positions for the multi-hot read of a terrain mask; not a
        # parameter, not in the state_dict.
        self.register_buffer("_terrain_bits", torch.arange(NUM_TERRAINS, dtype=torch.int64),
                             persistent=False)
        self.modifier_proj  = nn.Linear(NUM_HEX_MODIFIERS, d_model, bias=False)
        # Dynamic flags (e.g. recruit_rejected) live in their own
        # projection -- old checkpoints lacking this Linear initialize
        # it from scratch under load_checkpoint(strict=False); the
        # static modifier projection above is unaffected.
        self.dynamic_flag_proj = nn.Linear(NUM_HEX_DYNAMIC_FLAGS, d_model, bias=False)

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
        return self.encode_from_raw(self.raw_of(game_state))

    def raw_of(self, game_state: GameState, *, type_to_id=None, faction_to_id=None) -> RawEncoded:
        """`encode_raw` under THIS encoder's switches (the hex basis,
        the fog gate, the terrain view): the one way to build a raw
        that `encode_from_raw` reads as `encode` would. Every caller
        that holds an encoder and a GameState goes through here; a
        bare `encode_raw` call with a subset of the switches encodes
        another observation (2026-09-19: the trainer built its raws
        with the basis alone, so a fresh network trained on one-class
        terrain tokens and played on the set). The vocab defaults to
        the encoder's own; the trainer passes its frozen snapshot."""
        return encode_raw(
            game_state,
            type_to_id=self.unit_type_to_id if type_to_id is None else type_to_id,
            faction_to_id=self.faction_to_id if faction_to_id is None else faction_to_id,
            relevant_set=self.relevant_set_hexes,
            fog_hides_enemy_villages=self.fog_hides_enemy_villages,
            terrain_multi_hot=self.terrain_multi_hot,
        )

    def terrain_tokens(self, hex_terrain: torch.Tensor) -> torch.Tensor:
        """The terrain term of the hex tokens: under `terrain_multi_hot`
        `hex_terrain` holds masks and the term is the sum of the table's
        rows for the set bits (a multi-hot against the table); else it
        holds one class id per hex and the term is that row."""
        if not self.terrain_multi_hot:
            return self.terrain_embed(hex_terrain)
        bits = ((hex_terrain.unsqueeze(-1) >> self._terrain_bits) & 1)
        return bits.to(self.terrain_embed.weight.dtype) @ self.terrain_embed.weight

    def freeze_vocab(self) -> None:
        """Lock `unit_type_to_id` / `faction_to_id` so future
        `register_names` calls cannot add new entries. Any new name
        encountered after freeze fires a one-shot warning (per name)
        and the encoding path's clamp falls back to the overflow
        bucket. Call this after pretrain so a regression that
        introduces a new unit type during self-play is loud rather
        than silently aliased.

        Idempotent. To re-open (e.g. when fine-tuning on a new era),
        set `self._vocab_frozen = False` directly.
        """
        self._vocab_frozen = True
        log.info(
            f"encoder vocab frozen at {len(self.unit_type_to_id)} unit "
            f"types and {len(self.faction_to_id)} factions"
        )

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

        Freeze: callers can `freeze_vocab()` post-pretrain so that
        any subsequent new name fires a per-name warning instead of
        being silently added (would shift embedding ids and
        invalidate the checkpoint).

        Thread-safety: the append-only growth is serialized by the
        process-wide `_VOCAB_LOCK` (training + inference encoders share
        the same dict objects; MCTS leaf expansion can register from a
        worker thread concurrently with the trainer). See the lock's
        definition for why a process-wide lock suffices.
        """
        with _VOCAB_LOCK:
            self._register_names_locked(game_state)

    def watch_vocab_growth(self) -> None:
        """Start logging (once per name) whenever `register_names` adds
        a NEW vocab entry. Call this after warm-start / checkpoint load:
        the bulk initial population then stays quiet, but a genuinely
        mid-run addition (a new unit type appearing during self-play)
        becomes a visible breadcrumb -- the new id maps to a fresh-init
        embedding row, which is intentional under dynamic growth but
        worth surfacing on a long unattended run. Does NOT stop growth
        (that's `freeze_vocab`)."""
        self._vocab_growth_watch = True
        if getattr(self, "_warned_new_name", None) is None:
            self._warned_new_name = set()
        log.info(
            f"encoder now watching mid-run vocab growth (currently "
            f"{len(self.unit_type_to_id)} unit types, "
            f"{len(self.faction_to_id)} factions)."
        )

    def _note_new_type(self, name: str) -> None:
        """One-shot (per name) log of a mid-run vocab addition; no-op
        unless `watch_vocab_growth()` armed it."""
        if not getattr(self, "_vocab_growth_watch", False):
            return
        seen = getattr(self, "_warned_new_name", None)
        if seen is None:
            seen = set()
            self._warned_new_name = seen
        if name not in seen:
            seen.add(name)
            log.warning(
                f"encoder vocab grew mid-run: new entry {name!r} "
                f"(fresh embedding row; intentional dynamic growth)."
            )

    def _register_names_locked(self, game_state: GameState) -> None:
        type_to_id = self.unit_type_to_id
        cap = MAX_UNIT_TYPES
        frozen = getattr(self, "_vocab_frozen", False)
        seen_new = getattr(self, "_warned_new_name", None)
        if seen_new is None:
            seen_new = set()
            self._warned_new_name = seen_new
        for u in game_state.map.units:
            if u.name not in type_to_id:
                if frozen:
                    if u.name not in seen_new:
                        seen_new.add(u.name)
                        log.warning(
                            f"encoder vocab is frozen but encountered new "
                            f"unit type {u.name!r}; aliasing to overflow "
                            f"bucket id {cap - 1}. Either pre-seed before "
                            f"freeze or unfreeze + retrain."
                        )
                    continue
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
                self._note_new_type(u.name)
        faction_to_id = self.faction_to_id
        for s in game_state.sides:
            if s.faction not in faction_to_id:
                if frozen:
                    key = f"<faction>{s.faction}"
                    if key not in seen_new:
                        seen_new.add(key)
                        log.warning(
                            f"encoder vocab frozen; new faction "
                            f"{s.faction!r} aliasing to overflow."
                        )
                    continue
                if len(faction_to_id) >= MAX_FACTIONS:
                    if not getattr(self, "_warned_faction_overflow", False):
                        log.warning(
                            f"faction vocab full (MAX_FACTIONS={MAX_FACTIONS}); "
                            f"new faction {s.faction!r} aliasing.")
                        self._warned_faction_overflow = True
                    continue
                faction_to_id[s.faction] = len(faction_to_id)
                self._note_new_type(s.faction)
            for r in s.recruits:
                if r not in type_to_id:
                    if frozen:
                        if r not in seen_new:
                            seen_new.add(r)
                            log.warning(
                                f"encoder vocab frozen; new recruit type "
                                f"{r!r} aliasing to overflow."
                            )
                        continue
                    if len(type_to_id) >= cap:
                        # Already warned above on the unit pass; quiet
                        # here to avoid log-spam on a single state.
                        continue
                    type_to_id[r] = len(type_to_id)
                    self._note_new_type(r)

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
        # Kept True on DML after the ablation -- see the matching
        # comment in encode_from_raw_batch.
        # [gpu-perf B3] On CUDA, non_blocking=True is SILENTLY SYNCHRONOUS
        # from pageable (un-pinned) numpy memory, so we pin the host buffer
        # first to get real async DMA overlap. pin_memory() is a host->
        # pinned copy (small cost) that only pays off if it overlaps real
        # compute -- profile on the GPU node before assuming a win.
        nb = device.type != "cpu"
        _pin = device.type == "cuda"

        def _to_dev(arr_or_tensor):
            t = torch.from_numpy(arr_or_tensor)
            if device.type != "cpu":
                if _pin:
                    t = t.pin_memory()
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
            hd = _to_dev(raw.hex_dynamic_flags)
            hex_tokens = (
                self.pos_x_embed(hx)
                + self.pos_y_embed(hy)
                + self.terrain_tokens(ht)
                + self.modifier_proj(hm)
                + self.dynamic_flag_proj(hd)
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
            hex_subset=raw.hex_subset,
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
            recruit_is_ours_np=raw.recruit_is_ours,  # zero-copy view
            visible_unit_ids=frozenset(raw.unit_ids),  # opt #3
            material=torch.tensor([[float(raw.material)]], dtype=torch.float32,
                                  device=global_token.device),
            observation=getattr(raw, "observation", None),
        )

    def encode_from_raw_padded(
        self,
        raws: List[RawEncoded],
        *,
        device: Optional[torch.device] = None,
    ):
        """Server fast path (2026-09-04): the padded token streams the
        model's `forward_streams` consumes, built straight from the
        RawEncodeds -- no per-sample EncodedState, no position dicts
        (the inference server never reads them; they cost ~1 ms per
        leaf of the serve thread). Returns (hex [B, H_max, d],
        unit [B, U_max, d], recruit [B, R_max, d], global [B, 1, d],
        end_turn [B, 1, d], sizes [(U, R, H)])."""
        if device is None:
            device = next(self.parameters()).device
        emb = self._embed_streams(raws, device)
        d = self.d_model
        B = len(raws)

        def _pad(cat, lengths):
            if cat is None:
                return torch.zeros(B, 0, d, device=device)
            return torch.nn.utils.rnn.pad_sequence(
                list(torch.split(cat, lengths)), batch_first=True)

        hex_b = _pad(emb["hex"], emb["Hs"])
        unit_b = _pad(emb["unit"], emb["Us"])
        recruit_b = _pad(emb["recruit"], emb["Rs"])
        global_b = emb["global"].unsqueeze(1)                         # [B, 1, d]
        end_b = self.end_turn_token.view(1, 1, -1).expand(B, 1, d)
        sizes = list(zip(emb["Us"], emb["Rs"], emb["Hs"]))
        return hex_b, unit_b, recruit_b, global_b, end_b, sizes

    def _hex_embedding(self, xs, ys, terrain_ids, modifier_flags, dynamic_flags):
        """One hex token per row. The sum order here is the one every
        batched path shares, so their embeddings are the same tensor."""
        return (self.pos_x_embed(xs) + self.pos_y_embed(ys)
                + self.terrain_tokens(terrain_ids) + self.modifier_proj(modifier_flags)
                + self.dynamic_flag_proj(dynamic_flags))

    def _unit_embedding(self, type_ids, side_ids, xs, ys, feats):
        """Unit tokens, and recruit tokens: a recruit is a phantom unit at
        its leader's keep, embedded by the same tables and projection so
        the actor head sees "would-be Fighter at our keep" the way it
        sees "Fighter standing here"."""
        return (self.unit_type_embed(type_ids) + self.side_embed(side_ids)
                + self.pos_x_embed(xs) + self.pos_y_embed(ys) + self.unit_feat_proj(feats))

    def _global_embedding(self, feats, our_faction_ids, their_faction_ids):
        return (self.global_proj(feats) + self.our_faction_embed(our_faction_ids)
                + self.their_faction_embed(their_faction_ids))

    # RawEncoded's numeric fields per stream: (name, dtype, per-row
    # shape). encode_from_raw_embedded concatenates each across the
    # batch into one buffer.
    _STREAM_FIELDS = (
        ("hex", (("hex_xs", torch.int64, ()), ("hex_ys", torch.int64, ()),
                 ("hex_terrain_ids", torch.int64, ()),
                 ("hex_modifier_flags", torch.float32, (NUM_HEX_MODIFIERS,)),
                 ("hex_dynamic_flags", torch.float32, (NUM_HEX_DYNAMIC_FLAGS,)))),
        ("unit", (("unit_type_ids", torch.int64, ()), ("unit_side_ids", torch.int64, ()),
                  ("unit_xs", torch.int64, ()), ("unit_ys", torch.int64, ()),
                  ("unit_feats", torch.float32, (UNIT_FEAT_DIM,)))),
        ("recruit", (("recruit_type_ids", torch.int64, ()), ("recruit_side_ids", torch.int64, ()),
                     ("recruit_xs", torch.int64, ()), ("recruit_ys", torch.int64, ()),
                     ("recruit_feats", torch.float32, (UNIT_FEAT_DIM,)))),
    )

    def encode_from_raw_embedded(
        self,
        raws: List[RawEncoded],
        *,
        device: Optional[torch.device] = None,
    ) -> EmbeddedStreams:
        """Server fast path (2026-09-05): the batch's token embeddings in
        stream order (packed_trunk.EmbeddedStreams) from ONE pinned host
        buffer and one non-blocking copy. Every numeric field of every
        RawEncoded is concatenated straight into the buffer
        (np.concatenate(out=)), the trained embeddings run on device
        views of it, and nothing here waits for the device. The padded
        path (_embed_streams) pins and copies its 17 fields one by one
        and moves the global features and the faction ids by blocking
        pageable copies, each a stream synchronization. Same
        expressions as _embed_streams, so the embeddings are the same;
        WesnothModel.forward_embedded consumes the result."""
        if device is None:
            device = next(self.parameters()).device
        B = len(raws)
        if B == 0:
            raise ValueError("encode_from_raw_embedded: empty batch")
        Hs = [r.hex_xs.shape[0] for r in raws]
        Us = [r.unit_xs.shape[0] for r in raws]
        Rs = [r.recruit_type_ids.shape[0] for r in raws]
        totals = {"hex": sum(Hs), "unit": sum(Us), "recruit": sum(Rs)}
        fields = [(name, dt, (totals[stream],) + shape)
                  for stream, spec in self._STREAM_FIELDS for name, dt, shape in spec]
        fields += [("global_feats", torch.float32, (B, GLOBAL_FEAT_DIM)),
                   ("our_faction_id", torch.int64, (B,)),
                   ("their_faction_id", torch.int64, (B,))]
        layout = FlatLayout(fields)
        host = torch.empty(layout.nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
        hv = layout.numpy_views(host.numpy())
        for stream, spec in self._STREAM_FIELDS:
            if totals[stream]:
                for name, _, _ in spec:
                    np.concatenate([getattr(r, name) for r in raws], out=hv[name])
        np.stack([r.global_feats for r in raws], out=hv["global_feats"])
        hv["our_faction_id"][:] = [r.our_faction_id for r in raws]
        hv["their_faction_id"][:] = [r.their_faction_id for r in raws]
        dev = host if device.type == "cpu" else host.to(device, non_blocking=True)
        v = layout.torch_views(dev)
        parts = []
        if totals["hex"]:
            parts.append(self._hex_embedding(v["hex_xs"], v["hex_ys"], v["hex_terrain_ids"],
                                             v["hex_modifier_flags"], v["hex_dynamic_flags"]))
        if totals["unit"]:
            parts.append(self._unit_embedding(v["unit_type_ids"], v["unit_side_ids"],
                                              v["unit_xs"], v["unit_ys"], v["unit_feats"]))
        if totals["recruit"]:
            parts.append(self._unit_embedding(v["recruit_type_ids"], v["recruit_side_ids"],
                                              v["recruit_xs"], v["recruit_ys"], v["recruit_feats"]))
        parts.append(self._global_embedding(v["global_feats"], v["our_faction_id"],
                                            v["their_faction_id"]))
        parts.append(self.end_turn_token.view(1, -1))
        return EmbeddedStreams(tokens=torch.cat(parts, dim=0), sizes=list(zip(Us, Rs, Hs)))

    def _embed_streams(self, raws: List[RawEncoded], device) -> dict:
        """The trained embeddings of every stream for a batch, as
        concatenated [total, d] tensors plus per-sample lengths (None
        for an all-empty stream). Shared by encode_from_raw_batch and
        encode_from_raw_padded.

        The batch crosses to the device in TWO transfers (one per
        dtype), not one per field. Seventeen separate
        `from_numpy -> pin_memory -> to(device)` round trips were most
        of the server's fixed per-batch host cost: a fresh pinned
        allocation is a synchronizing call, and the server pays it once
        per batch, not once per leaf (docs/gpu_forward_design_20260904.md
        section 1.1). The values are untouched -- each field is a view
        into the coalesced buffer at its own offset -- and every field
        starts on a 512-byte boundary, the alignment a fresh block from
        torch's caching allocator has, so the projections see operands
        aligned exactly as on the per-field path and cuBLAS picks the
        same kernels (an operand's alignment is one of its kernel
        selection inputs). Bit-identity with the per-field path is
        pinned on CPU (tests/test_encoder_transfer.py); the alignment
        is what carries it to CUDA."""
        nb = device.type != "cpu"
        _pin = device.type == "cuda"   # [gpu-perf B3] see encode_from_raw
        Hs = [r.hex_xs.shape[0] for r in raws]
        Us = [r.unit_xs.shape[0] for r in raws]
        Rs = [r.recruit_type_ids.shape[0] for r in raws]

        # Gather every field of one dtype into one buffer, remembering
        # where each lands so it can be sliced back out on the device.
        plan: Dict[str, List[tuple]] = {"i": [], "f": []}      # key -> (name, rows, width)
        pieces: Dict[str, List[np.ndarray]] = {"i": [], "f": []}

        def _stage(name: str, arrays: List[np.ndarray], kind: str, width: int) -> None:
            cat = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
            if cat.shape[1:] != ((width,) if width > 1 else ()):
                raise ValueError(f"{name}: field width {cat.shape[1:]} is not {width}; "
                                 f"a cached RawEncoded from another feature layout?")
            plan[kind].append((name, cat.shape[0], width))
            pieces[kind].append(cat.reshape(-1))

        if sum(Hs):
            for nm, f in (("hx", "hex_xs"), ("hy", "hex_ys"), ("ht", "hex_terrain_ids")):
                _stage(nm, [getattr(r, f) for r in raws], "i", 1)
            _stage("hm", [r.hex_modifier_flags for r in raws], "f", NUM_HEX_MODIFIERS)
            _stage("hd", [r.hex_dynamic_flags for r in raws], "f", NUM_HEX_DYNAMIC_FLAGS)
        if sum(Us):
            for nm, f in (("ut", "unit_type_ids"), ("us", "unit_side_ids"),
                          ("ux", "unit_xs"), ("uy", "unit_ys")):
                _stage(nm, [getattr(r, f) for r in raws], "i", 1)
            _stage("uf", [r.unit_feats for r in raws], "f", UNIT_FEAT_DIM)
            _stage("ui", [r.unit_is_ours for r in raws], "f", 1)
        if sum(Rs):
            for nm, f in (("rt", "recruit_type_ids"), ("rs", "recruit_side_ids"),
                          ("rx", "recruit_xs"), ("ry", "recruit_ys")):
                _stage(nm, [getattr(r, f) for r in raws], "i", 1)
            _stage("rf", [r.recruit_feats for r in raws], "f", UNIT_FEAT_DIM)
            _stage("ri", [r.recruit_is_ours for r in raws], "f", 1)
        _stage("gf", [np.stack([r.global_feats for r in raws])], "f", GLOBAL_FEAT_DIM)
        _stage("ofi", [np.array([r.our_faction_id for r in raws], dtype=np.int64)], "i", 1)
        _stage("tfi", [np.array([r.their_faction_id for r in raws], dtype=np.int64)], "i", 1)

        t: Dict[str, torch.Tensor] = {}
        for kind, dtype in (("i", np.int64), ("f", np.float32)):
            if not pieces[kind]:
                continue
            align = _ALLOCATOR_ALIGNMENT // np.dtype(dtype).itemsize   # elements per boundary
            offsets: List[int] = []
            total = 0
            for piece in pieces[kind]:
                total = -(-total // align) * align
                offsets.append(total)
                total += piece.size
            buf = np.zeros(total, dtype=dtype)
            for piece, off in zip(pieces[kind], offsets):
                buf[off:off + piece.size] = piece
            dev_buf = torch.from_numpy(buf)
            if device.type != "cpu":
                if _pin:
                    dev_buf = dev_buf.pin_memory()
                dev_buf = dev_buf.to(device, non_blocking=nb)
            for (name, rows, width), off in zip(plan[kind], offsets):
                n = rows * width
                view = dev_buf[off:off + n]
                t[name] = view.view(rows, width) if width > 1 else view

        out = {"Hs": Hs, "Us": Us, "Rs": Rs, "hex": None, "unit": None,
               "recruit": None, "unit_is": None, "recruit_is": None}
        if sum(Hs):
            out["hex"] = self._hex_embedding(t["hx"], t["hy"], t["ht"], t["hm"], t["hd"])
        if sum(Us):
            out["unit"] = self._unit_embedding(t["ut"], t["us"], t["ux"], t["uy"], t["uf"])
            out["unit_is"] = t["ui"]
        if sum(Rs):
            out["recruit"] = self._unit_embedding(t["rt"], t["rs"], t["rx"], t["ry"], t["rf"])
            out["recruit_is"] = t["ri"]
        out["global"] = self._global_embedding(t["gf"], t["ofi"], t["tfi"])   # [B, d]
        return out

    def encode_from_raw_batch(
        self,
        raws: List[RawEncoded],
        *,
        device: Optional[torch.device] = None,
    ) -> List[EncodedState]:
        """Batched version of `encode_from_raw`.

        For each embedding/projection (`pos_x_embed`, `terrain_embed`,
        `unit_type_embed`, etc.), the per-sample path pays a fixed
        kernel-launch overhead (~30 µs on GPU, ~5 µs on CPU). Across
        the 5 hex streams + 5 unit streams + 5 recruit streams + 4
        global ops, that's ~70 launches per state. At a typical
        training chunk of 8 states, fusing the launches concatenates
        all per-state arrays into one big call (5 + 5 + 5 + 4 = 19
        launches per CHUNK rather than 19×B), trimming ~3 ms of
        launch overhead per train_step on GPU.
        Result identity: each returned EncodedState matches what
        `encode_from_raw` would produce for that sample (verified by
        `test_encode_from_raw_batch_parity`). Downstream code is
        unchanged.
        """
        if not raws:
            return []
        if len(raws) == 1:
            return [self.encode_from_raw(raws[0], device=device)]

        if device is None:
            device = next(self.parameters()).device
        d = self.d_model
        B = len(raws)
        emb = self._embed_streams(raws, device)
        Hs, Us, Rs = emb["Hs"], emb["Us"], emb["Rs"]

        def _per(cat, lengths):
            if cat is None:
                return [torch.zeros(1, 0, d, device=device) for _ in raws]
            return [t.unsqueeze(0) for t in torch.split(cat, lengths)]

        def _per_flag(cat, lengths):
            if cat is None:
                return [torch.zeros(1, 0, device=device, dtype=torch.float32) for _ in raws]
            return [t.unsqueeze(0) for t in torch.split(cat, lengths)]

        hex_per = _per(emb["hex"], Hs)
        unit_per = _per(emb["unit"], Us)
        recruit_per = _per(emb["recruit"], Rs)
        unit_is_per = _per_flag(emb["unit_is"], Us)
        recruit_is_per = _per_flag(emb["recruit_is"], Rs)
        global_emb = emb["global"]

        # ---- assemble per-sample EncodedState objects ----
        end_turn_token = self.end_turn_token.view(1, 1, -1)
        results: List[EncodedState] = []
        for b in range(B):
            raw = raws[b]
            pos_to_hex = {
                (p.x, p.y): j for j, p in enumerate(raw.hex_positions)
            }
            results.append(EncodedState(
                hex_subset=raw.hex_subset,
                hex_tokens=hex_per[b],
                hex_positions=raw.hex_positions,
                pos_to_hex=pos_to_hex,
                unit_tokens=unit_per[b],
                unit_is_ours=unit_is_per[b],
                unit_positions=raw.unit_positions,
                unit_ids=raw.unit_ids,
                recruit_tokens=recruit_per[b],
                recruit_is_ours=recruit_is_per[b],
                recruit_types=raw.recruit_types,
                global_token=global_emb[b:b + 1].unsqueeze(1),  # [1, 1, d]
                end_turn_token=end_turn_token,
                recruit_is_ours_np=raw.recruit_is_ours,
                visible_unit_ids=frozenset(raw.unit_ids),  # opt #3
                material=torch.tensor([[float(raw.material)]], dtype=torch.float32,
                                      device=global_emb.device),
                observation=getattr(raw, "observation", None),
            ))
        return results


# ---------------------------------------------------------------------
# encode_raw — phase-1 of encoding. Pure Python, vocab read-only.
# ---------------------------------------------------------------------

def _clamp_pos(v: int) -> int:
    return max(0, min(v, MAX_MAP_SIZE - 1))


def names_on_overflow_row(type_to_id: Dict[str, int]) -> List[str]:
    """The type names whose id reaches the overflow row (MAX_UNIT_TYPES
    - 1): the encoder clamps them onto it, so they share one embedding
    row with each other and with every unknown name."""
    return sorted(name for name, i in type_to_id.items() if i >= MAX_UNIT_TYPES - 1)


def _lookup_id(name: str, table: Dict[str, int], maxn: int) -> int:
    """Read-only vocab lookup. Out-of-vocab → overflow bucket (maxn-1).

    Mirrors the clamping behavior of the old `_name_id` / `_faction_id`
    methods, but never mutates the dict — safe to call from worker
    processes that share a frozen view of the vocab.
    """
    return min(table.get(name, maxn - 1), maxn - 1)


def _side_info(sides: List[SideInfo], side: int) -> Optional[SideInfo]:
    """The SideInfo of side number `side`, or None when the state
    lists fewer sides."""
    return sides[side - 1] if 0 < side <= len(sides) else None


def encode_raw(
    game_state: GameState,
    *,
    type_to_id: Dict[str, int],
    faction_to_id: Dict[str, int],
    relevant_set: bool = False,
    fog_hides_enemy_villages: bool = False,
    terrain_multi_hot: bool = False,
) -> RawEncoded:
    """Build a `RawEncoded` from a GameState using read-only vocab.
    `terrain_multi_hot`: the hex stream carries each hex's terrain
    mask (Hex.terrain_mask) instead of its one class id.

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

    Python gathers the facts (slot orderings, vocab ids, fog and
    ownership predicates); the arrays are then composed either by one
    call into the Rust wheel (docs/rust_port_plan.md phase 2b, see
    `_rust_encode_kernel`) or by the `_python_*` builders below, which
    are the reference the wheel is certified against
    (tests/test_rust_encode_raw.py).

    The side to move must be a player's (`classes.PLAYER_SIDES`); the
    global features describe the other player's side as the enemy.
    A state whose side to move is not a player's raises ValueError.
    """
    current_side = game_state.global_info.current_side
    them_side = opponent_of(current_side)
    sides = game_state.sides

    # ---- hexes ----
    # Fog hexes are RETAINED. Wesnoth's fog of war hides UNITS on a
    # hex from sides that don't have vision there, but the TERRAIN
    # is still visible (the player saw the map at scenario start).
    # Dropping fog hexes from our hex token stream silently makes
    # them ineligible for the recruit hex mask (the BFS over the
    # leader's castle network would skip them) -- so the policy
    # could never attempt to recruit on fog castle hexes, even
    # though Wesnoth would happily accept the attempt and bounce
    # only if an enemy is actually there. Per the legality-mask
    # contract in CLAUDE.md, we want fog hexes attemptable; the
    # rejection-history feature handles the bounce case.
    #
    # We DON'T need to filter units in fog -- Wesnoth's state
    # collector already excludes invisible enemy units from
    # `gs.map.units` for the side we're encoding for, so any
    # hidden unit isn't in the input we see anyway.
    # Hex stream: the FULL board, or (opt-in) only the hexes that can
    # matter this decision. T2-A measured the relevant set at mean 0.30 of
    # the board with ZERO superset violations over 1,840 decisions, which
    # buys a 4.3-4.8x rollout forward -- the sequence length, not the
    # parameter count, is what gates scaling here. Both orderings come from
    # the SAME canonical (y,x) sort (relevant_hexes_in_slot_order FILTERS
    # it rather than re-sorting), so slot indices stay deterministic --
    # load-bearing, because the trainer replays target_idx against
    # re-encoded states.
    # One observation per decision through the Rust kernels (wesnoth_ai/
    # observe.py): the seen hexes, the visible units, the mask builder's
    # reach context and, in the relevant-set basis, the acting units'
    # landable rows and the relevant hex set; None on the Python path.
    from wesnoth_ai.observe import observe as _observe
    observation = _observe(game_state, current_side, reach=relevant_set)
    if relevant_set:
        if observation is not None and observation.relevant is not None:
            # The kernel's relevant mask over the cached full-board
            # arrays; the token index goes to the mask builder.
            static, observation.tok_of_hex = _relevant_subset_static(game_state, observation)
            hexes = static.hexes
            hex_positions = static.positions
        else:
            hexes = relevant_hexes_in_slot_order(game_state)   # slot contract
            hex_positions = [h.position for h in hexes]
            static = _build_static_hex_arrays(hexes) if hexes else None
    else:
        # Full board: the slot ordering and the static arrays are
        # cached per hex set (see _static_hex_arrays).
        static = _static_hex_arrays(game_state)
        hexes = static.hexes
        hex_positions = static.positions
        if observation is not None:
            observation.tok_of_hex = observation.geometry.full_slot
    H = len(hex_positions)

    # Per-turn rejection set (hexes a previous recruit attempt
    # bounced this turn). Stashed on global_info by the harness;
    # absent on fresh states. See CLAUDE.md legality-mask contract.
    rejected_hexes = (
        getattr(game_state.global_info, "_recruit_rejected_hexes", None)
        or set()
    )

    # The hexes the side to move sees, shared between the
    # village-ownership fog gate below and `units_visible_to` (which
    # otherwise reads them again) -- read lazily, at most ONCE per
    # encode.
    # Fog toggle: underscore attr so GlobalInfo.__deepcopy__ carries
    # it through MCTS state copies (non-underscore attrs are
    # dropped; adversarial review 2026-07-11).
    fog_on = getattr(game_state.global_info, "_fog", True)
    _seen_cache: list = []

    def _seen_hexes():
        if not _seen_cache:
            if observation is not None:
                _seen_cache.append(observation.seen_set())
            else:
                from wesnoth_ai.visibility import visible_hexes_for
                _seen_cache.append(
                    visible_hexes_for(game_state, current_side))
        return _seen_cache[0]

    # Static per-map arrays come from a cache keyed on the hex set's
    # identity (2026-09-04: the per-hex Python loop was ~1 ms of the
    # 1.35 ms encode; terrain never changes within a self-play game
    # -- morph events REPLACE gs.map.hexes, so the key changes with
    # them). Only the dynamic bits (village ownership as the mover
    # sees it, recruit rejections) are gathered per encode, over the
    # village hexes alone.
    if H == 0:
        village_entries: List[Tuple[int, int, bool]] = []
        rejected_slots: List[int] = []
    else:
        village_entries = _village_entries(
            static, game_state, current_side, fog_on, _seen_hexes)
        rejected_slots = _rejected_slots(static, rejected_hexes)

    # ---- units ----
    # Fog-of-war filter: the policy must only see units that the
    # current side could see in real Wesnoth. The sim runs god-
    # view internally (combat math etc. need ground truth), but
    # for the encoder's observation we MUST filter -- otherwise
    # the policy learns to use enemy positions it wouldn't have
    # access to at deploy time. `units_visible_to` honors the
    # three Wesnoth rules: own units always visible; enemy units
    # on hexes the side does not see hidden; enemy units with an
    # active hide-cover ability (ambush/concealment/submerge/
    # nightstalk) hidden until uncovered (sim manages the
    # `_uncovered_units` set per ambush-trigger). See
    # `visibility.py` for the full contract.
    # Slot contract (visibility.visible_units_in_slot_order): the
    # label builder and any other slot consumer share THIS
    # enumeration -- do not inline a sort here again.
    if observation is not None:
        # The kernel's visibility, in the slot contract's order.
        units = sorted(observation.visible_units(),
                       key=lambda u: (u.position.y, u.position.x, u.id))
    else:
        units = visible_units_in_slot_order(
            game_state, current_side,
            # Reuse the seen hexes if the village fog gate already read
            # them; None lets the filter read them lazily.
            vis_set=_seen_cache[0] if _seen_cache else None,
        )
    unit_positions = [u.position for u in units]
    unit_ids       = [u.id for u in units]

    # ---- recruits ----
    # Fog-of-war filter: only the CURRENT side's recruit phantoms
    # are emitted. In real Wesnoth a player never sees the enemy's
    # recruit list (or even confirms which units they CAN recruit
    # until one appears on the board). Previously we emitted
    # phantoms for every side -- a fog leak on two counts:
    #   1. The recruit type ids leaked which faction the enemy
    #      picked (visible to humans from the lobby anyway, but
    #      the policy shouldn't get a free token for it).
    #   2. The phantom's positional coords (lx, ly) leaked the
    #      enemy LEADER's keep coordinates -- god-view info that
    #      Wesnoth's fog hides until you scout it.
    # Looking up the current side's leader is straightforward;
    # we no longer need a per-side dict because we only need OUR
    # leader's position to coord-anchor OUR recruit phantoms.
    own_leader_xy: Tuple[int, int] = (0, 0)
    for u in game_state.map.units:
        if u.is_leader and u.side == current_side:
            own_leader_xy = (u.position.x, u.position.y)
            break
    # Recruit phantoms via the slot contract
    # (visibility.own_recruit_types): CURRENT side only, side_info
    # order -- the label builder resolves recruit slots through the
    # same function.
    own_recruits = own_recruit_types(game_state, current_side)

    # ---- global ----
    # The enemy is the other player's side, looked up by its number: a
    # replayed game lists a SideInfo for every side its scenario
    # declares, statues and tentacles included.
    gi = game_state.global_info
    us = _side_info(sides, current_side)
    them = _side_info(sides, them_side)
    our_gold       = us.current_gold if us else 0
    our_income     = us.base_income if us else 0
    our_villages   = us.nb_villages_controlled if us else 0
    their_villages = them.nb_villages_controlled if them else 0
    if fog_hides_enemy_villages and fog_on:
        # Global feature 5 was the enemy's TRUE village count on every
        # path (2026-09-08 contamination review): a player under fog
        # never sees it (visibility.enemy_villages_visible_to cites
        # the engine). Behind a checkpoint flag: the seed was trained
        # with the count, its encoding stays byte-identical.
        from wesnoth_ai.visibility import enemy_villages_visible_to
        their_villages = enemy_villages_visible_to(game_state, current_side, _seen_hexes())

    our_fac  = us.faction if us else ""
    them_fac = them.faction if them else ""
    our_faction_id   = _lookup_id(our_fac,  faction_to_id, MAX_FACTIONS)
    their_faction_id = _lookup_id(them_fac, faction_to_id, MAX_FACTIONS)

    # ---- arrays ----
    # The time of day the board is under, and the one it moves to next.
    # `_tod_start_offset` carries the slot a random-start scenario drew,
    # so this is the phase the game is actually in rather than the one
    # the turn number would imply.
    from tools.replay_dataset import _lawful_bonus_for_turn
    tod_offset = int(getattr(gi, "_tod_start_offset", 0) or 0)
    lawful_bonus = _lawful_bonus_for_turn(gi.turn_number, tod_offset)
    next_lawful_bonus = _lawful_bonus_for_turn(gi.turn_number + 1, tod_offset)

    kernel = _rust_encode_kernel()
    if kernel is not None:
        hex_arrays, unit_arrays, recruit_arrays, global_feats_np = _rust_streams(
            kernel, static, H, village_entries, rejected_slots, units,
            current_side, type_to_id, own_recruits, own_leader_xy,
            (gi.turn_number, current_side, our_gold, our_income,
             our_villages, their_villages, lawful_bonus, next_lawful_bonus))
    else:
        hex_arrays = _python_hex_arrays(static, H, village_entries, rejected_slots)
        unit_arrays = _python_unit_arrays(units, current_side, type_to_id)
        recruit_arrays = _python_recruit_arrays(own_recruits, own_leader_xy, type_to_id)
        global_feats_np = _python_global_feats(
            gi.turn_number, current_side, our_gold, our_income,
            our_villages, their_villages, lawful_bonus, next_lawful_bonus)
    hex_modifier_flags_np, hex_dynamic_flags_np = hex_arrays
    (unit_is_ours_np, unit_type_ids_np, unit_side_ids_np,
     unit_xs_np, unit_ys_np, unit_feats_np) = unit_arrays
    (recruit_is_ours_np, recruit_type_ids_np, recruit_side_ids_np,
     recruit_xs_np, recruit_ys_np, recruit_feats_np) = recruit_arrays

    return RawEncoded(
        hex_subset=relevant_set,
        hex_positions=hex_positions,
        hex_xs=static.xs if H else np.empty(0, dtype=np.int64),
        hex_ys=static.ys if H else np.empty(0, dtype=np.int64),
        hex_terrain_ids=((static.terrain_masks if terrain_multi_hot else static.terrain_ids)
                         if H else np.empty(0, dtype=np.int64)),
        hex_modifier_flags=hex_modifier_flags_np,
        hex_dynamic_flags=hex_dynamic_flags_np,
        unit_positions=unit_positions,
        unit_ids=unit_ids,
        unit_is_ours=unit_is_ours_np,
        unit_type_ids=unit_type_ids_np,
        unit_side_ids=unit_side_ids_np,
        unit_xs=unit_xs_np,
        unit_ys=unit_ys_np,
        unit_feats=unit_feats_np,
        recruit_types=own_recruits,
        recruit_is_ours=recruit_is_ours_np,
        recruit_type_ids=recruit_type_ids_np,
        recruit_side_ids=recruit_side_ids_np,
        recruit_xs=recruit_xs_np,
        recruit_ys=recruit_ys_np,
        recruit_feats=recruit_feats_np,
        global_feats=global_feats_np,
        our_faction_id=our_faction_id,
        their_faction_id=their_faction_id,
        material=material_of_units(units, current_side),
        observation=observation.detached() if observation is not None else None,
    )


# ---------------------------------------------------------------------
# encode_raw facts: the per-encode predicates Python owns
# ---------------------------------------------------------------------

def _village_entries(static, game_state, current_side, fog_on,
                     seen_hexes) -> List[Tuple[int, int, bool]]:
    """Village ownership as the mover sees it, one (hex slot, owner
    code, owner visible) per candidate hex. Candidates are the hexes
    carrying the village MODIFIER (the static village bit) plus every
    owner-map entry on the board (an owned hex without the modifier
    gets the village bit too). Owner code: 1 = ours, 2 = another
    side's, 0 = neutral. Owner visible is the fog gate: own villages
    always, others when fog is off or the mover sees the hex --
    evaluated in that order, so the seen hexes are read only when a
    candidate that is not ours needs them."""
    village_owner_map = (getattr(
        game_state.global_info, "_village_owner", None) or {})
    cand = set(static.village_idx)
    if village_owner_map:
        pos_index = static.pos_index
        for key in village_owner_map:
            j = pos_index.get(key)
            if j is not None:
                cand.add(j)
    entries: List[Tuple[int, int, bool]] = []
    for i in cand:
        key = static.keys[i]
        owner = village_owner_map.get(key, 0)
        ours = owner == current_side
        visible = ours or not fog_on or key in seen_hexes()
        code = 1 if ours else (2 if owner not in (0, current_side) else 0)
        entries.append((i, code, visible))
    return entries


def _rejected_slots(static, rejected_hexes) -> List[int]:
    """Hex slots of this turn's bounced recruit attempts."""
    if not rejected_hexes:
        return []
    pos_index = static.pos_index
    return [j for j in (pos_index.get(key) for key in rejected_hexes)
            if j is not None]


# ---------------------------------------------------------------------
# encode_raw arrays, Rust path (docs/rust_port_plan.md phase 2b)
# ---------------------------------------------------------------------

_EMPTY_HEX_MODIFIERS = np.zeros((0, NUM_HEX_MODIFIERS), dtype=np.float32)
_EMPTY_I64 = np.zeros(0, dtype=np.int64)
_EMPTY_F64 = np.zeros(0, dtype=np.float64)


# The wheel phase whose `encode_raw_streams` composes the feature
# widths this module expects. A kernel older than this emits a
# narrower row -- a phase-10 wheel writes 6 global features where
# GLOBAL_FEAT_DIM is now 8 -- and numpy would broadcast or raise far
# from the cause, so the mismatch is caught here and the Python
# builders take over.
_ENCODE_KERNEL_PHASE = 11
_warned_stale_kernel = False


def _rust_encode_kernel():
    """The wheel's `encode_raw_streams`, selected exactly as
    tools.pathfind_sim selects its kernels (wheel importable and
    WESNOTH_RUST != 0), else None for the Python builders. A wheel
    built before phase 2b lacks the function, and one built before
    `_ENCODE_KERNEL_PHASE` composes the wrong widths; both take the
    Python path."""
    global _warned_stale_kernel
    from tools import pathfind_sim
    kernel = getattr(pathfind_sim._RUST, "encode_raw_streams", None)
    if kernel is None:
        return None
    phase = int(getattr(pathfind_sim._RUST, "__phase__", 0) or 0)
    if phase < _ENCODE_KERNEL_PHASE:
        if not _warned_stale_kernel:
            _warned_stale_kernel = True
            log.warning(
                "wesnoth_core is phase %d; the encode kernel composes the "
                "feature widths of phase %d or later (GLOBAL_FEAT_DIM=%d). "
                "Using the PYTHON encoders, which are slower but current. "
                "Rebuild the wheel: pip install rust/wesnoth_core.",
                phase, _ENCODE_KERNEL_PHASE, GLOBAL_FEAT_DIM)
        return None
    return kernel


def _rust_streams(kernel, static, H, village_entries, rejected_slots,
                  units, current_side, type_to_id, own_recruits,
                  own_leader_xy, global_values):
    """Every RawEncoded array in one wheel call: Python gathers the
    per-object facts as flat arrays, the kernel composes the features
    in the reference builders' float order
    (rust/wesnoth_core/src/encode.rs)."""
    unit_ints, unit_stats = _unit_rows(units, current_side, type_to_id)
    recruit_ids, recruit_stats = _recruit_rows(own_recruits, type_to_id)
    leader_x, leader_y = own_leader_xy
    return kernel(
        static.modifier_flags if H else _EMPTY_HEX_MODIFIERS,
        _flat_i64([v for entry in village_entries for v in entry]),
        _flat_i64(rejected_slots),
        _flat_i64(unit_ints), _flat_f64(unit_stats),
        _flat_i64(recruit_ids), _flat_f64(recruit_stats),
        leader_x, leader_y, global_values,
        (HP_NORM, MOVES_NORM, EXP_NORM, COST_NORM,
         GOLD_NORM, INCOME_NORM, VILLAGES_NORM, TURN_NORM),
        MAX_MAP_SIZE - 1, NUM_ALIGNMENTS)


def _flat_i64(values) -> np.ndarray:
    return np.array(values, dtype=np.int64) if values else _EMPTY_I64


def _flat_f64(values) -> np.ndarray:
    return np.array(values, dtype=np.float64) if values else _EMPTY_F64


def _unit_rows(units, current_side, type_to_id):
    """Per visible unit, the facts the kernel composes (its column
    constants): (type id, side code, x, y, alignment, is_leader,
    has_attacked) and (max_hp, current_hp, max_moves, current_moves,
    max_exp, current_exp, cost). Side code as `_python_unit_arrays`:
    scenery is neutral (2) even on our side; armed side>=3 units are
    enemies (1)."""
    overflow = MAX_UNIT_TYPES - 1
    ints: List[int] = []
    stats: List[float] = []
    for u in units:
        p = u.position
        ints += (min(type_to_id.get(u.name, overflow), overflow),
                 2 if is_scenery_unit(u)
                 else (0 if u.side == current_side else 1),
                 p.x, p.y, u.alignment.value,
                 1 if u.is_leader else 0, 1 if u.has_attacked else 0)
        stats += (u.max_hp, u.current_hp, u.max_moves, u.current_moves,
                  u.max_exp, u.current_exp, u.cost)
    return ints, stats


def _recruit_rows(own_recruits, type_to_id):
    """Per recruit option: the vocab id and `_recruit_stats_for`."""
    overflow = MAX_UNIT_TYPES - 1
    ids = [min(type_to_id.get(name, overflow), overflow)
           for name in own_recruits]
    stats: List[float] = []
    for name in own_recruits:
        stats += _recruit_stats_for(name)
    return ids, stats


# ---------------------------------------------------------------------
# encode_raw arrays, Python path: the reference the wheel is
# certified against (tests/test_rust_encode_raw.py). Keep verbatim.
# ---------------------------------------------------------------------

def _python_hex_arrays(static, H, village_entries, rejected_slots):
    if H == 0:
        return (np.empty((0, NUM_HEX_MODIFIERS), dtype=np.float32),
                np.empty((0, NUM_HEX_DYNAMIC_FLAGS), dtype=np.float32))
    hex_modifier_flags_np = static.modifier_flags.copy()
    hex_dynamic_flags_np = np.zeros((H, NUM_HEX_DYNAMIC_FLAGS), dtype=np.float32)
    for i, code, visible in village_entries:
        if visible:
            hex_modifier_flags_np[i, 0] = 1.0
        if code == 1:
            hex_dynamic_flags_np[i, 1] = 1.0
        elif code == 2 and visible:
            hex_dynamic_flags_np[i, 2] = 1.0
    for j in rejected_slots:
        hex_dynamic_flags_np[j, 0] = 1.0
    return hex_modifier_flags_np, hex_dynamic_flags_np


def _python_unit_arrays(units, current_side, type_to_id):
    MAP_LIMIT = MAX_MAP_SIZE - 1   # avoid attribute lookup in tight loops
    U = len(units)
    unit_is_ours_np  = np.empty(U, dtype=np.float32)
    unit_type_ids_np = np.empty(U, dtype=np.int64)
    unit_side_ids_np = np.empty(U, dtype=np.int64)
    unit_xs_np       = np.empty(U, dtype=np.int64)
    unit_ys_np       = np.empty(U, dtype=np.int64)
    unit_feats_np    = np.empty((U, UNIT_FEAT_DIM), dtype=np.float32)
    type_overflow    = MAX_UNIT_TYPES - 1
    for i, u in enumerate(units):
        is_neutral = is_scenery_unit(u)
        # Scenery is board furniture even if nominally on our side:
        # neutral code AND is_ours=0 (matches the legality mask,
        # where it is inert and never an actor). Armed side>=3
        # combatants (tentacles) encode as ENEMIES (code 1) -- they
        # are hostile and attackable (2026-07-14).
        is_ours = u.side == current_side and not is_neutral
        unit_is_ours_np[i]  = 1.0 if is_ours else 0.0
        unit_side_ids_np[i] = (2 if is_neutral
                               else (0 if is_ours else 1))
        unit_type_ids_np[i] = min(
            type_to_id.get(u.name, type_overflow), type_overflow
        )
        ux, uy = u.position.x, u.position.y
        unit_xs_np[i] = 0 if ux < 0 else (MAP_LIMIT if ux > MAP_LIMIT else ux)
        unit_ys_np[i] = 0 if uy < 0 else (MAP_LIMIT if uy > MAP_LIMIT else uy)
        unit_feats_np[i] = _unit_features(u)
    return (unit_is_ours_np, unit_type_ids_np, unit_side_ids_np,
            unit_xs_np, unit_ys_np, unit_feats_np)


def _python_recruit_arrays(own_recruits, own_leader_xy, type_to_id):
    MAP_LIMIT = MAX_MAP_SIZE - 1
    type_overflow = MAX_UNIT_TYPES - 1
    recruit_is_ours: List[float] = []
    recruit_type_ids: List[int]  = []
    recruit_side_ids: List[int]  = []
    recruit_xs: List[int] = []
    recruit_ys: List[int] = []
    recruit_feats_rows: List[np.ndarray] = []
    if own_recruits:
        lx, ly = own_leader_xy
        lx_clamped = 0 if lx < 0 else (MAP_LIMIT if lx > MAP_LIMIT else lx)
        ly_clamped = 0 if ly < 0 else (MAP_LIMIT if ly > MAP_LIMIT else ly)
        for name in own_recruits:
            recruit_is_ours.append(1.0)
            recruit_type_ids.append(
                min(type_to_id.get(name, type_overflow), type_overflow)
            )
            recruit_side_ids.append(0)   # 0 = ours
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
    return (recruit_is_ours_np, recruit_type_ids_np, recruit_side_ids_np,
            recruit_xs_np, recruit_ys_np, recruit_feats_np)


def _python_global_feats(turn_number, current_side, our_gold, our_income,
                         our_villages, their_villages,
                         lawful_bonus, next_lawful_bonus) -> np.ndarray:
    return np.array([
        turn_number / TURN_NORM,
        (current_side - 1.5) * 2.0,   # 1 → -1, 2 → +1
        our_gold       / GOLD_NORM,
        our_income     / INCOME_NORM,
        our_villages   / VILLAGES_NORM,
        their_villages / VILLAGES_NORM,
        lawful_bonus      / LAWFUL_BONUS_NORM,
        next_lawful_bonus / LAWFUL_BONUS_NORM,
    ], dtype=np.float32)


# ---------------------------------------------------------------------
# Static per-map hex arrays (encode_raw fast path)
# ---------------------------------------------------------------------

@dataclass
class _StaticHexArrays:
    xs: np.ndarray
    ys: np.ndarray
    terrain_ids: np.ndarray          # one class per hex (the legacy view)
    terrain_masks: np.ndarray        # the hex's terrain set as a bitmask; a hex
                                     # with no resolved set carries its class bit
    modifier_flags: np.ndarray       # [H, NUM_HEX_MODIFIERS]; column 0 (owned
                                     # village) is left 0 and set per encode
    village_idx: List[int]           # hex indices carrying the village terrain
                                     # or modifier
    village_flags: np.ndarray        # [H] bool, the same fact per slot
    keys: List[Tuple[int, int]]      # (x, y) per hex index
    pos_index: Dict[Tuple[int, int], int]
    hex_set: object                  # the set the entry was built from: a
                                     # strong reference, so its id cannot be
                                     # recycled while the entry exists
    n_hexes: int
    hexes: List                      # the hexes in slot order
    positions: List                  # their Position objects


_STATIC_HEX_CACHE: Dict[int, _StaticHexArrays] = {}


def _build_static_hex_arrays(hexes, hex_set=None) -> _StaticHexArrays:
    MAP_LIMIT = MAX_MAP_SIZE - 1
    H = len(hexes)
    xs = np.empty(H, dtype=np.int64)
    ys = np.empty(H, dtype=np.int64)
    tids = np.empty(H, dtype=np.int64)
    tmasks = np.empty(H, dtype=np.int64)
    mods_np = np.zeros((H, NUM_HEX_MODIFIERS), dtype=np.float32)
    village_idx: List[int] = []
    village_flags = np.zeros(H, dtype=bool)
    keys: List[Tuple[int, int]] = []
    terrain_village = Terrain.VILLAGE
    terrain_castle = Terrain.CASTLE
    terrain_flat_v = Terrain.FLAT.value
    for i, h in enumerate(hexes):
        p = h.position
        keys.append((p.x, p.y))
        xs[i] = 0 if p.x < 0 else (MAP_LIMIT if p.x > MAP_LIMIT else p.x)
        ys[i] = 0 if p.y < 0 else (MAP_LIMIT if p.y > MAP_LIMIT else p.y)
        tt = h.terrain_types
        if not tt:
            tids[i] = terrain_flat_v
        elif terrain_village in tt:
            tids[i] = terrain_village.value
        elif terrain_castle in tt:
            tids[i] = terrain_castle.value
        else:
            tids[i] = next(iter(tt)).value
        mask = int(getattr(h, "terrain_mask", 0) or 0)
        tmasks[i] = mask if mask else (1 << int(tids[i]))
        mods = h.modifiers
        if TerrainModifiers.VILLAGE in mods:
            village_idx.append(i)
            village_flags[i] = True
        if TerrainModifiers.KEEP in mods:
            mods_np[i, 1] = 1.0
        if TerrainModifiers.CASTLE in mods:
            mods_np[i, 2] = 1.0
    return _StaticHexArrays(
        xs=xs, ys=ys, terrain_ids=tids, terrain_masks=tmasks, modifier_flags=mods_np,
        village_idx=village_idx, village_flags=village_flags, keys=keys,
        pos_index={k: i for i, k in enumerate(keys)},
        hex_set=hex_set, n_hexes=H,
        hexes=list(hexes), positions=[h.position for h in hexes])


def _subset_static(full: _StaticHexArrays, idx: np.ndarray) -> _StaticHexArrays:
    """The static arrays of the slots `idx` (ascending, so the subset
    keeps the full board's row-major order) gathered from the cached
    full-board arrays: no per-hex Python."""
    keys = [full.keys[i] for i in idx.tolist()]
    return _StaticHexArrays(
        xs=full.xs[idx], ys=full.ys[idx], terrain_ids=full.terrain_ids[idx],
        terrain_masks=full.terrain_masks[idx],
        modifier_flags=full.modifier_flags[idx],
        village_idx=np.flatnonzero(full.village_flags[idx]).tolist(),
        village_flags=full.village_flags[idx], keys=keys,
        pos_index={k: j for j, k in enumerate(keys)},
        hex_set=None, n_hexes=len(keys),
        hexes=[full.hexes[i] for i in idx.tolist()],
        positions=[full.positions[i] for i in idx.tolist()])


def _relevant_subset_static(game_state, observation) -> Tuple[_StaticHexArrays, np.ndarray]:
    """The relevant subset's static arrays and the map-to-token index
    from the observation's relevant mask (wesnoth_ai/observe.py): the
    subset in the full board's slot order, as
    `visibility.relevant_hexes_in_slot_order` filters it."""
    full = _static_hex_arrays(game_state)
    geom = observation.geometry
    if len(geom.keys) != full.n_hexes:
        raise ValueError("observation geometry and static hex arrays disagree")
    rel_slot = np.zeros(full.n_hexes, dtype=bool)
    rel_slot[geom.full_slot[observation.relevant != 0]] = True
    idx = np.flatnonzero(rel_slot)
    sub_of_full = np.full(full.n_hexes, -1, dtype=np.int64)
    sub_of_full[idx] = np.arange(len(idx), dtype=np.int64)
    return _subset_static(full, idx), sub_of_full[geom.full_slot]


def _static_hex_arrays(game_state) -> _StaticHexArrays:
    """Cached slot ordering + static arrays for the full board, keyed
    on the identity of `game_state.map.hexes` (aliased across forks;
    replaced, never mutated, by terrain-morph events). The entry holds
    the set itself, so a hit is an identity match (`is`) and a freed
    set's address can never serve another map (2026-09-04 review: the
    earlier weakref anchor was kept alive by the entry's own hex list,
    which made that guard vacuous)."""
    hex_set = game_state.map.hexes
    key = id(hex_set)
    hit = _STATIC_HEX_CACHE.get(key)
    if hit is not None and hit.hex_set is hex_set and hit.n_hexes == len(hex_set):
        return hit
    built = _build_static_hex_arrays(hexes_in_slot_order(game_state), hex_set)
    if len(_STATIC_HEX_CACHE) >= 64:      # a few maps per process
        _STATIC_HEX_CACHE.clear()
    _STATIC_HEX_CACHE[key] = built
    return built

# ---------------------------------------------------------------------
# Plain-python helpers (no torch) — keep them out of the module so
# they're easy to unit-test.
# ---------------------------------------------------------------------

def _first_terrain_id(terrain_types) -> int:
    """Pick one terrain id per hex; Hex.terrain_types can have several.

    Priority: VILLAGE > CASTLE > the first member the set yields, FLAT
    for an empty set. The set holds IntEnum members, whose hash is their
    value, so "first" does not depend on the process's hash seed.
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

_RECRUIT_STATS_CACHE: Dict[str, Tuple[float, float, float, float, int]] = {}
_RECRUIT_FEATS_CACHE: Dict[str, np.ndarray] = {}
_RECRUIT_DB_WARN_FIRED: bool = False  # one-shot flag for unit-DB load fallback
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


def _recruit_stats_for(unit_type: str) -> Tuple[float, float, float, float, int]:
    """(max_hp, max_moves, max_exp, cost, alignment index) of a
    recruit option of `unit_type`, from the unit-stats DB. Cached to
    avoid re-reading the DB on every encoding call."""
    cached = _RECRUIT_STATS_CACHE.get(unit_type)
    if cached is not None:
        return cached
    # Lazy import: tools.replay_dataset already loads the unit DB on
    # first access; reuse it rather than re-parsing the JSON.
    # Narrow except: legitimate failures here are the lazy import
    # itself (ImportError -- tools/ not on path) or the unit-db load
    # propagating an OS / JSON error past replay_dataset's
    # FileNotFoundError handler. Anything else is a bug we want loud.
    try:
        from tools.replay_dataset import _stats_for, _load_unit_db
        _load_unit_db()
        stats = _stats_for(unit_type)
    except (ImportError, OSError, ValueError) as exc:
        # ValueError covers json.JSONDecodeError (subclass) for a
        # corrupt unit_stats.json. Warn once -- if this fires every
        # encoding call we'd flood logs.
        if not _RECRUIT_DB_WARN_FIRED:
            globals()["_RECRUIT_DB_WARN_FIRED"] = True
            log.warning(
                "Falling back to default recruit stats for %s: %s: %s. "
                "All recruit feature embeddings will use FALLBACK values "
                "until unit_stats.json is fixed.",
                unit_type, type(exc).__name__, exc,
            )
        stats = _FALLBACK_RECRUIT_STATS
    out = (float(stats.get("hitpoints", 33)),
           float(stats.get("moves", 5)),
           float(stats.get("experience", 50)),
           float(stats.get("cost", 14)),
           _alignment_value(stats.get("alignment", "neutral")))
    _RECRUIT_STATS_CACHE[unit_type] = out
    return out


def _recruit_features_for(unit_type: str) -> np.ndarray:
    """Return a [UNIT_FEAT_DIM] float32 phantom feature vector for
    a recruit option of `unit_type`; cached per type."""
    cached = _RECRUIT_FEATS_CACHE.get(unit_type)
    if cached is not None:
        return cached
    max_hp, max_mv, max_xp, cost, align = _recruit_stats_for(unit_type)
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
    _RECRUIT_FEATS_CACHE[unit_type] = out
    return out
