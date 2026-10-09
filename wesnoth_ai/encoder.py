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
from collections import Counter
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
    Terrain,
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
from wesnoth_ai.visibility import (  # noqa: E402
    hexes_in_slot_order)


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
#                          BOTH flags 0 = appears neutral, as the
#                          game draws it: `display::get_flag` shows
#                          an enemy's flag only on an unfogged hex
#                          (src/display.cpp:339-356, 1.18.4).
#   Neutral / non-village hexes carry 0/0.
#
# Moves have no rejection flag: a move onto a hex a hidden unit holds
# stops next to it and reveals it (rust/wesnoth_core/src/core_move.rs).
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

# ---------------------------------------------------------------------
# The parity observation: the widths a network built with the checkpoint
# flag `observation_parity` reads (2026-09-29,
# docs/parity_memory_design_20260929.md). Without the flag a network reads
# the widths above, which are obs8's.
# ---------------------------------------------------------------------
PARITY_WEAPON_SLOTS = 3        # the most attacks of the 190 reachable unit types
PARITY_DAMAGE_TYPES = ("blade", "pierce", "impact", "fire", "cold", "arcane")
PARITY_SPECIALS = ("magical", "poison", "slow", "marksman", "firststrike",
                   "drains", "backstab", "charge", "plague", "berserk")
PARITY_TRAITS = ("strong", "quick", "intelligent", "resilient", "healthy", "dextrous",
                 "weak", "slow", "dim", "fearless", "undead", "feral", "elemental")
PARITY_ABILITIES = ("leadership", "skirmisher", "regenerate", "submerge", "cures",
                    "ambush", "heals_4", "steadfast", "feeding", "illuminates",
                    "nightstalk", "concealment", "teleport")
# One weapon slot: present, damage, strikes, ranged, the damage type
# (one-hot) and the specials (multi-hot).
PARITY_WEAPON_COLS = 4 + len(PARITY_DAMAGE_TYPES) + len(PARITY_SPECIALS)      # 20
# Weapon slots, resistances, traits, abilities, poisoned and slowed, the
# time of day at the unit's hex, the leadership it receives.
PARITY_UNIT_EXTRA = (PARITY_WEAPON_SLOTS * PARITY_WEAPON_COLS + len(PARITY_DAMAGE_TYPES)
                     + len(PARITY_TRAITS) + len(PARITY_ABILITIES) + 2 + 1 + 1)  # 96
UNIT_FEAT_DIM_PARITY = UNIT_FEAT_DIM + PARITY_UNIT_EXTRA                      # 109
WEAPON_DAMAGE_NORM = 40.0      # Dwarvish Dragonguard, the most of the reachable types
WEAPON_STRIKES_NORM = 6.0      # Inferno Drake, likewise
# + the side sees the hex now (the fog overlay), + the hex's lawful bonus
# minus the board's / LAWFUL_BONUS_NORM.
NUM_HEX_DYNAMIC_FLAGS_PARITY = NUM_HEX_DYNAMIC_FLAGS + 2                      # 5
# + village gold, village support, own net income, fog on, and with fog off
# the enemy's gold, net income and upkeep.
GLOBAL_FEAT_DIM_PARITY = GLOBAL_FEAT_DIM + 7                                   # 15
NUM_TERRAINS_PARITY = 16       # + FUNGUS (mushroom grove), REEF
# + 3: an enemy unit remembered from the watched turn (a sighting token).
NUM_SIDE_CODES_PARITY = 4
SIGHT_SIDE_CODE = 3
SIGHT_FEAT_DIM = 2             # hp / max_hp, max_hp / HP_NORM
LEADERSHIP_NORM = 25.0         # leadership gives 25 percent per level of difference
VILLAGE_GOLD_NORM = 8.0
VILLAGE_SUPPORT_NORM = 4.0

# The parity unit row, column by column (GameCore.encode_streams builds it,
# rust/wesnoth_core/src/core_parity.rs):
#     0-12    obs8's 13 columns (UNIT_NUMERIC_FEATS, then the alignment one-hot)
#    13-72    weapon slots 0, 1 and 2, the unit's attacks in attack order (the
#             weapon head's index); slot k starts at 13 + 20k:
#               +0      present
#               +1      damage per strike after traits and objects / WEAPON_DAMAGE_NORM
#               +2      strikes / WEAPON_STRIKES_NORM
#               +3      ranged
#               +4..9   damage type, one-hot in PARITY_DAMAGE_TYPES order
#               +10..19 specials, multi-hot in PARITY_SPECIALS order
#    73-78    resistances, (100 - damage percent) / 100, PARITY_DAMAGE_TYPES order
#    79-91    traits, multi-hot in PARITY_TRAITS order
#    92-104   abilities, multi-hot in PARITY_ABILITIES order
#    105      poisoned
#    106      slowed
#    107      the combat modifier of the unit's alignment and fearless trait under
#             the illuminated time of day at its hex / LAWFUL_BONUS_NORM
#    108      the leadership bonus the unit fights with now / LEADERSHIP_NORM
# A recruit row has the same layout from its type's base values: full hit
# points, 0 moves, has_attacked 1, the experience cap scaled by the game's
# experience modifier, the alignment coded as board units code it; the type's
# weapons, resistances and abilities; traits (rolled when recruited),
# statuses, time of day and leadership 0.
PARITY_WEAPON_AT = UNIT_FEAT_DIM                                                   # 13
PARITY_RESIST_AT = PARITY_WEAPON_AT + PARITY_WEAPON_SLOTS * PARITY_WEAPON_COLS      # 73
PARITY_TRAIT_AT = PARITY_RESIST_AT + len(PARITY_DAMAGE_TYPES)                       # 79
PARITY_ABILITY_AT = PARITY_TRAIT_AT + len(PARITY_TRAITS)                            # 92
PARITY_POISONED_AT = PARITY_ABILITY_AT + len(PARITY_ABILITIES)                      # 105
PARITY_SLOWED_AT = PARITY_POISONED_AT + 1                                           # 106
PARITY_TOD_AT = PARITY_SLOWED_AT + 1                                                # 107
PARITY_LEADERSHIP_AT = PARITY_TOD_AT + 1                                            # 108
# The parity hex dynamic columns: 0-2 obs8's (the recruit-rejected column is
# written 0), 3 the side sees the hex now (every hex with fog off), 4 the
# hex's lawful bonus minus the board's (its time area, lit terrain, and the
# illumination of the units the side sees) / LAWFUL_BONUS_NORM. The static
# village column (hex_modifier_flags 0) is set on every village hex, and the
# terrain stream is a mask over NUM_TERRAINS_PARITY classes.
PARITY_HEX_SEEN_AT = NUM_HEX_DYNAMIC_FLAGS                                          # 3
PARITY_HEX_TOD_AT = NUM_HEX_DYNAMIC_FLAGS + 1                                       # 4
# The parity global row: 0-7 obs8's, then
#     8   village gold / VILLAGE_GOLD_NORM
#     9   village support / VILLAGE_SUPPORT_NORM
#    10   own net income as the status table shows it / INCOME_NORM
#    11   fog on
#    12   the enemy's gold / GOLD_NORM            (fog off; 0 under fog)
#    13   the enemy's net income / INCOME_NORM    (fog off; 0 under fog)
#    14   the enemy's upkeep / INCOME_NORM        (fog off; 0 under fog)
# The sighting stream (RawEncoded.sight_*): each enemy unit the side saw since
# its last end_turn and does not see now, at the last hex it was seen on, in
# the unit stream's (y, x, id) order.


@dataclass(frozen=True)
class ObservationWidths:
    """The input widths of one observation, which size the encoder's
    tables and projections."""
    terrains: int
    hex_dynamic: int
    unit_feats: int
    side_codes: int
    global_feats: int


# obs8's widths, and the parity observation's (the constants above).
LEGACY_WIDTHS = ObservationWidths(NUM_TERRAINS, NUM_HEX_DYNAMIC_FLAGS, UNIT_FEAT_DIM,
                                  NUM_SIDE_CODES, GLOBAL_FEAT_DIM)
PARITY_WIDTHS = ObservationWidths(NUM_TERRAINS_PARITY, NUM_HEX_DYNAMIC_FLAGS_PARITY,
                                  UNIT_FEAT_DIM_PARITY, NUM_SIDE_CODES_PARITY,
                                  GLOBAL_FEAT_DIM_PARITY)

# Normalization divisors. Re-exported from `constants.py` so era
# mods can override them in one place; see the comment block in
# constants.py for scale rationale. The core's encoding reads them here
# (`game_core.CoreState.encode_raw`).
from wesnoth_ai.constants import (  # noqa: E402, F401 -- re-export point documented above
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

    # The parity observation's sighting stream ([1, S, d], S >= 0): enemy
    # units seen during the enemy's last turn and not visible now. None
    # from an encoder without `observation_parity`. Never an actor or a
    # target: the model reads these tokens and points at nothing in them.
    sighting_tokens: Optional[torch.Tensor] = None


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
    # of the recruit's side; recruit_feats = the unit features of a
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

    # The parity observation's inputs; None without `observation_parity`.
    # The posterior over the faction vocabulary for the opponent, and the
    # sighting stream: enemy units the side saw during the enemy's last turn
    # and cannot see now, each at the last hex it was seen on
    # (docs/parity_memory_design_20260929.md "The watched turn").
    their_faction_probs: Optional[np.ndarray] = None   # float32 [MAX_FACTIONS]
    sight_type_ids:      Optional[np.ndarray] = None   # int64 [S]
    sight_xs:            Optional[np.ndarray] = None   # int64 [S]
    sight_ys:            Optional[np.ndarray] = None   # int64 [S]
    sight_feats:         Optional[np.ndarray] = None   # float32 [S, SIGHT_FEAT_DIM]


@dataclass
class StagedBatch:
    """A batch's numeric fields on the device before any learned layer
    (`GameStateEncoder.stage_raws`): device views of one buffer, each
    stream's token total, the per-sample (U, R, H), the sighting counts
    and the name of the opponent's faction field."""
    views: Dict[str, torch.Tensor]
    totals: Dict[str, int]
    sizes: List[Tuple[int, int, int]]
    sighting_counts: Optional[List[int]]
    their_field: str


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
        observation_parity: bool = False,
        relevant_set_version: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        # The parity observation (docs/parity_memory_design_20260929.md,
        # "What the network observes"): its input widths, the sighting
        # stream and the faction posterior. Without it every table and
        # projection is obs8's. It rides the checkpoint like the terrain
        # view, and sizes parameters, so it is fixed at construction.
        self.observation_parity = bool(observation_parity)
        self.widths = PARITY_WIDTHS if self.observation_parity else LEGACY_WIDTHS
        # Which relevant set the hex stream holds when relevant_set_hexes
        # is on: 1, obs8's; 2, the parity recipe's (the neighbours of
        # every own unit and the unseen hexes near them added). Data
        # flow, like the fog gate: it rides the checkpoint.
        self.relevant_set_version = int(relevant_set_version)
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

        widths = self.widths
        # --- hex embeddings -------------------------------------------
        self.terrain_embed  = nn.Embedding(widths.terrains, d_model)
        # Bit positions for the multi-hot read of a terrain mask; not a
        # parameter, not in the state_dict.
        self.register_buffer("_terrain_bits", torch.arange(widths.terrains, dtype=torch.int64),
                             persistent=False)
        self.modifier_proj  = nn.Linear(NUM_HEX_MODIFIERS, d_model, bias=False)
        # Dynamic flags (e.g. recruit_rejected) live in their own
        # projection -- old checkpoints lacking this Linear initialize
        # it from scratch under load_checkpoint(strict=False); the
        # static modifier projection above is unaffected.
        self.dynamic_flag_proj = nn.Linear(widths.hex_dynamic, d_model, bias=False)

        # --- shared position embeddings --------------------------------
        self.pos_x_embed = nn.Embedding(MAX_MAP_SIZE, d_model)
        self.pos_y_embed = nn.Embedding(MAX_MAP_SIZE, d_model)

        # --- unit embeddings ------------------------------------------
        self.unit_type_embed = nn.Embedding(MAX_UNIT_TYPES, d_model)
        self.unit_feat_proj  = nn.Linear(widths.unit_feats, d_model)
        self.side_embed      = nn.Embedding(widths.side_codes, d_model)

        # --- global ---------------------------------------------------
        self.global_proj = nn.Linear(widths.global_feats, d_model)

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

        # --- sightings (the parity observation) ------------------------
        # A sighting token is a unit token of side code SIGHT_SIDE_CODE:
        # the type, the position, and hit points in place of the unit's
        # columns. Built last, so the tables above draw as without it.
        self.sight_feat_proj = (nn.Linear(SIGHT_FEAT_DIM, d_model)
                                if self.observation_parity else None)

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
        the encoder's own; the trainer passes its frozen snapshot. The
        parity observation and the relevant set's version 2 are built by
        the Rust core only (`encode_raw` refuses them for an unbound state)."""
        return encode_raw(
            game_state,
            type_to_id=self.unit_type_to_id if type_to_id is None else type_to_id,
            faction_to_id=self.faction_to_id if faction_to_id is None else faction_to_id,
            relevant_set=self.relevant_set_hexes,
            fog_hides_enemy_villages=self.fog_hides_enemy_villages,
            terrain_multi_hot=self.terrain_multi_hot,
            observation_parity=self.observation_parity,
            relevant_set_version=self.relevant_set_version,
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
        self._check_raw(raw)
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

        # ---- sightings (the parity observation) ----
        sighting_tokens = None
        if self.observation_parity:
            if raw.sight_type_ids.shape[0] == 0:
                sighting_tokens = torch.zeros(1, 0, d, device=device)
            else:
                sighting_tokens = self._sighting_embedding(
                    _to_dev(raw.sight_type_ids), _to_dev(raw.sight_xs),
                    _to_dev(raw.sight_ys), _to_dev(raw.sight_feats)).unsqueeze(0)  # [1, S, d]

        # ---- global ----
        # A small float vector + two ints — small enough that
        # torch.tensor scalar paths are fine; from_numpy on an 8-element
        # array would be a wash.
        gf = torch.from_numpy(raw.global_feats).unsqueeze(0)
        if device.type != "cpu":
            gf = gf.to(device, non_blocking=nb)
        our_fid  = torch.tensor([raw.our_faction_id],  device=device, dtype=torch.long)
        if self.observation_parity:
            them = _to_dev(raw.their_faction_probs).unsqueeze(0)            # [1, MAX_FACTIONS]
        else:
            them = torch.tensor([raw.their_faction_id], device=device, dtype=torch.long)
        global_token = self._global_embedding(gf, our_fid, them).unsqueeze(0)  # [1, 1, d]

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
            material=torch.tensor([[float(raw.material)]], dtype=torch.float32,
                                  device=global_token.device),
            observation=getattr(raw, "observation", None),
            sighting_tokens=sighting_tokens,
        )

    def _check_raw(self, raw: RawEncoded) -> None:
        """A record of this encoder's observation: the parity fields
        present exactly under `observation_parity`, every width the one
        its projections read. Refuses records from the other encoding
        with a message instead of a matrix shape error."""
        has_parity_fields = getattr(raw, "their_faction_probs", None) is not None
        if has_parity_fields != self.observation_parity:
            raise ValueError(
                f"a RawEncoded {'with' if has_parity_fields else 'without'} the parity "
                f"observation's fields reached an encoder "
                f"{'with' if self.observation_parity else 'without'} observation_parity")
        w = self.widths
        checks = [("unit_feats", raw.unit_feats, w.unit_feats),
                  ("recruit_feats", raw.recruit_feats, w.unit_feats),
                  ("hex_dynamic_flags", raw.hex_dynamic_flags, w.hex_dynamic),
                  ("global_feats", raw.global_feats, w.global_feats)]
        if self.observation_parity:
            checks += [("their_faction_probs", raw.their_faction_probs, MAX_FACTIONS),
                       ("sight_feats", raw.sight_feats, SIGHT_FEAT_DIM)]
            if raw.sight_type_ids is None or raw.sight_xs is None or raw.sight_ys is None:
                raise ValueError("a parity RawEncoded without its sighting stream")
        for name, arr, width in checks:
            if arr is None or (arr.size and arr.shape[-1] != width):
                raise ValueError(f"RawEncoded.{name}: width "
                                 f"{None if arr is None else arr.shape[-1]}, this encoder reads "
                                 f"{width}; a record from another feature layout?")

    def _check_raws(self, raws: List[RawEncoded]) -> None:
        """`_check_raw` on a batch's first record; the others must carry
        the same observation. One producer builds a batch, and the batched
        paths' concatenations refuse a width that differs within it, so
        the serve thread pays one attribute read per further record."""
        if not raws:
            return
        self._check_raw(raws[0])
        for r in raws[1:]:
            if (getattr(r, "their_faction_probs", None) is not None) != self.observation_parity:
                raise ValueError("a batch mixes records with and without the parity "
                                 "observation's fields")

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
        if self.observation_parity:
            raise ValueError("encode_from_raw_padded carries no sighting stream; a parity "
                             "encoder's batches go through encode_from_raw_embedded or "
                             "encode_from_raw_batch")
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

    def _sighting_embedding(self, type_ids, xs, ys, feats):
        """Sighting tokens: a unit token of side code SIGHT_SIDE_CODE, with
        the hit points it was last seen with in place of the unit's
        columns."""
        return (self.unit_type_embed(type_ids) + self.side_embed.weight[SIGHT_SIDE_CODE]
                + self.pos_x_embed(xs) + self.pos_y_embed(ys) + self.sight_feat_proj(feats))

    def _global_embedding(self, feats, our_faction_ids, their):
        """`their`: the opponent's faction ids [B]; under the parity
        observation, its faction posterior [B, MAX_FACTIONS]."""
        return (self.global_proj(feats) + self.our_faction_embed(our_faction_ids)
                + self._their_faction_term(their))

    def _their_faction_term(self, their: torch.Tensor) -> torch.Tensor:
        """The opponent's faction row, or under the parity observation the
        posterior-weighted sum of the rows (design "The enemy's faction"),
        which is exactly that row for a one-hot posterior. Elementwise
        products and a sum, so no autocast lowers it below float32."""
        if not self.observation_parity:
            return self.their_faction_embed(their)
        return (their.unsqueeze(-1) * self.their_faction_embed.weight).sum(dim=-2)

    # RawEncoded's numeric fields per stream at obs8's widths: (name,
    # dtype, per-row shape). encode_from_raw_embedded concatenates each
    # across the batch into one buffer (`_stream_fields` gives this
    # encoder's own widths and streams).
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

    def _stream_fields(self):
        """`_STREAM_FIELDS` at this encoder's widths, plus the sighting
        stream under the parity observation."""
        if not self.observation_parity:
            return self._STREAM_FIELDS
        w = self.widths
        i64, f32 = torch.int64, torch.float32
        return (
            ("hex", (("hex_xs", i64, ()), ("hex_ys", i64, ()), ("hex_terrain_ids", i64, ()),
                     ("hex_modifier_flags", f32, (NUM_HEX_MODIFIERS,)),
                     ("hex_dynamic_flags", f32, (w.hex_dynamic,)))),
            ("unit", (("unit_type_ids", i64, ()), ("unit_side_ids", i64, ()),
                      ("unit_xs", i64, ()), ("unit_ys", i64, ()),
                      ("unit_feats", f32, (w.unit_feats,)))),
            ("recruit", (("recruit_type_ids", i64, ()), ("recruit_side_ids", i64, ()),
                         ("recruit_xs", i64, ()), ("recruit_ys", i64, ()),
                         ("recruit_feats", f32, (w.unit_feats,)))),
            ("sighting", (("sight_type_ids", i64, ()), ("sight_xs", i64, ()),
                          ("sight_ys", i64, ()), ("sight_feats", f32, (SIGHT_FEAT_DIM,)))),
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
        return self.embed_staged(self.stage_raws(raws, device=device))

    def stage_raws(self, raws: List[RawEncoded], *,
                   device: Optional[torch.device] = None) -> "StagedBatch":
        """The first half of `encode_from_raw_embedded`: the batch's
        numeric fields in one pinned host buffer, copied to the device,
        no learned layer involved. A trainer that recomputes its forward
        in the backward pass (activation checkpointing) stages once and
        embeds twice."""
        if device is None:
            device = next(self.parameters()).device
        B = len(raws)
        if B == 0:
            raise ValueError("encode_from_raw_embedded: empty batch")
        self._check_raws(raws)
        spec = self._stream_fields()
        counts = {stream: [getattr(r, fields[0][0]).shape[0] for r in raws]
                  for stream, fields in spec}
        totals = {stream: sum(c) for stream, c in counts.items()}
        their = (("their_faction_probs", torch.float32, (B, MAX_FACTIONS))
                 if self.observation_parity else ("their_faction_id", torch.int64, (B,)))
        fields = [(name, dt, (totals[stream],) + shape)
                  for stream, stream_fields in spec for name, dt, shape in stream_fields]
        fields += [("global_feats", torch.float32, (B, self.widths.global_feats)),
                   ("our_faction_id", torch.int64, (B,)), their]
        layout = FlatLayout(fields)
        host = torch.empty(layout.nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
        hv = layout.numpy_views(host.numpy())
        for stream, stream_fields in spec:
            if totals[stream]:
                for name, _, _ in stream_fields:
                    np.concatenate([getattr(r, name) for r in raws], out=hv[name])
        np.stack([r.global_feats for r in raws], out=hv["global_feats"])
        hv["our_faction_id"][:] = [r.our_faction_id for r in raws]
        if self.observation_parity:
            np.stack([r.their_faction_probs for r in raws], out=hv["their_faction_probs"])
        else:
            hv["their_faction_id"][:] = [r.their_faction_id for r in raws]
        dev = host if device.type == "cpu" else host.to(device, non_blocking=True)
        return StagedBatch(views=layout.torch_views(dev), totals=totals,
                           sizes=list(zip(counts["unit"], counts["recruit"], counts["hex"])),
                           sighting_counts=counts.get("sighting"), their_field=their[0])

    def embed_staged(self, staged: "StagedBatch") -> EmbeddedStreams:
        """The second half of `encode_from_raw_embedded`: the learned
        embeddings of a staged batch, in stream order."""
        v, totals, their = staged.views, staged.totals, (staged.their_field,)
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
        if totals.get("sighting"):
            parts.append(self._sighting_embedding(v["sight_type_ids"], v["sight_xs"],
                                                  v["sight_ys"], v["sight_feats"]))
        parts.append(self._global_embedding(v["global_feats"], v["our_faction_id"], v[their[0]]))
        parts.append(self.end_turn_token.view(1, -1))
        return EmbeddedStreams(tokens=torch.cat(parts, dim=0), sizes=list(staged.sizes),
                               sighting_counts=staged.sighting_counts)

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
        self._check_raws(raws)
        w = self.widths
        parity = self.observation_parity
        Hs = [r.hex_xs.shape[0] for r in raws]
        Us = [r.unit_xs.shape[0] for r in raws]
        Rs = [r.recruit_type_ids.shape[0] for r in raws]
        Ss = [r.sight_type_ids.shape[0] for r in raws] if parity else [0] * len(raws)

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
            _stage("hd", [r.hex_dynamic_flags for r in raws], "f", w.hex_dynamic)
        if sum(Us):
            for nm, f in (("ut", "unit_type_ids"), ("us", "unit_side_ids"),
                          ("ux", "unit_xs"), ("uy", "unit_ys")):
                _stage(nm, [getattr(r, f) for r in raws], "i", 1)
            _stage("uf", [r.unit_feats for r in raws], "f", w.unit_feats)
            _stage("ui", [r.unit_is_ours for r in raws], "f", 1)
        if sum(Rs):
            for nm, f in (("rt", "recruit_type_ids"), ("rs", "recruit_side_ids"),
                          ("rx", "recruit_xs"), ("ry", "recruit_ys")):
                _stage(nm, [getattr(r, f) for r in raws], "i", 1)
            _stage("rf", [r.recruit_feats for r in raws], "f", w.unit_feats)
            _stage("ri", [r.recruit_is_ours for r in raws], "f", 1)
        if sum(Ss):
            for nm, f in (("st", "sight_type_ids"), ("sx", "sight_xs"), ("sy", "sight_ys")):
                _stage(nm, [getattr(r, f) for r in raws], "i", 1)
            _stage("sf", [r.sight_feats for r in raws], "f", SIGHT_FEAT_DIM)
        _stage("gf", [np.stack([r.global_feats for r in raws])], "f", w.global_feats)
        _stage("ofi", [np.array([r.our_faction_id for r in raws], dtype=np.int64)], "i", 1)
        if parity:
            _stage("tfp", [np.stack([r.their_faction_probs for r in raws])], "f", MAX_FACTIONS)
        else:
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

        out = {"Hs": Hs, "Us": Us, "Rs": Rs, "Ss": Ss, "hex": None, "unit": None,
               "recruit": None, "sighting": None, "unit_is": None, "recruit_is": None}
        if sum(Hs):
            out["hex"] = self._hex_embedding(t["hx"], t["hy"], t["ht"], t["hm"], t["hd"])
        if sum(Us):
            out["unit"] = self._unit_embedding(t["ut"], t["us"], t["ux"], t["uy"], t["uf"])
            out["unit_is"] = t["ui"]
        if sum(Rs):
            out["recruit"] = self._unit_embedding(t["rt"], t["rs"], t["rx"], t["ry"], t["rf"])
            out["recruit_is"] = t["ri"]
        if sum(Ss):
            out["sighting"] = self._sighting_embedding(t["st"], t["sx"], t["sy"], t["sf"])
        their = t["tfp"] if parity else t["tfi"]
        out["global"] = self._global_embedding(t["gf"], t["ofi"], their)   # [B, d]
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
        sighting_per = (_per(emb["sighting"], emb["Ss"]) if self.observation_parity
                        else [None] * B)
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
                material=torch.tensor([[float(raw.material)]], dtype=torch.float32,
                                      device=global_emb.device),
                observation=getattr(raw, "observation", None),
                sighting_tokens=sighting_per[b],
            ))
        return results


# ---------------------------------------------------------------------
# encode_raw — phase-1 of encoding. Pure Python, vocab read-only.
# ---------------------------------------------------------------------

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


# Unit-type names absent from the vocabulary, looked up onto the overflow
# row of the type embedding (where they share one row with each other),
# with the number of lookups since the process started. Each name is
# warned about once; the pre-encoding manifest reads the counts
# (`unknown_type_counts`).
_UNKNOWN_TYPE_LOOKUPS: Counter = Counter()


def unknown_type_counts() -> Dict[str, int]:
    """The unit-type names that took the overflow row for want of a
    vocabulary entry, with their lookup counts, since the process
    started."""
    return dict(_UNKNOWN_TYPE_LOOKUPS)


def type_row(name: str, type_to_id: Dict[str, int]) -> int:
    """The type-embedding row of a unit type: its vocab id clamped to the
    overflow row. A name absent from the vocabulary takes the overflow
    row, is counted and is warned about once."""
    overflow = MAX_UNIT_TYPES - 1
    i = type_to_id.get(name)
    if i is None:
        if name not in _UNKNOWN_TYPE_LOOKUPS:
            log.warning("unit type %r is not in the encoder's vocabulary: it takes the overflow "
                        "row %d, which every unknown name shares", name, overflow)
        _UNKNOWN_TYPE_LOOKUPS[name] += 1
        return overflow
    return min(i, overflow)


def encode_raw(
    game_state: GameState,
    *,
    type_to_id: Dict[str, int],
    faction_to_id: Dict[str, int],
    relevant_set: bool = False,
    fog_hides_enemy_villages: bool = False,
    terrain_multi_hot: bool = False,
    observation_parity: bool = False,
    relevant_set_version: int = 1,
) -> RawEncoded:
    """Build a `RawEncoded` from a GameState using read-only vocab, by
    the Rust core that answers for the state (`game_core.core_for`: a
    view's own core, or one built from a state made by hand or copied).
    `terrain_multi_hot`: the hex stream carries each hex's terrain
    mask (Hex.terrain_mask) instead of its one class id.
    `observation_parity`: the parity observation (the layout above,
    "The parity observation"), which reads the sighting records of the
    view's own core: a state not bound to a core raises ValueError
    (encode a core's view; a deep copy is unbound,
    `game_core.snapshot_view` keeps the binding).

    Self-contained: no torch, no nn modules, no GPU. The result is
    picklable, so workers can call this and ship results back to the
    trainer over a multiprocessing queue.

    The caller is responsible for keeping `type_to_id` / `faction_to_id`
    in sync between workers and the encoder owning the embedding tables
    — typically by pre-seeding before spawning workers and never
    growing during training.

    Output bulk fields are numpy arrays (int64 / float32) so the
    trainer's `encode_from_raw` can wrap them with `torch.from_numpy`
    in zero-copy O(1) time.

    The side to move must be a player's (`classes.PLAYER_SIDES`); the
    global features describe the other player's side as the enemy.
    A state whose side to move is not a player's raises ValueError.
    """
    from wesnoth_ai.game_core import core_for, core_of
    if (observation_parity or relevant_set_version != 1) and core_of(game_state) is None:
        raise ValueError(
            "observation_parity and relevant_set_version 2 read the sighting records of the view's "
            "own core, and this state is not a view bound to a core: encode the simulator's view or a "
            "replay pair's, or keep a copy with game_core.snapshot_view (copy.deepcopy drops the "
            "binding)")
    return core_for(game_state).encode_raw(
        type_to_id=type_to_id, faction_to_id=faction_to_id, relevant_set=relevant_set,
        fog_hides_enemy_villages=fog_hides_enemy_villages, terrain_multi_hot=terrain_multi_hot,
        observation_parity=observation_parity, relevant_set_version=relevant_set_version)


# ---------------------------------------------------------------------
# The recruit options' inputs to the core's encoding
# ---------------------------------------------------------------------

def _recruit_rows(own_recruits, type_to_id):
    """Per recruit option: the vocab id and `_recruit_stats_for`."""
    ids = [type_row(name, type_to_id) for name in own_recruits]
    stats: List[float] = []
    for name in own_recruits:
        stats += _recruit_stats_for(name)
    return ids, stats


# ---------------------------------------------------------------------
# The full board's slot order, per hex set
# ---------------------------------------------------------------------

@dataclass
class _StaticHexArrays:
    keys: List[Tuple[int, int]]      # (x, y) per hex slot
    hex_set: object                  # the set the entry was built from: a
                                     # strong reference, so its id cannot be
                                     # recycled while the entry exists
    n_hexes: int
    hexes: List                      # the hexes in slot order
    positions: List                  # their Position objects


_STATIC_HEX_CACHE: Dict[int, _StaticHexArrays] = {}


def _build_static_hex_arrays(hexes, hex_set=None) -> _StaticHexArrays:
    return _StaticHexArrays(
        keys=[(h.position.x, h.position.y) for h in hexes], hex_set=hex_set, n_hexes=len(hexes),
        hexes=list(hexes), positions=[h.position for h in hexes])


def _static_hex_arrays(game_state) -> _StaticHexArrays:
    """Cached slot ordering for the full board (the hex positions of the
    core's encoding), keyed on the identity of `game_state.map.hexes`
    (aliased across forks;
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


# ---------------------------------------------------------------------
# Recruit phantom-unit features
# ---------------------------------------------------------------------
# A "recruit option" doesn't have a Unit instance until it's spawned,
# but the model reads it through the unit feature layout, so recruits and
# on-board units are treated consistently. The core composes the phantom
# feature vector from these stats (scraped from wesnoth_src): HP / moves
# / xp / cost / alignment come from the stats; current_* fields are spawn
# defaults (full HP, 0 MP since spawn turn, 0 XP); is_leader=False,
# has_attacked=False.

_RECRUIT_STATS_CACHE: Dict[str, Tuple[float, float, float, float, int]] = {}
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
