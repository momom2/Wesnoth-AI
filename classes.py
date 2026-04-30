# classes.py
# Data structures for Wesnoth AI

from dataclasses import dataclass
from typing import List, Optional, Set
from enum import IntEnum

class Alignment(IntEnum):
    """Unit alignment affecting ToD damage."""
    LAWFUL = 0
    NEUTRAL = 1
    CHAOTIC = 2
    LIMINAL = 3

class UnitTrait(IntEnum):
    """Traits units can have."""
    INTELLIGENT = 0
    QUICK = 1
    RESILIENT = 2
    STRONG = 3
    DEXTROUS = 4
    FEARLESS = 5
    FERAL = 6
    HEALTHY = 7
    DIM = 8
    SLOW = 9
    UNDEAD = 10
    WEAK = 11

class UnitAbility(IntEnum):
    """Special abilities."""
    AMBUSH = 0
    CONCEALMENT = 1
    CURES = 2
    FEEDING = 3
    HEALS4 = 4
    HEALS8 = 5
    ILLUMINATES = 6
    LEADERSHIP = 7
    NIGHTSTALK = 8
    REGENERATES = 9
    SKIRMISHER = 10
    STEADFAST = 11
    SUBMERGE = 12
    TELEPORT = 13

class UnitStatus(IntEnum):
    """Status effects."""
    POISONED = 0
    SLOW = 1
    PETRIFIED = 2
    STUNNED = 3

class AttackSpecial(IntEnum):
    """Weapon special properties."""
    BACKSTAB = 0
    BERSERK = 1
    CHARGE = 2
    DRAIN = 3
    FIRSTSTRIKE = 4
    MAGICAL = 5
    MARKSMAN = 6
    PLAGUE = 7
    POISON = 8
    SLOW = 9

class DamageType(IntEnum):
    """Damage types."""
    SLASH = 0
    PIERCE = 1
    IMPACT = 2
    FIRE = 3
    COLD = 4
    ARCANE = 5

class Terrain(IntEnum):
    """Terrain types."""
    CASTLE = 0
    CAVE = 1
    DEEPWATER = 2
    FLAT = 3
    FOREST = 4
    FROZEN = 5
    HILLS = 6
    IMPASSABLE = 7
    MOUNTAINS = 8
    SAND = 9
    SHALLOWWATER = 10
    SWAMP = 11
    UNWALKABLE = 12
    VILLAGE = 13

class TerrainModifiers(IntEnum):
    """Special terrain modifiers."""
    VILLAGE = 0
    KEEP = 1
    CASTLE = 2
    ILLUMINATED = 3
    SHADOWED = 4

@dataclass
class Position:
    """Position on hex grid."""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))

@dataclass
class Attack:
    """Unit attack."""
    type_id: DamageType
    number_strikes: int
    damage_per_strike: int
    is_ranged: bool
    weapon_specials: Set[AttackSpecial]

@dataclass(eq=False)
class Unit:
    """Complete unit information.

    Equality + hash are based on (id, side) only -- the "logical
    identity" of a unit on the board, not its current state. This
    matters because Unit is the element type of `gs.map.units`, a
    Python set, and our mutation pattern is "create a NEW Unit with
    the changed fields, then `set.discard(old) + set.add(new)`":

      - `__hash__((id, side))` makes the discard/add hit the right
        bucket regardless of how many fields changed.
      - `__eq__((id, side))` makes `discard(old)` actually find the
        member even if `old` was *itself* mutated in place between
        the original add and the discard. With dataclass auto-eq
        (every field), such mutations would make `discard` silently
        no-op and the next `add` would land BOTH old and new in the
        set -- the cause of the historical "hex (0,0) occupied by
        both 'u22' and 'u22'" sim invariant failures (verified
        2026-04-30).

    Field-level equality is never needed: callers compare unit IDs
    explicitly when they need to (`u1.id == u2.id`), and
    `state_key` hashes the full state when canonical content
    comparison is required.
    """
    # Identity
    id: str
    name: str
    name_id: int  # Numeric ID for embedding
    side: int
    is_leader: bool
    position: Position

    # Permanent stats
    max_hp: int
    max_moves: int
    max_exp: int
    cost: int
    alignment: Alignment
    levelup_names: List[str]

    # Current state
    current_hp: int
    current_moves: int
    current_exp: int
    has_attacked: bool

    # Combat
    attacks: List[Attack]
    resistances: List[float]
    defenses: List[float]
    movement_costs: List[int]
    abilities: Set[UnitAbility]
    traits: Set[UnitTrait]
    statuses: Set[UnitStatus]

    def __hash__(self):
        return hash((self.id, self.side))

    def __eq__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        return self.id == other.id and self.side == other.side

@dataclass
class Hex:
    """Single hex on map."""
    position: Position
    terrain_types: Set[Terrain]
    modifiers: Set[TerrainModifiers]
    
    def __hash__(self):
        return hash(self.position)

@dataclass
class Map:
    """Complete map state."""
    size_x: int
    size_y: int
    mask: Set[Position]  # Off-board hexes
    fog: Set[Position]   # Fogged hexes
    hexes: Set[Hex]      # All visible hexes
    units: Set[Unit]     # All visible units

    def __deepcopy__(self, memo):
        """Fast-path deepcopy for MCTS-style branching, where we
        clone the map dozens or hundreds of times per move and most
        fields don't change.

        Treats the following as IMMUTABLE for the purposes of cloning:
          - mask, fog: never mutated by self-play (no side moves them).
          - hexes: terrain_types and modifiers ARE Sets that *could*
            mutate (scenario events like Aethermaw morph; village
            capture re-adds VILLAGE), but in 2p ladder games they
            don't, and `_capture_village`'s add() is idempotent on
            actual villages anyway. **Non-self-play callers that
            run terrain-mutating events should `copy.deepcopy` each
            hex explicitly OR use `Map.deep_clone()` (slow path).**

        Treats `units` as MUTABLE: the set is rebuilt every step via
        `_replace_unit` (new frozen-style Unit, set membership
        replaced). A new set is required so add/remove on one copy
        doesn't leak to the other; the Unit *contents* are shared
        because the codebase never mutates a Unit in place.

        Net: ~10x faster than the default deepcopy for typical
        2p mid-game state.
        """
        # memo could have a pre-cached copy from a parent deepcopy;
        # honor it so cyclic graphs work even though we don't have
        # them. Mirrors copy.deepcopy's contract.
        if id(self) in memo:
            return memo[id(self)]
        new = Map(
            size_x = self.size_x,
            size_y = self.size_y,
            mask   = self.mask,    # alias (immutable in self-play)
            fog    = self.fog,     # alias
            hexes  = self.hexes,   # alias (see docstring caveat)
            units  = set(self.units),  # NEW set, same Unit refs
        )
        memo[id(self)] = new
        return new

    def deep_clone(self) -> "Map":
        """Slow-path full deepcopy that copies hexes too. Use for
        scenarios where terrain or hex modifiers actually mutate
        (Aethermaw morph events, terrain `[modify_terrain]` events,
        etc.). ~10x slower than __deepcopy__ but correct for those
        cases."""
        import copy as _copy
        return Map(
            size_x = self.size_x,
            size_y = self.size_y,
            mask   = set(self.mask),
            fog    = set(self.fog),
            hexes  = {_copy.deepcopy(h) for h in self.hexes},
            units  = set(self.units),
        )

@dataclass
class GlobalInfo:
    """Global game information."""
    current_side: int
    turn_number: int
    time_of_day: str
    village_gold: int
    village_upkeep: int
    base_income: int

    def __deepcopy__(self, memo):
        """Fast-path deepcopy for GlobalInfo. Scalar fields can be
        aliased (int / str are immutable). Stashed mutable side-data
        (`_village_owner`, `_uncovered_units`, `_terrain_codes`,
        `_rng_request_counter`, `_scenario_events`, etc.) is added
        ad-hoc by the sim / replay-recon and would silently leak
        across MCTS branches if not copied. Copy each by type:

          - dict      -> dict(...) (shallow; values are typically
                         scalars / tuples)
          - set       -> set(...)
          - list      -> list(...)
          - other     -> alias (assumed immutable)

        If you add a new mutable stash attr, extend the copy logic
        below or set it AFTER the deepcopy at the call site.
        """
        if id(self) in memo:
            return memo[id(self)]
        new = GlobalInfo(
            current_side   = self.current_side,
            turn_number    = self.turn_number,
            time_of_day    = self.time_of_day,
            village_gold   = self.village_gold,
            village_upkeep = self.village_upkeep,
            base_income    = self.base_income,
        )
        # Copy stashed (`_`-prefixed) attrs that the sim or replay
        # reconstruction has accumulated. We can't predict the full
        # set, so iterate __dict__ and dispatch by container type.
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                continue
            if isinstance(v, dict):
                setattr(new, k, dict(v))
            elif isinstance(v, set):
                setattr(new, k, set(v))
            elif isinstance(v, list):
                setattr(new, k, list(v))
            else:
                setattr(new, k, v)  # immutable / unknown -> alias
        memo[id(self)] = new
        return new

@dataclass
class SideInfo:
    """Per-side information."""
    player: str
    recruits: List[str]
    current_gold: int
    base_income: int
    nb_villages_controlled: int
    # Default-era faction name ("Drakes", "Knalgan Alliance", etc.) or
    # "Custom" / "" when unknown. Used by the encoder's faction
    # embedding so one model can learn all matchups. Absent from pre-
    # faction-conditioning checkpoints — state_converter and the replay
    # DataLoader default to "" for backwards compatibility.
    faction: str = ""

@dataclass
class GameState:
    """Complete game state."""
    game_id: str
    map: Map
    global_info: GlobalInfo
    sides: List[SideInfo]
    game_over: bool = False
    winner: Optional[int] = None


# ---------------------------------------------------------------------
# Canonical state key for MCTS transposition tables
# ---------------------------------------------------------------------
# Two semantically-identical states (same unit positions / HPs / MPs /
# statuses, same gold, same turn, same side-to-move) produce the same
# 64-bit hash. Used by MCTS to share visit statistics across paths
# that converge on the same position. Without this, "move A then B"
# and "move B then A" would expand and search the same node twice.
#
# Excludes immutables: terrain, map shape, faction lists. Including
# them in every key would just waste cycles -- they're constant for
# the game session, so they can't differentiate two states.
#
# Includes per-village ownership (mutable: village capture changes
# it) by reading `gs.global_info._village_owner` if present.

def state_key(gs: "GameState") -> int:
    """Return an order-independent 64-bit content hash of `gs`.

    Designed for MCTS transposition: two states differing in exactly
    one unit's HP, position, MP, status set, has_attacked flag, or
    XP produce different keys; two states differing only in unit-set
    iteration order produce the SAME key.

    Cost: O(U + V + S) where U=#units, V=#villages, S=#sides.
    Typical 2p mid-game: ~30 units, ~10 villages, 2 sides; runs in
    well under 0.1ms.
    """
    # Note: u.statuses is annotated `Set[UnitStatus]` but the live
    # codebase uses string keys ("slowed", "poisoned", "resting",
    # "petrified", "loyal", "uncovered", ...) -- the enum is unused.
    # Sort by `str(s)` so int-enum AND string entries both order.
    units_key = tuple(sorted(
        (
            u.id, u.side, u.position.x, u.position.y,
            u.current_hp, u.current_moves, u.current_exp,
            u.has_attacked,
            tuple(sorted(str(s) for s in u.statuses)),
            u.name,             # cheap proxy for "unit type"
            u.is_leader,
        )
        for u in gs.map.units
    ))
    sides_key = tuple(
        (s.faction, s.current_gold, s.base_income,
         s.nb_villages_controlled, tuple(s.recruits))
        for s in gs.sides
    )
    village_owner = getattr(gs.global_info, "_village_owner", None) or {}
    villages_key = tuple(sorted(village_owner.items()))
    global_key = (
        gs.global_info.current_side,
        gs.global_info.turn_number,
        gs.global_info.time_of_day,
        gs.global_info.village_gold,
        gs.global_info.village_upkeep,
        gs.global_info.base_income,
        # Sim's RNG counter -- two states with the same unit layout
        # but different counter values would produce different
        # downstream traits / damage rolls, so they're NOT the same
        # MCTS node. Pull from the optional field set by WesnothSim.
        getattr(gs.global_info, "_rng_request_counter", 0),
    )
    return hash((units_key, sides_key, villages_key, global_key))
