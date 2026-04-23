# classes.py
# Data structures for Wesnoth AI

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
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

@dataclass
class Unit:
    """Complete unit information."""
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

@dataclass
class GlobalInfo:
    """Global game information."""
    current_side: int
    turn_number: int
    time_of_day: str
    village_gold: int
    village_upkeep: int
    base_income: int

@dataclass
class SideInfo:
    """Per-side information."""
    player: str
    recruits: List[str]
    current_gold: int
    base_income: int
    nb_villages_controlled: int

@dataclass
class GameState:
    """Complete game state."""
    game_id: str
    map: Map
    global_info: GlobalInfo
    sides: List[SideInfo]
    game_over: bool = False
    winner: Optional[int] = None

@dataclass
class Experience:
    """Training experience. `action` is the on-wire dict passed to Lua."""
    game_id: str
    state: GameState
    action: Dict
    value: float
    reward: Optional[float]
    turn_number: int
    action_number: int
