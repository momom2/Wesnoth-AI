# classes.py

from dataclasses import dataclass
from typing import List, Set, Optional
from enum import IntEnum

class Alignment(IntEnum):
    """Determines how units are affected by time of day."""
    LAWFUL = 0
    NEUTRAL = 1
    CHAOTIC = 2
    LIMINAL = 3

class UnitTrait(IntEnum):
    """
    Traits that units can have. These are typically assigned randomly at recruitment,
    though some units have fixed traits.
    """
    INTELLIGENT = 0     # -20% exp required
    QUICK = 1           # +1 mp, -5% hp
    RESILIENT = 2       # +4 hp, +1hp per level
    STRONG = 3          # +1 damage for melee attacks, +1 hp
    DEXTROUS = 4        # +1 damage for ranged attacks
    FEARLESS = 5        # No malus at bad ToD
    FERAL = 6           # 50% max def on villages; pretend trait
    HEALTHY = 7         # Always rest heals (regens 2hp per turn)
    DIM = 8             # +20% exp required
    SLOW = 9            # -1 mp, +5% hp
    UNDEAD = 10         # Immune to drain, plague, poison, feeding, etc.
    WEAK = 11           # -1 damage for melee attacks, -1 hp
    AGED = 12           # Not used in standard 1v1
    ELEMENTAL = 13      # Not used in standard 1v1
    LOYAL = 14          # Not used in standard 1v1
    MECHANICAL = 15     # Not used in standard 1v1

class UnitAbility(IntEnum):
    """Special abilities that units can have."""
    AMBUSH = 0          # Hidden in forest
    CONCEALMENT = 1     # Hidden in village
    CURES = 2           # Cures nearby allied units of poison
    FEEDING = 3         # +1 max and current hp on kill
    HEALS4 = 4          # Heals nearby allied units for 4 hp per turn
    HEALS8 = 5          # Heals nearby allied units for 8 hp per turn
    ILLUMINATES = 6     # Adjacent hexes are brighter.
    LEADERSHIP = 7      # +25% dmg for adjacent units per level of difference
    NIGHTSTALK = 8      # Hidden at night
    REGENERATES = 9     # Heals self +8 every turn or cures self of poison
    SKIRMISHER = 10     # Ignores enemy zones of control
    STEADFAST = 11      # Double resistance in defence (up to 50%)
    SUBMERGE = 12       # Hidden in deep water
    TELEPORT = 13       # Can teleport between villages for 1mp

class AttackSpecial(IntEnum):
    """Special properties that individual attacks can have."""
    BACKSTAB = 0        # Double damage dealt if enemy unit behind
    BERSERK = 1         # Repeats attack up to 50 times - or until death
    CHARGE = 2          # Double damage dealt and received on attack
    DRAIN = 3           # Heals self for half damage dealt
    FIRSTSTRIKE = 4     # Strikes first on defense
    MAGICAL = 5         # Always 70% chance to hit
    MARKSMAN = 6        # At least 60% chance to hit
    PLAGUE = 7          # Creates walking corpse on kill
    POISON = 8          # Poisons the target
    SLOW = 9            # Hinders movement and divides damage by 2
    PETRIFY = 10        # Petrifies the target - Not used in standard 1v1
    SWARM = 11          # Damage depends on own HP - Not used in standard 1v1

class DamageType(IntEnum):
    """The six basic damage types in Wesnoth."""
    SLASH = 0
    PIERCE = 1
    IMPACT = 2
    FIRE = 3
    COLD = 4
    ARCANE = 5

class Terrain(IntEnum):
    """All basic terrain types in Wesnoth."""
    CASTLE = 0
    CAVE = 1
    COASTALREEF = 2
    DEEPWATER = 3
    FLAT = 4
    FOREST = 5
    FROZEN = 6
    FUNGUS = 7
    HILLS = 8
    IMPASSABLE = 9
    MOUNTAINS = 10
    SAND = 11
    SHALLOWWATER = 12
    SWAMP = 13
    UNWALKABLE = 14
    VILLAGE = 15
    VOID = 16        # For hexes outside the map

class TerrainModifiers(IntEnum):
    """Special modifiers that can affect terrain."""
    OASIS = 0
    OBSTRUCTED = 1   # Statue, fire, etc.
    SHADOWED = 2     # Darker than normal
    ILLUMINATED = 3   # Brighter than normal
    FIXEDLIGHTSHADOWY = 4     # Always dark
    FIXEDLIGHTNEUTRAL = 5     # No ToD effect
    FIXEDLIGHTILLUMINATED = 6  # Always bright

class TimeOfDay(IntEnum):
    """Different times of day affecting unit alignment."""
    DAWN = 0
    MORNING = 1
    AFTERNOON = 2
    DUSK = 3
    FIRSTWATCH = 4    # First night phase
    SECONDWATCH = 5   # Second night phase, followed by dawn

@dataclass
class Position:
    """Represents a position on the hex grid."""
    x: int
    y: int

@dataclass
class Attack:
    """
    Represents a single attack that a unit can perform.
    Each unit typically has 1-3 different attacks.
    """
    type_id: DamageType
    number_strikes: int
    damage_per_strike: int
    is_ranged: bool
    weapon_specials: Set[AttackSpecial]

@dataclass
class Unit:
    """
    Represents an actual unit on the map, with both
    permanent and current properties.
    """
    # Identity
    name: str           # Will be converted to ID
    side: int           # Player number (1 or 2)
    is_leader: bool     # Is this the leader unit?
    position: Position

    # Permanent stats
    max_hp: int
    max_moves: int
    max_exp: int
    cost: int
    alignment: Alignment
    levelup_names: List[str]  # Will be converted to IDs

    # Current state
    current_hp: int
    current_moves: int
    current_exp: int
    has_attacked: bool  # Reset at start of turn

    # Combat properties
    attacks: List[Attack]
    resistances: List[float]  # 6 values, -1.0 to 1.0
    defenses: List[float]    # 16 values, 0.0 to 1.0
    movement_costs: List[int] # 16 values, 1 to 10 (like def, once per terrain)
    abilities: Set[UnitAbility]
    traits: Set[UnitTrait]

    def __post_init__(self):
        """Validates unit properties are within expected ranges."""
        assert self.side in [1, 2], "Only 1v1 games supported"
        assert 0 <= self.current_hp <= self.max_hp
        assert 0 <= self.current_moves <= self.max_moves
        assert 0 <= self.current_exp < self.max_exp
        assert len(self.resistances) == 6
        assert len(self.defenses) == 16
        assert all(-1.0 <= x <= 1.0 for x in self.resistances)
        assert all(0.0 <= x <= 1.0 for x in self.defenses)

@dataclass
class PartialUnit:
    """Template for a unit that can be recruited."""
    name: str
    hp: int
    moves: int
    exp: int
    cost: int
    alignment: Alignment
    levelup_names: List[str]
    attacks: List[Attack]
    resistances: List[float]
    defenses: List[float]
    abilities: Set[UnitAbility]
    traits: Set[UnitTrait]

@dataclass
class Hex:
    """Represents a single hex on the map."""
    position: Position
    terrain_types: Set[Terrain]
    modifiers: Set[TerrainModifiers]

@dataclass
class Map:
    """Represents the complete game map state for a 1v1."""
    size_x: int
    size_y: int
    mask: Set[Position]                 # Hexes that are not used in the map: void, unplayable.
    fog: Set[Position]                  # Fogged hexes: covered by the fog of war. Doesn't include void hexes.
    hexes: Set[Hex]                     # All playable hexes: what we know about the non-void non-fogged hexes.
    units: Set[Unit]                    # All visible units

@dataclass
class Memory:
    """
    The AI's memory of past states.
    Exact structure will be learned by the model.
    """
    state: List[float]

@dataclass
class Input:
    """Complete input state for the AI to make a decision."""
    map: Map
    recruits: List[PartialUnit]
    memory: Memory

@dataclass
class Action:
    """
    Represents a single action the AI can take.
    This can be movement, attack, or recruitment.
    """
    start_hex: Position     # Where to act from
    target_hex: Position    # Where to act to
    attack_index: int       # Which attack to use (-1 for none)
    recruit_unit: int       # Which unit to recruit (-1 for none)