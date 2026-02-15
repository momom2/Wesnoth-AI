"""Intruder archetype definitions — 8 distinct adventurer types.

Each archetype is a frozen dataclass defining the base stats, abilities,
and behavioral parameters for one category of intruder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class IntruderObjective(Enum):
    """High-level goal an intruder is pursuing."""

    DESTROY_CORE = auto()
    EXPLORE = auto()
    PILLAGE = auto()


@dataclass(frozen=True)
class ArchetypeStats:
    """Immutable stat block shared by all intruders of one archetype.

    Instances are module-level constants (e.g. VANGUARD, SHADOWBLADE).
    Individual :class:`Intruder` objects hold a *reference* to their
    archetype so there is zero per-intruder duplication.
    """

    name: str

    # Combat
    hp: int
    speed: int          # 1-4 (higher = faster)
    damage: int
    attack_interval: int  # ticks between core attacks
    attack_range: int     # 1 = melee, 3 = ranged (Pyremancer)

    # Perception
    perception_range: int   # LOS radius in cells
    darkvision_range: int   # extra range in dark (future-proof)
    arcane_sight_range: int  # see-through-walls range (Gloomseer)
    spike_detect_range: int  # detect extended spikes N cells away

    # Movement
    move_interval: int  # ticks between moves

    # Behavioral
    retreat_threshold: float  # HP fraction to trigger retreat (0 = never)
    greed: float              # 0-1, probability of betrayal for treasure
    loyalty: float            # 0-1, resistance to leaving party
    cunning: float            # 0-1, hazard avoidance intelligence

    # Ability flags
    can_fly: bool
    can_dig: bool
    can_bash_door: bool
    can_lockpick: bool
    fire_immune: bool

    # Special behavior
    frenzy_threshold: float  # HP fraction to trigger frenzy (0 = never)
    never_retreats: bool
    healer: bool

    # Objective weights (destroy_core, explore, pillage)
    objective_weights: tuple[float, float, float]


# ── Archetype instances ─────────────────────────────────────────────

VANGUARD = ArchetypeStats(
    name="Vanguard",
    hp=120, speed=1, damage=8,
    attack_interval=20, attack_range=1,
    perception_range=4, darkvision_range=0,
    arcane_sight_range=0, spike_detect_range=0,
    move_interval=10,
    retreat_threshold=0.15, greed=0.0, loyalty=0.9, cunning=0.0,
    can_fly=False, can_dig=False, can_bash_door=True, can_lockpick=False,
    fire_immune=False,
    frenzy_threshold=0.0, never_retreats=False, healer=False,
    objective_weights=(1.0, 0.0, 0.0),
)

SHADOWBLADE = ArchetypeStats(
    name="Shadowblade",
    hp=40, speed=3, damage=4,
    attack_interval=20, attack_range=1,
    perception_range=6, darkvision_range=3,
    arcane_sight_range=0, spike_detect_range=2,
    move_interval=3,
    retreat_threshold=0.3, greed=0.9, loyalty=0.2, cunning=0.8,
    can_fly=False, can_dig=False, can_bash_door=False, can_lockpick=True,
    fire_immune=False,
    frenzy_threshold=0.0, never_retreats=False, healer=False,
    objective_weights=(0.1, 0.3, 0.6),
)

TUNNELER = ArchetypeStats(
    name="Tunneler",
    hp=70, speed=1, damage=3,
    attack_interval=20, attack_range=1,
    perception_range=3, darkvision_range=2,
    arcane_sight_range=0, spike_detect_range=0,
    move_interval=10,
    retreat_threshold=0.2, greed=0.0, loyalty=0.7, cunning=0.3,
    can_fly=False, can_dig=True, can_bash_door=True, can_lockpick=False,
    fire_immune=False,
    frenzy_threshold=0.0, never_retreats=False, healer=False,
    objective_weights=(0.8, 0.2, 0.0),
)

PYREMANCER = ArchetypeStats(
    name="Pyremancer",
    hp=50, speed=2, damage=6,
    attack_interval=20, attack_range=3,
    perception_range=5, darkvision_range=0,
    arcane_sight_range=0, spike_detect_range=0,
    move_interval=5,
    retreat_threshold=0.25, greed=0.1, loyalty=0.6, cunning=0.5,
    can_fly=False, can_dig=False, can_bash_door=False, can_lockpick=False,
    fire_immune=True,
    frenzy_threshold=0.0, never_retreats=False, healer=False,
    objective_weights=(0.9, 0.1, 0.0),
)

WINDCALLER = ArchetypeStats(
    name="Windcaller",
    hp=35, speed=4, damage=2,
    attack_interval=20, attack_range=1,
    perception_range=8, darkvision_range=0,
    arcane_sight_range=0, spike_detect_range=0,
    move_interval=2,
    retreat_threshold=0.5, greed=0.0, loyalty=0.5, cunning=0.6,
    can_fly=True, can_dig=False, can_bash_door=False, can_lockpick=False,
    fire_immune=False,
    frenzy_threshold=0.0, never_retreats=False, healer=False,
    objective_weights=(0.2, 0.8, 0.0),
)

WARDEN = ArchetypeStats(
    name="Warden",
    hp=60, speed=2, damage=2,
    attack_interval=20, attack_range=1,
    perception_range=5, darkvision_range=0,
    arcane_sight_range=0, spike_detect_range=0,
    move_interval=5,
    retreat_threshold=0.3, greed=0.0, loyalty=1.0, cunning=0.4,
    can_fly=False, can_dig=False, can_bash_door=False, can_lockpick=False,
    fire_immune=False,
    frenzy_threshold=0.0, never_retreats=False, healer=True,
    objective_weights=(0.5, 0.3, 0.2),
)

GORECLAW = ArchetypeStats(
    name="Goreclaw",
    hp=90, speed=2, damage=15,
    attack_interval=15, attack_range=1,
    perception_range=3, darkvision_range=0,
    arcane_sight_range=0, spike_detect_range=0,
    move_interval=5,
    retreat_threshold=0.0, greed=0.0, loyalty=0.4, cunning=0.0,
    can_fly=False, can_dig=False, can_bash_door=True, can_lockpick=False,
    fire_immune=False,
    frenzy_threshold=0.5, never_retreats=True, healer=False,
    objective_weights=(1.0, 0.0, 0.0),
)

GLOOMSEER = ArchetypeStats(
    name="Gloomseer",
    hp=45, speed=2, damage=3,
    attack_interval=20, attack_range=1,
    perception_range=4, darkvision_range=4,
    arcane_sight_range=6, spike_detect_range=0,
    move_interval=5,
    retreat_threshold=0.4, greed=0.1, loyalty=0.6, cunning=0.7,
    can_fly=False, can_dig=False, can_bash_door=False, can_lockpick=False,
    fire_immune=False,
    frenzy_threshold=0.0, never_retreats=False, healer=False,
    objective_weights=(0.3, 0.6, 0.1),
)

# All archetypes in a tuple for iteration / lookup
ALL_ARCHETYPES: tuple[ArchetypeStats, ...] = (
    VANGUARD, SHADOWBLADE, TUNNELER, PYREMANCER,
    WINDCALLER, WARDEN, GORECLAW, GLOOMSEER,
)

ARCHETYPE_BY_NAME: dict[str, ArchetypeStats] = {a.name: a for a in ALL_ARCHETYPES}
