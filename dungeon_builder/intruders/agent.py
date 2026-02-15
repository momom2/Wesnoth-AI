"""Individual intruder data model and state machine."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dungeon_builder.intruders.archetypes import ArchetypeStats, IntruderObjective
    from dungeon_builder.intruders.personal_map import PersonalMap


class IntruderState(Enum):
    SPAWNING = auto()
    ADVANCING = auto()       # Moving toward current objective target
    INTERACTING = auto()     # Handling a block interaction (bash, dig, etc.)
    ATTACKING = auto()       # At core, dealing damage
    RETREATING = auto()      # Low HP / objective complete, heading to surface
    PILLAGING = auto()       # Heading to / collecting treasure
    DEAD = auto()
    ESCAPED = auto()         # Reached surface while retreating


class Intruder:
    """A single intruder agent inside the dungeon.

    Each intruder is defined by its *archetype* (shared, immutable stats)
    plus per-instance mutable state (position, HP, personal map, etc.).
    """

    __slots__ = (
        # Identity
        "id",
        "archetype",
        # Position
        "x", "y", "z",
        # Combat
        "hp", "max_hp",
        # State machine
        "state",
        "objective",
        # Path following
        "path", "path_index",
        # Movement timing
        "ticks_since_move", "move_interval",
        # Attack timing
        "ticks_since_attack", "attack_interval",
        # Fog of war
        "personal_map",
        # Party
        "party_id",
        "loyalty_modifier",
        # Interaction state
        "interaction_type",
        "interaction_target",
        "interaction_ticks",
        # Frenzy
        "frenzy_active",
        # Loot
        "loot_count",
        # Tunneling progress: maps (x,y,z) -> ticks spent digging
        "dig_progress",
    )

    def __init__(
        self,
        intruder_id: int,
        x: int,
        y: int,
        z: int,
        archetype: ArchetypeStats,
        objective: IntruderObjective,
        personal_map: PersonalMap,
        party_id: int | None = None,
    ) -> None:
        self.id = intruder_id
        self.archetype = archetype

        self.x = x
        self.y = y
        self.z = z

        self.hp = archetype.hp
        self.max_hp = archetype.hp

        self.state = IntruderState.SPAWNING
        self.objective = objective

        self.path: list[tuple[int, int, int]] | None = None
        self.path_index: int = 0

        self.ticks_since_move: int = 0
        self.move_interval: int = archetype.move_interval
        self.ticks_since_attack: int = 0
        self.attack_interval: int = archetype.attack_interval

        self.personal_map = personal_map
        self.party_id = party_id
        self.loyalty_modifier: float = 0.0

        self.interaction_type: str | None = None
        self.interaction_target: tuple[int, int, int] | None = None
        self.interaction_ticks: int = 0

        self.frenzy_active: bool = False
        self.loot_count: int = 0
        self.dig_progress: dict[tuple[int, int, int], int] = {}

    # ── Convenience properties ──────────────────────────────────────

    @property
    def pos(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @property
    def alive(self) -> bool:
        return self.state not in (IntruderState.DEAD, IntruderState.ESCAPED)

    @property
    def effective_loyalty(self) -> float:
        return min(1.0, max(0.0, self.archetype.loyalty + self.loyalty_modifier))

    @property
    def effective_speed(self) -> int:
        if self.frenzy_active:
            return self.archetype.speed * 2
        return self.archetype.speed

    @property
    def effective_damage(self) -> int:
        if self.frenzy_active:
            return int(self.archetype.damage * 1.5)
        return self.archetype.damage

    @property
    def effective_move_interval(self) -> int:
        if self.frenzy_active:
            return max(1, self.move_interval // 2)
        return self.move_interval

    def take_damage(self, amount: int) -> None:
        """Apply damage, clamping HP to 0."""
        self.hp = max(0, self.hp - amount)
        if self.hp == 0:
            self.state = IntruderState.DEAD

    def __repr__(self) -> str:
        return (
            f"Intruder(id={self.id}, {self.archetype.name}, "
            f"pos=({self.x},{self.y},{self.z}), "
            f"hp={self.hp}/{self.max_hp}, state={self.state.name})"
        )
