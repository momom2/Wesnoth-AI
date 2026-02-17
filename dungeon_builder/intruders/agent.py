"""Individual intruder data model and state machine."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import dungeon_builder.config as _cfg
from dungeon_builder.config import (
    MORALE_LOW_THRESHOLD,
    MORALE_HIGH_THRESHOLD,
    MORALE_SLOW_FACTOR,
    MORALE_FAST_FACTOR,
    MORALE_DAMAGE_BONUS,
)

if TYPE_CHECKING:
    from dungeon_builder.intruders.archetypes import ArchetypeStats, IntruderObjective, IntruderStatus
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
        # Vision cache: True when intruder has moved and needs re-scan
        "_vision_dirty",
        # Path cache: (start, goal, map_generation) → avoids repathing
        "_path_cache_key",
        # Origin faction
        "is_underworlder",
        # Social dynamics
        "level",
        "status",
        "morale",
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
        is_underworlder: bool = False,
        level: int = 1,
        status: IntruderStatus | None = None,
    ) -> None:
        from dungeon_builder.intruders.archetypes import IntruderStatus as _IS

        self.id = intruder_id
        self.archetype = archetype

        self.x = x
        self.y = y
        self.z = z

        # Level stat scaling (applied to mutable instance fields, not frozen archetype)
        self.level: int = level
        self.status: IntruderStatus = status if status is not None else _IS.GRUNT
        hp_mult = 1.0 + (level - 1) * _cfg.LEVEL_HP_SCALE
        self.hp = int(archetype.hp * hp_mult)
        self.max_hp = self.hp

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
        self._vision_dirty: bool = True
        self._path_cache_key: tuple | None = None
        self.is_underworlder: bool = is_underworlder
        self.morale: float = _cfg.MORALE_BASE

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
        # Level scaling applied to base archetype damage
        damage_mult = 1.0 + (self.level - 1) * _cfg.LEVEL_DAMAGE_SCALE
        base = int(self.archetype.damage * damage_mult)
        if self.frenzy_active:
            base = int(base * 1.5)
        if self.morale > MORALE_HIGH_THRESHOLD:
            base = int(base * MORALE_DAMAGE_BONUS)
        return base

    @property
    def effective_move_interval(self) -> int:
        base = self.move_interval
        if self.frenzy_active:
            base = max(1, base // 2)
        if self.morale < MORALE_LOW_THRESHOLD:
            base = int(base * MORALE_SLOW_FACTOR)
        elif self.morale > MORALE_HIGH_THRESHOLD:
            base = max(1, int(base * MORALE_FAST_FACTOR))
        return max(1, base)

    def take_damage(self, amount: int) -> None:
        """Apply damage, clamping HP to 0."""
        self.hp = max(0, self.hp - amount)
        if self.hp == 0:
            self.state = IntruderState.DEAD

    def __repr__(self) -> str:
        return (
            f"Intruder(id={self.id}, {self.archetype.name}, "
            f"L{self.level} {self.status.name}, "
            f"pos=({self.x},{self.y},{self.z}), "
            f"hp={self.hp}/{self.max_hp}, morale={self.morale:.2f}, "
            f"state={self.state.name})"
        )
