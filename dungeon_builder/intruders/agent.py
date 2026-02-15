"""Individual intruder data model and state machine."""

from __future__ import annotations

from enum import Enum, auto

from dungeon_builder.config import (
    INTRUDER_DEFAULT_HP,
    INTRUDER_DEFAULT_SPEED,
    INTRUDER_DEFAULT_DAMAGE,
)


class IntruderState(Enum):
    SPAWNING = auto()
    ADVANCING = auto()     # Moving toward core
    ATTACKING = auto()     # At core, dealing damage
    RETREATING = auto()    # Low HP, heading back to surface
    DEAD = auto()
    ESCAPED = auto()       # Reached surface while retreating


class Intruder:
    """A single intruder agent inside the dungeon."""

    __slots__ = (
        "id", "x", "y", "z",
        "hp", "max_hp", "speed", "damage",
        "state", "path", "path_index",
        "ticks_since_move", "move_interval",
        "ticks_since_attack", "attack_interval",
    )

    def __init__(
        self,
        intruder_id: int,
        x: int,
        y: int,
        z: int,
        hp: int = INTRUDER_DEFAULT_HP,
        speed: int = INTRUDER_DEFAULT_SPEED,
        damage: int = INTRUDER_DEFAULT_DAMAGE,
    ) -> None:
        self.id = intruder_id
        self.x = x
        self.y = y
        self.z = z
        self.hp = hp
        self.max_hp = hp
        self.speed = speed
        self.damage = damage
        self.state = IntruderState.SPAWNING

        # Path following
        self.path: list[tuple[int, int, int]] | None = None
        self.path_index: int = 0

        # Movement timing: higher speed = less ticks between moves
        self.ticks_since_move: int = 0
        self.move_interval: int = max(1, 10 // speed)

        # Attack timing
        self.ticks_since_attack: int = 0
        self.attack_interval: int = 20  # 1 second at 20 tps

    def __repr__(self) -> str:
        return (
            f"Intruder(id={self.id}, pos=({self.x},{self.y},{self.z}), "
            f"hp={self.hp}/{self.max_hp}, state={self.state.name})"
        )
