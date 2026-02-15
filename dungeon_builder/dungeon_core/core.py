"""The dungeon core: the player's central objective to defend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.config import CORE_DEFAULT_HP

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus

logger = logging.getLogger("dungeon_builder.core")


class DungeonCore:
    """The magical core that intruders seek to destroy.

    Has HP. When HP reaches 0, fires game_over event.
    """

    def __init__(
        self,
        event_bus: EventBus,
        x: int,
        y: int,
        z: int,
        hp: int = CORE_DEFAULT_HP,
    ) -> None:
        self.event_bus = event_bus
        self.x = x
        self.y = y
        self.z = z
        self.hp = hp
        self.max_hp = hp
        self.alive = True

    def take_damage(self, amount: int) -> None:
        if not self.alive:
            return
        self.hp = max(0, self.hp - amount)
        self.event_bus.publish("core_damaged", hp=self.hp, max_hp=self.max_hp)
        logger.info("Core took %d damage (HP: %d/%d)", amount, self.hp, self.max_hp)
        if self.hp <= 0:
            self.alive = False
            self.event_bus.publish("game_over", reason="core_destroyed")
            logger.warning("GAME OVER: Core destroyed!")
