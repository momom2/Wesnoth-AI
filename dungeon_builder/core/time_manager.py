"""Tick-based time management. All game simulation advances in discrete ticks."""

from __future__ import annotations

from dungeon_builder.config import TICKS_PER_SECOND, SPEED_MULTIPLIERS


class TimeManager:
    """Accumulates real time and fires discrete ticks at a configured rate.

    At 1x speed, fires TICKS_PER_SECOND ticks per real second.
    At 0x (paused), fires no ticks.
    At higher speeds, fires proportionally more ticks.
    """

    def __init__(self, event_bus) -> None:
        self.event_bus = event_bus
        self.speed: int = 0  # Start paused; 0=paused, 1=normal, 2=fast
        self.tick_count: int = 0
        self._accumulator: float = 0.0
        self._tick_interval: float = 1.0 / TICKS_PER_SECOND

    def update(self, dt: float) -> None:
        """Called every frame with real delta time in seconds."""
        multiplier = SPEED_MULTIPLIERS.get(self.speed, 0.0)
        if multiplier <= 0.0:
            return

        self._accumulator += dt * multiplier
        # Cap accumulator to prevent spiral of death
        max_ticks_per_frame = TICKS_PER_SECOND * 2
        ticks_this_frame = 0

        while self._accumulator >= self._tick_interval and ticks_this_frame < max_ticks_per_frame:
            self._accumulator -= self._tick_interval
            self.tick_count += 1
            ticks_this_frame += 1
            self.event_bus.publish("tick", tick=self.tick_count)

        if ticks_this_frame >= max_ticks_per_frame:
            self._accumulator = 0.0  # Drop excess

    def set_speed(self, speed: int) -> None:
        old_speed = self.speed
        self.speed = speed
        self._accumulator = 0.0
        self.event_bus.publish("speed_changed", speed=speed, old_speed=old_speed)

    @property
    def paused(self) -> bool:
        return self.speed == 0
