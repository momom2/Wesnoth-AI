"""Intruder AI: spawning, pathfinding, decision-making."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.config import (
    VOXEL_AIR,
    GRID_WIDTH,
    GRID_DEPTH,
    INTRUDER_RETREAT_THRESHOLD,
    INTRUDER_SPAWN_INTERVAL,
    MAX_INTRUDERS,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.world.pathfinding import AStarPathfinder
    from dungeon_builder.dungeon_core.core import DungeonCore
    from dungeon_builder.utils.rng import SeededRNG

logger = logging.getLogger("dungeon_builder.intruders")


class IntruderAI:
    """Manages intruder spawning, movement, and decision-making."""

    def __init__(
        self,
        event_bus: EventBus,
        voxel_grid: VoxelGrid,
        pathfinder: AStarPathfinder,
        core: DungeonCore,
        rng: SeededRNG,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.pathfinder = pathfinder
        self.core = core
        self.rng = rng

        self.intruders: list[Intruder] = []
        self._next_id = 1
        self._spawn_timer = 0
        self._spawn_interval = INTRUDER_SPAWN_INTERVAL

        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("intruder_needs_repath", self._on_needs_repath)
        event_bus.subscribe("game_over", self._on_game_over)

        self._game_over = False

    def _on_game_over(self, **kwargs) -> None:
        self._game_over = True

    def _on_tick(self, tick: int) -> None:
        if self._game_over:
            return

        # Spawn new intruders periodically
        self._spawn_timer += 1
        if self._spawn_timer >= self._spawn_interval:
            self._spawn_timer = 0
            alive_count = sum(
                1 for i in self.intruders
                if i.state not in (IntruderState.DEAD, IntruderState.ESCAPED)
            )
            if alive_count < MAX_INTRUDERS:
                self._spawn_intruder()

        # Update each intruder
        for intruder in self.intruders:
            if intruder.state in (IntruderState.DEAD, IntruderState.ESCAPED):
                continue
            self._update_intruder(intruder)

        # Clean up dead/escaped intruders periodically
        if tick % 100 == 0:
            self.intruders = [
                i for i in self.intruders
                if i.state not in (IntruderState.DEAD, IntruderState.ESCAPED)
            ]

    def _spawn_intruder(self) -> None:
        """Spawn an intruder at a random surface edge position."""
        spawn_pos = self._find_spawn_position()
        if spawn_pos is None:
            logger.debug("No valid spawn position found")
            return

        sx, sy, sz = spawn_pos
        intruder = Intruder(self._next_id, sx, sy, sz)
        self._next_id += 1

        # Find path to core
        path = self.pathfinder.find_path(
            (intruder.x, intruder.y, intruder.z),
            (self.core.x, self.core.y, self.core.z),
        )

        if path is None:
            logger.debug("No path from spawn to core for intruder #%d", intruder.id)
            return

        intruder.path = path
        intruder.path_index = 1  # Skip the start position (we're already there)
        intruder.state = IntruderState.ADVANCING
        self.intruders.append(intruder)
        self.event_bus.publish("intruder_spawned", intruder=intruder)
        logger.info("Spawned intruder #%d at (%d, %d, %d)", intruder.id, sx, sy, sz)

    def _find_spawn_position(self) -> tuple[int, int, int] | None:
        """Find an air cell on Z=0 at or near the map edge."""
        # Try random edge positions
        edges = []
        for x in range(self.voxel_grid.width):
            for y in [0, self.voxel_grid.depth - 1]:
                if self.voxel_grid.get(x, y, 0) == VOXEL_AIR:
                    edges.append((x, y, 0))
        for y in range(self.voxel_grid.depth):
            for x in [0, self.voxel_grid.width - 1]:
                if self.voxel_grid.get(x, y, 0) == VOXEL_AIR:
                    edges.append((x, y, 0))

        # Also include any air cells at z=0 that connect to a path downward
        # For MVP, just use edge cells
        if not edges:
            # Fallback: any air on surface
            for x in range(self.voxel_grid.width):
                for y in range(self.voxel_grid.depth):
                    if self.voxel_grid.get(x, y, 0) == VOXEL_AIR:
                        edges.append((x, y, 0))

        if not edges:
            return None

        return self.rng.choice(edges)

    def _update_intruder(self, intruder: Intruder) -> None:
        if intruder.state == IntruderState.ADVANCING:
            self._update_advancing(intruder)
        elif intruder.state == IntruderState.ATTACKING:
            self._update_attacking(intruder)
        elif intruder.state == IntruderState.RETREATING:
            self._update_retreating(intruder)

    def _update_advancing(self, intruder: Intruder) -> None:
        # Check retreat condition
        if intruder.hp < intruder.max_hp * INTRUDER_RETREAT_THRESHOLD:
            self._start_retreat(intruder)
            return

        intruder.ticks_since_move += 1
        if intruder.ticks_since_move < intruder.move_interval:
            return
        intruder.ticks_since_move = 0

        self._advance_along_path(intruder)

        # Check if at core
        if (intruder.x, intruder.y, intruder.z) == (self.core.x, self.core.y, self.core.z):
            intruder.state = IntruderState.ATTACKING
            logger.info("Intruder #%d reached the core!", intruder.id)

    def _update_attacking(self, intruder: Intruder) -> None:
        intruder.ticks_since_attack += 1
        if intruder.ticks_since_attack >= intruder.attack_interval:
            intruder.ticks_since_attack = 0
            self.core.take_damage(intruder.damage)
            logger.debug("Intruder #%d attacks core for %d damage", intruder.id, intruder.damage)

    def _update_retreating(self, intruder: Intruder) -> None:
        intruder.ticks_since_move += 1
        if intruder.ticks_since_move < intruder.move_interval:
            return
        intruder.ticks_since_move = 0

        self._advance_along_path(intruder)

        # Check if reached surface
        if intruder.z == 0:
            intruder.state = IntruderState.ESCAPED
            self.event_bus.publish("intruder_escaped", intruder=intruder)
            logger.info("Intruder #%d escaped to the surface!", intruder.id)

    def _start_retreat(self, intruder: Intruder) -> None:
        intruder.state = IntruderState.RETREATING
        # Path back to surface (any surface cell)
        path = self.pathfinder.find_path(
            (intruder.x, intruder.y, intruder.z),
            (intruder.x, intruder.y, 0),
        )
        if path:
            intruder.path = path
            intruder.path_index = 1
        else:
            # Can't retreat — fight to the death
            intruder.state = IntruderState.ADVANCING
        logger.info("Intruder #%d retreating (HP: %d)", intruder.id, intruder.hp)

    def _advance_along_path(self, intruder: Intruder) -> None:
        if intruder.path is None:
            return
        if intruder.path_index >= len(intruder.path):
            # Reached end of path
            if intruder.state == IntruderState.ADVANCING:
                intruder.state = IntruderState.ATTACKING
            return

        next_pos = intruder.path[intruder.path_index]

        # Verify next position is still air (dungeon might have changed)
        if self.voxel_grid.get(*next_pos) != VOXEL_AIR:
            # Path blocked — repath
            self._repath_intruder(intruder)
            return

        intruder.x, intruder.y, intruder.z = next_pos
        intruder.path_index += 1
        self.event_bus.publish("intruder_moved", intruder=intruder)

    def _on_needs_repath(self, intruder: Intruder) -> None:
        self._repath_intruder(intruder)

    def _repath_intruder(self, intruder: Intruder) -> None:
        if intruder.state == IntruderState.ADVANCING:
            goal = (self.core.x, self.core.y, self.core.z)
        elif intruder.state == IntruderState.RETREATING:
            goal = (intruder.x, intruder.y, 0)
        else:
            return

        path = self.pathfinder.find_path(
            (intruder.x, intruder.y, intruder.z), goal
        )
        if path:
            intruder.path = path
            intruder.path_index = 1
        else:
            logger.debug("Intruder #%d cannot find path, becoming stuck", intruder.id)
            intruder.path = None
