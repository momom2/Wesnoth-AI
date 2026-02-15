"""Central game state container. Holds references to all subsystems."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.time_manager import TimeManager
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.dungeon_core.core import DungeonCore
    from dungeon_builder.building.build_system import BuildSystem
    from dungeon_builder.building.move_system import MoveSystem
    from dungeon_builder.world.pathfinding import AStarPathfinder


class GameState:
    """Container for shared game state references.

    Subsystems are assigned after construction during initialization.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.event_bus: EventBus | None = None
        self.time_manager: TimeManager | None = None
        self.voxel_grid: VoxelGrid | None = None
        self.core: DungeonCore | None = None
        self.build_system: BuildSystem | None = None
        self.move_system: MoveSystem | None = None
        self.pathfinder: AStarPathfinder | None = None

        # Current build mode for mouse interaction
        self.build_mode: str = "dig"
        self.game_over: bool = False
