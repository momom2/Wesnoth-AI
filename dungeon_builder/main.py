"""Entry point: Panda3D application initialization and system wiring."""

from __future__ import annotations

import logging

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    WindowProperties,
    AmbientLight,
    DirectionalLight,
    LVector4f,
    LVector3f,
    AntialiasAttrib,
    loadPrcFileData,
)

# Set window size before ShowBase init (must happen before window creation)
loadPrcFileData("", "win-size 1280 720")

from dungeon_builder.config import (
    DEFAULT_SEED,
    GRID_WIDTH,
    GRID_DEPTH,
    VOXEL_AIR,
    CORE_X,
    CORE_Y,
    CORE_Z,
    CORE_DEFAULT_HP,
)
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.time_manager import TimeManager
from dungeon_builder.core.game_state import GameState
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.geology import GeologyGenerator
from dungeon_builder.world.room_detection import RoomDetector
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.world.physics.temperature import TemperaturePhysics
from dungeon_builder.world.physics.humidity import HumidityPhysics
from dungeon_builder.world.physics.gravity import GravityPhysics
from dungeon_builder.world.physics.structural import StructuralIntegrityPhysics
from dungeon_builder.building.build_system import BuildSystem
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.building.crafting_system import CraftingSystem
from dungeon_builder.intruders.decision import IntruderAI
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.rendering.voxel_renderer import VoxelWorldRenderer
from dungeon_builder.rendering.layer_slice import LayerSliceManager
from dungeon_builder.rendering.camera import CameraController
from dungeon_builder.rendering.intruder_renderer import IntruderRenderer
from dungeon_builder.rendering.effects import EffectsRenderer
from dungeon_builder.ui.hud import HUD
from dungeon_builder.ui.render_mode_selector import RenderModeSelector
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.utils.logging import setup_logging

logger = logging.getLogger("dungeon_builder")


class DungeonApp(ShowBase):
    """Main game application."""

    def __init__(self) -> None:
        ShowBase.__init__(self)
        self.disableMouse()
        self.setBackgroundColor(0.08, 0.08, 0.12, 1.0)

        setup_logging(logging.INFO)

        # Window title
        props = WindowProperties()
        props.set_title("Dungeon Builder")
        self.win.request_properties(props)

        # Anti-aliasing
        self.render.set_antialias(AntialiasAttrib.M_auto)

        # ── Core systems ──
        event_bus = EventBus()
        rng = SeededRNG(DEFAULT_SEED)

        game_state = GameState(DEFAULT_SEED)
        game_state.event_bus = event_bus
        game_state.time_manager = TimeManager(event_bus)

        # ── World generation ──
        voxel_grid = VoxelGrid()
        GeologyGenerator(rng).generate(voxel_grid)
        game_state.voxel_grid = voxel_grid

        # Carve entrance and core room
        self._carve_initial_dungeon(voxel_grid)

        # ── Dungeon core ──
        core = DungeonCore(event_bus, CORE_X, CORE_Y, CORE_Z, hp=CORE_DEFAULT_HP)
        game_state.core = core

        # ── Simulation systems ──
        pathfinder = AStarPathfinder(voxel_grid)
        game_state.pathfinder = pathfinder

        room_detector = RoomDetector(event_bus, voxel_grid)

        build_system = BuildSystem(event_bus, voxel_grid)
        game_state.build_system = build_system

        move_system = MoveSystem(event_bus, voxel_grid)
        game_state.move_system = move_system

        crafting_system = CraftingSystem(event_bus, voxel_grid, move_system)
        move_system.crafting_system = crafting_system

        temperature_physics = TemperaturePhysics(event_bus, voxel_grid)
        humidity_physics = HumidityPhysics(event_bus, voxel_grid)
        gravity_physics = GravityPhysics(event_bus, voxel_grid)
        structural_physics = StructuralIntegrityPhysics(event_bus, voxel_grid)

        intruder_ai = IntruderAI(event_bus, voxel_grid, pathfinder, core, rng)

        # ── Lighting ──
        self._setup_lighting()

        # ── Rendering ──
        layer_manager = LayerSliceManager(self.render)

        world_renderer = VoxelWorldRenderer(event_bus, voxel_grid, layer_manager)
        world_renderer.build_system = build_system
        world_renderer.build_all_chunks()

        camera_ctrl = CameraController(self, event_bus, game_state, layer_manager)
        intruder_renderer = IntruderRenderer(self, event_bus)

        effects_renderer = EffectsRenderer(self, event_bus)
        effects_renderer.place_core_marker(CORE_X, CORE_Y, CORE_Z)

        # ── UI ──
        hud = HUD(self, event_bus, game_state)
        render_mode_selector = RenderModeSelector(
            self, event_bus, world_renderer
        )

        # ── Main game loop ──
        def game_loop(task):
            dt = globalClock.get_dt()
            game_state.time_manager.update(dt)
            return task.cont

        self.taskMgr.add(game_loop, "game_loop", sort=10)

        # ── Escape to quit ──
        self.accept("escape", self.userExit)

        # Store references to prevent garbage collection
        self._game_state = game_state
        self._subsystems = {
            "event_bus": event_bus,
            "rng": rng,
            "voxel_grid": voxel_grid,
            "core": core,
            "pathfinder": pathfinder,
            "room_detector": room_detector,
            "build_system": build_system,
            "move_system": move_system,
            "crafting_system": crafting_system,
            "temperature_physics": temperature_physics,
            "humidity_physics": humidity_physics,
            "gravity_physics": gravity_physics,
            "structural_physics": structural_physics,
            "intruder_ai": intruder_ai,
            "layer_manager": layer_manager,
            "world_renderer": world_renderer,
            "camera_ctrl": camera_ctrl,
            "intruder_renderer": intruder_renderer,
            "effects_renderer": effects_renderer,
            "hud": hud,
            "render_mode_selector": render_mode_selector,
        }

        logger.info("Dungeon Builder initialized. Press SPACE to start!")

    def _carve_initial_dungeon(self, voxel_grid: VoxelGrid) -> None:
        """Carve a vertical shaft and corridor so intruders can reach the core."""
        grid = voxel_grid

        # Clear 3x3 room around core
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if grid.in_bounds(CORE_X + dx, CORE_Y + dy, CORE_Z):
                    grid.grid[CORE_X + dx, CORE_Y + dy, CORE_Z] = VOXEL_AIR

        # Vertical shaft from surface (z=0) down to core level (z=CORE_Z)
        # Located at (CORE_X, 0) to (CORE_X+1, 1)
        shaft_x, shaft_y = CORE_X, 0
        for z in range(0, CORE_Z + 1):
            for dx in range(2):
                for dy in range(2):
                    x, y = shaft_x + dx, shaft_y + dy
                    if grid.in_bounds(x, y, z):
                        grid.grid[x, y, z] = VOXEL_AIR

        # Horizontal corridor at core level from shaft to core room
        for y in range(0, CORE_Y + 1):
            for dx in range(2):
                x = shaft_x + dx
                if grid.in_bounds(x, y, CORE_Z):
                    grid.grid[x, y, CORE_Z] = VOXEL_AIR

        # Also clear surface around entrance for spawning
        for dx in range(-2, 4):
            for dy in range(4):
                x, y = shaft_x + dx, dy
                if grid.in_bounds(x, y, 0):
                    grid.grid[x, y, 0] = VOXEL_AIR

    def _setup_lighting(self) -> None:
        """Add ambient and directional lights for the voxel world."""
        # Ambient light
        alight = AmbientLight("ambient")
        alight.set_color(LVector4f(0.3, 0.3, 0.35, 1.0))
        alnp = self.render.attach_new_node(alight)
        self.render.set_light(alnp)

        # Directional light (sun-like, from above-front)
        dlight = DirectionalLight("directional")
        dlight.set_color(LVector4f(0.8, 0.8, 0.75, 1.0))
        dlnp = self.render.attach_new_node(dlight)
        dlnp.set_hpr(45, -60, 0)
        self.render.set_light(dlnp)


def main() -> None:
    app = DungeonApp()
    app.run()


if __name__ == "__main__":
    main()
