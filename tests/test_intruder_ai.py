"""Tests for the intruder AI system."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.decision import IntruderAI
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.config import VOXEL_AIR, VOXEL_STONE


def _make_simple_dungeon():
    """Grid with surface, shaft, and core room."""
    grid = VoxelGrid(width=10, depth=10, height=5)
    grid.grid[:] = VOXEL_STONE
    # Surface air
    grid.grid[:, :, 0] = VOXEL_AIR
    # Shaft from (5,0) z=0 to z=3
    for z in range(0, 4):
        grid.grid[5, 0, z] = VOXEL_AIR
    # Corridor at z=3 from (5,0) to (5,5)
    for y in range(0, 6):
        grid.grid[5, y, 3] = VOXEL_AIR
    return grid


def test_intruder_spawns():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    spawned = []
    bus.subscribe("intruder_spawned", lambda **kw: spawned.append(kw))

    # Force spawn
    ai._spawn_intruder()

    assert len(ai.intruders) == 1
    assert ai.intruders[0].state == IntruderState.ADVANCING


def test_intruder_follows_path():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    # Manually create an intruder with a known path
    intruder = Intruder(1, 5, 0, 0, hp=100)
    path = pf.find_path((5, 0, 0), (5, 5, 3))
    assert path is not None
    intruder.path = path
    intruder.path_index = 1
    intruder.state = IntruderState.ADVANCING
    intruder.move_interval = 1  # Move every tick for faster testing
    ai.intruders.append(intruder)

    # Advance enough ticks for the intruder to reach the core
    moves = []
    bus.subscribe("intruder_moved", lambda **kw: moves.append(1))

    for tick in range(1, 100):
        ai._update_intruder(intruder)
        if intruder.state == IntruderState.ATTACKING:
            break

    assert intruder.state == IntruderState.ATTACKING
    assert (intruder.x, intruder.y, intruder.z) == (5, 5, 3)


def test_intruder_attacks_core():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    intruder = Intruder(1, 5, 5, 3, hp=100, damage=10)
    intruder.state = IntruderState.ATTACKING
    intruder.attack_interval = 1  # Attack every tick
    ai.intruders.append(intruder)

    for tick in range(1, 6):
        ai._update_intruder(intruder)

    assert core.hp == 50  # 5 attacks x 10 damage


def test_intruder_retreats_at_low_hp():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    intruder = Intruder(1, 5, 3, 3, hp=100)
    intruder.state = IntruderState.ADVANCING
    intruder.path = pf.find_path((5, 3, 3), (5, 5, 3))
    intruder.path_index = 1
    intruder.move_interval = 1
    ai.intruders.append(intruder)

    # Reduce HP below retreat threshold (20%)
    intruder.hp = 10

    ai._update_intruder(intruder)
    assert intruder.state == IntruderState.RETREATING
