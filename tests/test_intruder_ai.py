"""Tests for the intruder AI system (rewritten for archetype-based agents)."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    VANGUARD,
    SHADOWBLADE,
)
from dungeon_builder.intruders.personal_map import PersonalMap
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


def _make_intruder(intruder_id=1, x=0, y=0, z=0, archetype=VANGUARD,
                   objective=IntruderObjective.DESTROY_CORE):
    """Helper to create an intruder with sensible defaults."""
    return Intruder(
        intruder_id=intruder_id,
        x=x, y=y, z=z,
        archetype=archetype,
        objective=objective,
        personal_map=PersonalMap(),
    )


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
    # Archetype should be VANGUARD (transitional default)
    assert ai.intruders[0].archetype.name == "Vanguard"


def test_intruder_follows_path():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    intruder = _make_intruder(1, 5, 0, 0)
    path = pf.find_path((5, 0, 0), (5, 5, 3))
    assert path is not None
    intruder.path = path
    intruder.path_index = 1
    intruder.state = IntruderState.ADVANCING
    intruder.move_interval = 1  # Move every tick for faster testing
    ai.intruders.append(intruder)

    moves = []
    bus.subscribe("intruder_moved", lambda **kw: moves.append(1))

    for tick in range(1, 100):
        ai._update_intruder(intruder)
        if intruder.state == IntruderState.ATTACKING:
            break

    assert intruder.state == IntruderState.ATTACKING
    # Vanguard has attack_range=1, so it attacks from within 1 cell of the core
    dist = abs(intruder.x - 5) + abs(intruder.y - 5) + abs(intruder.z - 3)
    assert dist <= intruder.archetype.attack_range


def test_intruder_attacks_core():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    # Vanguard has damage=8 and attack_interval=20
    # Use a custom approach: set attack_interval to 1 for fast testing
    intruder = _make_intruder(1, 5, 5, 3)
    intruder.state = IntruderState.ATTACKING
    intruder.attack_interval = 1  # Attack every tick
    ai.intruders.append(intruder)

    for tick in range(1, 6):
        ai._update_intruder(intruder)

    # 5 attacks × 8 damage (VANGUARD damage) = 40
    assert core.hp == 60


def test_intruder_retreats_at_low_hp():
    bus = EventBus()
    grid = _make_simple_dungeon()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, 5, 5, 3, hp=100)
    rng = SeededRNG(42)

    ai = IntruderAI(bus, grid, pf, core, rng)

    intruder = _make_intruder(1, 5, 3, 3)
    intruder.state = IntruderState.ADVANCING
    intruder.path = pf.find_path((5, 3, 3), (5, 5, 3))
    intruder.path_index = 1
    intruder.move_interval = 1
    ai.intruders.append(intruder)

    # VANGUARD retreat_threshold = 0.15, max_hp = 120
    # 120 * 0.15 = 18 → must be below 18
    intruder.hp = 10

    ai._update_intruder(intruder)
    assert intruder.state == IntruderState.RETREATING
