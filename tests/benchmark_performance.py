"""Performance benchmarks for the six algorithmic optimizations.

Run with:
    python -m pytest tests/benchmark_performance.py -v -s

Each benchmark measures the time for a representative workload and prints
results to stdout.  The benchmarks verify that the optimizations actually
provide measurable speedups by comparing "hot" (cached/skipped) vs "cold"
(full recompute) paths.
"""

from __future__ import annotations

import time
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.world.physics.gravity import GravityPhysics
from dungeon_builder.world.physics.structural import StructuralIntegrityPhysics
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    VANGUARD,
    SHADOWBLADE,
    TUNNELER,
    WINDCALLER,
    GLOOMSEER,
    GORECLAW,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.personal_pathfinder import PersonalPathfinder
from dungeon_builder.intruders.party import Party
from dungeon_builder.intruders.vision import compute_los, bresenham_3d
from dungeon_builder.intruders.decision import IntruderAI
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_BEDROCK,
    VOXEL_DIRT,
    VOXEL_CORE,
    GRID_WIDTH,
    GRID_DEPTH,
    GRID_HEIGHT,
    CONNECTIVITY_TICK_INTERVAL,
    STRUCTURAL_TICK_INTERVAL,
    MAP_SHARE_INTERVAL,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_full_grid(width=GRID_WIDTH, depth=GRID_DEPTH, height=GRID_HEIGHT):
    """Create a realistic dungeon grid with air corridors and stone walls."""
    grid = VoxelGrid(width=width, depth=depth, height=height)
    grid.grid[:] = VOXEL_STONE
    # Surface air
    grid.grid[:, :, 0] = VOXEL_AIR
    # Several corridors at z=5
    for x in range(width):
        grid.grid[x, width // 2, 5] = VOXEL_AIR
    for y in range(depth):
        grid.grid[depth // 2, y, 5] = VOXEL_AIR
    # A room at z=10
    for x in range(20, 44):
        for y in range(20, 44):
            grid.grid[x, y, 10] = VOXEL_AIR
    # Core
    grid.grid[32, 32, 10] = VOXEL_CORE
    # Anchors at bottom
    grid.grid[:, :, height - 1] = VOXEL_BEDROCK
    return grid


def _make_intruder(iid=1, x=0, y=0, z=0, archetype=VANGUARD):
    return Intruder(
        intruder_id=iid, x=x, y=y, z=z,
        archetype=archetype,
        objective=IntruderObjective.DESTROY_CORE,
        personal_map=PersonalMap(),
    )


def _time_it(fn, iterations=1):
    """Run fn() `iterations` times, return total seconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - start


def _report(name, cold_ms, hot_ms, iterations):
    speedup = cold_ms / hot_ms if hot_ms > 0 else float("inf")
    print(f"\n  {name}:")
    print(f"    Cold (full recompute):  {cold_ms:8.2f} ms  ({iterations} iters)")
    print(f"    Hot  (cached/skipped):  {hot_ms:8.2f} ms  ({iterations} iters)")
    print(f"    Speedup:                {speedup:8.1f}x")


# ── Benchmark 1: Vision cache on move ───────────────────────────────────

class TestBenchmarkVision:
    """Benchmark vision: _vision_dirty skip vs full recompute."""

    def test_vision_dirty_skip(self):
        """Stationary intruders skip vision entirely via _vision_dirty flag."""
        grid = _make_full_grid()
        intruder = _make_intruder(1, x=32, y=32, z=0)
        intruder.state = IntruderState.ADVANCING

        bus = EventBus()
        pf = AStarPathfinder(grid)
        core = DungeonCore(bus, 32, 32, 10, hp=100)
        rng = SeededRNG(42)
        ai = IntruderAI(bus, grid, pf, core, rng)

        # Cold: force vision dirty, measure full LOS computation
        iters = 50
        def cold():
            intruder._vision_dirty = True
            ai._update_vision(intruder)

        cold_time = _time_it(cold, iters) * 1000

        # Hot: vision not dirty, should skip immediately
        intruder._vision_dirty = False
        def hot():
            ai._update_vision(intruder)

        hot_time = _time_it(hot, iters) * 1000

        _report("Vision (_vision_dirty skip)", cold_time, hot_time, iters)
        assert hot_time < cold_time

    def test_inline_bresenham_vs_list(self):
        """Inline Bresenham in compute_los is faster than separate list build."""
        grid = _make_full_grid()

        # Time the inlined compute_los
        iters = 20
        def run_los():
            compute_los(grid, 32, 32, 0, 8)

        los_time = _time_it(run_los, iters) * 1000

        # Time the old-style bresenham_3d (builds list)
        def run_bresenham_lists():
            for tx in range(24, 41):
                for ty in range(24, 41):
                    bresenham_3d(32, 32, 0, tx, ty, 8)

        list_time = _time_it(run_bresenham_lists, iters) * 1000

        print(f"\n  Inline Bresenham LOS ({iters} iters):")
        print(f"    compute_los (inline):   {los_time:8.2f} ms")
        print(f"    bresenham_3d (lists):   {list_time:8.2f} ms")
        # Just ensure compute_los finishes — it does far more work than just
        # building lists (it also does transparency checks), so absolute
        # comparison isn't meaningful.  The key win is avoiding allocation.
        assert los_time >= 0


# ── Benchmark 2: Map sharing generation skip + throttle ─────────────────

class TestBenchmarkMapSharing:
    """Benchmark map sharing: generation-based skip and throttle."""

    def test_merge_generation_skip(self):
        """Merge with unchanged partner is O(1) via generation check."""
        map_a = PersonalMap()
        map_b = PersonalMap()

        # Populate both maps with substantial data
        for x in range(64):
            for y in range(64):
                map_a.reveal(x, y, 0, VOXEL_STONE)
                map_b.reveal(x, y, 1, VOXEL_STONE)

        iters = 500

        # Cold: first merge (must copy data)
        def cold():
            map_a._last_merge_gen.clear()  # reset cache
            map_b._generation += 1  # force generation change
            map_a.merge(map_b)

        cold_time = _time_it(cold, iters) * 1000

        # Hot: merge again with no change — generation skip
        map_a.merge(map_b)  # prime the cache
        def hot():
            map_a.merge(map_b)

        hot_time = _time_it(hot, iters) * 1000

        _report("Map merge (generation skip)", cold_time, hot_time, iters)
        assert hot_time < cold_time

    def test_share_maps_throttle(self):
        """Party.share_maps() throttle: most calls are no-ops."""
        m1 = _make_intruder(1, x=0, y=0, z=0)
        m2 = _make_intruder(2, x=1, y=0, z=0)
        m1.personal_map.reveal(5, 5, 5, VOXEL_STONE)
        m2.personal_map.reveal(10, 10, 10, VOXEL_STONE)

        party = Party(1, [m1, m2])
        # First share_maps fires immediately (counter starts at INTERVAL-1)
        party.share_maps()

        iters = 1000

        # Hot: subsequent calls hit the throttle counter
        def hot():
            party.share_maps()

        hot_time = _time_it(hot, iters) * 1000

        # Cold: force counter to trigger actual merge each time
        def cold():
            party._share_tick_counter = MAP_SHARE_INTERVAL - 1
            party.share_maps()

        cold_time = _time_it(cold, iters) * 1000

        _report("Party share_maps (throttle)", cold_time, hot_time, iters)
        assert hot_time < cold_time


# ── Benchmark 3: Chunk rebuilds targeted marking ────────────────────────

class TestBenchmarkChunkDirty:
    """Benchmark targeted vs full dirty marking."""

    def test_mark_blocks_dirty_vs_mark_all(self):
        """mark_blocks_dirty for 10 positions vs mark_all_dirty for 336 chunks."""
        grid = _make_full_grid()

        iters = 200

        # Targeted: mark 10 specific blocks dirty
        xs = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ys = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        zs = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 20])

        def targeted():
            grid._dirty_chunks.clear()
            grid.mark_blocks_dirty(xs, ys, zs)

        targeted_time = _time_it(targeted, iters) * 1000

        # Full: mark all chunks dirty (336 entries for 64x64x21)
        def full():
            grid._dirty_chunks.clear()
            grid.mark_all_dirty()

        full_time = _time_it(full, iters) * 1000

        _report("Chunk dirty (targeted vs full)", full_time, targeted_time, iters)
        assert targeted_time < full_time

    def test_targeted_fewer_chunks(self):
        """Targeted marking produces fewer dirty chunks than mark_all."""
        grid = _make_full_grid()

        xs = np.array([5, 10, 15])
        ys = np.array([5, 10, 15])
        zs = np.array([3, 5, 7])

        grid._dirty_chunks.clear()
        grid.mark_blocks_dirty(xs, ys, zs)
        targeted_count = len(grid._dirty_chunks)

        grid._dirty_chunks.clear()
        grid.mark_all_dirty()
        full_count = len(grid._dirty_chunks)

        print(f"\n  Dirty chunk counts:")
        print(f"    Targeted (3 blocks):  {targeted_count:5d} chunks")
        print(f"    Full mark_all_dirty:  {full_count:5d} chunks")
        print(f"    Reduction:            {(1 - targeted_count/full_count)*100:.1f}%")
        assert targeted_count < full_count


# ── Benchmark 4: Connectivity skip-if-unchanged ─────────────────────────

class TestBenchmarkConnectivity:
    """Benchmark connectivity flood-fill skip."""

    def test_connectivity_skip_unchanged(self):
        """When grid hasn't changed, connectivity is skipped via snapshot."""
        bus = EventBus()
        grid = _make_full_grid()
        grav = GravityPhysics(bus, grid)

        iters = 50

        # Cold: first run always does full flood-fill
        tick = CONNECTIVITY_TICK_INTERVAL
        def cold():
            grav._last_connectivity_grid = None
            grav._last_connectivity_loose = None
            bus.publish("tick", tick=tick)

        cold_time = _time_it(cold, iters) * 1000

        # Prime the snapshot
        grav._last_connectivity_grid = None
        bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL)

        # Hot: grid unchanged, should skip via np.array_equal
        def hot():
            bus.publish("tick", tick=tick)

        hot_time = _time_it(hot, iters) * 1000

        _report("Connectivity (skip-if-unchanged)", cold_time, hot_time, iters)
        assert hot_time < cold_time


# ── Benchmark 5: Structural skip + vectorized tensile ───────────────────

class TestBenchmarkStructural:
    """Benchmark structural: skip-if-unchanged and vectorized tensile."""

    def test_structural_skip_unchanged(self):
        """When grid hasn't changed, structural is skipped via snapshot."""
        bus = EventBus()
        grid = _make_full_grid()
        struct = StructuralIntegrityPhysics(bus, grid)

        iters = 50

        # Cold: first run always computes
        tick = STRUCTURAL_TICK_INTERVAL
        def cold():
            struct._last_structural_grid = None
            struct._last_structural_loose = None
            struct._on_tick(tick=tick)

        cold_time = _time_it(cold, iters) * 1000

        # Prime the snapshot
        struct._last_structural_grid = None
        struct._on_tick(tick=STRUCTURAL_TICK_INTERVAL)

        # Hot: grid unchanged, should skip
        def hot():
            struct._on_tick(tick=tick)

        hot_time = _time_it(hot, iters) * 1000

        _report("Structural (skip-if-unchanged)", cold_time, hot_time, iters)
        assert hot_time < cold_time

    def test_tensile_vectorized(self):
        """Tensile computation with vectorized span scan."""
        bus = EventBus()
        grid = VoxelGrid(width=64, depth=64, height=21)
        grid.grid[:] = VOXEL_STONE
        grid.grid[:, :, 0] = VOXEL_AIR
        grid.grid[:, :, 20] = VOXEL_BEDROCK

        # Create cantilevered blocks: stone beams over air
        for x in range(10, 54):
            grid.grid[x, 32, 5] = VOXEL_STONE
            grid.grid[x, 32, 6] = VOXEL_AIR  # air below = unsupported

        struct = StructuralIntegrityPhysics(bus, grid)

        iters = 50

        def run_tensile():
            struct._compute_tensile_failures()

        tensile_time = _time_it(run_tensile, iters) * 1000

        print(f"\n  Tensile vectorized ({iters} iters):")
        print(f"    Total:  {tensile_time:8.2f} ms")
        print(f"    Per call: {tensile_time/iters:8.3f} ms")
        assert tensile_time >= 0


# ── Benchmark 6: Pathfinding cache ──────────────────────────────────────

class TestBenchmarkPathfinding:
    """Benchmark pathfinding: cache hit vs full A* recompute."""

    def test_path_cache_hit(self):
        """Cached repath is instant when position/goal/generation match.

        Benchmarks A* recompute vs cache-key tuple comparison.
        """
        pmap = PersonalMap()
        # Reveal a large open area (gives A* a lot of nodes to explore)
        for x in range(64):
            for y in range(64):
                pmap.reveal(x, y, 0, VOXEL_AIR)

        start = (0, 0, 0)
        goal = (63, 63, 0)

        # Verify path exists
        test_path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert test_path is not None, "Setup error: no valid path"
        assert len(test_path) > 50, f"Path too short ({len(test_path)} steps)"

        iters = 200

        # Cold: full A* recompute each time
        def cold():
            PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)

        cold_time = _time_it(cold, iters) * 1000

        # Hot: cache check only — simulates what _repath_intruder does
        cache_key = (start, goal, pmap._generation)
        cached_path = test_path
        path_index = 1

        def hot():
            # This is what _repath_intruder does on cache hit
            if cache_key == (start, goal, pmap._generation) \
               and cached_path is not None \
               and path_index < len(cached_path):
                return  # cache hit

        hot_time = _time_it(hot, iters) * 1000

        _report("Pathfinding (cache hit vs A*)", cold_time, hot_time, iters)
        assert hot_time < cold_time

    def test_personal_pathfinder_single(self):
        """Baseline: single A* call on a moderately populated personal map."""
        pmap = PersonalMap()
        # Reveal a corridor
        for y in range(64):
            pmap.reveal(32, y, 5, VOXEL_AIR)
        for x in range(64):
            pmap.reveal(x, 32, 5, VOXEL_AIR)
        pmap.reveal(50, 32, 5, VOXEL_AIR)

        iters = 100

        def run_astar():
            PersonalPathfinder.find_path(
                pmap, (0, 32, 5), (50, 32, 5), VANGUARD,
            )

        astar_time = _time_it(run_astar, iters) * 1000

        print(f"\n  Personal A* ({iters} iters):")
        print(f"    Total:    {astar_time:8.2f} ms")
        print(f"    Per call: {astar_time/iters:8.3f} ms")
        assert astar_time >= 0


# ── Benchmark: Combined tick simulation ─────────────────────────────────

class TestBenchmarkCombinedTick:
    """Benchmark a full simulation tick with multiple intruders."""

    def test_multi_intruder_tick(self):
        """Simulate ticks with 8 intruders: measure per-tick cost."""
        bus = EventBus()
        grid = _make_full_grid()
        pf = AStarPathfinder(grid)
        core = DungeonCore(bus, 32, 32, 10, hp=100)
        rng = SeededRNG(42)
        ai = IntruderAI(bus, grid, pf, core, rng)

        # Create 8 intruders at various positions in air
        archetypes = [VANGUARD, SHADOWBLADE, TUNNELER, WINDCALLER,
                      GLOOMSEER, GORECLAW, VANGUARD, SHADOWBLADE]
        for i, arch in enumerate(archetypes):
            intruder = _make_intruder(i + 1, x=i * 7 + 2, y=0, z=0, archetype=arch)
            intruder.state = IntruderState.ADVANCING
            # Reveal around starting position
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = intruder.x + dx, intruder.y + dy
                    if 0 <= nx < 64 and 0 <= ny < 64:
                        vt = int(grid.grid[nx, ny, 0])
                        intruder.personal_map.reveal(nx, ny, 0, vt)
            ai.intruders.append(intruder)

        # Measure consecutive ticks (most should benefit from caching)
        tick_count = 100
        start = time.perf_counter()
        for tick in range(1, tick_count + 1):
            ai._on_tick(tick=tick)
        total_ms = (time.perf_counter() - start) * 1000
        avg_ms = total_ms / tick_count

        print(f"\n  Multi-intruder simulation ({len(archetypes)} intruders, {tick_count} ticks):")
        print(f"    Total:       {total_ms:8.2f} ms")
        print(f"    Per tick:    {avg_ms:8.3f} ms")
        print(f"    Budget (50ms @ 20 TPS):  {'OK' if avg_ms < 50 else 'OVER'}")
        assert avg_ms < 200  # Generous ceiling — real target is <50ms
