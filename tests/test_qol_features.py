"""Tests for QoL features: pending digs, cancel dig, auto-switch, x-ray, etc."""

import pytest
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.build_system import BuildSystem
from dungeon_builder.world.claimed_territory import ClaimedTerritorySystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DIRT,
    VOXEL_BEDROCK,
    VOXEL_CORE,
    VOXEL_IRON_ORE,
    VOXEL_GOLD_ORE,
    VOXEL_MANA_CRYSTAL,
    VOXEL_LAVA,
    VOXEL_REINFORCED_WALL,
    DEFAULT_SEED,
    DIG_DURATION,
    NON_DIGGABLE,
)


# ══════════════════════════════════════════════════════════════════════
# Pending digs
# ══════════════════════════════════════════════════════════════════════


class TestPendingDigs:
    """Invisible blocks go to pending_digs, promoted on visibility."""

    def _setup(self):
        bus = EventBus()
        grid = VoxelGrid(width=8, depth=8, height=4)
        grid.grid[:] = VOXEL_STONE
        bs = BuildSystem(bus, grid)
        return bus, grid, bs

    def test_invisible_block_goes_pending(self):
        bus, grid, bs = self._setup()
        # Not visible → pending
        assert bs.queue_dig(4, 4, 2) is True
        assert len(bs.pending_digs) == 1
        assert len(bs.dig_queue) == 0

    def test_pending_publishes_event(self):
        bus, grid, bs = self._setup()
        events = []
        bus.subscribe("dig_pending", lambda **kw: events.append(kw))
        bs.queue_dig(4, 4, 2)
        assert len(events) == 1
        assert events[0]["x"] == 4

    def test_is_being_dug_includes_pending(self):
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        assert bs.is_being_dug(4, 4, 2) is True
        assert bs.is_pending_dig(4, 4, 2) is True

    def test_pending_progress_is_zero(self):
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        assert bs.get_dig_progress(4, 4, 2) == pytest.approx(0.0)

    def test_no_double_queue_pending(self):
        bus, grid, bs = self._setup()
        assert bs.queue_dig(4, 4, 2) is True
        assert bs.queue_dig(4, 4, 2) is False  # duplicate

    def test_pending_promoted_when_visible(self):
        """When block becomes visible, pending dig moves to dig_queue."""
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        assert len(bs.pending_digs) == 1

        # Make it visible and trigger check
        grid.visible[4, 4, 2] = True
        bs._check_pending_digs()

        assert len(bs.pending_digs) == 0
        assert len(bs.dig_queue) == 1
        # Duration should be set for stone
        job = bs.dig_queue[0]
        assert job.total_ticks == DIG_DURATION[VOXEL_STONE]

    def test_pending_cancelled_when_visible_air(self):
        """Pending dig on air block is cancelled when revealed."""
        bus, grid, bs = self._setup()
        grid.grid[4, 4, 2] = VOXEL_AIR
        # Block isn't visible yet — goes to pending
        assert bs.queue_dig(4, 4, 2) is True

        cancelled = []
        bus.subscribe("dig_cancelled", lambda **kw: cancelled.append(kw))

        grid.visible[4, 4, 2] = True
        bs._check_pending_digs()

        assert len(bs.pending_digs) == 0
        assert len(bs.dig_queue) == 0
        assert len(cancelled) == 1

    def test_pending_cancelled_when_visible_non_diggable(self):
        """Pending dig on bedrock is cancelled when revealed."""
        bus, grid, bs = self._setup()
        grid.grid[4, 4, 2] = VOXEL_BEDROCK
        assert bs.queue_dig(4, 4, 2) is True

        cancelled = []
        bus.subscribe("dig_cancelled", lambda **kw: cancelled.append(kw))

        grid.visible[4, 4, 2] = True
        bs._check_pending_digs()

        assert len(bs.pending_digs) == 0
        assert len(cancelled) == 1

    def test_pending_stays_pending_if_still_invisible(self):
        """Blocks that are still invisible stay in pending_digs."""
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        bs._check_pending_digs()
        assert len(bs.pending_digs) == 1

    def test_claimed_territory_changed_triggers_check(self):
        """The 'claimed_territory_changed' event triggers _check_pending_digs."""
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        # Make visible before firing event
        grid.visible[4, 4, 2] = True
        bus.publish("claimed_territory_changed")
        assert len(bs.pending_digs) == 0
        assert len(bs.dig_queue) == 1


# ══════════════════════════════════════════════════════════════════════
# Cancel dig
# ══════════════════════════════════════════════════════════════════════


class TestCancelDig:
    """Click-to-cancel: clicking a block being dug cancels it fully."""

    def _setup(self):
        bus = EventBus()
        grid = VoxelGrid(width=8, depth=8, height=4)
        grid.grid[:] = VOXEL_STONE
        grid.visible[:] = True
        bs = BuildSystem(bus, grid)
        return bus, grid, bs

    def test_cancel_queued_dig(self):
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        assert bs.cancel_dig(4, 4, 2) is True
        assert not bs.is_being_dug(4, 4, 2)

    def test_cancel_active_dig(self):
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)
        bus.publish("tick", tick=1)  # promotes to active
        assert len(bs.active_digs) == 1

        assert bs.cancel_dig(4, 4, 2) is True
        assert len(bs.active_digs) == 0
        assert not bs.is_being_dug(4, 4, 2)

    def test_cancel_pending_dig(self):
        bus, grid, bs = self._setup()
        grid.visible[4, 4, 2] = False  # make invisible → goes to pending
        bs.queue_dig(4, 4, 2)
        assert len(bs.pending_digs) == 1

        assert bs.cancel_dig(4, 4, 2) is True
        assert len(bs.pending_digs) == 0

    def test_cancel_publishes_event(self):
        bus, grid, bs = self._setup()
        bs.queue_dig(4, 4, 2)

        events = []
        bus.subscribe("dig_cancelled", lambda **kw: events.append(kw))
        bs.cancel_dig(4, 4, 2)

        assert len(events) == 1
        assert events[0]["x"] == 4

    def test_cancel_nonexistent_returns_false(self):
        bus, grid, bs = self._setup()
        assert bs.cancel_dig(4, 4, 2) is False

    def test_left_click_toggles_dig(self):
        """Left-clicking a block being dug should cancel it."""
        bus, grid, bs = self._setup()
        bus.publish("voxel_left_clicked", x=4, y=4, z=2, mode="dig")
        assert bs.is_being_dug(4, 4, 2) is True

        bus.publish("voxel_left_clicked", x=4, y=4, z=2, mode="dig")
        assert bs.is_being_dug(4, 4, 2) is False

    def test_block_not_loose_after_cancel(self):
        """Cancelled active dig does not make block loose."""
        bus, grid, bs = self._setup()
        grid.grid[4, 4, 2] = VOXEL_DIRT
        bs.queue_dig(4, 4, 2)
        # Simulate some ticks
        for i in range(1, 11):
            bus.publish("tick", tick=i)
        # Partially dug — cancel
        bs.cancel_dig(4, 4, 2)
        assert not grid.is_loose(4, 4, 2)


# ══════════════════════════════════════════════════════════════════════
# Multiple concurrent digs
# ══════════════════════════════════════════════════════════════════════


class TestMultipleDigs:
    """Multiple blocks can be dug simultaneously."""

    def test_many_digs_concurrent(self):
        bus = EventBus()
        grid = VoxelGrid(width=8, depth=8, height=4)
        grid.grid[:] = VOXEL_DIRT
        grid.visible[:] = True
        bs = BuildSystem(bus, grid)

        # Queue 10 digs
        for i in range(10):
            assert bs.queue_dig(i % 8, i // 8, 0) is True

        # After one tick, all should be active (MAX_CONCURRENT_DIGS=999)
        bus.publish("tick", tick=1)
        assert len(bs.active_digs) == 10


# ══════════════════════════════════════════════════════════════════════
# Ore X-ray visibility
# ══════════════════════════════════════════════════════════════════════


class TestOreXray:
    """Ores within PLAYER_XRAY_RANGE of visible blocks become visible."""

    def _setup(self):
        bus = EventBus()
        grid = VoxelGrid(width=10, depth=10, height=5)
        grid.grid[:] = VOXEL_STONE
        return bus, grid

    def test_ore_visible_through_stone(self):
        """Iron ore 2 blocks from a visible block becomes visible."""
        import dungeon_builder.config as _cfg
        old_range = _cfg.PLAYER_XRAY_RANGE
        _cfg.PLAYER_XRAY_RANGE = 3
        try:
            bus, grid = self._setup()
            # Set up core + claimed territory
            grid.grid[5, 5, 2] = VOXEL_CORE
            grid.grid[5, 5, 1] = VOXEL_AIR  # above core
            grid.grid[5, 5, 3] = VOXEL_STONE
            # Place ore 2 blocks away through solid stone
            grid.grid[7, 5, 2] = VOXEL_IRON_ORE  # 2 blocks from border

            ct = ClaimedTerritorySystem(bus, grid, 5, 5, 2)
            ct.recompute()

            # The ore should be visible via x-ray dilation
            assert bool(grid.visible[7, 5, 2]) is True
        finally:
            _cfg.PLAYER_XRAY_RANGE = old_range

    def test_non_ore_not_revealed_by_xray(self):
        """Regular stone blocks are NOT revealed by x-ray."""
        import dungeon_builder.config as _cfg
        old_range = _cfg.PLAYER_XRAY_RANGE
        _cfg.PLAYER_XRAY_RANGE = 3
        try:
            bus, grid = self._setup()
            grid.grid[5, 5, 2] = VOXEL_CORE
            grid.grid[5, 5, 1] = VOXEL_AIR
            # Stone at distance 2 — NOT an ore type
            grid.grid[7, 5, 2] = VOXEL_STONE

            ct = ClaimedTerritorySystem(bus, grid, 5, 5, 2)
            ct.recompute()

            # Regular stone at distance 2 should NOT be visible
            # (only the immediate neighbors of claimed air are visible)
            if not grid.claimed[6, 5, 2]:
                assert bool(grid.visible[7, 5, 2]) is False
        finally:
            _cfg.PLAYER_XRAY_RANGE = old_range

    def test_ore_beyond_range_not_visible(self):
        """Ore too far from visible blocks is NOT revealed."""
        import dungeon_builder.config as _cfg
        old_range = _cfg.PLAYER_XRAY_RANGE
        _cfg.PLAYER_XRAY_RANGE = 2
        try:
            bus, grid = self._setup()
            grid.grid[5, 5, 2] = VOXEL_CORE
            grid.grid[5, 5, 1] = VOXEL_AIR
            # Ore at distance 4 from nearest visible block
            grid.grid[9, 5, 2] = VOXEL_IRON_ORE

            ct = ClaimedTerritorySystem(bus, grid, 5, 5, 2)
            ct.recompute()

            # Beyond x-ray range — should NOT be visible
            assert bool(grid.visible[9, 5, 2]) is False
        finally:
            _cfg.PLAYER_XRAY_RANGE = old_range

    def test_xray_range_zero_disables(self):
        """Setting PLAYER_XRAY_RANGE=0 disables ore x-ray."""
        import dungeon_builder.config as _cfg
        old_range = _cfg.PLAYER_XRAY_RANGE
        _cfg.PLAYER_XRAY_RANGE = 0
        try:
            bus, grid = self._setup()
            grid.grid[5, 5, 2] = VOXEL_CORE
            grid.grid[5, 5, 1] = VOXEL_AIR
            grid.grid[7, 5, 2] = VOXEL_IRON_ORE

            ct = ClaimedTerritorySystem(bus, grid, 5, 5, 2)
            ct.recompute()

            # Should NOT be visible with range=0
            assert bool(grid.visible[7, 5, 2]) is False
        finally:
            _cfg.PLAYER_XRAY_RANGE = old_range

    def test_mana_crystal_visible_through_stone(self):
        """Mana crystal (in XRAY_VISIBLE_TYPES) is revealed through stone."""
        import dungeon_builder.config as _cfg
        old_range = _cfg.PLAYER_XRAY_RANGE
        _cfg.PLAYER_XRAY_RANGE = 3
        try:
            bus, grid = self._setup()
            grid.grid[5, 5, 2] = VOXEL_CORE
            grid.grid[5, 5, 1] = VOXEL_AIR
            grid.grid[7, 5, 2] = VOXEL_MANA_CRYSTAL

            ct = ClaimedTerritorySystem(bus, grid, 5, 5, 2)
            ct.recompute()

            assert bool(grid.visible[7, 5, 2]) is True
        finally:
            _cfg.PLAYER_XRAY_RANGE = old_range


# ══════════════════════════════════════════════════════════════════════
# Build system: dig_complete triggers pending check
# ══════════════════════════════════════════════════════════════════════


class TestDigCompleteTriggersPendingCheck:
    """When a dig completes, adjacent blocks may become visible → pending digs promote."""

    def test_dig_complete_event_triggers_check(self):
        bus = EventBus()
        grid = VoxelGrid(width=8, depth=8, height=4)
        grid.grid[:] = VOXEL_STONE
        bs = BuildSystem(bus, grid)

        # Add a pending dig
        bs.queue_dig(4, 4, 2)
        assert len(bs.pending_digs) == 1

        # Make it visible and simulate a dig_complete
        grid.visible[4, 4, 2] = True
        bus.publish("dig_complete", x=0, y=0, z=0)

        # The pending dig should have been promoted
        assert len(bs.pending_digs) == 0
        assert len(bs.dig_queue) == 1
