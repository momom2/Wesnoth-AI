"""Tests for dig overlay golden color in the renderer."""

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.build_system import BuildSystem
from dungeon_builder.rendering.voxel_renderer import ChunkMeshBuilder
from dungeon_builder.config import (
    VOXEL_STONE,
    VOXEL_DIRT,
    VOXEL_COLORS,
    RENDER_MODE_MATTER,
)


def _setup(width=4, depth=4, height=4):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    # Mark all blocks visible so fog-of-war doesn't interfere with overlay tests
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    builder = ChunkMeshBuilder()
    return bus, grid, bs, builder


class TestDigOverlayColor:
    def test_normal_block_no_overlay(self):
        """Non-digging blocks return their normal color."""
        _, grid, bs, builder = _setup()
        grid.grid[1, 1, 1] = VOXEL_STONE

        color = builder._get_color(grid, 1, 1, 1, VOXEL_STONE, RENDER_MODE_MATTER, bs)
        expected = VOXEL_COLORS[VOXEL_STONE]
        assert color == pytest.approx(expected)

    def test_queued_dig_golden_overlay(self):
        """Queued digs get a golden overlay blended with base color."""
        bus, grid, bs, builder = _setup()
        grid.grid[1, 1, 1] = VOXEL_STONE
        bs.queue_dig(1, 1, 1)

        color = builder._get_color(grid, 1, 1, 1, VOXEL_STONE, RENDER_MODE_MATTER, bs)
        base = VOXEL_COLORS[VOXEL_STONE]

        # Should not be the same as base color (golden tint applied)
        assert color != pytest.approx(base)

        # Gold channel (r) should be higher than base
        # At t=0.4: r = base_r * 0.6 + 1.0 * 0.4
        assert color[0] > base[0]

    def test_active_dig_stronger_overlay(self):
        """Active digs with progress have a stronger golden blend."""
        bus, grid, bs, builder = _setup()
        grid.grid[1, 1, 1] = VOXEL_DIRT  # 20 ticks
        bs.queue_dig(1, 1, 1)

        # Queued color
        queued_color = builder._get_color(
            grid, 1, 1, 1, VOXEL_DIRT, RENDER_MODE_MATTER, bs
        )

        # Advance 10 ticks (50% progress)
        for i in range(1, 11):
            bus.publish("tick", tick=i)

        active_color = builder._get_color(
            grid, 1, 1, 1, VOXEL_DIRT, RENDER_MODE_MATTER, bs
        )

        # Active color should have more gold than queued
        assert active_color[0] > queued_color[0]  # More red (gold)

    def test_completed_dig_no_overlay(self):
        """After dig completes, golden overlay is gone (loose dim applied instead)."""
        bus, grid, bs, builder = _setup()
        grid.grid[1, 1, 1] = VOXEL_DIRT  # 20 ticks
        bs.queue_dig(1, 1, 1)

        for i in range(1, 21):
            bus.publish("tick", tick=i)

        color = builder._get_color(grid, 1, 1, 1, VOXEL_DIRT, RENDER_MODE_MATTER, bs)
        base = VOXEL_COLORS[VOXEL_DIRT]

        # Should be the dimmed loose color, not gold
        expected = (base[0] * 0.7, base[1] * 0.7, base[2] * 0.7, base[3])
        assert color == pytest.approx(expected)

    def test_no_build_system_no_overlay(self):
        """Without build_system reference, no golden overlay is applied."""
        _, grid, _, builder = _setup()
        grid.grid[1, 1, 1] = VOXEL_STONE

        color = builder._get_color(
            grid, 1, 1, 1, VOXEL_STONE, RENDER_MODE_MATTER, None
        )
        expected = VOXEL_COLORS[VOXEL_STONE]
        assert color == pytest.approx(expected)

    def test_overlay_alpha_preserved(self):
        """Golden overlay preserves the original alpha value."""
        bus, grid, bs, builder = _setup()
        grid.grid[1, 1, 1] = VOXEL_STONE
        bs.queue_dig(1, 1, 1)

        color = builder._get_color(grid, 1, 1, 1, VOXEL_STONE, RENDER_MODE_MATTER, bs)
        base = VOXEL_COLORS[VOXEL_STONE]

        assert color[3] == pytest.approx(base[3])
