"""Tests for hover voxel type display in the HUD.

Covers:
- _VTYPE_NAMES completeness (all voxel types have names)
- HUD subscribes to voxel_hover / voxel_hover_clear events
- Hover handler source inspection
"""

from __future__ import annotations

import inspect

import pytest

from dungeon_builder.config import (
    VOXEL_DIRT, VOXEL_STONE, VOXEL_BEDROCK, VOXEL_CORE,
    VOXEL_SANDSTONE, VOXEL_LIMESTONE, VOXEL_SHALE, VOXEL_CHALK,
    VOXEL_SLATE, VOXEL_MARBLE, VOXEL_GNEISS,
    VOXEL_GRANITE, VOXEL_BASALT, VOXEL_OBSIDIAN,
    VOXEL_IRON_ORE, VOXEL_COPPER_ORE, VOXEL_GOLD_ORE, VOXEL_MANA_CRYSTAL,
    VOXEL_LAVA, VOXEL_WATER,
    VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT, VOXEL_ENCHANTED_METAL,
    VOXEL_REINFORCED_WALL, VOXEL_SPIKE, VOXEL_DOOR, VOXEL_TREASURE,
    VOXEL_ROLLING_STONE, VOXEL_TARP, VOXEL_SLOPE, VOXEL_STAIRS,
)


class TestVTypeNames:
    """Verify the _VTYPE_NAMES dict covers all voxel types."""

    def test_all_solid_types_have_names(self):
        """Every non-air voxel type should have a friendly name."""
        from dungeon_builder.ui.hud import _VTYPE_NAMES

        expected_types = [
            VOXEL_DIRT, VOXEL_STONE, VOXEL_BEDROCK, VOXEL_CORE,
            VOXEL_SANDSTONE, VOXEL_LIMESTONE, VOXEL_SHALE, VOXEL_CHALK,
            VOXEL_SLATE, VOXEL_MARBLE, VOXEL_GNEISS,
            VOXEL_GRANITE, VOXEL_BASALT, VOXEL_OBSIDIAN,
            VOXEL_IRON_ORE, VOXEL_COPPER_ORE, VOXEL_GOLD_ORE, VOXEL_MANA_CRYSTAL,
            VOXEL_LAVA, VOXEL_WATER,
            VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT,
            VOXEL_ENCHANTED_METAL,
            VOXEL_REINFORCED_WALL, VOXEL_SPIKE, VOXEL_DOOR, VOXEL_TREASURE,
            VOXEL_ROLLING_STONE, VOXEL_TARP, VOXEL_SLOPE, VOXEL_STAIRS,
        ]
        for vtype in expected_types:
            assert vtype in _VTYPE_NAMES, (
                f"Voxel type {vtype} missing from _VTYPE_NAMES"
            )

    def test_functional_blocks_present(self):
        """All functional block types should have names."""
        from dungeon_builder.ui.hud import _VTYPE_NAMES

        assert _VTYPE_NAMES[VOXEL_SLOPE] == "Slope"
        assert _VTYPE_NAMES[VOXEL_STAIRS] == "Stairs"
        assert _VTYPE_NAMES[VOXEL_DOOR] == "Door"
        assert _VTYPE_NAMES[VOXEL_SPIKE] == "Spike"
        assert _VTYPE_NAMES[VOXEL_TREASURE] == "Treasure"
        assert _VTYPE_NAMES[VOXEL_TARP] == "Tarp"
        assert _VTYPE_NAMES[VOXEL_ROLLING_STONE] == "Rolling Stone"
        assert _VTYPE_NAMES[VOXEL_REINFORCED_WALL] == "Reinforced Wall"

    def test_water_present(self):
        """Water should be in _VTYPE_NAMES."""
        from dungeon_builder.ui.hud import _VTYPE_NAMES
        assert _VTYPE_NAMES[VOXEL_WATER] == "Water"


class TestHUDHoverSubscription:
    """HUD should subscribe to hover events."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from dungeon_builder.ui.hud import HUD
        self.cls = HUD
        self.init_src = inspect.getsource(HUD.__init__)

    def test_subscribes_to_voxel_hover(self):
        """HUD.__init__ should subscribe to 'voxel_hover'."""
        assert "voxel_hover" in self.init_src

    def test_subscribes_to_voxel_hover_clear(self):
        """HUD.__init__ should subscribe to 'voxel_hover_clear'."""
        assert "voxel_hover_clear" in self.init_src

    def test_hover_handler_exists(self):
        """HUD should have a _on_voxel_hover method."""
        assert hasattr(self.cls, "_on_voxel_hover")

    def test_hover_clear_handler_exists(self):
        """HUD should have a _on_voxel_hover_clear method."""
        assert hasattr(self.cls, "_on_voxel_hover_clear")

    def test_hover_label_created(self):
        """HUD should create a hover_label widget."""
        assert "hover_label" in self.init_src


class TestHoverHandlerLogic:
    """Verify hover handler logic via source inspection."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from dungeon_builder.ui.hud import HUD
        self.hover_src = inspect.getsource(HUD._on_voxel_hover)
        self.clear_src = inspect.getsource(HUD._on_voxel_hover_clear)

    def test_hover_looks_up_vtype(self):
        """Hover handler should look up voxel type from grid."""
        assert "grid.get(" in self.hover_src

    def test_hover_uses_vtype_names(self):
        """Hover handler should use _VTYPE_NAMES dict."""
        assert "_VTYPE_NAMES" in self.hover_src

    def test_hover_updates_label(self):
        """Hover handler should update hover_label text."""
        assert 'hover_label["text"]' in self.hover_src or "hover_label['text']" in self.hover_src

    def test_clear_empties_label(self):
        """Clear handler should set hover_label text to empty."""
        assert 'hover_label["text"]' in self.clear_src or "hover_label['text']" in self.clear_src
