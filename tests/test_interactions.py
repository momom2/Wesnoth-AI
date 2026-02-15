"""Tests for intruder micro-interaction system."""

import pytest

from dungeon_builder.intruders.interactions import (
    handle_block,
    InteractionResult,
    InteractionInfo,
)
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    VANGUARD, SHADOWBLADE, TUNNELER, PYREMANCER,
    WINDCALLER, WARDEN, GORECLAW, GLOOMSEER,
    ALL_ARCHETYPES,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.config import (
    VOXEL_AIR, VOXEL_DOOR, VOXEL_SPIKE, VOXEL_TREASURE,
    VOXEL_TARP, VOXEL_ROLLING_STONE, VOXEL_REINFORCED_WALL,
    VOXEL_LAVA, VOXEL_WATER, VOXEL_SLOPE, VOXEL_STAIRS,
    VOXEL_STONE,
    SPIKE_DAMAGE, ROLLING_STONE_DAMAGE,
    DOOR_BASH_TICKS, DOOR_BASH_TICKS_GORECLAW, DOOR_LOCKPICK_TICKS,
)


def _make(arch, objective=IntruderObjective.DESTROY_CORE):
    return Intruder(1, 0, 0, 0, arch, objective, PersonalMap())


# ── Air / Slope / Stairs — always CONTINUE ──────────────────────────


class TestPassThrough:
    @pytest.mark.parametrize("vtype", [VOXEL_AIR, VOXEL_SLOPE, VOXEL_STAIRS])
    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_always_continue(self, vtype, arch):
        info = handle_block(_make(arch), vtype, 0)
        assert info.result == InteractionResult.CONTINUE


# ── Door interactions ───────────────────────────────────────────────


class TestDoorInteractions:
    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_open_door_continue(self, arch):
        info = handle_block(_make(arch), VOXEL_DOOR, block_state=0)
        assert info.result == InteractionResult.CONTINUE

    def test_vanguard_bashes_closed_door(self):
        info = handle_block(_make(VANGUARD), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.INTERACT
        assert info.ticks == DOOR_BASH_TICKS
        assert info.interaction_type == "bash_door"

    def test_shadowblade_lockpicks_closed_door(self):
        info = handle_block(_make(SHADOWBLADE), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.INTERACT
        assert info.ticks == DOOR_LOCKPICK_TICKS
        assert info.interaction_type == "lockpick"

    def test_goreclaw_bashes_closed_door_faster(self):
        info = handle_block(_make(GORECLAW), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.INTERACT
        assert info.ticks == DOOR_BASH_TICKS_GORECLAW

    def test_tunneler_bashes_closed_door(self):
        info = handle_block(_make(TUNNELER), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.INTERACT
        assert info.interaction_type == "bash_door"

    def test_pyremancer_repaths_closed_door(self):
        info = handle_block(_make(PYREMANCER), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.REPATH

    def test_windcaller_repaths_closed_door(self):
        info = handle_block(_make(WINDCALLER), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.REPATH

    def test_warden_repaths_closed_door(self):
        info = handle_block(_make(WARDEN), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.REPATH

    def test_gloomseer_repaths_closed_door(self):
        info = handle_block(_make(GLOOMSEER), VOXEL_DOOR, block_state=1)
        assert info.result == InteractionResult.REPATH


# ── Spike interactions ──────────────────────────────────────────────


class TestSpikeInteractions:
    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_retracted_spike_continue(self, arch):
        info = handle_block(_make(arch), VOXEL_SPIKE, block_state=0)
        assert info.result == InteractionResult.CONTINUE

    def test_vanguard_takes_half_damage(self):
        info = handle_block(_make(VANGUARD), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == SPIKE_DAMAGE // 2

    def test_shadowblade_detects_and_avoids(self):
        info = handle_block(_make(SHADOWBLADE), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.REPATH

    def test_goreclaw_smashes_spike(self):
        info = handle_block(_make(GORECLAW), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.DESTROY_BLOCK
        assert info.damage == SPIKE_DAMAGE // 2  # Takes 10 damage

    def test_windcaller_flies_over_spike(self):
        info = handle_block(_make(WINDCALLER), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.CONTINUE

    def test_warden_takes_full_damage(self):
        info = handle_block(_make(WARDEN), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == SPIKE_DAMAGE

    def test_tunneler_takes_full_damage(self):
        info = handle_block(_make(TUNNELER), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == SPIKE_DAMAGE

    def test_pyremancer_takes_full_damage(self):
        info = handle_block(_make(PYREMANCER), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == SPIKE_DAMAGE

    def test_gloomseer_takes_full_damage(self):
        info = handle_block(_make(GLOOMSEER), VOXEL_SPIKE, block_state=1)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == SPIKE_DAMAGE


# ── Treasure interactions ───────────────────────────────────────────


class TestTreasureInteractions:
    def test_shadowblade_collects(self):
        info = handle_block(_make(SHADOWBLADE), VOXEL_TREASURE, 0)
        assert info.result == InteractionResult.COLLECT
        assert info.interaction_type == "grab_treasure"

    def test_vanguard_ignores(self):
        info = handle_block(_make(VANGUARD), VOXEL_TREASURE, 0)
        assert info.result == InteractionResult.CONTINUE  # greed=0.0

    def test_goreclaw_ignores(self):
        info = handle_block(_make(GORECLAW), VOXEL_TREASURE, 0)
        assert info.result == InteractionResult.CONTINUE  # greed=0.0

    def test_pyremancer_collects_slightly(self):
        # Pyremancer has greed=0.1 (> 0)
        info = handle_block(_make(PYREMANCER), VOXEL_TREASURE, 0)
        assert info.result == InteractionResult.COLLECT

    def test_gloomseer_collects(self):
        # greed=0.1
        info = handle_block(_make(GLOOMSEER), VOXEL_TREASURE, 0)
        assert info.result == InteractionResult.COLLECT


# ── Tarp interactions ───────────────────────────────────────────────


class TestTarpInteractions:
    def test_windcaller_flies_over(self):
        info = handle_block(_make(WINDCALLER), VOXEL_TARP, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_gloomseer_detects_via_arcane(self):
        info = handle_block(_make(GLOOMSEER), VOXEL_TARP, 0)
        assert info.result == InteractionResult.REPATH

    def test_shadowblade_detects_high_cunning(self):
        # Shadowblade cunning=0.8 >= 0.5
        info = handle_block(_make(SHADOWBLADE), VOXEL_TARP, 0)
        assert info.result == InteractionResult.REPATH

    def test_vanguard_falls_through(self):
        # Vanguard cunning=0.0
        info = handle_block(_make(VANGUARD), VOXEL_TARP, 0)
        assert info.result == InteractionResult.FALL

    def test_goreclaw_falls_through(self):
        info = handle_block(_make(GORECLAW), VOXEL_TARP, 0)
        assert info.result == InteractionResult.FALL

    def test_tunneler_falls_through(self):
        # Tunneler cunning=0.3 < 0.5
        info = handle_block(_make(TUNNELER), VOXEL_TARP, 0)
        assert info.result == InteractionResult.FALL

    def test_warden_falls_through(self):
        # Warden cunning=0.4 < 0.5
        info = handle_block(_make(WARDEN), VOXEL_TARP, 0)
        assert info.result == InteractionResult.FALL


# ── Rolling stone interactions ──────────────────────────────────────


class TestRollingStoneInteractions:
    def test_windcaller_flies_over(self):
        info = handle_block(_make(WINDCALLER), VOXEL_ROLLING_STONE, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_shadowblade_dodges_high_speed(self):
        # Shadowblade speed=3 >= 3
        info = handle_block(_make(SHADOWBLADE), VOXEL_ROLLING_STONE, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_windcaller_dodges_high_speed(self):
        # Windcaller speed=4 >= 3
        info = handle_block(_make(WINDCALLER), VOXEL_ROLLING_STONE, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_vanguard_takes_damage(self):
        info = handle_block(_make(VANGUARD), VOXEL_ROLLING_STONE, 0)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == ROLLING_STONE_DAMAGE

    def test_goreclaw_takes_damage(self):
        info = handle_block(_make(GORECLAW), VOXEL_ROLLING_STONE, 0)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == ROLLING_STONE_DAMAGE


# ── Reinforced wall ─────────────────────────────────────────────────


class TestReinforcedWall:
    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_always_repath(self, arch):
        info = handle_block(_make(arch), VOXEL_REINFORCED_WALL, 0)
        assert info.result == InteractionResult.REPATH


# ── Lava interactions ───────────────────────────────────────────────


class TestLavaInteractions:
    def test_pyremancer_walks_through(self):
        info = handle_block(_make(PYREMANCER), VOXEL_LAVA, 0)
        assert info.result == InteractionResult.CONTINUE

    @pytest.mark.parametrize("arch", [VANGUARD, SHADOWBLADE, TUNNELER,
                                       WINDCALLER, WARDEN, GORECLAW, GLOOMSEER],
                             ids=lambda a: a.name)
    def test_non_immune_death(self, arch):
        info = handle_block(_make(arch), VOXEL_LAVA, 0)
        assert info.result == InteractionResult.DEATH


# ── Water interactions ──────────────────────────────────────────────


class TestWaterInteractions:
    def test_pyremancer_dies_in_water(self):
        info = handle_block(_make(PYREMANCER), VOXEL_WATER, 0)
        assert info.result == InteractionResult.DEATH

    @pytest.mark.parametrize("arch", [VANGUARD, SHADOWBLADE, TUNNELER,
                                       WINDCALLER, WARDEN, GORECLAW, GLOOMSEER],
                             ids=lambda a: a.name)
    def test_non_fire_repath(self, arch):
        info = handle_block(_make(arch), VOXEL_WATER, 0)
        assert info.result == InteractionResult.REPATH


# ── Other solid blocks ──────────────────────────────────────────────


class TestOtherSolids:
    def test_stone_repaths(self):
        info = handle_block(_make(VANGUARD), VOXEL_STONE, 0)
        assert info.result == InteractionResult.REPATH
