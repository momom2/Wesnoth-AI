"""Tests for the 10 new block interaction handlers in interactions.py.

Covers: Gold Bait, Heat Beacon, Pressure Plate, Iron Bars, Floodgate,
Alarm Bell, Fragile Floor, Pipe, Pump, Steam Vent.
"""

from dataclasses import dataclass

from dungeon_builder.config import (
    GOLD_BAIT_INTERACT_TICKS,
    HEAT_BEACON_DAMAGE,
    STEAM_VENT_DAMAGE,
    TARP_DETECT_CUNNING,
    VOXEL_ALARM_BELL,
    VOXEL_FLOODGATE,
    VOXEL_FRAGILE_FLOOR,
    VOXEL_GOLD_BAIT,
    VOXEL_HEAT_BEACON,
    VOXEL_IRON_BARS,
    VOXEL_PIPE,
    VOXEL_PRESSURE_PLATE,
    VOXEL_PUMP,
    VOXEL_STEAM_VENT,
)
from dungeon_builder.intruders.interactions import (
    InteractionResult,
    handle_block,
)


@dataclass
class MockArch:
    name: str = "Vanguard"
    greed: float = 0.5
    fire_immune: bool = False
    can_fly: bool = False
    can_dig: bool = False
    can_bash_door: bool = False
    can_lockpick: bool = False
    arcane_sight_range: int = 0
    spike_detect_range: int = 0
    cunning: float = 0.3
    frenzy_threshold: float = 0.0
    speed: int = 2


class MockIntruder:
    def __init__(self, **kwargs):
        self.archetype = MockArch(**kwargs)


# ── Gold Bait ────────────────────────────────────────────────────────


class TestGoldBait:
    """Gold Bait: REPATH if arcane_sight; COLLECT if greedy; CONTINUE otherwise."""

    def test_arcane_sight_detects_bait(self):
        """Intruder with arcane sight sees through the bait and repaths."""
        intruder = MockIntruder(arcane_sight_range=3, greed=0.8)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, 0)
        assert info.result == InteractionResult.REPATH

    def test_greedy_intruder_collects_bait(self):
        """Greedy intruder without arcane sight grabs the bait."""
        intruder = MockIntruder(greed=0.5, arcane_sight_range=0)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, 0)
        assert info.result == InteractionResult.COLLECT
        assert info.ticks == GOLD_BAIT_INTERACT_TICKS
        assert info.interaction_type == "grab_bait"

    def test_no_greed_continues(self):
        """Non-greedy intruder without arcane sight walks past."""
        intruder = MockIntruder(greed=0.0, arcane_sight_range=0)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_arcane_sight_takes_priority_over_greed(self):
        """Arcane sight check comes before greed check."""
        intruder = MockIntruder(arcane_sight_range=1, greed=1.0)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, 0)
        assert info.result == InteractionResult.REPATH


# ── Heat Beacon ──────────────────────────────────────────────────────


class TestHeatBeacon:
    """Heat Beacon: CONTINUE if fire_immune; DAMAGE otherwise."""

    def test_fire_immune_ignores_beacon(self):
        intruder = MockIntruder(fire_immune=True)
        info = handle_block(intruder, VOXEL_HEAT_BEACON, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_non_immune_takes_damage(self):
        intruder = MockIntruder(fire_immune=False)
        info = handle_block(intruder, VOXEL_HEAT_BEACON, 0)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == HEAT_BEACON_DAMAGE

    def test_damage_amount_matches_config(self):
        """Verify the damage value is the exact config constant."""
        intruder = MockIntruder(fire_immune=False)
        info = handle_block(intruder, VOXEL_HEAT_BEACON, 0)
        assert info.damage == 15  # HEAT_BEACON_DAMAGE


# ── Pressure Plate ───────────────────────────────────────────────────


class TestPressurePlate:
    """Pressure Plate: always CONTINUE (activation handled in decision.py)."""

    def test_default_continues(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_PRESSURE_PLATE, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_any_archetype_continues(self):
        """Even special archetypes just walk over it."""
        intruder = MockIntruder(can_fly=True, arcane_sight_range=5)
        info = handle_block(intruder, VOXEL_PRESSURE_PLATE, 0)
        assert info.result == InteractionResult.CONTINUE


# ── Iron Bars ────────────────────────────────────────────────────────


class TestIronBars:
    """Iron Bars: always REPATH."""

    def test_default_repaths(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_IRON_BARS, 0)
        assert info.result == InteractionResult.REPATH

    def test_strong_intruder_still_repaths(self):
        """Even bash-capable intruders cannot get through iron bars."""
        intruder = MockIntruder(can_bash_door=True, can_dig=True)
        info = handle_block(intruder, VOXEL_IRON_BARS, 0)
        assert info.result == InteractionResult.REPATH


# ── Floodgate ────────────────────────────────────────────────────────


class TestFloodgate:
    """Floodgate: CONTINUE if open (state=0); REPATH if closed (state!=0)."""

    def test_open_floodgate_continues(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_FLOODGATE, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_closed_floodgate_repaths(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_FLOODGATE, 1)
        assert info.result == InteractionResult.REPATH

    def test_closed_floodgate_nonzero_state(self):
        """Any non-zero block_state means closed."""
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_FLOODGATE, 2)
        assert info.result == InteractionResult.REPATH


# ── Alarm Bell ───────────────────────────────────────────────────────


class TestAlarmBell:
    """Alarm Bell: always CONTINUE."""

    def test_default_continues(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_ALARM_BELL, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_any_archetype_continues(self):
        intruder = MockIntruder(can_fly=True, fire_immune=True)
        info = handle_block(intruder, VOXEL_ALARM_BELL, 0)
        assert info.result == InteractionResult.CONTINUE


# ── Fragile Floor ────────────────────────────────────────────────────


class TestFragileFloor:
    """Fragile Floor: CONTINUE if fly; REPATH if arcane/cunning; CONTINUE otherwise."""

    def test_flyer_passes_safely(self):
        intruder = MockIntruder(can_fly=True)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_arcane_sight_detects(self):
        intruder = MockIntruder(arcane_sight_range=2)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.REPATH

    def test_high_cunning_detects(self):
        intruder = MockIntruder(cunning=TARP_DETECT_CUNNING)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.REPATH

    def test_cunning_above_threshold_detects(self):
        intruder = MockIntruder(cunning=TARP_DETECT_CUNNING + 0.1)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.REPATH

    def test_cunning_below_threshold_continues(self):
        intruder = MockIntruder(cunning=TARP_DETECT_CUNNING - 0.1)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_default_walks_on_it(self):
        """Default intruder (low cunning, no arcane, no fly) walks onto it."""
        intruder = MockIntruder(cunning=0.3, arcane_sight_range=0, can_fly=False)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_fly_takes_priority_over_arcane(self):
        """Even with arcane sight, flyer gets CONTINUE (not REPATH)."""
        intruder = MockIntruder(can_fly=True, arcane_sight_range=3)
        info = handle_block(intruder, VOXEL_FRAGILE_FLOOR, 0)
        assert info.result == InteractionResult.CONTINUE


# ── Pipe ─────────────────────────────────────────────────────────────


class TestPipe:
    """Pipe: always REPATH."""

    def test_default_repaths(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_PIPE, 0)
        assert info.result == InteractionResult.REPATH

    def test_digger_still_repaths(self):
        intruder = MockIntruder(can_dig=True)
        info = handle_block(intruder, VOXEL_PIPE, 0)
        assert info.result == InteractionResult.REPATH


# ── Pump ─────────────────────────────────────────────────────────────


class TestPump:
    """Pump: always REPATH."""

    def test_default_repaths(self):
        intruder = MockIntruder()
        info = handle_block(intruder, VOXEL_PUMP, 0)
        assert info.result == InteractionResult.REPATH

    def test_any_archetype_repaths(self):
        intruder = MockIntruder(can_fly=True, can_dig=True)
        info = handle_block(intruder, VOXEL_PUMP, 0)
        assert info.result == InteractionResult.REPATH


# ── Steam Vent ───────────────────────────────────────────────────────


class TestSteamVent:
    """Steam Vent: CONTINUE if fire_immune or can_fly; DAMAGE otherwise."""

    def test_fire_immune_ignores(self):
        intruder = MockIntruder(fire_immune=True, can_fly=False)
        info = handle_block(intruder, VOXEL_STEAM_VENT, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_flyer_ignores(self):
        intruder = MockIntruder(fire_immune=False, can_fly=True)
        info = handle_block(intruder, VOXEL_STEAM_VENT, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_fire_immune_and_flyer_continues(self):
        intruder = MockIntruder(fire_immune=True, can_fly=True)
        info = handle_block(intruder, VOXEL_STEAM_VENT, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_non_immune_non_flyer_takes_damage(self):
        intruder = MockIntruder(fire_immune=False, can_fly=False)
        info = handle_block(intruder, VOXEL_STEAM_VENT, 0)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == STEAM_VENT_DAMAGE

    def test_damage_amount_matches_config(self):
        intruder = MockIntruder(fire_immune=False, can_fly=False)
        info = handle_block(intruder, VOXEL_STEAM_VENT, 0)
        assert info.damage == 10  # STEAM_VENT_DAMAGE
