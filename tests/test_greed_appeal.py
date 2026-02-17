"""Tests for greed appeal mechanics in intruder interactions.

Greedy intruders (archetype.greed > 0) are attracted to gold bait and treasure,
while non-greedy intruders ignore them.  Arcane sight reveals gold bait as a trap.
"""

from dataclasses import dataclass

import pytest

from dungeon_builder.intruders.interactions import handle_block, InteractionResult
from dungeon_builder.config import (
    VOXEL_GOLD_BAIT,
    VOXEL_TREASURE,
    GOLD_BAIT_INTERACT_TICKS,
    TREASURE_GRAB_TICKS,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gold Bait tests
# ---------------------------------------------------------------------------

class TestGoldBaitGreed:
    """Greedy intruders are attracted to gold bait; non-greedy ones walk past."""

    def test_greedy_intruder_attracted_to_gold_bait(self):
        """An intruder with greed > 0 gets COLLECT on gold bait."""
        intruder = MockIntruder(greed=0.5)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, block_state=0)
        assert info.result is InteractionResult.COLLECT

    def test_non_greedy_intruder_ignores_gold_bait(self):
        """An intruder with greed == 0 gets CONTINUE on gold bait."""
        intruder = MockIntruder(greed=0.0)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, block_state=0)
        assert info.result is InteractionResult.CONTINUE

    def test_arcane_sight_sees_through_gold_bait(self):
        """Arcane sight (range > 0) reveals gold bait as a trap -> REPATH."""
        intruder = MockIntruder(greed=0.8, arcane_sight_range=3)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, block_state=0)
        assert info.result is InteractionResult.REPATH

    def test_gold_bait_interaction_type_is_grab_bait(self):
        """COLLECT result on gold bait carries interaction_type='grab_bait'."""
        intruder = MockIntruder(greed=0.5)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, block_state=0)
        assert info.interaction_type == "grab_bait"

    def test_gold_bait_interaction_ticks(self):
        """COLLECT result on gold bait uses GOLD_BAIT_INTERACT_TICKS."""
        intruder = MockIntruder(greed=0.5)
        info = handle_block(intruder, VOXEL_GOLD_BAIT, block_state=0)
        assert info.ticks == GOLD_BAIT_INTERACT_TICKS


# ---------------------------------------------------------------------------
# Treasure tests
# ---------------------------------------------------------------------------

class TestTreasureGreed:
    """Greedy intruders grab treasure; non-greedy ones walk past."""

    def test_greedy_intruder_attracted_to_treasure(self):
        """An intruder with greed > 0 gets COLLECT on treasure."""
        intruder = MockIntruder(greed=0.5)
        info = handle_block(intruder, VOXEL_TREASURE, block_state=0)
        assert info.result is InteractionResult.COLLECT

    def test_non_greedy_intruder_ignores_treasure(self):
        """An intruder with greed == 0 gets CONTINUE on treasure."""
        intruder = MockIntruder(greed=0.0)
        info = handle_block(intruder, VOXEL_TREASURE, block_state=0)
        assert info.result is InteractionResult.CONTINUE

    def test_treasure_interaction_type_is_grab_treasure(self):
        """COLLECT result on treasure carries interaction_type='grab_treasure'."""
        intruder = MockIntruder(greed=0.5)
        info = handle_block(intruder, VOXEL_TREASURE, block_state=0)
        assert info.interaction_type == "grab_treasure"
